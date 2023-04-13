import os, time
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from score_po.score_matching import ScoreEstimator
from score_po.costs import Cost
from score_po.dynamical_system import DynamicalSystem
from score_po.policy import Policy
from score_po.nn import WandbParams


@dataclass
class PolicyOptimizerParams:
    cost: Cost
    dynamical_system: DynamicalSystem
    policy: Policy
    T: int
    x0_upper: torch.Tensor
    x0_lower: torch.Tensor
    batch_size: int
    std: float  # TODO change to torch.Tensor in the future.
    lr: float
    max_iters: int
    wandb_params: WandbParams

    save_best_model: Optional[str] = None
    device: str = "cuda"
    torch_optimizer: torch.optim.Optimizer = torch.optim.Adam

    def __init__(self):
        self.wandb_params = WandbParams()

    def load_from_config(self, cfg: DictConfig):
        self.T = cfg.policy.T
        self.x0_upper = torch.Tensor(cfg.policy.x0_upper).to(cfg.policy.device)
        self.x0_lower = torch.Tensor(cfg.policy.x0_lower).to(cfg.policy.device)
        self.batch_size = cfg.policy.batch_size
        if isinstance(cfg.policy.std, float):
            self.std = cfg.policy.std
        else:
            raise NotImplementedError("Currently we only support std being a float.")
        self.lr = cfg.policy.lr
        self.max_iters = cfg.policy.max_iters
        self.save_best_model = cfg.policy.save_best_model
        self.load_ckpt = cfg.policy.load_ckpt
        self.device = cfg.policy.device

        self.wandb_params.load_from_config(cfg, field="policy")

    def to_device(self, device):
        self.cost.params_to_device(device)
        self.policy = self.policy.to(self.device)
        self.x0_upper = self.x0_upper.to(device)
        self.x0_lower = self.x0_lower.to(device)
        self.device = device


class PolicyOptimizer:
    def __init__(self, params: PolicyOptimizerParams, **kwargs):
        self.params = params
        self.cost = params.cost
        self.ds = params.dynamical_system
        self.policy = params.policy

    def sample_initial_state_batch(self):
        """
        Given some batch size, sample initial states (B, dim_x).
        """
        initial = torch.rand(
            self.params.batch_size, self.ds.dim_x, device=self.params.device
        )
        initial = (
            self.params.x0_upper - self.params.x0_lower
        ) * initial + self.params.x0_lower
        return initial.to(self.params.device)

    def rollout_policy(self, x0, noise_trj):
        """
        Rollout policy, given
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj = torch.zeros((self.params.T + 1, self.ds.dim_x)).to(self.params.device)
        u_trj = torch.zeros((self.params.T, self.ds.dim_u)).to(self.params.device)
        x_trj[0] = x0.to(self.params.device)

        for t in range(self.params.T):
            # we assume both dynamics and actions silently handle device
            # depending on which device x_trj and u_trj were on.
            u_trj[t] = self.policy(x_trj[t], t) + noise_trj[t]
            x_trj[t + 1] = self.ds.dynamics(x_trj[t], u_trj[t])
        return x_trj, u_trj

    def rollout_policy_batch(
        self, x0_batch: torch.Tensor, noise_trj_batch: torch.Tensor
    ):
        """
        Rollout policy in batch, given
            x0_batch: initial condition of shape (B, dim_x)
            noise_trj_batch: (B, T, dim_u) noise output on the output of trajectory.
        """
        B = x0_batch.shape[0]
        x0_batch = x0_batch.to(self.params.device)
        x_trj_batch = torch.zeros((B, 0, self.ds.dim_x)).to(self.params.device)
        u_trj_batch = torch.zeros((B, 0, self.ds.dim_u)).to(self.params.device)
        x_trj_batch = torch.hstack((x_trj_batch, x0_batch[:, None, :]))

        for t in range(self.params.T):
            u_t_batch = self.policy(x_trj_batch[:, t, :], t) + noise_trj_batch[:, t, :]
            u_trj_batch = torch.hstack((u_trj_batch, u_t_batch[:, None, :]))
            x_next_batch = self.ds.dynamics_batch(
                x_trj_batch[:, t, :], u_trj_batch[:, t, :]
            )
            x_trj_batch = torch.hstack((x_trj_batch, x_next_batch[:, None, :]))

        return x_trj_batch, u_trj_batch

    def evaluate_cost(self, x0, noise_trj):
        """
        Evaluate the cost given:
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj, u_trj = self.rollout_policy(x0, noise_trj)
        cost = 0.0
        for t in range(self.T):
            cost += self.cost.get_running_cost(x_trj[t], u_trj[t])
        cost += self.cost.get_terminal_cost(x_trj[self.params.T])
        return cost

    def evaluate_cost_batch(
        self, x0_batch: torch.Tensor, noise_trj_batch: torch.Tensor
    ):
        """
        Evaluate the cost given:
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x_trj_batch, u_trj_batch = self.rollout_policy_batch(x0_batch, noise_trj_batch)
        B = x_trj_batch.shape[0]
        cost = torch.zeros(B).to(x0_batch.device)
        for t in range(self.params.T):
            cost += self.cost.get_running_cost_batch(
                x_trj_batch[:, t, :], u_trj_batch[:, t, :]
            )
        cost += self.cost.get_terminal_cost_batch(x_trj_batch[:, self.params.T, :])
        return cost

    def evaluate_policy_cost(self, x0_batch):
        """
        Given x0_batch, obtain the policy objective.
        """
        B = x0_batch.shape[0]
        noise_trj_batch = torch.normal(
            0, self.params.std, size=(B, self.params.T, self.ds.dim_u)
        ).to(self.params.device)

        cost_mean = torch.mean(self.evaluate_cost_batch(x0_batch, noise_trj_batch))
        return cost_mean

    def get_value_gradient(self, x0_batch, policy_params):
        """
        Obtain a batch of x0_batch and the currnet policy parameters,
        obtain gradients of the cost w.r.t policy.
        """
        raise NotImplementedError("this function is virtual.")

    def get_policy_gradient(self, x0_batch, policy_params):
        """
        By default, policy gradient equals value gradient.
        """
        return self.get_value_gradient(x0_batch, policy_params)

    def iterate(self):
        if self.params.wandb_params.enabled:
            if self.params.wandb_params.dir is not None and not os.path.exists(
                self.params.wandb_params.dir
            ):
                self.os.makedirs(self.params.wandb_params.dir, exist_ok=True)
            wandb.init(
                project=self.params.wandb_params.project,
                name=self.params.wandb_params.name,
                dir=self.params.wandb_params.dir,
                config=self.params.wandb_params.config,
                entity=self.params.wandb_params.entity,
            )

        zero_noise_trj = torch.zeros(
            (self.params.batch_size, self.params.T, self.ds.dim_u)
        ).to(self.params.device)
        self.policy = self.policy.to(self.params.device)
        self.policy.train()

        x0_batch = self.sample_initial_state_batch()
        cost = torch.mean(self.evaluate_cost_batch(x0_batch, zero_noise_trj))

        start_time = time.time()
        print("Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(0, cost, 0))
        self.cost_lst = np.zeros(self.params.max_iters)

        best_cost = np.inf
        self.cost_lst[0] = cost

        optimizer = self.params.torch_optimizer(
            self.policy.parameters(), lr=self.params.lr
        )

        for iter in range(self.params.max_iters - 1):
            optimizer.zero_grad()
            x0_batch = self.sample_initial_state_batch()
            cost_mean = self.evaluate_policy_cost(x0_batch)
            cost_mean.backward()
            optimizer.step()

            cost = self.evaluate_policy_cost(x0_batch)
            self.cost_lst[iter + 1] = cost.item()

            if self.params.wandb_params.enabled:
                wandb.log({"policy_loss": cost.item()})
            if self.params.save_best_model is not None and cost.item() < best_cost:
                model_path = os.path.join(os.getcwd(), self.params.save_best_model)
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                self.policy.save_parameters(model_path)
                best_cost = cost.item()

            print(
                "Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(
                    iter + 1, cost.item(), time.time() - start_time
                )
            )

        return self.cost_lst

    def plot_iterations(self):
        cost_history_np = self.cost_history.clone().detach().numpy()
        plt.figure()
        plt.plot(np.arange(self.params.max_iters), cost_history_np)
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
        plt.close()


@dataclass
class DRiskScorePolicyOptimizerParams(PolicyOptimizerParams):
    beta: float = 1.0
    sf: ScoreEstimator = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.policy.beta


class DRiskScorePolicyOptimizer(PolicyOptimizer):
    def __init__(self, params: DRiskScorePolicyOptimizerParams, **kwargs):
        super().__init__(params=params, **kwargs)
        self.beta = params.beta
        self.sf = params.sf

    def get_score_cost(self, x0_batch, noise_trj_batch):
        """
        Here, our goal is to compute a quantity such that when autodiffed w.r.t. θ, we
        obtain the score w.r.t parameters, compute ∇_θ log p(z). Using the chain rule,
        we break down the gradient into ∇_θ log p(z) = ∇_z log p(z) *  ∇_θ z.
        Since we don't want to compute ∇_θ z manually, we calculate the quantity
        (∇_z log p(z) * z) and and detach ∇_z log p(z) from the computation graph
        so that ∇_θ(∇_z log p(z) * z) = ∇_z log p(z) * ∇_θ z.
        """
        B = x0_batch.shape[0]

        x_trj_batch, u_trj_batch = self.rollout_policy_batch(x0_batch, noise_trj_batch)
        z_trj_batch = torch.cat((x_trj_batch[:, :-1, :], u_trj_batch), dim=2)

        sz_trj_batch = torch.zeros(B, self.params.T, self.ds.dim_x + self.ds.dim_u).to(
            self.params.device
        )

        for t in range(self.params.T):
            zt_batch = z_trj_batch[:, t, :]
            sz_trj_batch[:, t, :] = self.sf.get_score_z_given_z(zt_batch)

        # Here, ∇_z log p(z) is detached from the computation graph so that we ignore
        # terms related to ∂(∇_z log p(z))/∂θ.
        sz_trj_batch = sz_trj_batch.clone().detach()

        score = (
            torch.einsum("bti,bti->bt", z_trj_batch, sz_trj_batch)
            .sum(dim=-1)
            .mean(dim=0)
        )

        return -score

    def evaluate_policy_cost(self, x0_batch):
        """
        Given x0_batch, obtain the policy objective.
        """
        B = x0_batch.shape[0]
        noise_trj_batch = torch.normal(
            0, self.params.std, size=(B, self.params.T, self.ds.dim_u)
        ).to(self.params.device)

        cost_mean = torch.mean(self.evaluate_cost_batch(x0_batch, noise_trj_batch))
        score_cost = self.get_score_cost(x0_batch, noise_trj_batch)
        return cost_mean + self.beta * score_cost
