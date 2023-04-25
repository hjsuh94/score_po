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
from score_po.nn import WandbParams, save_module


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
    first_order: bool = True

    save_best_model: Optional[str] = None
    device: str = "cuda"
    torch_optimizer: torch.optim.Optimizer = torch.optim.Adam

    def __init__(self):
        self.wandb_params = WandbParams()

    def load_from_config(self, cfg: DictConfig):
        self.T = cfg.policy.T
        self.x0_upper = torch.Tensor(cfg.policy.x0_upper)
        self.x0_lower = torch.Tensor(cfg.policy.x0_lower)
        self.batch_size = cfg.policy.batch_size
        if isinstance(cfg.policy.std, float):
            self.std = cfg.policy.std
        else:
            raise NotImplementedError("Currently we only support std being a float.")
        self.lr = cfg.policy.lr
        self.max_iters = cfg.policy.max_iters
        if hasattr(cfg.policy, "first_order"):
            self.first_order = cfg.policy.first_order
        self.save_best_model = cfg.policy.save_best_model
        self.load_ckpt = cfg.policy.load_ckpt
        self.device = cfg.policy.device

        self.wandb_params.load_from_config(cfg, field="policy")

    def to_device(self, device):
        self.cost.to(device)
        self.policy = self.policy.to(self.device)
        if isinstance(self.dynamical_system, torch.nn.Module):
            self.dynamical_system.to(device)
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

    def sample_noise_trj_batch(self):
        noise_trj_batch = torch.normal(
            0,
            self.params.std,
            size=(self.params.batch_size, self.params.T, self.ds.dim_u),
        ).to(self.params.device)
        return noise_trj_batch

    def rollout_policy(self, x0, noise_trj):
        """
        Rollout policy, given
            x0: initial condition of shape (dim_x)
            noise_trj: (T, dim_u) noise output on the output of trajectory.
        """
        x0_batch = x0.unsqueeze(0)
        noise_trj_batch = noise_trj.unsqueeze(0)
        x_trj, u_trj = self.rollout_policy_batch(x0_batch, noise_trj_batch)
        return x_trj.squeeze(0), u_trj.squeeze(0)

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

    def evaluate_value_loss_first_order(self, x0_batch, noise_trj_batch):
        """
        We compute a loss so that when autodiffed w.r.t. the policy parameters,
        the resulting gradient gives the first-order gradient.
        This requires the system the cost, system, and policy to be differentiable.
        """
        cost_mean = torch.mean(self.evaluate_cost_batch(x0_batch, noise_trj_batch))
        return cost_mean

    def evaluate_value_loss_zeroth_order(self, x0_batch, noise_trj_batch):
        """
        We compute a loss so that when autodiffed w.r.t. the policy parameters,
        the resulting gradient gives the zeroth-order gradient with Gaussian noise:
        1/N \sum_i^N (1/sigma^2) * Vbar(x_i0, w_it) * [\sum_t (D\pi(x_it, theta)^T w_it)],
        where Vbar(x_i0, w_it) = V(x_i0, w_it) - V(x_i0, 0).
        The notation is as follows,
            - x_i0: the ith sample of the initial state.
            - w_it: the ith sample of the noise trajectory at time t.
            - x_it: the state at time t of rollouts from policy theta, with noise
                    trajectory w_it and initial condition x_i0.
            - u_it: the input at time t of rollouts from policy theta, with noise
                    trajectory w_it and initial condition x_i0.
            - V(x_i0, w_it): cost obtained by policy with initial condition x_i0
                             and noise trajectory w_it
            - V(x_i0, 0): cost obtained by policy with initial condition x_i0 and zero noise.
            - \pi(x_it, theta): action evaluated by a static feedback policy on state x_it,
                                and parametrized by theta.
            - D\pi(x_it, theta): the Jacobian d\pi(x_it, theta)/ dtheta

        For mathematical correctness of this gradient, refer to proof of
        Proposition A.11 from "Do Differentiable Simulators Give Better Policy Gradients?"

        Note that we formulate the loss as
        1/N \sum_i^N (1/sigma^2) * Vbar(x_i0, w_it) * [\sum_t (u_it - w_it)^T w_it],
        after detaching Vbar(x_i0, w_it) and ask autodiff to compute the gradient.
        Since u_it = \pi(x_it, theta) + w_it, this will compute D_theta \pi(x_it, theta).
        """

        if self.params.std <= 0.0:
            raise ValueError("Zeroth order optimizer needs noise to be positive.")

        B = x0_batch.shape[0]
        zero_noise_trj = torch.zeros(B, self.params.T, self.ds.dim_u).to(
            self.params.device
        )

        # 1. Rollout the policy and get costs.
        _, u_trj_batch = self.rollout_policy_batch(x0_batch, noise_trj_batch)
        cost_batch = (
            self.evaluate_cost_batch(x0_batch, noise_trj_batch).clone().detach()
        )
        cost_baseline = (
            self.evaluate_cost_batch(x0_batch, zero_noise_trj).clone().detach()
        )

        # 2. Compute zeroth-order loss
        # We subtract u_trj_batch - noise_trj_batch to obtain \pi(x_it, theta).
        jcb_sum = torch.einsum(
            "btu,btu->b", u_trj_batch - noise_trj_batch, noise_trj_batch
        )
        return torch.mean((cost_batch - cost_baseline) * jcb_sum) / (
            self.params.std**2
        )

    def evaluate_value_loss(self, x0_batch, noise_trj_batch):
        if self.params.first_order:
            return self.evaluate_value_loss_first_order(x0_batch, noise_trj_batch)
        else:
            return self.evaluate_value_loss_zeroth_order(x0_batch, noise_trj_batch)

    def evaluate_policy_loss(self, x0_batch, noise_trj_batch):
        return self.evaluate_value_loss(x0_batch, noise_trj_batch)

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
            noise_trj_batch = self.sample_noise_trj_batch()
            loss = self.evaluate_policy_loss(x0_batch, noise_trj_batch)
            loss.backward()
            optimizer.step()

            cost = torch.mean(self.evaluate_cost_batch(x0_batch, zero_noise_trj))
            self.cost_lst[iter + 1] = cost.item()

            if self.params.wandb_params.enabled:
                wandb.log({"policy_loss": cost.item()})
            if self.params.save_best_model is not None and cost.item() < best_cost:
                model_path = os.path.join(os.getcwd(), self.params.save_best_model)
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                save_module(self.policy, model_path)
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

    def evaluate_score_loss(self, x0_batch, noise_trj_batch):
        """
        We compute a quantity such that when autodiffed w.r.t. θ, we
        obtain the score w.r.t parameters, e.g. computes ∇_θ log p(z). Using the chain rule,
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

    def evaluate_policy_loss(self, x0_batch, noise_trj_batch):
        """
        Given x0_batch, obtain the policy objective.
        """
        value_loss = self.evaluate_value_loss(x0_batch, noise_trj_batch)
        score_loss = self.evaluate_score_loss(x0_batch, noise_trj_batch)
        return value_loss + self.beta * score_loss
