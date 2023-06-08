import os, time
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

from score_po.dynamical_system import (
    DynamicalSystem,
    NNEnsembleDynamicalSystem,
    sim_openloop,
)
from score_po.score_matching import (
    ScoreEstimatorXux,
    ScoreEstimatorXu,
    NoiseConditionedScoreEstimatorXu,
    NoiseConditionedScoreEstimatorXux,
)
from score_po.data_distance import DataDistanceEstimatorXux, DataDistanceEstimatorXu
from score_po.costs import Cost
from score_po.trajectory import Trajectory, IVPTrajectory, BVPTrajectory, SSTrajectory
from score_po.nn import (
    WandbParams,
    save_module,
    tensor_linspace,
    generate_cosine_schedule,
    get_current_sigma,
)


@dataclass
class TrajectoryOptimizerParams:
    cost: Cost
    trj: Trajectory
    T: int
    wandb_params: WandbParams
    ivp: True  # if false, we will assume bvp.
    lr: float = 1e-3
    max_iters: int = 1000
    load_ckpt: Optional[str] = None
    save_best_model: Optional[str] = None
    saving_period: Optional[int] = 100
    device: str = "cuda"
    torch_optimizer: torch.optim.Optimizer = torch.optim.Adam
    verbose: bool = True

    def __init__(self):
        self.wandb_params = WandbParams()

    def load_from_config(self, cfg: DictConfig):
        self.T = cfg.trj.T
        self.lr = cfg.trj.lr
        self.max_iters = cfg.trj.max_iters
        self.save_best_model = cfg.trj.save_best_model
        self.saving_period = cfg.trj.saving_period
        self.load_ckpt = cfg.trj.load_ckpt
        self.device = cfg.trj.device
        self.wandb_params.load_from_config(cfg, field="trj")

    def to_device(self, device):
        self.cost.to(device)
        self.trj.to(self.device)
        self.device = device


class TrajectoryOptimizer:
    def __init__(self, params: TrajectoryOptimizerParams, **kwargs):
        self.params = params
        self.cost = params.cost
        self.trj = params.trj
        self.iter = 0

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.trj.get_full_trajectory()
        cost = self.cost.get_running_cost_batch(x_trj[:-1], u_trj[:]).sum()
        cost += self.cost.get_terminal_cost(x_trj[-1])
        return cost

    def get_penalty_loss(self):
        return 0.0

    def modify_gradients(self):
        pass

    def compute_loss(self):
        return self.get_value_loss()

    def initialize(self, x_trj_guess, u_trj_guess):
        if not isinstance(self.trj, SSTrajectory):
            if x_trj_guess is None:
                if isinstance(self.trj, BVPTrajectory):
                    xnext_trj_init = tensor_linspace(
                        self.trj.x0, self.trj.xT, steps=self.trj.xnext_trj.shape[0]
                    ).T
                else:
                    xnext_trj_init = torch.cat(self.trj.T * [self.trj.x0[None, :]])
                self.trj.xnext_trj = torch.nn.Parameter(xnext_trj_init)
            else:
                self.trj.xnext_trj = torch.nn.Parameter(x_trj_guess[1:])

        if u_trj_guess is None:
            u_trj_init = torch.zeros(self.trj.u_trj.shape).to(self.params.device)
            u_trj_init += torch.randn_like(u_trj_init) * 0.1
            self.trj.u_trj = torch.nn.Parameter(u_trj_init)
        else:
            self.trj.u_trj = torch.nn.Parameter(u_trj_guess)

    def iterate(self, callback=None, x_trj_guess=None, u_trj_guess=None):
        """
        Callback is a function that can be called with signature
        f(self, loss, iter)
        """
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

        self.trj = self.trj.to(self.params.device)
        self.initialize(x_trj_guess, u_trj_guess)
        self.trj.train()
        loss = self.compute_loss()

        start_time = time.time()
        if self.params.verbose:
            print("Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(0, loss, 0))
        self.cost_lst = np.zeros(self.params.max_iters)

        best_cost = np.inf
        # self.cost_lst[0] = loss.item()

        optimizer = self.params.torch_optimizer(
            self.trj.parameters(), lr=self.params.lr
        )

        for iter in range(self.params.max_iters - 1):
            if callback is not None:
                callback(self, loss, iter)
            optimizer.zero_grad()
            loss = self.compute_loss()
            value_loss = self.get_value_loss()
            loss.backward()
            self.modify_gradients()
            optimizer.step()
            self.cost_lst[iter + 1] = loss.item()
            

            if self.params.wandb_params.enabled:
                wandb.log({"trj_loss": loss.item(), "value": value_loss.item()})
            save_model = False
            if loss.item() < best_cost:
                best_cost = loss.item()
                save_model = True
            if (
                self.params.saving_period is not None
                and iter % self.params.saving_period == 0
            ):
                save_model = True
            if self.params.save_best_model and save_model:
                model_path = os.path.join(os.getcwd(), self.params.save_best_model)
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                save_module(self.trj, model_path)

            if self.params.verbose:
                print(
                    "Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(
                        iter + 1, loss.item(), time.time() - start_time
                    )
                )

            self.iter += 1

        return self.cost_lst

    def plot_iterations(self):
        cost_history_np = self.cost_history.clone().detach().numpy()
        plt.figure()
        plt.plot(np.arange(self.params.max_iters), cost_history_np)
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()
        plt.close()


""" TrajectoryOptimizer with First Order + Dircol + SF"""


@dataclass
class TrajectoryOptimizerSFParams(TrajectoryOptimizerParams):
    beta: float = 1.0
    sf: ScoreEstimatorXux = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        self.sf.to(device)


class TrajectoryOptimizerSF(TrajectoryOptimizer):
    def __init__(self, params: TrajectoryOptimizerSFParams, **kwargs):
        super().__init__(params)
        self.sf = params.sf
        assert isinstance(self.sf, ScoreEstimatorXux)

    def modify_gradients(self):
        # Modify value loss by applying score function.
        # TODO: Changing this to batch implementation will
        # result in a much better speedup.
        x_trj, u_trj = self.trj.get_full_trajectory()

        z_trj = torch.cat((x_trj[:-1], u_trj, x_trj[1:]), dim=1)
        sz_trj = self.sf.get_score_z_given_z(z_trj)
        sx_trj, su_trj, sxnext_trj = self.sf.get_xux_from_z(sz_trj)

        weight = -self.params.sf.sigma**2 * self.params.beta

        if isinstance(self.trj, BVPTrajectory):
            self.trj.xnext_trj.grad += weight * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * sxnext_trj[:-1]
            self.trj.u_trj.grad += weight * su_trj
        elif isinstance(self.trj, IVPTrajectory):
            self.trj.xnext_trj.grad[:-1] += weight * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * sxnext_trj
            self.trj.u_trj.grad += weight * su_trj
        else:
            raise ValueError("Must be BVPTrajectory or IVPTrajectory")


""" Variant of TrajectoryOptimizerSF with Variance annealing """


class TrajectoryOptimizerNCSF(TrajectoryOptimizerSF):
    def __init__(self, params: TrajectoryOptimizerSFParams, **kwargs):
        super().__init__(params)
        self.sf = params.sf
        self.sigma_lst = params.sf.sigma_lst
        assert isinstance(self.sf, NoiseConditionedScoreEstimatorXux)

    def modify_gradients(self):
        # Modify value loss by applying score function.
        sigma_idx, sigma = get_current_sigma(
            self.sf.sigma_lst, self.iter, self.params.max_iters
        )
        # weight = -1 / sigma ** 2 * self.params.beta
        weight = -(sigma**2) * self.params.beta
        x_trj, u_trj = self.trj.get_full_trajectory()

        z_trj = torch.cat((x_trj[:-1], u_trj, x_trj[1:]), dim=1)
        sz_trj = self.sf.get_score_z_given_z(z_trj, sigma_idx)
        sx_trj, su_trj, sxnext_trj = self.sf.get_xux_from_z(sz_trj)

        if isinstance(self.trj, BVPTrajectory):
            self.trj.xnext_trj.grad += weight * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * sxnext_trj[:-1]
            self.trj.u_trj.grad += weight * su_trj
        elif isinstance(self.trj, IVPTrajectory):
            self.trj.xnext_trj.grad[:-1] += weight * sx_trj[1:]
            self.trj.xnext_trj.grad += weight * sxnext_trj
            self.trj.u_trj.grad += weight * su_trj
        else:
            raise ValueError("Must be BVPTrajectory or IVPTrajectory")


""" TrajectoryOptimizer with First Order + Dircol + DDE"""


@dataclass
class TrajectoryOptimizerDDEParams(TrajectoryOptimizerParams):
    beta: float = 1.0
    dde: DataDistanceEstimatorXux = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        self.dde.to(device)


class TrajectoryOptimizerDDE(TrajectoryOptimizer):
    def __init__(self, params: TrajectoryOptimizerDDEParams, **kwargs):
        super().__init__(params)
        self.dde = params.dde
        assert isinstance(self.dde, DataDistanceEstimatorXux)

    def get_penalty_loss(self):
        x_trj, u_trj = self.trj.get_full_trajectory()
        z_trj = torch.cat((x_trj[:-1], u_trj, x_trj[1:]), dim=1)
        return self.dde.get_energy_to_data(z_trj).sum()

    def compute_loss(self):
        return self.get_value_loss() + self.params.beta * self.get_penalty_loss()


""" TrajectoryOptimizer with First Order + SS + SF"""


@dataclass
class TrajectoryOptimizerSSParams(TrajectoryOptimizerParams):
    beta: float = 1.0
    beta_min: float = 1.0
    sf: ScoreEstimatorXu = None
    ds: DynamicalSystem = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta
        if hasattr(cfg.trj, "beta_min"):
            self.beta_min = cfg.trj.beta_min
        else:
            self.beta_min = self.beta

    def to_device(self, device):
        super().to_device(device)
        self.sf.to(device)
        if isinstance(self.ds, torch.nn.Module):
            self.ds.to(device)


class TrajectoryOptimizerSS(TrajectoryOptimizer):
    def __init__(self, params: TrajectoryOptimizerSSParams, **kwargs):
        super().__init__(params)
        self.sf = params.sf
        self.ds = params.ds
        assert isinstance(self.trj, SSTrajectory)
        assert isinstance(self.sf, ScoreEstimatorXu)

        self.beta_schedule = generate_cosine_schedule(
            self.params.beta, self.params.beta_min, self.params.max_iters
        )

    def initialize(self, x_trj_guess, u_trj_guess):
        if u_trj_guess is None:
            u_trj_init = torch.zeros(self.trj.u_trj.shape).to(self.params.device)
            u_trj_init += torch.randn_like(u_trj_init) * 0.0
        else:
            u_trj_init = u_trj_guess
        self.trj.u_trj = torch.nn.Parameter(u_trj_init)

    def rollout_trajectory(self):
        """
        Rollout policy in batch, given
            x0_batch: initial condition of shape (B, dim_x)
            noise_trj_batch: (B, T, dim_u) noise output on the output of trajectory.
        """
        x_trj = sim_openloop(self.ds, self.trj.x0, self.trj.u_trj, noise_trj=None)
        return x_trj, self.trj.u_trj

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.rollout_trajectory()
        cost = self.cost.get_running_cost_batch(x_trj[:-1], u_trj[:]).sum()
        cost += self.cost.get_terminal_cost(x_trj[-1])
        return cost

    def get_penalty_loss(self):
        """
        We compute a quantity such that when autodiffed w.r.t. θ, we
        obtain the score w.r.t parameters, e.g. computes ∇_θ log p(z). Using the chain rule,
        we break down the gradient into ∇_θ log p(z) = ∇_z log p(z) *  ∇_θ z.
        Since we don't want to compute ∇_θ z manually, we calculate the quantity
        (∇_z log p(z) * z) and and detach ∇_z log p(z) from the computation graph
        so that ∇_θ(∇_z log p(z) * z) = ∇_z log p(z) * ∇_θ z.
        """
        x_trj, u_trj = self.rollout_trajectory()
        z_trj = torch.cat((x_trj[:-1], u_trj), dim=1)

        sz_trj = self.sf.get_score_z_given_z(z_trj)

        # Here, ∇_z log p(z) is detached from the computation graph so that we ignore
        # terms related to ∂(∇_z log p(z))/∂θ.
        sz_trj = sz_trj.clone().detach()

        score = torch.einsum("ti,ti->t", z_trj, sz_trj).sum()
        return -score

    def compute_loss(self):
        beta = self.beta_schedule[self.iter]
        return self.get_value_loss() + beta * self.get_penalty_loss()


""" Variant of TrajectoryOptimizerSS with Variance annealing """


class TrajectoryOptimizerNCSS(TrajectoryOptimizerSS):
    def __init__(self, params: TrajectoryOptimizerSFParams, **kwargs):
        super().__init__(params)
        self.sigma_lst = params.sf.sigma_lst
        assert isinstance(self.sf, NoiseConditionedScoreEstimatorXu)

    def get_penalty_loss(self):
        """
        We compute a quantity such that when autodiffed w.r.t. θ, we
        obtain the score w.r.t parameters, e.g. computes ∇_θ log p(z). Using the chain rule,
        we break down the gradient into ∇_θ log p(z) = ∇_z log p(z) *  ∇_θ z.
        Since we don't want to compute ∇_θ z manually, we calculate the quantity
        (∇_z log p(z) * z) and and detach ∇_z log p(z) from the computation graph
        so that ∇_θ(∇_z log p(z) * z) = ∇_z log p(z) * ∇_θ z.
        """
        idx, sigma = get_current_sigma(
            self.sf.sigma_lst, self.iter, self.params.max_iters
        )
        x_trj, u_trj = self.rollout_trajectory()
        z_trj = torch.cat((x_trj[:-1], u_trj), dim=1)

        sz_trj = self.sf.get_score_z_given_z(z_trj, idx)

        # Here, ∇_z log p(z) is detached from the computation graph so that we ignore
        # terms related to ∂(∇_z log p(z))/∂θ.
        sz_trj = sz_trj.clone().detach()

        score = torch.einsum("ti,ti->t", z_trj, sz_trj).sum()
        return -score


@dataclass
class TrajectoryOptimizerDirTranParams(TrajectoryOptimizerParams):
    alpha: float = 100.0
    beta: float = 1.0
    sf: ScoreEstimatorXu = None
    ds: DynamicalSystem = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.alpha = cfg.trj.alpha
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        self.sf.to(device)
        if isinstance(self.ds, torch.nn.Module):
            self.ds.to(device)


class TrajectoryOptimizerDirTran(TrajectoryOptimizer):
    def __init__(self, params: TrajectoryOptimizerDirTranParams, **kwargs):
        super().__init__(params)
        self.sf = params.sf
        self.ds = params.ds
        assert isinstance(self.sf, ScoreEstimatorXu)
        assert not isinstance(self.trj, SSTrajectory)

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.trj.get_full_trajectory()
        cost = self.cost.get_running_cost_batch(x_trj[:-1], u_trj[:]).sum()
        cost += self.cost.get_terminal_cost(x_trj[-1])
        return cost

    def get_dynamics_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.trj.get_full_trajectory()

        # Compute one-step rollouts.
        # T x dim_x
        xnext_true = self.ds.dynamics_batch(x_trj[:-1], u_trj)
        error = xnext_true - x_trj[1:]
        x_weights = 1 / self.ds.x_normalizer.k
        return torch.einsum("Tx,x,Tx->T", error, x_weights, error).sum()

    def modify_gradients(self):
        # Modify value loss by applying score function.
        # TODO: Changing this to batch implementation will
        # result in a much better speedup.
        x_trj, u_trj = self.trj.get_full_trajectory()

        z_trj = torch.cat((x_trj[:-1], u_trj), dim=1)
        sz_trj = self.sf.get_score_z_given_z(z_trj)
        sx_trj, su_trj = self.sf.get_xu_from_z(sz_trj)

        weight = -self.params.sf.sigma**2 * self.params.beta

        if isinstance(self.trj, BVPTrajectory):
            self.trj.xnext_trj.grad += weight * sx_trj[1:]
            self.trj.u_trj.grad += weight * su_trj
        elif isinstance(self.trj, IVPTrajectory):
            self.trj.xnext_trj.grad[:-1] += weight * sx_trj[1:]
            self.trj.u_trj.grad += weight * su_trj
        else:
            raise ValueError("Must be BVPTrajectory or IVPTrajectory")

    def compute_loss(self):
        # alpha = self.params.alpha * (self.iter / self.params.max_iters)
        return self.get_value_loss() + self.params.alpha * self.get_dynamics_loss()


class TrajectoryOptimizerNCDirTran(TrajectoryOptimizerDirTran):
    def __init__(self, params: TrajectoryOptimizerDirTranParams, **kwargs):
        super().__init__(params)
        self.sigma_lst = params.sf.sigma_lst
        assert isinstance(self.sf, NoiseConditionedScoreEstimatorXu)

    def modify_gradients(self):
        # Modify value loss by applying score function.
        sigma_idx, sigma = get_current_sigma(
            self.sf.sigma_lst, self.iter, self.params.max_iters
        )
        # weight = -1 / sigma ** 2 * self.params.beta
        weight = -(sigma**2) * self.params.beta
        x_trj, u_trj = self.trj.get_full_trajectory()

        z_trj = torch.cat((x_trj[:-1], u_trj), dim=1)
        sz_trj = self.sf.get_score_z_given_z(z_trj, sigma_idx)
        sx_trj, su_trj = self.sf.get_xu_from_z(sz_trj)

        if isinstance(self.trj, BVPTrajectory):
            self.trj.xnext_trj.grad += weight * sx_trj[1:]
            self.trj.u_trj.grad += weight * su_trj
        elif isinstance(self.trj, IVPTrajectory):
            self.trj.xnext_trj.grad[:-1] += weight * sx_trj[1:]
            self.trj.u_trj.grad += weight * su_trj
        else:
            raise ValueError("Must be BVPTrajectory or IVPTrajectory")


@dataclass
class TrajectoryOptimizerSSEnsembleParams(TrajectoryOptimizerParams):
    beta: float = 1.0
    ds: DynamicalSystem = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        if isinstance(self.ds, torch.nn.Module):
            self.ds.to(device)


class TrajectoryOptimizerSSEnsemble(TrajectoryOptimizer):
    def __init__(self, params: TrajectoryOptimizerSSEnsembleParams, **kwargs):
        super().__init__(params)
        self.ds = params.ds
        self.K = len(self.ds.ds_lst)
        assert isinstance(self.trj, SSTrajectory)
        assert isinstance(self.ds, NNEnsembleDynamicalSystem)

    def initialize(self, x_trj_guess=None, u_trj_guess=None):
        if u_trj_guess is None:
            u_trj_init = torch.zeros(self.trj.u_trj.shape).to(self.params.device)
        else:
            u_trj_init = u_trj_guess.to(self.device)
        self.trj.u_trj = torch.nn.Parameter(u_trj_init)

    def rollout_trajectory(self):
        """
        Rollout trajectory with ensembles.
        """
        x_trj = torch.zeros((self.K, self.trj.T + 1, self.ds.dim_x)).to(
            self.params.device
        )
        u_trj = torch.cat(self.K * [self.trj.u_trj[None, :, :]])
        x_trj[:, 0, :] = self.trj.x0

        for t in range(self.trj.T):
            x_trj[:, t + 1, :] = self.ds.dynamics_batch(
                x_trj[:, t, :][:, None, :], u_trj[:, t, :][:, None, :]
            )[:, 0, :]

        return x_trj, u_trj

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.rollout_trajectory()
        cost = torch.zeros(self.K).to(self.params.device)
        for k in range(self.K):
            cost[k] += self.cost.get_running_cost_batch(
                x_trj[k, :-1, :], u_trj[k, :, :]
            ).sum()
            cost[k] += self.cost.get_terminal_cost(x_trj[k, -1, :])
        return cost.mean()

    def get_penalty_loss(self):
        x_trj, u_trj = self.rollout_trajectory()
        cost = torch.zeros(self.K).to(self.params.device)
        for k in range(self.K):
            cost[k] += self.cost.get_running_cost_batch(
                x_trj[k, :-1, :], u_trj[k, :, :]
            ).sum()
            cost[k] += self.cost.get_terminal_cost(x_trj[k, -1, :])
        return cost.var()

    def compute_loss(self):
        return self.get_value_loss() + self.params.beta * self.get_penalty_loss()


@dataclass
class CEMParams(TrajectoryOptimizerParams):
    n_elite: int = 3
    batch_size: int = 10
    std: float = 0.05
    ds: DynamicalSystem = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        if hasattr(cfg.trj, "n_elite"):
            self.n_elite = cfg.trj.n_elite
        if hasattr(cfg.trj, "batch_size"):
            self.batch_size = cfg.trj.batch_size
        self.std = cfg.trj.std

    def to_device(self, device):
        super().to_device(device)


class CEM(TrajectoryOptimizer):
    params: CEMParams
    ds: DynamicalSystem

    def __init__(self, params: CEMParams, **kwargs):
        super().__init__(params)
        self.ds = params.ds

    def initialize(self):
        if self.trj is None:
            u_trj_init = torch.zeros(self.trj.u_trj.shape).to(self.params.device)
            u_trj_init += torch.randn_like(u_trj_init) * 0.0
            self.trj.u_trj = torch.nn.Parameter(u_trj_init)

    def rollout_trajectory(self):
        """
        Rollout policy in batch, given
            x0_batch: initial condition of shape (B, dim_x)
            noise_trj_batch: (B, T, dim_u) noise output on the output of trajectory.
        """
        x_trj = sim_openloop(self.ds, self.trj.x0, self.trj.u_trj, noise_trj=None)
        return x_trj, self.trj.u_trj

    def _get_value_loss(self, x_trj, u_trj):
        cost = self.cost.get_running_cost_batch(x_trj[:-1], u_trj[:]).sum()
        cost += self.cost.get_terminal_cost(x_trj[-1])
        return cost

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.rollout_trajectory()
        return self._get_value_loss(x_trj, u_trj)

    def compute_loss(self):
        return self.get_value_loss()

    def update_trajectory(self):
        u_trj_batch = self.trj.u_trj.unsqueeze(0).repeat(
            [self.params.batch_size] + [1] * self.trj.u_trj.ndim
        )
        u_trj_batch += torch.randn_like(u_trj_batch) * self.params.std
        cost_lst = torch.zeros(self.params.batch_size).to(self.params.device)
        for b in range(self.params.batch_size):
            self.trj.u_trj = torch.nn.Parameter(u_trj_batch[b])
            cost_lst[b] = self.compute_loss()
        _, ind = torch.topk(cost_lst, self.params.n_elite, largest=False)
        next_trj = u_trj_batch[ind].mean(dim=0)
        self.trj.u_trj = torch.nn.Parameter(next_trj)

    def compute_loss(self):
        self.get_value_loss() + self.params.beta * self.get_penalty_loss()

    def iterate(self, callback=None):
        """
        Callback is a function that can be called with signature
        f(self, loss, iter)
        """
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

        self.trj = self.trj.to(self.params.device)
        self.initialize()
        self.trj.train()

        loss = self.compute_loss()

        start_time = time.time()
        print("Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(0, loss, 0))
        self.cost_lst = np.zeros(self.params.max_iters)
        self.cost_lst[0] = loss.item()

        for iter in range(self.params.max_iters - 1):
            if callback is not None:
                callback(self, loss.item(), iter)

            self.update_trajectory()
            value = self.get_value_loss()
            loss = self.compute_loss()
            self.cost_lst[iter + 1] = loss.item()

            if self.params.wandb_params.enabled:
                wandb.log({"trj_loss": loss.item(), "value": value.item()})
            if (
                self.params.saving_period is not None
                and iter % self.params.saving_period == 0
            ):
                model_path = os.path.join(os.getcwd(), self.params.save_best_model)
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                save_module(self.trj, model_path)

            print(
                "Iteration: {:04d} | Cost: {:.3f} | Time: {:.3f}".format(
                    iter + 1, loss.item(), time.time() - start_time
                )
            )

            self.iter += 1

        return self.cost_lst


@dataclass
class CEMEnsembleParams(CEMParams):
    beta: float = 1.0
    ds: DynamicalSystem = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        if isinstance(self.ds, torch.nn.Module):
            self.ds.to(device)


class CEMEnsemble(CEM, TrajectoryOptimizerSSEnsemble):
    # Both CEM and TrajectoryOptimizerSSEnsemble define iterate() function, but we
    # want to use CEM's iterate function, so we put CEM before
    # TrajectoryOptimizerSSEnsemble
    def __init__(self, params: CEMEnsembleParams, **kwargs):
        super().__init__(params)

    def rollout_trajectory(self):
        """
        Rollout trajectory with ensembles.
        """
        x_trj = torch.zeros((self.K, self.trj.T + 1, self.ds.dim_x)).to(
            self.params.device
        )
        u_trj = torch.cat(self.K * [self.trj.u_trj[None, :, :]])
        x_trj[:, 0, :] = self.trj.x0

        for t in range(self.trj.T):
            x_trj[:, t + 1, :] = self.ds.dynamics_batch(
                x_trj[:, t, :][:, None, :], u_trj[:, t, :][:, None, :]
            )[:, 0, :]

        return x_trj, u_trj

    def get_value_loss(self):
        # Loop over trajectories to compute reward loss.
        x_trj, u_trj = self.rollout_trajectory()
        cost = torch.zeros(self.K).to(self.params.device)
        for k in range(self.K):
            cost[k] += self.cost.get_running_cost_batch(
                x_trj[k, :-1, :], u_trj[k, :, :]
            ).sum()
            cost[k] += self.cost.get_terminal_cost(x_trj[k, -1, :])
        return cost.mean()

    def get_penalty_loss(self):
        x_trj, u_trj = self.rollout_trajectory()
        cost = torch.zeros(self.K).to(self.params.device)
        for k in range(self.K):
            cost[k] += self.cost.get_running_cost_batch(
                x_trj[k, :-1, :], u_trj[k, :, :]
            ).sum()
            cost[k] += self.cost.get_terminal_cost(x_trj[k, -1, :])
        return cost.var()

    def compute_loss(self):
        return self.get_value_loss() + self.params.beta * self.get_penalty_loss()

    def update_trajectory(self):
        u_trj_batch = self.trj.u_trj.unsqueeze(0).repeat(
            [self.params.batch_size] + [1] * self.trj.u_trj.ndim
        )
        u_trj_batch += torch.randn_like(u_trj_batch) * self.params.std
        cost_lst = torch.zeros(self.params.batch_size).to(self.params.device)
        for b in range(self.params.batch_size):
            self.trj.u_trj = torch.nn.Parameter(u_trj_batch[b])
            cost_lst[b] = self.compute_loss()
        _, ind = torch.topk(cost_lst, self.params.n_elite, largest=False)
        next_trj = u_trj_batch[ind].mean(dim=0)
        self.trj.u_trj = torch.nn.Parameter(next_trj)


@dataclass
class CEMDataDistanceEstimatorParams(CEMParams):
    beta: float = 1.0
    dde: DataDistanceEstimatorXu = None

    def __init__(self):
        super().__init__()

    def load_from_config(self, cfg: DictConfig):
        super().load_from_config(cfg)
        self.beta = cfg.trj.beta

    def to_device(self, device):
        super().to_device(device)
        if isinstance(self.ds, torch.nn.Module):
            self.dde.to(device)


class CEMDataDistanceEstimator(CEM):
    def __init__(self, params: CEMDataDistanceEstimatorParams, **kwargs):
        super().__init__(params)
        self.dde = params.dde

    def _get_penalty_loss(self, x_trj, u_trj):
        z_trj = torch.cat((x_trj[:-1], u_trj), dim=1)
        return self.dde.get_energy_to_data(z_trj).sum()

    def compute_loss(self):
        x_trj, u_trj = self.rollout_trajectory()
        return self._get_value_loss(
            x_trj, u_trj
        ) + self.params.beta * self._get_penalty_loss(x_trj, u_trj)
