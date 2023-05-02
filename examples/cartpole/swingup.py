from omegaconf import DictConfig, OmegaConf
import os
from typing import Optional, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpoleNNDynamicalSystem, CartpolePlant
from score_po.dynamical_system import DynamicalSystem, sim_openloop
from score_po.nn import AdamOptimizerParams, WandbParams, Normalizer
from score_po.trajectory import SSTrajectory, BVPTrajectory
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.trajectory_optimizer import (
    TrajectoryOptimizerSF,
    TrajectoryOptimizerSFParams,
    TrajectoryOptimizerSSParams,
    TrajectoryOptimizerSS,
)
from score_po.score_matching import ScoreEstimatorXu, ScoreEstimatorXux
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy, Clamper
from examples.cartpole.score_training import get_score_network


def plot_result(
    traj_optimizer: Union[TrajectoryOptimizerSS, TrajectoryOptimizerSF],
    plant: CartpolePlant,
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    u_max: float,
    dt: float,
):
    device = traj_optimizer.params.device
    u_trj = traj_optimizer.trj.u_trj.data
    x0 = torch.zeros((4,), device=device)
    if isinstance(traj_optimizer, TrajectoryOptimizerSS):
        # Simulate with the optimizer dynamics.
        x_trj_plan = sim_openloop(traj_optimizer.ds, x0, u_trj, None)
    elif isinstance(traj_optimizer, TrajectoryOptimizerSF):
        x_trj_plan, u_trj = traj_optimizer.trj.get_full_trajectory()
    # Simulate with the true dynamics.
    x_trj_sim = sim_openloop(plant, x0, u_trj, None)
    x_trj_plan_np = x_trj_plan.cpu().detach().numpy()
    x_trj_sim_np = x_trj_sim.cpu().detach().numpy()
    u_trj_np = u_trj.cpu().detach().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(x_trj_plan_np[:, 0], x_trj_plan_np[:, 1], label="plan", color="b")
    ax1.plot(x_trj_sim_np[:, 0], x_trj_sim_np[:, 1], label="sim", color="g")
    ax1.plot([0], [np.pi], "*", color="r")
    x_lo_np = x_lo.cpu().detach().numpy()
    x_up_np = x_up.cpu().detach().numpy()
    ax1.plot(
        [x_lo_np[0], x_up_np[0], x_up_np[0], x_lo_np[0], x_lo_np[0]],
        [x_lo_np[1], x_lo_np[1], x_up_np[1], x_up_np[1], x_lo_np[1]],
        linestyle="--",
        color="r",
    )
    ax1.legend()
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel(r"$\theta$")
    beta_val = 0
    beta_val = traj_optimizer.params.beta
    ax1.set_title(r"$\beta$" + f"={beta_val}")

    ax2 = fig.add_subplot(122)
    ax2.plot(x_trj_plan_np[:, 2], x_trj_plan_np[:, 3], label="plan", color="b")
    ax2.plot(x_trj_sim_np[:, 2], x_trj_sim_np[:, 3], label="sim", color="g")
    ax2.plot(
        [x_lo_np[2], x_up_np[2], x_up_np[2], x_lo_np[2], x_lo_np[2]],
        [x_lo_np[3], x_lo_np[3], x_up_np[3], x_up_np[3], x_lo_np[3]],
        linestyle="--",
        color="r",
    )
    ax2.plot([0], [0], "*", color="r")
    ax2.legend()
    ax2.set_xlabel(r"$\dot{x}$")
    ax2.set_ylabel(r"$\dot{\theta}$")
    ax2.set_title(r"$\beta$" + f"={beta_val}")

    fig.tight_layout()

    fig.savefig(
        os.path.join(os.getcwd(), f"swingup_state_beta{beta_val}.png"),
        format="png",
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.arange(u_trj_np.shape[0]) * dt, u_trj_np)
    ax.plot([0, dt * u_trj_np.shape[0]], [u_max, u_max], linestyle="--", color="r")
    ax.plot([0, dt * u_trj_np.shape[0]], [-u_max, -u_max], linestyle="--", color="r")
    ax.set_xlabel("time")
    ax.set_ylabel("u")
    ax.set_title(r"$\beta$" + f"={beta_val}")
    fig.savefig(
        os.path.join(os.getcwd(), f"swingup_u_beta{beta_val}.png"),
        format="png",
    )


def single_shooting(
    dynamical_system: CartpoleNNDynamicalSystem,
    x0: torch.Tensor,
    score_estimator: ScoreEstimatorXu,
    cfg: DictConfig,
):
    """
    Find a swing up trajectory by single shooting method.

    return:
      u_traj: Of shape (T-1, 1). u_traj[i]
    """
    device = x0.device
    params = TrajectoryOptimizerSSParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), params.save_best_model)
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4)),
        R=torch.tensor([[0.000]]),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1])),
        xd=torch.tensor([0, np.pi, 0, 0]),
    )
    params.trj = SSTrajectory(
        dim_x=dynamical_system.dim_x,
        dim_u=dynamical_system.dim_u,
        T=params.T,
        x0=x0,
    )
    params.ivp = True
    params.sf = score_estimator
    params.ds = dynamical_system
    if cfg.trj.load_ckpt is not None:
        params.trj.load_state_dict(torch.load(cfg.trj.load_ckpt))

    params.to_device(device)

    traj_optimizer = TrajectoryOptimizerSS(params)
    traj_optimizer.iterate()
    return traj_optimizer


def dircol(
    x0: torch.Tensor,
    score_estimator: ScoreEstimatorXux,
    cfg: DictConfig,
):
    device = x0.device
    params = TrajectoryOptimizerSFParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), params.save_best_model)
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4)),
        R=torch.tensor([[0.000]]),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1])),
        xd=torch.tensor([0, np.pi, 0, 0]),
    )
    params.trj = BVPTrajectory(
        dim_x=4,
        dim_u=1,
        T=params.T,
        x0=x0,
        xT=torch.tensor([0, np.pi, 0, 0], device=device),
    )
    params.ivp = False
    params.sf = score_estimator
    if cfg.trj.load_ckpt is not None:
        params.trj.load_state_dict(torch.load(cfg.trj.load_ckpt))

    params.to_device(device)

    traj_optimizer = TrajectoryOptimizerSF(params)
    traj_optimizer.iterate()
    return traj_optimizer


@hydra.main(config_path="./config", config_name="swingup")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    device = cfg.device
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.plant_param.hidden_layers,
        device=device,
    )

    nn_plant.load_state_dict(torch.load(cfg.dynamics_load_ckpt))
    plant = CartpolePlant(dt=cfg.plant_param.dt)

    score_network = get_score_network(xu=cfg.single_shooting)
    score_estimator_cls = ScoreEstimatorXu if cfg.single_shooting else ScoreEstimatorXux
    sf = score_estimator_cls(dim_x=4, dim_u=1, network=score_network).to(device)
    if cfg.single_shooting:
        sf.load_state_dict(torch.load(cfg.score_estimator_xu_load_ckpt))
    else:
        sf.load_state_dict(torch.load(cfg.score_estimator_xux_load_ckpt))

    if cfg.single_shooting:
        traj_optimizer = single_shooting(
            nn_plant,
            x0=torch.tensor([0, 0, 0, 0.0], device=device),
            score_estimator=sf,
            cfg=cfg,
        )
    else:
        traj_optimizer = dircol(
            x0=torch.tensor([0, 0, 0, 0.0], device=device),
            score_estimator=sf,
            cfg=cfg,
        )
    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    u_max = cfg.plant_param.u_max
    dt = cfg.plant_param.dt
    plot_result(traj_optimizer, plant, x_lo, x_up, u_max, dt)


if __name__ == "__main__":
    main()
