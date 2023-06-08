from omegaconf import DictConfig, OmegaConf
import os
from typing import List, Optional, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpoleNNDynamicalSystem, CartpolePlant
from score_po.dynamical_system import (
    DynamicalSystem,
    sim_openloop,
    NNEnsembleDynamicalSystem,
)
from score_po.nn import AdamOptimizerParams, WandbParams, Normalizer
from score_po.trajectory import SSTrajectory, BVPTrajectory
from score_po.trajectory_optimizer import (
    TrajectoryOptimizerSF,
    TrajectoryOptimizerNCSF,
    TrajectoryOptimizerSFParams,
    TrajectoryOptimizerSSParams,
    TrajectoryOptimizerSS,
    TrajectoryOptimizerSSEnsembleParams,
    TrajectoryOptimizerSSEnsemble,
)
from score_po.score_matching import (
    ScoreEstimatorXu,
    ScoreEstimatorXux,
    NoiseConditionedScoreEstimatorXux,
)
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy, Clamper
from examples.cartpole.score_training import get_score_network
from examples.cartpole.swingup import plot_result
from examples.cartpole.generate_video_util import generate_video_snapshots

def plot_result(
    traj_optimizer: TrajectoryOptimizerSSEnsemble,
    plant: CartpolePlant,
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    u_max: float,
    dt: float,
    plot_all_model: bool = True,
    snapshot_indices=List[int],
):
    device = traj_optimizer.params.device
    u_trj = traj_optimizer.trj.u_trj.data
    x0 = torch.zeros((4,), device=device)
    x_trj_plan = [None] * len(traj_optimizer.ds.ds_lst)
    for i, ds_i in enumerate(traj_optimizer.ds.ds_lst):
        x_trj_plan[i] = sim_openloop(ds_i, x0, u_trj, None)
    # Simulate with the true dynamics.
    x_trj_sim = sim_openloop(plant, x0, u_trj, None)
    x_trj_plan_np = [x_trj_plan_i.cpu().detach().numpy() for x_trj_plan_i in x_trj_plan]
    x_trj_sim_np = x_trj_sim.cpu().detach().numpy()
    u_trj_np = u_trj.cpu().detach().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    if plot_all_model:
        for x_trj_plan_np_i in x_trj_plan_np:
            ax1.plot(
                x_trj_plan_np_i[:, 0], x_trj_plan_np_i[:, 1], label="plan", color="b"
            )
    else:
        ax1.plot(
            x_trj_plan_np[0][:, 0], x_trj_plan_np[0][:, 1], label="plan", color="b"
        )
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
    ax1.legend(fontsize=16)
    ax1.set_xlabel("x (m)", fontsize=16)
    ax1.set_ylabel(r"$\theta$", fontsize=16)
    beta_val = 0
    beta_val = traj_optimizer.params.beta
    ax1.set_title(r"$\beta$" + f"={beta_val}", fontsize=16)
    ax1.tick_params(axis="both", labelsize=15)

    ax2 = fig.add_subplot(122)
    if plot_all_model:
        for x_trj_plan_np_i in x_trj_plan_np:
            ax2.plot(
                x_trj_plan_np_i[:, 2], x_trj_plan_np_i[:, 3], label="plan", color="b"
            )
    else:
        ax2.plot(
            x_trj_plan_np[0][:, 2], x_trj_plan_np[0][:, 3], label="plan", color="b"
        )
    ax2.plot(x_trj_sim_np[:, 2], x_trj_sim_np[:, 3], label="sim", color="g")
    ax2.plot(
        [x_lo_np[2], x_up_np[2], x_up_np[2], x_lo_np[2], x_lo_np[2]],
        [x_lo_np[3], x_lo_np[3], x_up_np[3], x_up_np[3], x_lo_np[3]],
        linestyle="--",
        color="r",
    )
    ax2.plot([0], [0], "*", color="r")
    ax2.legend(fontsize=16)
    ax2.set_xlabel(r"$\dot{x}$", fontsize=16)
    ax2.set_ylabel(r"$\dot{\theta}$", fontsize=16)
    ax2.set_title(r"$\beta$" + f"={beta_val}", fontsize=16)
    ax2.tick_params(axis="both", labelsize=15)

    fig.tight_layout()

    for fig_format in ("pdf", "png", "svg"):
        fig.savefig(
            os.path.join(os.getcwd(), f"swingup_state_beta{beta_val}.{fig_format}"),
            format=fig_format,
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

    # Draw snapshots of the trajectory
    print(f"x_trj_sim_end: {x_trj_sim_np[-1]}")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_ylim([-0.5, 0.75])
    ax.set_xlabel("x(m)", fontsize=16)
    ax.set_ylabel("z(m)", fontsize=16)
    num_snapshots = len(snapshot_indices)
    colors = np.zeros((num_snapshots, 4))
    colors[:, 3] = np.linspace(0.3, 1, num_snapshots)
    for i in range(num_snapshots):
        x_snapshot = x_trj_sim_np[snapshot_indices[i]]
        plant.visualize(ax, x_snapshot, color=colors[i])
        text_y_pos = -0.1 if i != num_snapshots - 1 else 0.64
        ax.text(
            x_snapshot[0] - 0.15,
            text_y_pos,
            f"t={snapshot_indices[i] * plant.dt:.1f}s",
            fontsize=6,
        )

    for fig_format in ("pdf", "png", "svg"):
        fig.savefig(
            os.path.join(os.getcwd(), f"swingup_snapshots.{fig_format}"),
            format=fig_format,
            bbox_inches="tight",
        )

    generate_video_snapshots(plant, x_trj_sim_np, dt, N_interpolate=5, videofolder="sim_video", video_name="ensemble_sim", title_prefix="Sim ")
    generate_video_snapshots(plant, x_trj_plan_np[0], dt, N_interpolate=5, videofolder="plan_video", video_name="ensemble_plan", title_prefix="Plan ")


def single_shooting(ds: NNEnsembleDynamicalSystem, x0: torch.Tensor, cfg: DictConfig):
    device = x0.device
    params = TrajectoryOptimizerSSEnsembleParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), params.save_best_model)
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4)),
        R=torch.tensor([[0.0]]),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1])),
        xd=torch.tensor([0, np.pi, 0, 0]),
    )
    params.trj = SSTrajectory(
        dim_x=ds.dim_x,
        dim_u=ds.dim_u,
        T=params.T,
        x0=x0,
    )
    params.ivp = True
    params.ds = ds
    if cfg.trj.load_ckpt is not None:
        params.trj.load_state_dict(torch.load(cfg.trj.load_ckpt))

    params.to_device(device)

    traj_optimizer = TrajectoryOptimizerSSEnsemble(params)
    if cfg.trj.train:
        traj_optimizer.iterate()
    return traj_optimizer


@hydra.main(config_path="./config", config_name="swingup_ensemble")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    device = cfg.device
    ensemble_size = cfg.plant_param.num_models
    ds_lst = [None] * ensemble_size
    for i in range(ensemble_size):
        ds_lst[i] = CartpoleNNDynamicalSystem(
            hidden_layers=cfg.plant_param.hidden_layers[i], device=device
        )

    ds = NNEnsembleDynamicalSystem(
        dim_x=4,
        dim_u=1,
        ds_lst=ds_lst,
        x_normalizer=None,
        u_normalizer=None,
    )
    ds.load_ensemble(cfg.load_ensemble_model)

    plant = CartpolePlant(dt=cfg.plant_param.dt)

    traj_optimizer = single_shooting(
        ds, x0=torch.tensor([0, 0, 0, 0.0], device=device), cfg=cfg
    )

    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    u_max = cfg.plant_param.u_max
    dt = cfg.plant_param.dt
    plot_result(
        traj_optimizer,
        plant,
        x_lo,
        x_up,
        u_max,
        dt,
        plot_all_model=False,
        snapshot_indices=[0, 6, 10, 16, 30, 60],
    )


if __name__ == "__main__":
    main()
