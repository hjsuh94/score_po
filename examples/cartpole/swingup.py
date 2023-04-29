from omegaconf import DictConfig, OmegaConf
import os
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpoleNNDynamicalSystem, CartpolePlant
from score_po.dynamical_system import DynamicalSystem, sim_openloop
from score_po.nn import AdamOptimizerParams, WandbParams, Normalizer
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.score_matching import ScoreEstimatorXu
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy, Clamper
from examples.cartpole.score_training import get_score_network


def plot_result(
    policy_optimizer: PolicyOptimizer,
    plant: CartpolePlant,
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    u_max: float,
    dt: float,
):
    device = policy_optimizer.params.device
    u_trj = policy_optimizer.policy.params.data
    x0 = torch.zeros((4,), device=device)
    # Simulate with dynamical system in policy_optimizer.
    x_trj_plan = sim_openloop(policy_optimizer.ds, x0, u_trj, None)
    # Simulate with the true dynamics.
    x_trj_sim = sim_openloop(plant, x0, u_trj, None)
    x_trj_plan_np = x_trj_plan.cpu().detach().numpy()
    x_trj_sim_np = x_trj_sim.cpu().detach().numpy()
    u_trj_np = u_trj.cpu().detach().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(x_trj_plan_np[:, 0], x_trj_plan_np[:, 1], label="plan", color="b")
    ax1.plot(x_trj_sim_np[:, 0], x_trj_sim_np[:, 1], label="sim", color="g")
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
    if isinstance(policy_optimizer.params, DRiskScorePolicyOptimizerParams):
        beta_val = policy_optimizer.params.beta
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
    score_estimator: Optional[ScoreEstimatorXu],
    cfg: DictConfig,
) -> torch.Tensor:
    """
    Find a swing up trajectory by single shooting method.

    return:
      u_traj: Of shape (T-1, 1). u_traj[i]
    """
    device = x0.device
    if score_estimator is None:
        params = PolicyOptimizerParams()
    else:
        params = DRiskScorePolicyOptimizerParams()
    params.load_from_config(cfg)
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4)),
        R=torch.tensor([[0.000]]),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1])),
        xd=torch.tensor([0, np.pi, 0, 0]),
    )
    if score_estimator is not None:
        params.sf = score_estimator
    params.dynamical_system = dynamical_system
    u_clip = Clamper(
        lower=torch.tensor(-cfg.plant_param.u_max),
        upper=torch.tensor(cfg.plant_param.u_max),
        method=Clamper.Method.HARD,
    )
    params.policy = TimeVaryingOpenLoopPolicy(
        dim_x=4, dim_u=1, T=params.T, u_clip=u_clip
    )
    if cfg.policy.load_ckpt is not None:
        params.policy.load_state_dict(torch.load(cfg.policy.load_ckpt))

    params.to_device(device)

    if score_estimator is None:
        policy_optimizer = PolicyOptimizer(params)
    else:
        policy_optimizer = DRiskScorePolicyOptimizer(params)
    policy_optimizer.iterate()
    return policy_optimizer


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

    if cfg.score_estimator_load_ckpt:
        score_network = get_score_network()
        sf = ScoreEstimatorXu(dim_x=4, dim_u=1, network=score_network).to(device)
        sf.load_state_dict(torch.load(cfg.score_estimator_load_ckpt))
    else:
        sf = None

    policy_optimizer = single_shooting(
        nn_plant,
        x0=torch.tensor([0, 0, 0, 0.0], device=device),
        score_estimator=sf,
        cfg=cfg,
    )
    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    u_max = cfg.plant_param.u_max
    dt = cfg.plant_param.dt
    plot_result(policy_optimizer, plant, x_lo, x_up, u_max, dt)


if __name__ == "__main__":
    main()
