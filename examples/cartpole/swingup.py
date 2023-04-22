from omegaconf import DictConfig, OmegaConf
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpoleNNDynamicalSystem, CartpolePlant
from score_po.nn import AdamOptimizerParams, WandbParams, Normalizer
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.score_matching import ScoreEstimator
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy
from examples.cartpole.score_training import get_score_network


def plot_result(policy_optimizer: PolicyOptimizer):
    device = policy_optimizer.params.device
    x_trj, u_trj = policy_optimizer.rollout_policy(
        x0=torch.zeros((4,), device=device),
        noise_trj=torch.zeros((policy_optimizer.params.T, 1), device=device),
    )
    x_trj_np = x_trj.cpu().detach().numpy()
    u_trj_np = u_trj.cpu().detach().numpy()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(x_trj_np[:, 0], x_trj_np[:, 1])
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel(r"$\theta$")

    ax2 = fig.add_subplot(122)
    ax2.plot(x_trj_np[:, 2], x_trj_np[:, 3])
    ax2.set_xlabel(r"$\dot{x}$")
    ax2.set_ylabel(r"$\dot{\theta}$")

    fig.savefig(os.path.join(os.getcwd(), "swingup_state.png"), format="png")

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(u_trj_np)
    ax.set_title("u")
    fig.savefig(os.path.join(os.getcwd(), "swingup_u.png"), format="png")


def single_shooting(
    dynamical_system: CartpoleNNDynamicalSystem,
    x0: torch.Tensor,
    score_estimator: ScoreEstimator,
    cfg: DictConfig,
) -> torch.Tensor:
    """
    Find a swing up trajectory by single shooting method.

    return:
      u_traj: Of shape (T-1, 1). u_traj[i]
    """
    device = x0.device
    params = DRiskScorePolicyOptimizerParams()
    params.load_from_config(cfg)
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4)),
        R=torch.tensor([[0.000]]),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1])),
        xd=torch.tensor([0, np.pi, 0, 0]),
    )
    params.sf = score_estimator
    params.dynamical_system = dynamical_system
    params.policy = TimeVaryingOpenLoopPolicy(dim_x=4, dim_u=1, T=params.T)
    if cfg.policy.load_ckpt is not None:
        params.policy.load_state_dict(torch.load(cfg.policy.load_ckpt))

    params.to_device(device)

    policy_optimizer = DRiskScorePolicyOptimizer(params)
    policy_optimizer.iterate()
    plot_result(policy_optimizer)


@hydra.main(config_path="./config", config_name="swingup")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    device = cfg.device
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.nn_plant.hidden_layers,
        device=device,
    )

    nn_plant.load_state_dict(torch.load(cfg.dynamics_load_ckpt))
    plant = CartpolePlant(dt=0.1)

    score_network = get_score_network()
    sf = ScoreEstimator(dim_x=4, dim_u=1, network=score_network).to(device)
    sf.load_state_dict(torch.load(cfg.score_estimator_load_ckpt))

    single_shooting(
        plant,
        x0=torch.tensor([0, 0, 0, 0.0], device=device),
        score_estimator=sf,
        cfg=cfg,
    )
    pass


if __name__ == "__main__":
    main()
