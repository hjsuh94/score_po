from omegaconf import DictConfig 
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpoleNNDynamicalSystem, CartpolePlant
from score_po.nn import AdamOptimizerParams, WandbParams
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    FirstOrderPolicyOptimizer,
)
from score_po.costs import QuadraticCost
from score_po.policy import TimeVaryingOpenLoopPolicy


def plot_result(policy_optimizer: PolicyOptimizer):
    device = policy_optimizer.policy_history.device
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
    T: int,
    u_max: float,
    cfg: DictConfig,
) -> torch.Tensor:
    """
    Find a swing up trajectory by single shooting method.

    return:
      u_traj: Of shape (T-1, 1). u_traj[i]
    """
    device = x0.device
    params = PolicyOptimizerParams()
    params.cost = QuadraticCost(
        Q=torch.zeros((4, 4), device=device),
        R=torch.tensor([[0.000]], device=device),
        Qd=torch.diag(torch.tensor([1, 1, 0.1, 0.1], device=device)),
        xd=torch.tensor([0, np.pi, 0, 0], device=device),
    )
    params.dynamical_system = dynamical_system
    params.policy = TimeVaryingOpenLoopPolicy(dim_x=4, dim_u=1, T=T)
    if cfg.load_swingup_policy is None:
        params.policy_params_0 = 0 * torch.ones((T,), device=device)
    else:
        params.policy.set_parameters(torch.load(cfg.load_swingup_policy))
        params.policy_params_0 = params.policy.get_parameters()
    params.T = T

    params.x0_upper = x0
    params.x0_lower = x0
    params.batch_size = 1

    params.std = 0
    params.lr = 6e-1
    params.max_iters = 20000
    params.wandb_params = WandbParams()
    params.wandb_params.load_from_config(cfg, "policy")
    params.save_best_model = cfg.save_swingup_policy

    policy_optimizer = FirstOrderPolicyOptimizer(params)
    policy_optimizer.iterate()
    plot_result(policy_optimizer)


@hydra.main(config_path="./config", config_name="swingup")
def main(cfg: DictConfig):
    device = cfg.device
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.nn_plant.hidden_layers,
        x_lo=torch.tensor([-1, -np.pi, -3, -12]),
        x_up=torch.tensor([1, 1.5 * np.pi, 3, 12]),
        u_lo=torch.tensor([-cfg.nn_plant.u_max]),
        u_up=torch.tensor([cfg.nn_plant.u_max]),
        device=device,
    )

    nn_plant.load_network_parameters(cfg.dynamics_load_ckpt)
    plant = CartpolePlant(dt=0.1)
    single_shooting(
        plant, x0=torch.tensor([0, 0, 0, 0.0], device=device), T=20, u_max=100, cfg=cfg
    )
    pass


if __name__ == "__main__":
    main()
