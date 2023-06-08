from omegaconf import DictConfig, OmegaConf
import os
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import torch

from score_po.costs import Cost
from score_po.dynamical_system import DynamicalSystem, NNDynamicalSystem
from score_po.nn import MLP, Normalizer
from score_po.policy_optimizer import (
    PolicyOptimizer,
    PolicyOptimizerParams,
    DRiskScorePolicyOptimizer,
    DRiskScorePolicyOptimizerParams,
)
from score_po.score_matching import ScoreEstimatorXu
from score_po.policy import TimeVaryingOpenLoopPolicy, Clamper
from examples.corridor.score_training import get_score_network
from examples.corridor.learn_model import draw_corridor
from examples.corridor.car_plant import SingleIntegrator


class PathCost(Cost):
    """
    The cost is in the form of
    ∑ᵢ₌₀ᵀ w1 * dist(x[t], x[t+1]) + u[t]ᵀRu[t] + (x[-1]-x_des)ᵀ*Qd*(x[-1]-x_des)
    where w1 is a positive scalar scalar.
    """

    def __init__(
        self,
        dynamical_system: DynamicalSystem,
        goal: torch.Tensor,
        path_length_weight: float,
        R: torch.Tensor,
        Qd: torch.Tensor,
    ):
        super().__init__()
        self.dynamical_system = dynamical_system
        self.goal = goal
        self.path_length_weight = path_length_weight
        self.R = R
        self.Qd = Qd

    def get_running_cost_batch(self, x, u):
        x_next = self.dynamical_system.dynamics_batch(x, u)
        x_diff = x_next - x
        dist = torch.sqrt(torch.einsum("bi,bi->b", x_diff, x_diff))
        uRu = torch.einsum("bi,ij,bj->b", u, self.R, u)
        return self.path_length_weight * dist + uRu

    def get_terminal_cost_batch(self, x):
        return torch.einsum("bi,ij,bj->b", x - self.goal, self.Qd, x - self.goal)


def plot_result(
    policy_optimizer: PolicyOptimizer,
    dynamical_system: DynamicalSystem,
    x_start: torch.Tensor,
    x_goal: torch.Tensor,
    corridor_width,
    horizontal_max,
    vertical_max,
):
    """
    We rollout the trajectory using the learned open-loop action and
    `dynamical_system`, then plot the rollout path.
    """
    device = policy_optimizer.params.device
    fig = plt.figure()
    ax = fig.add_subplot()
    draw_corridor(ax, corridor_width, horizontal_max, vertical_max, color="k")
    ax.plot(x_goal[0].item(), x_goal[1].item(), marker="*", color="r")

    T = policy_optimizer.params.T
    x_trj = torch.zeros((T, dynamical_system.dim_x), device=device)
    u_trj = torch.zeros((T - 1, dynamical_system.dim_u), device=device)
    x_trj[0] = x_start
    for t in range(T-1):
        u_trj[t] = policy_optimizer.policy(x_trj[t].unsqueeze(0), t).squeeze(0)
        x_trj[t + 1] = dynamical_system.dynamics(x_trj[t], u_trj[t])

    x_trj_np = x_trj.cpu().detach().numpy()
    ax.plot(x_trj_np[:, 0], x_trj_np[:, 1])
    return fig, ax


def optimize_path(
    dynamical_system: DynamicalSystem,
    x_goal: torch.Tensor,
    score_estimator: Optional[ScoreEstimatorXu],
    path_length_weight: float,
    R: torch.Tensor,
    Qd: torch.Tensor,
    cfg: DictConfig,
):
    device = cfg.device
    u_clip = Clamper(
        torch.tensor(cfg.u_lo), torch.tensor(cfg.u_up), method=Clamper.Method.HARD
    )
    if score_estimator is None:
        params = PolicyOptimizerParams()
    else:
        params = DRiskScorePolicyOptimizerParams()
        params.sf = score_estimator
    params.load_from_config(cfg)
    params.cost = PathCost(dynamical_system, x_goal, path_length_weight, R, Qd)
    params.dynamical_system = dynamical_system
    params.policy = TimeVaryingOpenLoopPolicy(
        dim_x=dynamical_system.dim_x,
        dim_u=dynamical_system.dim_u,
        T=cfg.policy.T,
        u_clip=u_clip,
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


@hydra.main(config_path="./config", config_name="optimize_path")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    device = cfg.device

    dim_x = 2
    plant_network = MLP(
        dim_in=dim_x + dim_x, dim_out=dim_x, hidden_layers=cfg.nn_plant.hidden_layers
    )
    nn_plant = NNDynamicalSystem(
        dim_x=dim_x,
        dim_u=dim_x,
        network=plant_network,
        x_normalizer=Normalizer(k=torch.ones(2), b=torch.zeros(2)),
        u_normalizer=Normalizer(k=torch.ones(2), b=torch.zeros(2)),
    ).to(device)
    nn_plant.load_state_dict(torch.load(cfg.dynamics_load_ckpt))

    score_network = get_score_network()
    sf = ScoreEstimatorXu(
        dim_x=dim_x,
        dim_u=dim_x,
        network=score_network,
    )
    sf.load_state_dict(torch.load(cfg.score_estimator_load_ckpt))

    x_start = torch.tensor(cfg.x_start, device=device)
    x_goal = torch.tensor(cfg.x_goal, device=device)

    path_length_weight = 0.1
    Qd = 100 * torch.eye(dim_x).to(device)
    R = 0.001 * torch.eye(dim_x).to(device)

    # This dt here should be the same as nn_plant.dt in learn_model.yaml
    # TODO: figure out how to load this dt from the yaml file.
    single_integrator = SingleIntegrator(dt=0.1, dim_x=2)

    # First optimize path without the score function regulization.
    policy_optimizer = optimize_path(
        nn_plant,
        x_goal,
        score_estimator=None,
        path_length_weight=path_length_weight,
        R=R,
        Qd=Qd,
        cfg=cfg,
    )
    fig, ax = plot_result(
        policy_optimizer,
        single_integrator,
        x_start,
        x_goal,
        cfg.corridor_width,
        cfg.horizontal_max,
        cfg.vertical_max,
    )
    fig.savefig(os.path.join(os.getcwd(), "no_score.png"), format="png")

    # Now optimize the path with score function regularization
    drisk_policy_optimizer = optimize_path(
        nn_plant,
        x_goal,
        score_estimator=sf,
        path_length_weight=path_length_weight,
        R=R,
        Qd=Qd,
        cfg=cfg,
    )
    fig, ax = plot_result(
        drisk_policy_optimizer,
        single_integrator,
        x_start,
        x_goal,
        cfg.corridor_width,
        cfg.horizontal_max,
        cfg.vertical_max,
    )
    fig.savefig(os.path.join(os.getcwd(), "with_score.png"), format="png")


if __name__ == "__main__":
    main()
