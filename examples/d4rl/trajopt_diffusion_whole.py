import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset
import os

import hydra
import pickle
from omegaconf import DictConfig
import gym, d4rl

from score_po.score_matching import NoiseConditionedScoreEstimatorXux
from score_po.nn import MLP, MLPwEmbedding
from score_po.trajectory_optimizer import (
    TrajectoryOptimizerNCSF,
    TrajectoryOptimizerSFParams,
)
from score_po.costs import Cost
from score_po.trajectory import IVPTrajectory
from score_po.costs import NNCost, QuadraticCost
from score_po.mpc import MPC

from gym_policy import GymPolicyEvaluator, TrajectoryVisualizer


class MujocoCost(Cost):
    def __init__(self):
        super().__init__()

    def get_running_cost_batch(self, z_batch, u_batch):
        cost_run = z_batch[:, 8]
        cost_ctrl = 0.1 * torch.square(u_batch).sum(dim=1)
        return cost_run + cost_ctrl

    def get_running_cost(self, z, u):
        cost_run = -10.0 * z[8]
        cost_ctrl = 0.1 * torch.square(u).sum()
        return cost_run + cost_ctrl

    def get_terminal_cost(self, z):
        return 0.0

    def get_terminal_cost_batch(self, z):
        return 0.0


@hydra.main(config_path="./config", config_name="trajopt")
def main(cfg: DictConfig):
    params = TrajectoryOptimizerSFParams()

    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]

    # Load learned score function.
    dim_z = dim_x + dim_u + dim_x
    network_sf = MLPwEmbedding(dim_z, dim_z, 4 * [2048], 10)
    sf = NoiseConditionedScoreEstimatorXux(dim_x, dim_u, network_sf)
    sf.load_state_dict(torch.load(cfg.score_weights))
    params.sf = sf

    # Load learned costs.
    # network_cost = MLP(dim_x + dim_u, 1, 4 * [1024], layer_norm=True)
    # cost = NNCost(dim_x, dim_u, network_cost)
    # cost.load_state_dict(torch.load(cfg.cost_weights))
    # params.cost = cost

    params.cost = MujocoCost()

    """
    cost = QuadraticCost()
    x_diag = torch.zeros(dim_x)
    x_diag[8] = 10.0
    cost.Q = torch.diag(x_diag)
    cost.R = 0.1 * torch.diag(torch.ones(dim_u))
    cost.Qd = torch.diag(torch.zeros(dim_x))
    cost.xd = torch.zeros(dim_x)
    params.cost = cost
    """

    # Define trajectory.
    trj = IVPTrajectory(dim_x, dim_u, cfg.trj.T, torch.zeros(dim_x))
    params.trj = trj

    # Set upoptimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)
    optimizer = TrajectoryOptimizerNCSF(params)
    optimizer.iterate()

    vis = TrajectoryVisualizer(cfg.env_name)
    if not os.path.isdir("whole"):
        os.mkdir("whole")

    x_trj, u_trj = optimizer.trj.get_full_trajectory()
    x_trj = x_trj.cpu().detach().numpy()
    u_trj = u_trj.cpu().detach().numpy()

    vis.render_trajectory(x_trj, u_trj, "whole")


if __name__ == "__main__":
    main()
