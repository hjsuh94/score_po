import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset
import logging

import hydra
import pickle
from omegaconf import DictConfig
import gym, d4rl

from score_po.score_matching import NoiseConditionedScoreEstimatorXu
from score_po.nn import MLP, MLPwEmbedding
from score_po.trajectory_optimizer import (
    TrajectoryOptimizerNCSS,
    TrajectoryOptimizerSSParams,
)
from score_po.dynamical_system import NNDynamicalSystem
from score_po.costs import Cost
from score_po.trajectory import SSTrajectory
from score_po.costs import NNCost, QuadraticCost
from score_po.mpc import MPC

from gym_policy import GymPolicyEvaluator


class MujocoCost(Cost):
    def __init__(self):
        super().__init__()

    def get_running_cost_batch(self, z_batch, u_batch):
        cost_run = -1.0 * z_batch[:, 8]
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


@hydra.main(config_path="./config", config_name="trajopt_ss")
def main(cfg: DictConfig):
    params = TrajectoryOptimizerSSParams()

    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]

    # Load learned score function.
    dim_z = dim_x + dim_u
    network_sf = MLPwEmbedding(dim_z, dim_z, 4 * [1024], 10)
    sf = NoiseConditionedScoreEstimatorXu(dim_x, dim_u, network_sf)
    sf.load_state_dict(torch.load(cfg.score_weights))
    params.sf = sf

    network_dyn = MLP(dim_x + dim_u, dim_x, 4 * [1024], layer_norm=True)
    dynamics = NNDynamicalSystem(dim_x, dim_u, network_dyn)
    dynamics.load_state_dict(torch.load(cfg.dynamics_weights))
    params.ds = dynamics

    # Load learned costs.
    network_cost = MLP(dim_x + dim_u, 1, 4 * [1024], layer_norm=True)
    cost = NNCost(dim_x, dim_u, network_cost)
    cost.load_state_dict(torch.load(cfg.cost_weights))
    params.cost = cost

    # params.cost = MujocoCost()

    """
    cost = QuadraticCost()
    x_diag = torch.zeros(dim_x)
    x_diag[8] = 10.0
    cost.Q = ttorch.diag(x_diag)
    cost.R = 0.1 * torch.diag(torch.ones(dim_u))
    cost.Qd = torch.diag(torch.zeros(dim_x))
    cost.xd = torch.zeros(dim_x)
    params.cost = cost
    """

    # Define trajectory.
    trj = SSTrajectory(dim_x, dim_u, cfg.trj.T, torch.zeros(dim_x))
    params.trj = trj

    # Set upoptimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)
    # params.torch_optimizer = torch.optim.RMSprop
    optimizer = TrajectoryOptimizerNCSS(params)

    print(cfg.warm_start)

    mpc = MPC(optimizer)
    mpc.warm_start = cfg.warm_start
    evaluator = GymPolicyEvaluator(cfg.env_name, mpc)

    logging.info("beta : {:.1f}".format(cfg.beta))
    logging.info("lr : {:.4f}".format(cfg.trj.lr))
    logging.info("T : {:02d}".format(cfg.trj.T))
    logging.info("warm_start: " + str(cfg.warm_start))
    logging.info("iters: " + str(cfg.max_iters))
    logging.info("Score: " + str(evaluator.get_policy_score()))


if __name__ == "__main__":
    main()
