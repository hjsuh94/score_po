"""
Runs manipulation_station example with Sony DualShock4 Controller for
teleoperating the end effector.
"""

import argparse
from enum import Enum
import os
import pprint
import sys
from textwrap import dedent
import webbrowser
import lcm
import time, pickle

import numpy as np
import torch

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from pydrake.common.value import AbstractValue
from pydrake.examples import (
    ManipulationStation,
    ManipulationStationHardwareInterface,
    CreateClutterClearingYcbObjectList,
    SchunkCollisionModel,
)
from pydrake.geometry import DrakeVisualizer, Meshcat, MeshcatVisualizer
from pydrake.multibody.plant import MultibodyPlant
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import FirstOrderLowPassFilter
from pydrake.all import (
    LoadModelDirectives,
    ProcessModelDirectives,
    Parser,
    DoDifferentialInverseKinematics,
    DifferentialInverseKinematicsStatus,
    JacobianWrtVariable,
)

from drake import lcmt_iiwa_status, lcmt_iiwa_command
import optitrack

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

import matplotlib.cm as cm


class KeypointCost(Cost):
    def __init__(self):
        super().__init__()
        self.goal = np.load(
            "~/Documents/score_po/examples/box_keypoints/goals/keypts_left.npy"
        )
        self.goal = torch.Tensor(self.goal).cuda()

    def get_running_cost_batch(self, z_batch, u_batch):
        cost_ctrl = torch.square(u_batch).sum(dim=1)
        return 0.1 * cost_ctrl

    def get_running_cost(self, z, u):
        z_batch = z[None, :]
        u_batch = u[None, :]
        return self.get_running_cost_batch(z_batch, u_batch)[0]

    def get_terminal_cost(self, z):
        z_batch = z[None, :]
        return self.get_terminal_cost_batch(z_batch)[0]

    def get_terminal_cost_batch(self, z_batch):
        keypts = z_batch[:, :10]
        cost = (keypts - self.goal[None, :]).square().sum(dim=1)
        return cost


def visualize_trajectory(x_trj, u_trj):
    T = x_trj.shape[0]
    keypts_trj = x_trj[:, :10].detach().cpu().numpy().reshape(T, 5, 2)
    pusher_trj = x_trj[:, 10:].detach().cpu().numpy().reshape(T, 2)

    colormap = cm.get_cmap("viridis")

    plt.figure(figsize=(8, 7))
    plt.rcParams.update({"font.size": 15})
    for t in range(T):
        plt.plot(
            keypts_trj[t, :, 0],
            keypts_trj[t, :, 1],
            linestyle="-",
            marker="o",
            alpha=0.8,
            color=colormap(t / 8),
            label="Predicted Keypoint Trajectory" if t == 0 else None,
        )
    plt.plot(
        pusher_trj[:, 0], pusher_trj[:, 1], "ro-", label="Predicted Pusher Trajectory"
    )

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("without_beta.png")
    plt.show()


@hydra.main(config_path="./config", config_name="trajopt_ss")
def main(cfg: DictConfig):
    with open(cfg.dataset_dir, "rb") as f:
        dataset = pickle.load(f)

    dim_x = 12
    dim_u = 2
    x = dataset.tensors[0]
    x0 = x[653]

    params = TrajectoryOptimizerSSParams()
    dim_x = 12
    dim_u = 2
    dim_z = dim_x + dim_u

    network_sf = MLPwEmbedding(dim_z, dim_z, 4 * [1024], 10)
    dim_z = dim_x + dim_u
    network_sf = MLPwEmbedding(dim_z, dim_z, 4 * [1024], 10)
    sf = NoiseConditionedScoreEstimatorXu(dim_x, dim_u, network_sf)
    sf.load_state_dict(torch.load(cfg.score_weights))
    params.sf = sf

    network_dyn = MLP(dim_x + dim_u, dim_x, 4 * [1024], layer_norm=True)
    dynamics = NNDynamicalSystem(dim_x, dim_u, network_dyn)
    dynamics.load_state_dict(torch.load(cfg.dynamics_weights))
    params.ds = dynamics

    params.cost = KeypointCost()

    # Define trajectory.
    trj = SSTrajectory(dim_x, dim_u, cfg.trj.T, torch.zeros(dim_x))
    params.trj = trj

    # Set upoptimizer
    params.load_from_config(cfg)
    params.to_device(cfg.trj.device)
    # params.torch_optimizer = torch.optim.RMSprop
    optimizer = TrajectoryOptimizerNCSS(params)

    mpc = MPC(optimizer)
    mpc.warm_start = cfg.warm_start

    u = mpc.get_action(torch.Tensor(x0).to(cfg.trj.device))
    x_trj, u_trj = mpc.opt.rollout_trajectory()

    visualize_trajectory(x_trj, u_trj)


if __name__ == "__main__":
    main()
