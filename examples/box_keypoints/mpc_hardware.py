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
import time

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


class KeypointCost(Cost):
    def __init__(self):
        super().__init__()

    def get_running_cost_batch(self, z_batch, u_batch):
        cost_ctrl = torch.square(u_batch).sum(dim=1)
        return 0.5 * cost_ctrl

    def get_running_cost(self, z, u):
        z_batch = z[None, :]
        u_batch = u[None, :]
        return self.get_running_cost_batch(z_batch, u_batch)[0]

    def get_terminal_cost(self, z):
        z_batch = z[None, :]
        return self.get_terminal_cost_batch(z_batch)

    def get_terminal_cost_batch(self, z_batch):
        keypts = z_batch[:, :10].reshape(z_batch.shape[0], 5, 2)
        mean_xy = keypts.mean(dim=1)
        x = mean_xy[:, 0]
        y = mean_xy[:, 1]
        cost_run = ((x - 0.8) ** 2.0).mean() + ((y - 0.2) ** 2.0).mean()
        return cost_run


class MPCDiffIK:
    def __init__(self, cfg):
        # Setup parameters for IK and DiffIK plants.
        self.robot = MultibodyPlant(1e-3)
        parser = Parser(self.robot)
        parser.package_map().PopulateFromFolder(os.path.dirname(__file__))
        directives = LoadModelDirectives(
            os.path.join(os.path.dirname(__file__), "models/iiwa_ctrl.yaml")
        )
        ProcessModelDirectives(directives, self.robot, parser)
        self.robot.Finalize()
        self.context = self.robot.CreateDefaultContext()
        self.diffik_params = DifferentialInverseKinematicsParameters(7, 7)
        self.diffik_params.set_time_step(1.0 / 200.0)
        self.velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        self.diffik_params.set_joint_velocity_limits(
            (-self.velocity_limits, self.velocity_limits)
        )
        self.diffik_params.set_nominal_joint_position(
            np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])
        )

        # Set up
        self.cfg = cfg
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

        print(params.beta)
        self.mpc = MPC(optimizer)
        self.mpc.warm_start = cfg.warm_start

        self.q = None
        self.v = None
        self.keypts = None
        self.lc = lcm.LCM()
        self.lc.subscribe("IIWA_STATUS", self.get_state)
        self.lc.subscribe("KEYPTS", self.get_keypts)
        self.wait_for_message()

    def forward_kinematics(self, q, v):
        self.robot.SetPositions(self.context, q)
        self.robot.SetVelocities(self.context, v)
        self.X_WT = self.robot.CalcRelativeTransform(
            self.context,
            self.robot.world_frame(),
            self.robot.GetFrameByName("iiwa_link_7"),
        )
        self.J_WT = self.robot.CalcJacobianSpatialVelocity(
            context=self.context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self.robot.GetFrameByName("iiwa_link_7"),
            p_BP=np.array([0.0, 0.0, 0.0]),
            frame_A=self.robot.world_frame(),
            frame_E=self.robot.world_frame(),
        )
        return self.X_WT, self.J_WT

    def compute_V_from_posediff(self, X_WT, X_WT_des):
        v_trans = X_WT_des.translation() - X_WT.translation()
        rot_err = (
            X_WT_des.rotation().multiply(X_WT.rotation().transpose())
        ).ToAngleAxis()
        v_rot = rot_err.axis() * rot_err.angle()
        return np.concatenate((v_rot, v_trans))

    def visualize_trajectory(self, x_trj, u_trj):
        T = x_trj.shape[0]
        keypts_trj = x_trj[:, :10].detach().cpu().numpy().reshape(T, 5, 2)
        pusher_trj = x_trj[:, 10:].detach().cpu().numpy().reshape(T, 2)

        plt.figure()
        for t in range(T):
            plt.plot(keypts_trj[t, :, 0], keypts_trj[t, :, 1], "ko")
        plt.plot(pusher_trj[:, 0], pusher_trj[:, 1], "ro-")
        plt.axis("equal")
        plt.show()
        plt.close()

    def get_command(self):
        z_state = np.concatenate((self.keypts, self.X_WT.translation()[:2]))
        u = self.mpc.get_action(torch.Tensor(z_state).to(self.cfg.trj.device))
        u = torch.clip(u, min=-0.05, max=0.05)
        u = u.cpu().detach().numpy()

        x_trj, u_trj = self.mpc.opt.rollout_trajectory()

        self.visualize_trajectory(x_trj, u_trj)

        return np.array([u[0], u[1], 0.0])

    def publish_command(self, joint_cmd):
        msg = lcmt_iiwa_command()
        msg.utime = int(time.time() * 1e6)
        msg.num_joints = 7
        msg.joint_position = joint_cmd
        msg.num_torques = 7
        msg.joint_torque = np.zeros(7)
        self.lc.publish("IIWA_COMMAND", msg.encode())

    def publish_fk(self):
        msg = lcmt_iiwa_command()
        msg.utime = int(time.time() * 1e6)
        msg.num_joints = 2
        msg.joint_position = self.X_WT.translation()[0:2]
        msg.num_torques = 2
        msg.joint_torque = np.zeros(2)
        self.lc.publish("PUSHER", msg.encode())

    def publish_action(self, action):
        msg = lcmt_iiwa_command()
        msg.utime = int(time.time() * 1e6)
        msg.num_joints = 2
        msg.joint_position = action
        msg.num_torques = 2
        msg.joint_torque = np.zeros(2)
        self.lc.publish("PUSHER_COMMAND", msg.encode())

    def diffik(self):
        if self.counter % 100 == 0:
            self.u = self.get_command()
            self.publish_action(self.u[:2])
        self.X_WT_ref = RigidTransform(
            self.X_WT_ref.rotation(), self.X_WT_ref.translation() + self.u / 100
        )
        V = self.compute_V_from_posediff(self.X_WT, self.X_WT_ref)
        result = DoDifferentialInverseKinematics(
            self.q, self.v, V, self.J_WT, self.diffik_params
        )
        if result.status != DifferentialInverseKinematicsStatus.kSolutionFound:
            print(result.status)
            print("DiffIK solution not found.")
            joint_cmd = self.q
        else:
            joint_cmd = self.q + 0.5 * result.joint_velocities
        self.publish_command(joint_cmd)

    def get_state(self, channel, data):
        msg = lcmt_iiwa_status.decode(data)
        self.q = np.array(msg.joint_position_measured)
        self.v = np.array(msg.joint_velocity_estimated)
        self.X_WT, self.J_WT = self.forward_kinematics(self.q, self.v)

    def get_keypts(self, channel, data):
        msg = optitrack.optitrack_frame_t.decode(data)
        self.keypts = np.array(msg.marker_sets[0].xyz)[:, :2].flatten()

    def wait_for_message(self):
        while True:
            self.lc.handle()
            if self.q is not None and self.keypts is not None:
                # Record X_WT_reference.
                self.X_WT_ref, _ = self.forward_kinematics(self.q, self.v)
                break

    def run(self):
        self.counter = 0
        while True:
            self.lc.handle()
            self.diffik()
            self.publish_fk()
            self.counter = (self.counter + 1) % 200


@hydra.main(config_path="./config", config_name="trajopt_ss")
def main(cfg: DictConfig):
    ctrl = MPCDiffIK(cfg)
    ctrl.run()


if __name__ == "__main__":
    main()
