import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

import hydra
from omegaconf import DictConfig

from hydra.utils import get_original_cwd

from pydrake.all import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    StartMeshcat,
    MeshcatPointCloudVisualizer,
    Rgba,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import (
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
)
from pydrake.perception import PointCloud
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import DiagramBuilder, Diagram
from pydrake.systems.primitives import TrajectorySource
from pydrake.systems.primitives import LogVectorOutput
from pydrake.trajectories import PiecewisePolynomial


from score_po.dynamical_system import DynamicalSystem


class PlanarPusherSystem(DynamicalSystem):
    """
    Planar pushing dynamical system, implemented in Drake.
    x: position of box and pusher, [x_box, y_box, theta_box, x_pusher, y_pusher]
    u: velocity command on the pusher.
    """

    def __init__(self):
        super().__init__(dim_x=5, dim_u=2)
        self.is_differentiable = False
        self.h = 3.0  # simulation duration.
        self.h_mbp = 1e-3
        self.meshcat = StartMeshcat()
        
        # These dimensions come from the ycb dataset on 004_sugar_box.sdf
        # The elements corresponds to box_width along [x,y,z] dimension.
        self.box_dim = np.array([0.0867, 0.1703, 0.0391])

        # Parse model directives and add mbp.
        builder = DiagramBuilder()
        self.mbp, self.sg = AddMultibodyPlantSceneGraph(builder, time_step=self.h_mbp)
        self.parser = Parser(self.mbp, self.sg)
        self.parser.package_map().PopulateFromFolder(
            os.path.join(get_original_cwd(), "examples/box_keypoints")
        )
        directives = LoadModelDirectives(
            os.path.join(
                get_original_cwd(), "examples/box_keypoints/models/box_pushing.yaml"
            )
        )
        ProcessModelDirectives(directives, self.mbp, self.parser)
        self.mbp.Finalize()

        # Get model instances for box and pusher.
        self.box = self.mbp.GetModelInstanceByName("box")
        self.pusher = self.mbp.GetModelInstanceByName("pusher")

        # Add visualizer.
        self.visualizer = MeshcatVisualizer.AddToBuilder(builder, self.sg, self.meshcat)

        # Add PID Controller.
        pid = builder.AddSystem(
            PidController(kp=[500, 500], kd=[150, 150], ki=[50, 50])
        )
        builder.Connect(
            pid.get_output_port_control(),
            self.mbp.get_actuation_input_port(self.pusher),
        )
        builder.Connect(
            self.mbp.get_state_output_port(self.pusher),
            pid.get_input_port_estimated_state(),
        )
        # builder.ExportInput(pid.get_input_port_desired_state(), "u")

        # Add trajectory Source.
        dummy_trj = PiecewisePolynomial.FirstOrderHold([0, 0.1], np.zeros((4, 2)))
        self.traj_source = builder.AddSystem(TrajectorySource(dummy_trj))
        builder.Connect(
            self.traj_source.get_output_port(), pid.get_input_port_desired_state()
        )

        self.pose_logger = LogVectorOutput(
            self.mbp.get_state_output_port(self.pusher), builder
        )
        self.force_logger = LogVectorOutput(pid.get_output_port_control(), builder)

        # Build and add simulator.
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)

    def planar_to_full_coordinates(self, x):
        """Given x in planar coordinates, convert to full coordinates."""
        theta = x[2]
        q_wxyz = Quaternion(
            RotationMatrix(RollPitchYaw([0, 0, theta])).matrix()).wxyz()
        p_xyz = np.array([x[0], x[1], self.box_dim[2]])
        return np.concatenate((q_wxyz, p_xyz))

    def full_to_planar_coordinates(self, x):
        """Given x in full coordinates, convert to planar coordinates."""
        q_wxyz = x[0:4] / np.linalg.norm(x[0:4])
        p_xyz = x[4:7]

        theta = RollPitchYaw(Quaternion(q_wxyz)).vector()[2]
        return np.concatenate((p_xyz[0:2], np.array([theta])))

    def reset_x(self, x):
        context = self.simulator.get_mutable_context()
        mbp_context = self.mbp.GetMyContextFromRoot(context)

        x_box = self.planar_to_full_coordinates(x[0:3])

        self.mbp.SetPositions(mbp_context, self.box, x_box)
        self.mbp.SetVelocities(mbp_context, self.box, np.zeros(6))
        self.mbp.SetPositions(mbp_context, self.pusher, x[3:5])
        self.mbp.SetVelocities(mbp_context, self.pusher, np.zeros(2))

    def is_in_collision(self, x):
        """Check if x is a colliding configuration."""
        context = self.simulator.get_mutable_context()
        mbp_context = self.mbp.GetMyContextFromRoot(context)

        self.reset_x(x)
        contact_results = self.mbp.get_contact_results_output_port().Eval(mbp_context)
        for i in range(contact_results.num_point_pair_contacts()):
            contact_info = contact_results.point_pair_contact_info(i)
            body_A = self.mbp.get_body(contact_info.bodyA_index()).name()
            body_B = self.mbp.get_body(contact_info.bodyB_index()).name()

            if (body_A == "base_link_sugar" and body_B == "cylinder") or (
                body_A == "cylinder" and body_B == "base_link_sugar"
            ):
                return True
        return False

    def get_keypoints(self, x, visualize=False, path="keypoints", noise_std=0.0):
        """
        Get keypoints given pose x.
        Optionally accepts whether to visualize, the path to meshcat,
        and injected noise.
        """
        canon_points = np.array(
            [
                [1, -1, -1, 1, 0],
                [1, 1, -1, -1, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        # These dimensions come from the ycb dataset on 004_sugar_box.sdf
        # Make homogeneous coordinates
        keypoints = np.zeros((4, 5))
        keypoints[0, :] = self.box_dim[0] * canon_points[0, :] / 2
        keypoints[1, :] = self.box_dim[1] * canon_points[1, :] / 2
        keypoints[2, :] = self.box_dim[2]
        keypoints[3, :] = 1

        # transform according to pose x.
        X_WB = RigidTransform(
            RollPitchYaw(np.array([0.0, 0.0, x[2]])), np.array([x[0], x[1], 0.0])
        ).GetAsMatrix4()

        X_WK = X_WB @ keypoints
        X_WK[0:2] = X_WK[0:2] + noise_std * np.random.randn(*X_WK[0:2].shape)

        if visualize:
            cloud = PointCloud(5)
            cloud.mutable_xyzs()[:] = X_WK[0:3, :]
            self.meshcat.SetObject(
                path=path, cloud=cloud, point_size=0.02, rgba=Rgba(0.5, 0.0, 0.0)
            )

        return X_WK[0:2, :]

    def dynamics(self, x, u, record=False):
        """Return dynamics. We assume that x is a non-colliding configuration."""
        assert not self.is_in_collision(x)

        if record:
            self.visualizer.StartRecording()

        context = self.simulator.get_mutable_context()
        mbp_context = self.mbp.GetMyContextFromRoot(context)

        x_box = self.planar_to_full_coordinates(x[0:3])

        self.mbp.SetPositions(mbp_context, self.box, x_box)
        self.mbp.SetVelocities(mbp_context, self.box, np.zeros(6))
        self.mbp.SetPositions(mbp_context, self.pusher, x[3:5])
        self.mbp.SetVelocities(mbp_context, self.pusher, np.zeros(2))

        setpoints = np.zeros((4, 3))
        setpoints[0:2, 0] = x[3:5]
        setpoints[0:2, 1] = x[3:5] + u
        setpoints[0:2, 2] = x[3:5] + u
        traj = PiecewisePolynomial.FirstOrderHold([0, self.h / 2, self.h], setpoints)
        self.traj_source.UpdateTrajectory(traj)

        t_now = self.simulator.get_context().get_time()
        self.simulator.AdvanceTo(t_now + self.h)

        if record:
            self.visualizer.StopRecording()
            self.visualizer.PublishRecording()

        x_box_next = self.full_to_planar_coordinates(
            self.mbp.GetPositions(mbp_context, self.box)
        )
        x_pusher_next = self.mbp.GetPositions(mbp_context, self.pusher)

        return np.concatenate((x_box_next, x_pusher_next))

    def dynamics_batch(self, x_batch, u_batch):
        """
        Return dynamics in a batch. Note that this can be parallelized with threading,
        but we simply do loops here due to two reasons:
        1. Drake doesn't have bindings for batch simulations in python.
        2. This class is only used for collecting data - since it's offline time,
           it's reasonable to have a not efficient implementation.
        """
        B = x_batch.shape[0]
        xnext_batch = np.zeros((B, 5))
        for b in range(B):
            xnext_batch = self.dynamics_batch(x_batch[b], u_batch[b])
        return xnext_batch
