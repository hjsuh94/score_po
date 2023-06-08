import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os, time
import lcm

import hydra
from omegaconf import DictConfig

from hydra.utils import get_original_cwd

from pydrake.systems.lcm import (
    LcmInterfaceSystem,
    LcmSubscriberSystem,
    LcmPublisherSystem,
)

from pydrake.all import Quaternion, IiwaStatusReceiver, IiwaCommandSender
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
from pydrake.lcm import DrakeLcm
from pydrake.perception import PointCloud
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import DiagramBuilder, Diagram
from pydrake.systems.primitives import (
    TrajectorySource,
    LogVectorOutput,
    Demultiplexer,
    PassThrough,
    Adder,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.manipulation.kuka_iiwa import IiwaStatusSender, IiwaCommandReceiver
from pydrake.all import (
    MultibodyPlant,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    LeafSystem,
    AbstractValue,
)

from drake import lcmt_iiwa_status, lcmt_iiwa_command, lcmt_robot_state
import optitrack


from score_po.dynamical_system import DynamicalSystem


def planar_to_full_coordinates(x):
    """Given x in planar coordinates, convert to full coordinates."""
    box_dim = np.array([0.0867, 0.1703, 0.0391])
    theta = x[2]
    q_wxyz = Quaternion(RotationMatrix(RollPitchYaw([0, 0, theta])).matrix()).wxyz()
    p_xyz = np.array([x[0], x[1], box_dim[2]])
    return np.concatenate((q_wxyz, p_xyz))


def full_to_planar_coordinates(x):
    """Given x in full coordinates, convert to planar coordinates."""
    q_wxyz = x[0:4] / np.linalg.norm(x[0:4])
    p_xyz = x[4:7]

    theta = RollPitchYaw(Quaternion(q_wxyz)).vector()[2]
    return np.concatenate((p_xyz[0:2], np.array([theta])))


def get_keypoints(x):
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
    box_dim = np.array([0.0867, 0.1703, 0.0391])
    # These dimensions come from the ycb dataset on 004_sugar_box.sdf
    # Make homogeneous coordinates
    keypoints = np.zeros((4, 5))
    keypoints[0, :] = box_dim[0] * canon_points[0, :] / 2
    keypoints[1, :] = box_dim[1] * canon_points[1, :] / 2
    keypoints[2, :] = box_dim[2]
    keypoints[3, :] = 1

    # transform according to pose x.
    X_WB = RigidTransform(
        RollPitchYaw(np.array([0.0, 0.0, x[2]])), np.array([x[0], x[1], 0.0])
    ).GetAsMatrix4()

    X_WK = X_WB @ keypoints
    X_WK[0:2] = X_WK[0:2]
    return X_WK[0:2, :]


class ManipulationDiagram(Diagram):
    def __init__(self):
        Diagram.__init__(self)

        self.h_mbp = 1e-3
        self.meshcat = StartMeshcat()

        # These dimensions come from the ycb dataset on 004_sugar_box.sdf
        # The elements corresponds to box_width along [x,y,z] dimension.
        self.box_dim = np.array([0.0867, 0.1703, 0.0391])
        self.meshcat.SetTransform(
            path="/Cameras/default",
            matrix=RigidTransform(
                RollPitchYaw([-np.pi / 8, 0.0, np.pi / 2]),
                0.01 * np.array([0.05, 0.0, 0.1]),
            ).GetAsMatrix4(),
        )

        # Parse model directives and add mbp.
        builder = DiagramBuilder()
        self.mbp, self.sg = AddMultibodyPlantSceneGraph(builder, time_step=self.h_mbp)
        self.parser = Parser(self.mbp, self.sg)
        self.parser.package_map().PopulateFromFolder(os.path.dirname(__file__))
        directives = LoadModelDirectives(
            os.path.join(os.path.dirname(__file__), "models/box_pushing_iiwa.yaml")
        )
        ProcessModelDirectives(directives, self.mbp, self.parser)
        self.mbp.Finalize()

        # Get model instances for box and pusher.
        self.box = self.mbp.GetModelInstanceByName("box")
        self.pusher = self.mbp.GetModelInstanceByName("pusher")
        self.iiwa = self.mbp.GetModelInstanceByName("iiwa")

        self.box_index = self.mbp.GetBodyByName("base_link_sugar").index()

        # Add visualizer.
        self.visualizer = MeshcatVisualizer.AddToBuilder(builder, self.sg, self.meshcat)

        self.add_controller(builder)
        # Export states
        builder.ExportOutput(self.mbp.get_body_poses_output_port(), "body_poses")
        builder.BuildInto(self)

    def add_controller(self, builder):
        ctrl_plant = MultibodyPlant(1e-3)
        parser = Parser(ctrl_plant)
        parser.package_map().PopulateFromFolder(os.path.dirname(__file__))
        directives = LoadModelDirectives(
            os.path.join(os.path.dirname(__file__), "models/iiwa_ctrl.yaml")
        )
        ProcessModelDirectives(directives, ctrl_plant, parser)
        ctrl_plant.Finalize()
        kp = 800 * np.ones(7)
        ki = 100 * np.ones(7)
        kd = 2 * np.sqrt(kp)
        arm_controller = builder.AddSystem(
            InverseDynamicsController(ctrl_plant, kp, ki, kd, False)
        )
        adder = builder.AddSystem(Adder(2, 7))
        state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(7, 1e-3, True)
        )
        arm_command = builder.AddSystem(PassThrough(7))
        state_split = builder.AddSystem(Demultiplexer(14, 7))

        # Export positions command
        builder.ExportInput(arm_command.get_input_port(0), "iiwa_position")
        builder.ExportOutput(arm_command.get_output_port(0), "iiwa_position_commanded")

        # Export arm state ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa), state_split.get_input_port(0)
        )
        builder.ExportOutput(state_split.get_output_port(0), "iiwa_position_measured")
        builder.ExportOutput(state_split.get_output_port(1), "iiwa_velocity_estimated")
        builder.ExportOutput(
            self.mbp.get_state_output_port(self.iiwa), "iiwa_state_measured"
        )

        # Export controller stack ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa),
            arm_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            arm_controller.get_output_port_control(), adder.get_input_port(0)
        )
        builder.Connect(
            adder.get_output_port(0),
            self.mbp.get_actuation_input_port(self.iiwa),
        )
        builder.Connect(
            state_from_position.get_output_port(0),
            arm_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            arm_command.get_output_port(0), state_from_position.get_input_port(0)
        )

        builder.ExportInput(adder.get_input_port(1), "iiwa_feedforward_torque")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_commanded")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_measured")

        builder.ExportOutput(
            self.mbp.get_generalized_contact_forces_output_port(self.iiwa),
            "iiwa_torque_external",
        )


class KeyptsLCM(LeafSystem):
    def __init__(self, box_index):
        LeafSystem.__init__(self)
        # self.Abstract
        self.box_index = box_index
        self.lc = lcm.LCM()
        self.input_port = self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclarePeriodicPublishEvent(1.0 / 200.0, 0.0, self.publish)

    def publish(self, context):
        X_WB = self.get_input_port().Eval(context)[self.box_index]
        q_wxyz = Quaternion(X_WB.rotation().matrix()).wxyz()
        p_xyz = X_WB.translation()
        qp_WB = np.concatenate((q_wxyz, p_xyz))
        xyt_WB = full_to_planar_coordinates(qp_WB)

        keypts = get_keypoints(xyt_WB)
        keypts = np.concatenate((keypts, np.zeros((1, 5))))

        sub_msg = optitrack.optitrack_marker_set_t()
        sub_msg.num_markers = 5
        sub_msg.xyz = keypts.T
        msg = optitrack.optitrack_frame_t()
        msg.utime = int(time.time() * 1e6)
        msg.num_marker_sets = 1
        msg.marker_sets = [sub_msg]
        self.lc.publish("KEYPTS", msg.encode())


class MockManipulation:
    """
    Planar pushing dynamical system, implemented in Drake.
    x: position of box and pusher, [x_box, y_box, theta_box, x_pusher, y_pusher]
    u: delta position command on the pusher.
    """

    def __init__(self):
        builder = DiagramBuilder()
        self.station = builder.AddSystem(ManipulationDiagram())
        self.connect_lcm(builder, self.station)
        self.keypts_lcm = builder.AddSystem(KeyptsLCM(self.station.box_index))
        builder.Connect(
            self.station.GetOutputPort("body_poses"), self.keypts_lcm.get_input_port()
        )

        diagram = builder.Build()
        self.simulator = Simulator(diagram)
        self.simulator.set_target_realtime_rate(1.0)

        self.set_default_joint_position()
        self.set_default_box_position()

    def run(self, timeout=1e8):
        self.simulator.AdvanceTo(timeout)

    def set_default_joint_position(self):
        context = self.simulator.get_mutable_context()
        mbp_context = self.station.mbp.GetMyContextFromRoot(context)
        self.default_joint_position = np.array(
            [0.666, 1.039, -0.7714, -2.0497, 1.3031, 0.6729, -1.0252]
        )
        self.station.mbp.SetPositions(
            mbp_context, self.station.iiwa, self.default_joint_position
        )

    def set_default_box_position(self):
        context = self.simulator.get_mutable_context()
        mbp_context = self.station.mbp.GetMyContextFromRoot(context)
        self.default_box_position = planar_to_full_coordinates(
            np.array([0.5, 0.0, 0.0])
        )
        self.station.mbp.SetPositions(
            mbp_context, self.station.box, self.default_box_position
        )

    def connect_lcm(self, builder, station):
        # Set up LCM publisher subscribers.
        lcm = DrakeLcm()
        lcm_system = builder.AddSystem(LcmInterfaceSystem(lcm))
        iiwa_command = builder.AddSystem(IiwaCommandReceiver())
        iiwa_command_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="IIWA_COMMAND",
                lcm_type=lcmt_iiwa_command,
                lcm=lcm,
                use_cpp_serializer=True,
            )
        )
        builder.Connect(
            iiwa_command_subscriber.get_output_port(),
            iiwa_command.get_message_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_position_measured"),
            iiwa_command.get_position_measured_input_port(),
        )
        builder.Connect(
            iiwa_command.get_commanded_position_output_port(),
            station.GetInputPort("iiwa_position"),
        )
        builder.Connect(
            iiwa_command.get_commanded_torque_output_port(),
            station.GetInputPort("iiwa_feedforward_torque"),
        )

        iiwa_status = builder.AddSystem(IiwaStatusSender())
        builder.Connect(
            station.GetOutputPort("iiwa_position_commanded"),
            iiwa_status.get_position_commanded_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_position_measured"),
            iiwa_status.get_position_measured_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_velocity_estimated"),
            iiwa_status.get_velocity_estimated_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_commanded"),
            iiwa_status.get_torque_commanded_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_measured"),
            iiwa_status.get_torque_measured_input_port(),
        )
        builder.Connect(
            station.GetOutputPort("iiwa_torque_external"),
            iiwa_status.get_torque_external_input_port(),
        )

        iiwa_status_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                "IIWA_STATUS",
                lcm_type=lcmt_iiwa_status,
                lcm=lcm,
                publish_period=0.005,
                use_cpp_serializer=True,
            )
        )
        builder.Connect(
            iiwa_status.get_output_port(), iiwa_status_publisher.get_input_port()
        )


a = MockManipulation()
a.run()
