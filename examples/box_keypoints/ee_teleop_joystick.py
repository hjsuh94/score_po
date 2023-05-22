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

# On macOS, our setup scripts do not provide pygame so we need to skip this
# program and its tests.  On Ubuntu, we do expect to have pygame.
if sys.platform == "darwin":
    try:
        import pygame
        from pygame.locals import *
    except ImportError:
        print("ERROR: missing pygame.  " "Please install pygame to use this example.")
        sys.exit(0)
else:
    import pygame
    from pygame.locals import *


def initialize_joystick(joystick_id):
    assert isinstance(joystick_id, (int, type(None)))
    pygame.init()
    try:
        pygame.joystick.init()
        if joystick_id is None:
            count = pygame.joystick.get_count()
            if count != 1:
                raise RuntimeError(
                    f"joystick_id=None, but there are {count} joysticks "
                    f"plugged in. Please specify --joystick_id, or ensure "
                    f"that exactly 1 joystick is plugged in"
                )
            joystick_id = 0
        joystick = pygame.joystick.Joystick(joystick_id)
        joystick.init()
        return joystick
    except pygame.error as e:
        raise Exception(
            "Make sure dualshock 4 controller is connected. "
            "Controller initialization failed "
            f"with: {e}"
        )


class DS4Buttons(Enum):
    X_BUTTON = 0
    O_BUTTON = 1
    TRIANGLE_BUTTON = 2
    SQUARE_BUTTON = 3
    L1_BUTTON = 4
    R1_BUTTON = 5
    L2_BUTTON = 6
    R2_BUTTON = 7


class DS4Axis(Enum):
    LEFTJOY_UP_DOWN = 0
    LEFTJOY_LEFT_RIGHT = 1
    RIGHTJOY_LEFT_RIGHT = 2
    RIGHTJOY_UP_DOWN = 3


def print_instructions():
    instructions = """\

        END EFFECTOR CONTROL
        -----------------------------------------
        +/- x-axis         - leftjoy left / right
        +/- y-axis         - leftjoy up / down
        +/- roll           - rightjoy up / down
        +/- pitch          - rightjoy left / right
        +/- z-axis         - l2 / r2
        +/- yaw            - l1 / r1

        GRIPPER CONTROL
        -----------------------------------------
        open / close       - square / circle (O)

        -----------------------------------------
        x button           - quit
    """
    print(dedent(instructions))


class TeleopDualShock4Manager:
    def __init__(self, joystick):
        self._joystick = joystick
        self._axis_data = list()
        self._button_data = list()
        self._name = joystick.get_name()
        print(f"Using Joystick: {self._name}")

        for i in range(self._joystick.get_numbuttons()):
            self._button_data.append(False)

        for i in range(self._joystick.get_numaxes()):
            self._axis_data.append(0.0)

    def get_events(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self._axis_data[event.axis] = round(event.value, 2)
            if event.type == pygame.JOYBUTTONDOWN:
                self._button_data[event.button] = True
            if event.type == pygame.JOYBUTTONUP:
                self._button_data[event.button] = False

        events = dict()
        # For example mappings, see:
        # https://www.pygame.org/docs/ref/joystick.html#controller-mappings
        if self._name == "Logitech Logitech Dual Action":
            events[DS4Axis.LEFTJOY_LEFT_RIGHT] = self._axis_data[0]
            events[DS4Axis.LEFTJOY_UP_DOWN] = self._axis_data[1]
            events[DS4Axis.RIGHTJOY_LEFT_RIGHT] = self._axis_data[2]
            events[DS4Axis.RIGHTJOY_UP_DOWN] = self._axis_data[3]
        else:
            events[DS4Axis.LEFTJOY_UP_DOWN] = self._axis_data[0]
            events[DS4Axis.LEFTJOY_LEFT_RIGHT] = self._axis_data[1]
            events[DS4Axis.RIGHTJOY_LEFT_RIGHT] = self._axis_data[3]
            events[DS4Axis.RIGHTJOY_UP_DOWN] = self._axis_data[4]
        events[DS4Buttons.X_BUTTON] = self._button_data[0]
        events[DS4Buttons.O_BUTTON] = self._button_data[1]
        events[DS4Buttons.SQUARE_BUTTON] = self._button_data[3]
        events[DS4Buttons.L1_BUTTON] = self._button_data[4]
        events[DS4Buttons.R1_BUTTON] = self._button_data[5]
        events[DS4Buttons.L2_BUTTON] = self._button_data[6]
        events[DS4Buttons.R2_BUTTON] = self._button_data[7]

        # TODO(eric.cousineau): Replace `sys.exit` with a status to
        # the Systems Framework.
        if events[DS4Buttons.X_BUTTON]:
            sys.exit(0)
        return events


class JoystickTeleopDiffIK:
    def __init__(self):
        # Setup parameters for telop
        self.joystick = initialize_joystick(1)
        self.teleop_manager = TeleopDualShock4Manager(self.joystick)

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

        self.q = None
        self.v = None
        self.lc = lcm.LCM()
        self.lc.subscribe("IIWA_STATUS", self.get_state)
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

    def get_command(self):
        events = self.teleop_manager.get_events()
        scale = 0.05
        delta_x = scale * events[DS4Axis.LEFTJOY_LEFT_RIGHT]
        delta_y = scale * events[DS4Axis.LEFTJOY_UP_DOWN]
        delta_z = scale * events[DS4Axis.RIGHTJOY_UP_DOWN]
        return np.array([delta_x, delta_y, delta_z])

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
        # Assume 200 Hz, so command is sent around every 0.5s
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

    def wait_for_message(self):
        while True:
            self.lc.handle()
            if self.q is not None:
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


a = JoystickTeleopDiffIK()
a.run()
