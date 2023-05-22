from future.utils import iteritems
import os
import math
import numpy as np

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.manipulation.schunk_wsg import SchunkWsgPositionController
from pydrake.multibody.parsing import (
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import (
    CoulombFriction,
    MultibodyPlant,
    ConnectContactResultsToDrakeVisualizer,
)
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.tree import (
    SpatialInertia,
    UnitInertia,
    RevoluteJoint,
    PrismaticJoint,
)
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.primitives import (
    Adder,
    Demultiplexer,
    MatrixGain,
    Multiplexer,
    PassThrough,
    StateInterpolatorWithDiscreteDerivative,
)


class ManipulationDiagram:
    def __init__(self):
        pass

    # === Property accessors ========================================
    @property
    def model_ids(self):
        return self._model_ids

    @property
    def port_names(self):
        return self._port_names

    def get_control_mbp(self, name):
        return self._control_mbp[name]

    def add_arm_gripper(
        self,
        arm_name,
        arm_path,
        arm_base,
        X_arm,
        gripper_path,
        arm_ee,
        gripper_base,
        X_gripper,
    ):
        # Add arm
        parser = Parser(self._mbp, self._sg)
        arm_model_id = parser.AddModelFromFile(arm_path, arm_name)
        arm_base_frame = self._mbp.GetFrameByName(arm_base, arm_model_id)
        self._mbp.WeldFrames(self._mbp.world_frame(), arm_base_frame, X_arm)
        self._model_ids[arm_name] = arm_model_id

        # Add gripper
        gripper_name = arm_name + "_gripper"
        arm_end_frame = self._mbp.GetFrameByName(arm_ee, arm_model_id)
        self.add_floating_gripper(
            gripper_name, gripper_path, arm_end_frame, gripper_base, X_gripper
        )

        # Add arm controller stack
        ctrl_plant = MultibodyPlant(self._config["env"]["mbp_dt"])
        parser = Parser(ctrl_plant)
        ctrl_arm_id = parser.AddModelFromFile(arm_path, arm_name)
        arm_base_frame = ctrl_plant.GetFrameByName(arm_base, ctrl_arm_id)
        ctrl_plant.WeldFrames(ctrl_plant.world_frame(), arm_base_frame, X_arm)

        gripper_equivalent = ctrl_plant.AddRigidBody(
            gripper_name + "_equivalent",
            ctrl_arm_id,
            self.calculate_ee_composite_inertia(gripper_path),
        )
        arm_end_frame = ctrl_plant.GetFrameByName(arm_ee, ctrl_arm_id)
        ctrl_plant.WeldFrames(
            arm_end_frame,
            gripper_equivalent.body_frame(),
            X_gripper.multiply(RigidTransform([0, 0, -0.1])),
        )

        ctrl_plant.Finalize()
        self._control_mbp[arm_name] = ctrl_plant
        arm_num_positions = ctrl_plant.num_positions(ctrl_arm_id)
        kp = 4000 * np.ones(arm_num_positions)
        ki = 0 * np.ones(arm_num_positions)
        kd = 5 * np.sqrt(kp)
        arm_controller = self._builder.AddSystem(
            InverseDynamicsController(ctrl_plant, kp, ki, kd, False)
        )
        adder = self._builder.AddSystem(Adder(2, arm_num_positions))
        state_from_position = self._builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                arm_num_positions, self._mbp.time_step(), True
            )
        )

        # Add command pass through and state splitter
        arm_command = self._builder.AddSystem(PassThrough(arm_num_positions))
        state_split = self._builder.AddSystem(
            Demultiplexer(2 * arm_num_positions, arm_num_positions)
        )

        def finalize_func():
            builder = self._builder

            # Export positions commanded
            command_input_name = arm_name + "_position"
            command_output_name = arm_name + "_position_commanded"
            self._port_names.extend([command_input_name, command_output_name])
            builder.ExportInput(arm_command.get_input_port(0), command_input_name)
            builder.ExportOutput(arm_command.get_output_port(0), command_output_name)

            # Export arm state ports
            builder.Connect(
                self._mbp.get_state_output_port(arm_model_id),
                state_split.get_input_port(0),
            )
            arm_q_name = arm_name + "_position_measured"
            arm_v_name = arm_name + "_velocity_estimated"
            arm_state_name = arm_name + "_state_measured"
            self._port_names.extend([arm_q_name, arm_v_name, arm_state_name])
            builder.ExportOutput(state_split.get_output_port(0), arm_q_name)
            builder.ExportOutput(state_split.get_output_port(1), arm_v_name)
            builder.ExportOutput(
                self._mbp.get_state_output_port(arm_model_id), arm_state_name
            )

            # Export controller stack ports
            builder.Connect(
                self._mbp.get_state_output_port(arm_model_id),
                arm_controller.get_input_port_estimated_state(),
            )
            builder.Connect(
                arm_controller.get_output_port_control(), adder.get_input_port(0)
            )
            builder.Connect(
                adder.get_output_port(0),
                self._mbp.get_actuation_input_port(arm_model_id),
            )
            builder.Connect(
                state_from_position.get_output_port(0),
                arm_controller.get_input_port_desired_state(),
            )
            builder.Connect(
                arm_command.get_output_port(0), state_from_position.get_input_port(0)
            )
            torque_input_name = arm_name + "_feedforward_torque"
            torque_output_cmd_name = arm_name + "_torque_commanded"
            torque_output_est_name = arm_name + "_torque_measured"
            self._port_names.extend(
                [torque_input_name, torque_output_cmd_name, torque_output_est_name]
            )
            builder.ExportInput(adder.get_input_port(1), torque_input_name)
            builder.ExportOutput(adder.get_output_port(0), torque_output_cmd_name)
            builder.ExportOutput(adder.get_output_port(0), torque_output_est_name)

            external_torque_name = arm_name + "_torque_external"
            self._port_names.append(external_torque_name)
            builder.ExportOutput(
                self._mbp.get_generalized_contact_forces_output_port(arm_model_id),
                external_torque_name,
            )

        self._finalize_functions.append(finalize_func)

    def add_object_from_file(self, object_name, object_path):
        parser = Parser(self._mbp, self._sg)
        self._model_ids[object_name] = parser.AddModelFromFile(object_path, object_name)
        # this should be masked
        self._model_names_to_mask.append(object_name)

        def finalize_func():
            state_output_port = self._mbp.get_state_output_port(
                self._model_ids[object_name]
            )
            port_state_output_name = object_name + "_state_output"
            self._port_names.append(port_state_output_name)
            self._builder.ExportOutput(state_output_port, port_state_output_name)

        self._finalize_functions.append(finalize_func)

    def process_model_directives(self):
        parser = Parser(self._mbp, self._sg)

        # Load all the required external packages from config file.
        for package in self._config["external_packages"]:
            package_path = self._config["external_packages"][package]["path"]
            if "drake" in package_path:
                package_path = FindResourceOrThrow(package_path)
            parser.package_map().Add(package, os.path.dirname(package_path))

        directive_file = self._config["directives"]

        # Read and process model directives.
        directives = LoadModelDirectives(directive_file)
        model_info_lst = ProcessModelDirectives(directives, self._mbp, parser)

        for model_info in model_info_lst:
            object_name = model_info.model_name
            object_id = model_info.model_instance
            self._model_ids[object_name] = object_id

        def finalize_func():
            for model_info in model_info_lst:
                state_output_port = self._mbp.get_state_output_port(
                    model_info.model_instance
                )
                port_state_output_name = model_info.model_name + "_state_output"
                self.port_names.append(port_state_output_name)
                self._builder.ExportOutput(state_output_port, port_state_output_name)

        self._finalize_functions.append(finalize_func)
        return model_info_lst

    def connect_contact_results(self):
        # Call this function last since it needs finalized plant.
        def finalize_func():
            ConnectContactResultsToDrakeVisualizer(self._builder, self._mbp)

        self._finalize_functions.append(finalize_func)

    # === Add Cameras ===============================================
    def add_rgbd_sensors_from_config(self, config):
        super(ManipulationDiagram, self).add_rgbd_sensors_from_config(config["env"])

    # === Finalize the completed diagram ============================
    def finalize(self):
        super(ManipulationDiagram, self).finalize()
        self._port_names.extend(
            [
                "pose_bundle",
                "contact_results",
                "reaction_forces",
                "plant_continuous_state",
                "geometry_poses",
                "spatial_input",
                "body_poses",
            ]
        )

    # === State getters & setters ===================================
    def get_model_state(self, context, model_name):
        assert self.is_finalized()
        model_idx = self._model_ids[model_name]
        mbp_context = self.GetMutableSubsystemContext(self._mbp, context)

        d = dict()
        d["position"] = np.copy(self._mbp.GetPositions(mbp_context, model_idx))
        d["velocity"] = np.copy(self._mbp.GetVelocities(mbp_context, model_idx))
        return d

    def set_model_state(self, context, model_name, q, v):
        assert self.is_finalized()
        model_idx = self._model_ids[model_name]
        mbp_context = self.GetMutableSubsystemContext(self._mbp, context)
        self._mbp.SetPositions(mbp_context, model_idx, q)
        self._mbp.SetVelocities(mbp_context, model_idx, v)
