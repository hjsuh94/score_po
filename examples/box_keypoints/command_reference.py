import numpy as np

from plan_runner_client.zmq_client import PlanManagerZmqClient
from plan_runner_client.calc_plan_msg import calc_task_space_plan_msg

from pydrake.all import RigidTransform, RollPitchYaw


def normalize(x):
    return x / np.linalg.norm(x)


zmq_client = PlanManagerZmqClient()
frame_E = zmq_client.plant.GetFrameByName("iiwa_link_7")
X_ET = RigidTransform()

q_now = zmq_client.get_current_joint_angles()
X_WE = zmq_client.get_current_ee_pose(frame_E)

t_knots = np.linspace(0, 5.0, 2)
X_WT_lst = []
X_WT_lst.append(zmq_client.get_current_ee_pose(frame_E))
X_WT_lst.append(
    RigidTransform(RollPitchYaw([0, -np.pi, 0]), np.array([0.4, 0.0, 0.15]))
)

plan_msg = calc_task_space_plan_msg(RigidTransform(), X_WT_lst, t_knots)
zmq_client.send_plan(plan_msg)
zmq_client.wait_for_plan_to_finish()
print(zmq_client.get_current_joint_angles())
