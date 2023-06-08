import os, sys
import numpy as np
import lcm
import torch
import pickle

from drake import lcmt_iiwa_command
from optitrack import optitrack_frame_t, optitrack_marker_set_t
from pydrake.all import PiecewisePolynomial
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt


def visualize_demonstration(x, u, xnext):
    for t in range(x.shape[0] - 1):
        x_now = x[t].numpy()
        u_now = u[t].numpy()
        x_next = xnext[t].numpy()

        plt.figure()

        circle = plt.Circle(x_now[10:12], radius=0.035, edgecolor="k", fill=False)
        plt.gca().add_patch(circle)

        circle = plt.Circle(x_next[10:12], radius=0.035, edgecolor="r", fill=False)
        plt.gca().add_patch(circle)

        plt.arrow(x_now[10], x_now[11], u_now[0], u_now[1])

        keypts_now = x_now[:10].reshape(5, 2)
        plt.plot(keypts_now[:, 0], keypts_now[:, 1], "ko")

        keypts_next = x_next[:10].reshape(5, 2)
        plt.plot(keypts_next[:, 0], keypts_next[:, 1], "ro")

        plt.axis("equal")
        plt.xlim([0.4, 0.9])
        plt.ylim([-0.25, 0.25])
        plt.savefig(
            os.path.join(
                os.getcwd(), "examples/box_keypoints/visual/{:05d}.png".format(t)
            )
        )

        plt.close()


x_lst = []
u_lst = []
xnext_lst = []

log_dir = os.path.join(os.getcwd(), "examples/box_keypoints/logs")
filenames = os.listdir(log_dir)

for i, log_name in enumerate(filenames):
    log_path = os.path.join(log_dir, log_name)
    log = lcm.EventLog(log_path, "r")

    marker_time = []
    marker_state = []

    pusher_time = []
    pusher_state = []

    command_time = []
    pusher_command = []

    # 1. Collect all events in the log into an array
    for event in log:
        if event.channel == "KEYPTS":
            msg = optitrack_frame_t.decode(event.data)

            time = float(msg.utime / 1e6)
            keypts = np.array(msg.marker_sets[0].xyz)[:, :2].flatten()

            marker_time.append(time)
            marker_state.append(keypts)

        if event.channel == "PUSHER":
            msg = lcmt_iiwa_command.decode(event.data)

            time = float(msg.utime / 1e6)
            state = np.array(msg.joint_position)

            pusher_time.append(time)
            pusher_state.append(state)

        if event.channel == "PUSHER_COMMAND":
            msg = lcmt_iiwa_command.decode(event.data)

            time = float(msg.utime / 1e6)
            command = np.array(msg.joint_position)

            command_time.append(time)
            pusher_command.append(command)

    marker_time = np.array(marker_time)
    marker_state = np.array(marker_state)
    pusher_time = np.array(pusher_time)
    pusher_state = np.array(pusher_state)
    command_time = np.array(command_time)
    pusher_command = np.array(pusher_command)

    # 2. Interpolate along command times.
    marker_trj = PiecewisePolynomial.FirstOrderHold(marker_time, marker_state.T)
    pusher_trj = PiecewisePolynomial.FirstOrderHold(pusher_time, pusher_state.T)

    # 3. Compute x and u array.

    for t in range(len(command_time) - 1):
        time_now = command_time[t]
        marker_now = marker_trj.value(time_now)
        pusher_now = pusher_trj.value(time_now)
        x_now = np.concatenate((marker_now, pusher_now))[:, 0]
        u_now = pusher_command[t]

        time_next = command_time[t + 1]
        marker_next = marker_trj.value(time_next)
        pusher_next = pusher_trj.value(time_next)
        x_next = np.concatenate((marker_next, pusher_next))[:, 0]

        x_lst.append(x_now)
        u_lst.append(u_now)
        xnext_lst.append(x_next)

x_tensor = torch.Tensor(np.array(x_lst))
u_tensor = torch.Tensor(np.array(u_lst))
xnext_tensor = torch.Tensor(np.array(xnext_lst))
visualize_demonstration(x_tensor, u_tensor, xnext_tensor)

dataset = TensorDataset(x_tensor, u_tensor, xnext_tensor)
with open("tensor_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
