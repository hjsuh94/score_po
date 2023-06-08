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
import matplotlib
from matplotlib.patches import Rectangle, Polygon
from pydrake.all import RotationMatrix


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


def plot_demonstration(
    ax,
    marker,
    pusher,
    marker_color="k",
    pusher_color="k",
    marker_alpha=0.01,
    pusher_alpha=0.01,
    linestyle="-",
    marker_label=None,
    pusher_label=None,
):
    marker_xy = marker.reshape(marker.shape[0], 5, 2)
    for j in range(5):
        ax.plot(
            0.1 + marker_xy[:, j, 0],
            marker_xy[:, j, 1],
            linestyle=linestyle,
            color=marker_color,
            alpha=marker_alpha,
            label=None if j < 4 else marker_label,
        )
        ax.plot(
            pusher[:, 0],
            pusher[:, 1],
            linestyle=linestyle,
            color=pusher_color,
            alpha=pusher_alpha,
            label=None if j < 4 else pusher_label,
        )


def process_logs(log_path):
    log = lcm.EventLog(log_path, "r")

    marker_time = []
    marker_state = []

    pusher_time = []
    pusher_state = []

    command_time = []
    pusher_command = []

    # 1. Collect all events in the log into an array
    for event in log:
        if event.channel == "OPTITRACK_FRAMES":
            msg = optitrack_frame_t.decode(event.data)

            for i, marker_set in enumerate(msg.marker_sets):
                if marker_set.name == "box":
                    keypts = np.array(marker_set.xyz)[:, :2].flatten()
                    if np.any(np.array(marker_set.xyz)[:, 0] == 0):
                        pass
                    else:
                        time = float(msg.utime / 1e6)
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

    marker_state = np.array(marker_state)
    pusher_state = np.array(pusher_state)
    return marker_state, pusher_state


def compute_transform(marker_now, marker_next):
    marker_now = marker_now.reshape(5, 2)
    marker_next = marker_next.reshape(5, 2)
    translation = np.mean(marker_next, axis=0) - np.mean(marker_now, axis=0)
    marker_now -= np.mean(marker_now, axis=0)
    marker_next -= np.mean(marker_next, axis=0)
    data_mat = marker_now.T @ marker_next
    U, S, V = np.linalg.svd(data_mat)
    D = np.diag(np.array([1, np.linalg.det(U @ V.T)]))
    rotation = U @ D @ V.T
    return rotation.T, translation


log_dir_training = os.path.join(
    os.getcwd(), "examples/box_keypoints/hardware_logs/logs_training"
)
filenames = os.listdir(log_dir_training)

plt.figure(figsize=(8, 7))
matplotlib.rcParams.update({"font.size": 15})

circle = plt.Circle(np.array([0.4246, 0.0]), radius=0.035, edgecolor="r", fill=False)
plt.gca().add_patch(circle)
box = Rectangle(
    np.array([0.4246 + 0.035, -0.1703 / 2]),
    0.0867,
    0.1703,
    edgecolor="k",
    facecolor="k",
    fill=True,
    alpha=0.1,
)
plt.gca().add_patch(box)
plt.plot(
    [0.78, 0.78],
    [-0.25, 0.25],
    color="royalblue",
    alpha=0.3,
    linewidth=4.0,
    # label="Goal Line",
)

"""
for i, log_name in enumerate(filenames):
    log_path = os.path.join(log_dir_training, log_name)
    marker_state, pusher_state = process_logs(log_path)
    if i < len(filenames) - 1:
        marker_label = None
        pusher_label = None
    else:
        marker_label = "Keypoint Trajectory"
        pusher_label = "Pusher Trajectory"
    plot_demonstration(
        plt.gca(),
        marker_state,
        pusher_state,
        marker_color="black",
        pusher_color="r",
        marker_alpha=0.1,
        pusher_alpha=0.1,
        marker_label=marker_label,
        pusher_label=pusher_label,
    )
"""

c_x = 0.0
t_x = 0.4246 + 0.035 + 0.0867 / 2
c_y = 0.0
l_x = 0.0867 / 2
l_y = 0.1703 / 2
canonical_box = np.array(
    [
        [c_x + l_x, c_x + l_x, c_x - l_x, c_x - l_x],
        [c_y + l_y, c_y - l_y, c_y - l_y, c_y + l_y],
    ]
)

log_dir = os.path.join(os.getcwd(), "examples/box_keypoints/hardware_logs/logs_center")
filename = os.path.join(log_dir, os.listdir(log_dir)[2])
marker_state, pusher_state = process_logs(filename)
plot_demonstration(
    plt.gca(),
    marker_state,
    pusher_state,
    marker_color="springgreen",
    pusher_color="r",
    marker_alpha=0.5,
    pusher_alpha=0.6,
    marker_label="Keypoint Trajectory (MBDP)",
    pusher_label="Pusher Trajectory (MBDP)",
)
marker_final = marker_state[-1].reshape(5, 2)
plt.plot(
    marker_final[:, 0] + 0.1,
    marker_final[:, 1],
    linestyle="",
    marker="o",
    color="springgreen",
)
marker_final = marker_state[0].reshape(5, 2)
plt.plot(
    marker_final[:, 0] + 0.1,
    marker_final[:, 1],
    linestyle="",
    marker="o",
    color="black",
)


R, t = compute_transform(marker_state[0], marker_state[-1])
next_box = R @ canonical_box + np.array([t_x, 0])[:, None] + t[:, None]
poly = Polygon(
    next_box.T,
    fill=True,
    closed=True,
    facecolor="springgreen",
    edgecolor="springgreen",
    alpha=0.1,
)
plt.gca().add_patch(poly)


log_dir = os.path.join(
    os.getcwd(), "examples/box_keypoints/hardware_logs/logs_center_nobeta"
)
filename = os.path.join(log_dir, os.listdir(log_dir)[1])
marker_state, pusher_state = process_logs(filename)
plot_demonstration(
    plt.gca(),
    marker_state,
    pusher_state,
    marker_color="magenta",
    pusher_color="r",
    marker_alpha=0.5,
    pusher_alpha=0.2,
    linestyle="--",
    marker_label="Keypoint Trajectory (Vanilla MBRL)",
    pusher_label="Pusher Trajectory (Vanilla MBRL)",
)
marker_final = marker_state[-1].reshape(5, 2)
plt.plot(
    marker_final[:, 0] + 0.1,
    marker_final[:, 1],
    linestyle="",
    marker="o",
    color="magenta",
)

R, t = compute_transform(marker_state[0], marker_state[-1])
next_box = R @ canonical_box + np.array([t_x, 0])[:, None] + t[:, None]
poly = Polygon(
    next_box.T,
    fill=True,
    closed=True,
    facecolor="magenta",
    edgecolor="magenta",
    alpha=0.1,
)
plt.gca().add_patch(poly)

keypts_goal = np.load("examples/box_keypoints/goals/keypts_center.npy")
keypts_goal = keypts_goal.reshape(5, 2)
plt.plot(keypts_goal[:, 0] + 0.1, keypts_goal[:, 1], "bo", label="Goal Keypoints")

plt.xlim([-0.35, 0.9])
plt.ylim([-0.3, 0.3])
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("center.png")
