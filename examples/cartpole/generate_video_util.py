import cv2
import os
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from examples.cartpole.cartpole_plant import CartpolePlant


def interpolate(x_trj: np.ndarray, N: int):
    nT = x_trj.shape[0]
    x_trj_interpolate = np.empty(((nT - 1) * N + 1, x_trj.shape[1]))
    x_trj_interpolate[0] = x_trj[0]
    for i in range(1, nT):
        x_trj_interpolate[(i - 1) * N : i * N, :] = np.linspace(
            x_trj[i - 1], x_trj[i], N+1
        )[:N]
    x_trj_interpolate[-1] = x_trj[-1]
    return x_trj_interpolate


def generate_video(image_folder: str, video_name, fps: int):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith("png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    height = int(np.floor(height / 2) * 2)
    width = int(np.floor(width/ 2) * 2)

    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    video = cv2.VideoWriter(
        os.path.join(image_folder, video_name), fourcc, fps, (width, height)
    )

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()

def generate_video_snapshots(plant: CartpolePlant, x_trj_np: np.ndarray, dt: float, N_interpolate, videofolder: str, video_name:str, title_prefix: str):
    x_trj_interpolated = interpolate(x_trj_np, N_interpolate)
    t_interpolate = dt / N_interpolate * np.arange(x_trj_interpolated.shape[0])
    # Now append some frames at the end of the video
    append_last_frame = 100
    x_trj_interpolated = np.concatenate(
        (
            x_trj_interpolated,
            x_trj_interpolated[-1].reshape((1, -1)).repeat(append_last_frame, axis=0),
        )
    )
    t_interpolate = np.concatenate((t_interpolate, np.ones(append_last_frame) * t_interpolate[-1]))

    for i in range(x_trj_interpolated.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        ax.set_ylim([-0.5, 0.75])
        x_trj_horizontal_range = [x_trj_interpolated[i, 0] - 1, x_trj_interpolated[i, 0] + 1]
        ax.set_xlim(x_trj_horizontal_range)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("z(m)", fontsize=16)
        plant.visualize(ax, x_trj_interpolated[i], color="g")
        ax.set_title(f"{title_prefix}t={t_interpolate[i]:.2f}s", fontsize=16)
        os.makedirs(videofolder, exist_ok=True)
        fig.savefig(
            os.path.join(videofolder, f"image{i:03d}.png"),
            format="png",
            bbox_inches="tight",
        )
    generate_video(videofolder, video_name, fps=100)