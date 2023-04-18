import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pickle
import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.dynamical_system import DynamicalSystem
from box_pushing_system import PlanarPusherSystem


@hydra.main(config_path="./config", config_name="data_collection")
def main(cfg: DictConfig):
    system = PlanarPusherSystem(get_original_cwd())

    x_data = np.zeros((cfg.dataset_size, 5))
    u_data = np.zeros((cfg.dataset_size, 2))
    xnext_data = np.zeros((cfg.dataset_size, 5))
    keypts_x_data = np.zeros((cfg.dataset_size, 10))
    keypts_xnext_data = np.zeros((cfg.dataset_size, 10))

    count = 0
    start_time = time.time()
    while count < cfg.dataset_size:
        if cfg.frame == "world":
            x = 0.8 * (np.random.rand(5) - 0.5)
        elif cfg.frame == "body":
            x = 0.8 * (np.random.rand(5) - 0.5)
            x[0:3] = 0.0
        else:
            raise ValueError

        if not system.is_in_collision(x):
            u = 0.2 * (np.random.rand(2) - 0.5)
            xnext = system.dynamics(x, u, record=False)
            keypts_x = system.get_keypoints(x, noise_std=cfg.noise_std)
            keypts_xnext = system.get_keypoints(xnext, noise_std=cfg.noise_std)

            x_data[count, :] = x
            u_data[count, :] = u
            xnext_data[count, :] = xnext
            keypts_x_data[count, :] = keypts_x.flatten()
            keypts_xnext_data[count, :] = keypts_xnext.flatten()

            count += 1
            print(
                "data_collected: {:07d} , time_elapsed: {:.2f}".format(
                    count, time.time() - start_time
                )
            )

    data_dict = {
        "x": x_data,
        "u": u_data,
        "xnext": xnext_data,
        "keypts_x": keypts_x_data,
        "keypts_xnext": keypts_xnext_data,
    }

    with open(cfg.dataset_dir, "wb") as f:
        pickle.dump(data_dict, f)


if __name__ == "__main__":
    main()
