import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    network = MLP(10, 10, cfg.nn_layers)
    sf = ScoreEstimatorXu(10, 0, network)
    sf.load_state_dict(
        torch.load(
            os.path.join(
                get_original_cwd(), "examples/box_keypoints/weights/checkpoint.pth"
            )
        )
    )

    plt.figure()
    pts_history = torch.zeros(cfg.max_iters, 10)
    pts_history[0, :] = torch.rand(10)
    pts = pts_history[0, :].reshape(2, 5).detach().cpu().numpy()
    plt.plot(pts[0], pts[1], "bo", label="initial point")

    for i in range(cfg.max_iters - 1):
        pts_history[i + 1, :] = pts_history[
            i, :
        ] + cfg.step_size * sf.get_score_z_given_z(pts_history[i, :])
        pts = pts_history[i + 1, :].reshape(2, 5).detach().cpu().numpy()
        plt.plot(pts[0], pts[1], "ro", alpha=0.3 + 0.5 * (i / cfg.max_iters))

    plt.plot(pts[0], pts[1], "ko", label="final point")
    plt.legend()

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
