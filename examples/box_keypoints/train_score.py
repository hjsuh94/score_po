import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams


def get_canon_points(cfg):
    canon_points = torch.Tensor(
        [
            [1, -1, -1, 1, 0],
            [1, 1, -1, -1, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    canon_points[0, :] = cfg.box_width * canon_points[0, :] / 2
    canon_points[1, :] = cfg.box_height * canon_points[1, :] / 2
    return canon_points


def pose_batch_to_points_batch(pose_batch, canon_points):
    """Convert a batch of B x 3 x 3 pose batch matrices into keypoints."""
    batch_pts = torch.einsum("bij,jp->bip", pose_batch, canon_points)
    return batch_pts


def xytheta_batch_to_pose_batch(xytheta_batch):
    """Convert a xytheta_batch of B x 3 x 3 pose batch matrices into keypoints."""
    B = xytheta_batch.shape[0]

    x_batch = xytheta_batch[:, 0]
    y_batch = xytheta_batch[:, 1]
    theta_batch = xytheta_batch[:, 2]
    pose_batch = torch.zeros((B, 3, 3))
    pose_batch[:, 0, 0] = torch.cos(theta_batch)
    pose_batch[:, 0, 1] = -torch.sin(theta_batch)
    pose_batch[:, 1, 1] = torch.cos(theta_batch)
    pose_batch[:, 1, 0] = torch.sin(theta_batch)
    pose_batch[:, 0, 2] = x_batch
    pose_batch[:, 1, 2] = y_batch
    pose_batch[:, 2, 2] = torch.ones(B)
    return pose_batch


def flatten_points_batch(pts_batch):
    """Convert pts_batch of shape"""
    B = pts_batch.shape[0]
    return pts_batch[:, :2, :].view(B, 10)


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    pts = get_canon_points(cfg)

    x_batch = 2.0 * torch.rand(cfg.dataset_size) - 1.0
    y_batch = 2.0 * torch.rand(cfg.dataset_size) - 1.0
    theta_batch = 2.0 * np.pi * torch.rand(cfg.dataset_size)

    xytheta_batch = torch.vstack((x_batch, y_batch, theta_batch)).T
    pose_batch = xytheta_batch_to_pose_batch(xytheta_batch)
    pts_batch = pose_batch_to_points_batch(pose_batch, pts)
    pts_flatten_batch = flatten_points_batch(pts_batch)

    if cfg.plot_data:
        test_batch = pts_batch[:1000]
        plt.figure()
        for i in range(1000):
            poly = Polygon(test_batch[i, :2, :4].T, fill=False)
            plt.gca().add_patch(poly)
            plt.plot(pts_batch[i, 0, :], pts_batch[i, 1, :], "ro")
        plt.axis("equal")
        plt.show()

    dataset = TensorDataset(pts_flatten_batch)
    network = MLP(10, 10, cfg.nn_layers)
    sf = ScoreEstimatorXu(10, 0, network)

    params = TrainParams()
    params.load_from_config(cfg)

    sf.train_network(dataset, params, 0.1, split=False)


if __name__ == "__main__":
    main()
