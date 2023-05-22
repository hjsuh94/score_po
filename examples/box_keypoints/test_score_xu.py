import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXu
from score_po.nn import MLPwEmbedding, train_network, TrainParams, Normalizer


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    with open(cfg.dataset_dir, "rb") as f:
        dataset = pickle.load(f)

    dim_x = 12
    dim_u = 2
    x = dataset.tensors[0]
    u = dataset.tensors[1]
    xnext = dataset.tensors[2]
    network = MLPwEmbedding(dim_x + dim_u, dim_x + dim_u, 4 * [1024], 10)

    sf = NoiseConditionedScoreEstimatorXu(
        dim_x=dim_x,
        dim_u=dim_u,
        network=network,
    )
    sf.load_state_dict(torch.load(cfg.score_weights))

    plt.figure()
    pts_history = torch.zeros(cfg.max_iters, 14)
    pts_history[0, :12] = x[327] + 0.1 * torch.randn_like(x[327])
    pts = pts_history[0, :10].reshape(5, 2).detach().cpu().numpy()
    plt.plot(pts[:, 0], pts[:, 1], "bo", label="initial point")

    for i in range(cfg.max_iters - 1):
        pts_history[i + 1, :12] = (
            pts_history[i, :12]
            + cfg.step_size * sf.get_score_z_given_z(pts_history[i, None, :], 9)[:, :12]
        )
        pts = pts_history[i + 1, :10].reshape(5, 2).detach().cpu().numpy()
        plt.plot(pts[:, 0], pts[:, 1], "ro", alpha=0.3 + 0.5 * (i / cfg.max_iters))

    plt.plot(pts[:, 0], pts[:, 1], "ko", label="final point")
    plt.legend()

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
