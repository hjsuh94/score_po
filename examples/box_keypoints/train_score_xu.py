import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import pickle
import hydra
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXu
from score_po.nn import MLPwEmbedding, TrainParams, Normalizer, generate_cosine_schedule


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    with open(cfg.dataset_dir, "rb") as f:
        dataset = pickle.load(f)

    x = dataset.tensors[0]
    u = dataset.tensors[1]
    xnext = dataset.tensors[2]
    dim_x = 12
    dim_u = 2

    x_max, _ = torch.max(x, axis=0)
    x_min, _ = torch.min(x, axis=0)
    k_x = x_max - x_min
    b_x = (x_max + x_min) / 2
    x_normalizer = Normalizer(k=k_x, b=b_x)

    u_max, _ = torch.max(u, axis=0)
    u_min, _ = torch.min(u, axis=0)
    k_u = u_max - u_min
    b_u = (u_max + u_min) / 2
    u_normalizer = Normalizer(k=k_u, b=b_u)

    # We need to append pusher coordinates to keypoints to fully define
    # the state.
    dataset = TensorDataset(x, u, xnext)
    network = MLPwEmbedding(dim_x + dim_u, dim_x + dim_u, 4 * [1024], 10)
    params = TrainParams()
    params.load_from_config(cfg)

    sf = NoiseConditionedScoreEstimatorXu(
        dim_x=dim_x,
        dim_u=dim_u,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )

    sf.train_network(
        dataset,
        params,
        sigma_lst=generate_cosine_schedule(0.2, 0.01, 10),
        split=False,
        sample_sigma=False,
    )


if __name__ == "__main__":
    main()
