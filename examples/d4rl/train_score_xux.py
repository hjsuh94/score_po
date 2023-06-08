import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import hydra
import pickle
from omegaconf import DictConfig
import gym, d4rl

from score_po.score_matching import NoiseConditionedScoreEstimatorXux
from score_po.nn import MLPwEmbedding, TrainParams, Normalizer, generate_cosine_schedule


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]
    dataset = env.get_dataset()

    x = torch.Tensor(dataset["observations"])
    u = torch.Tensor(dataset["actions"])
    xnext = torch.Tensor(dataset["next_observations"])

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
    network = MLPwEmbedding(
        dim_x + dim_u + dim_x, dim_x + dim_u + dim_x, 4 * [1024], 10
    )
    params = TrainParams()
    params.load_from_config(cfg)

    sf = NoiseConditionedScoreEstimatorXux(
        dim_x=dim_x,
        dim_u=dim_u,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
        alpha=10,
    )

    sf.train_network(
        dataset,
        params,
        # sigma_lst=generate_cosine_schedule(0.4, 0.001, 10),
        sigma_lst=torch.logspace(1.0, -1, 10),
        split=False,
        sample_sigma=True,
    )


if __name__ == "__main__":
    main()
