import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torch.utils.data import TensorDataset

import hydra
import pickle
from omegaconf import DictConfig
import gym, d4rl

from score_po.score_matching import ScoreEstimator
from score_po.nn import MLP, TrainParams, Normalizer


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)

    x = torch.Tensor(dataset["observations"])
    u = torch.Tensor(dataset["actions"])
    z = torch.cat((x, u), dim=1)
    
    z_max, _ = torch.max(z, axis=0)
    z_min, _ = torch.min(z, axis=0)
    k_z = z_max - z_min
    b_z = (z_max + z_min) / 2
    z_normalizer = Normalizer(k=k_z, b=b_z)

    # We need to append pusher coordinates to keypoints to fully define
    # the state.
    dataset = TensorDataset(z)
    network = MLP(dim_x + dim_u, dim_x + dim_u, cfg.nn_layers)
    params = TrainParams()
    params.load_from_config(cfg)

    sf = ScoreEstimator(dim_x=dim_x, dim_u=dim_u, network=network,
                        z_normalizer=z_normalizer)
    sf.train_network(
        dataset, params, sigma=cfg.sigma, mode="denoising", split=False)


if __name__ == "__main__":
    main()
