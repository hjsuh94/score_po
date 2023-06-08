import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

import gym, d4rl

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, train_network, TrainParams, Normalizer


@hydra.main(config_path="./config", config_name="dynamics_training")
def main(cfg: DictConfig):
    env = gym.make(cfg.env_name)
    dim_x = env.observation_space.shape[0]
    dim_u = env.action_space.shape[0]
    dataset = env.get_dataset()

    x = torch.Tensor(dataset["observations"])
    xnext = torch.Tensor(dataset["next_observations"])
    u = torch.Tensor(dataset["actions"])

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

    dataset = TensorDataset(x, u, xnext)
    network = MLP(dim_x + dim_u, dim_x, 4 * [1024], layer_norm=True)
    params = TrainParams()
    params.load_from_config(cfg)

    nn_dynamics = NNDynamicalSystem(
        dim_x=dim_x,
        dim_u=dim_u,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    nn_dynamics.train_network(dataset, params, sigma=0)


if __name__ == "__main__":
    main()
