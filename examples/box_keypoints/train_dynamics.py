import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, train_network, TrainParams, Normalizer


@hydra.main(config_path="./config", config_name="dynamics_training")
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
    nn_dynamics.train_network(dataset, params)


if __name__ == "__main__":
    main()
