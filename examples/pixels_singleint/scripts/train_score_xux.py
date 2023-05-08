import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.score_matching import ScoreEstimatorXux
from score_po.nn import MLP, TrainParams, Normalizer

from examples.pixels_singleint.dynamics import SingleIntegratorPixels
from examples.pixels_singleint.environment import Environment


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    env = Environment()
    dynamics = SingleIntegratorPixels()
    env.add_ellipse([0, 0], 0.4, 0.4)
    nx = 32 # image dimension
    nu = 2 # control dimension

    u_normalizer = Normalizer(k=0.2 * torch.ones(2), b=torch.zeros(2))
    if not cfg.load_data:
        y_batch, x_batch = env.sample_image(cfg.dataset_size, nx, filter_obs=True)
        u_batch = 0.1 * 2.0 * (torch.rand(x_batch.shape[0], 2) - 0.5)
        xnext_batch = dynamics.dynamics_batch(x_batch, u_batch)
        ynext_batch, _ = env.sample_image(cfg.dataset_size, nx, samples=xnext_batch, filter_obs=False)
    else:
        y_batch, u_batch, ynext_batch = torch.load(cfg.load_data_path)
    dataset = TensorDataset(y_batch.reshape(y_batch.shape[0], -1).float().to(cfg.device),
                            u_batch.float().to(cfg.device),
                            ynext_batch.reshape(y_batch.shape[0], -1).float().to(cfg.device)
                            )

    params = TrainParams()
    params.load_from_config(cfg)

    network = MLP(2*int(nx**2) + nu, 2*int(nx**2) + nu, cfg.nn_layers)
    sf = ScoreEstimatorXux(
        int(nx**2), nu, network, x_normalizer=None, u_normalizer=u_normalizer)
    sf.train_network(dataset, params, sigma=0.1 * torch.ones(1))


if __name__ == "__main__":
    main()