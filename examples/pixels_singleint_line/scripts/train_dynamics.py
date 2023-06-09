import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.dynamical_system import NNDynamicalSystem_Image
from score_po.nn import MLP, TrainParams, Normalizer
from score_po.image_nn import NN_ImageU_Dynamics

from examples.pixels_singleint_line.dynamics import SingleIntegratorPixels
from examples.pixels_singleint_line.environment import Environment

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    env = Environment()
    dynamics = SingleIntegratorPixels()
    env.add_ellipse([0, 0], 0.4, 0.4)
    nx = 32  # image width/height
    nu = 32  # number of control inputs
    u_normalizer = Normalizer(k=0.2 * torch.ones(2), b=torch.zeros(2))
    if not cfg.load_data:
        y_batch, x_batch = env.sample_image(cfg.dataset_size, nx, filter_obs=True)
        u_batch = 0.1 * 2.0 * (torch.rand(x_batch.shape[0], 2) - 0.5)
        xnext_batch = dynamics.dynamics_batch(x_batch, u_batch)
        u_batch_im = env.sample_control_image(nx, u_batch, buf=2.)
        ynext_batch, _ = env.sample_image(cfg.dataset_size, nx, samples=xnext_batch, filter_obs=False)
    else:
        y_batch, u_batch_im, ynext_batch, _, _, _ = torch.load(cfg.load_data_path)

    dataset = TensorDataset(y_batch.reshape(y_batch.shape[0], -1).float().to(cfg.train.device),
                            u_batch_im.reshape(y_batch.shape[0], -1).float().to(cfg.train.device),
                            ynext_batch.reshape(y_batch.shape[0], -1).float().to(cfg.train.device)
                            )
    params = TrainParams()
    params.load_from_config(cfg)

    network = NN_ImageU_Dynamics(cfg).to(cfg.train.device)

    params = TrainParams()
    params.load_from_config(cfg)

    dynamics = NNDynamicalSystem_Image(int(nx**2), int(nu**2), network, x_normalizer=None, u_normalizer=None)

    dynamics.train_network(dataset, params)


if __name__ == "__main__":
    main()