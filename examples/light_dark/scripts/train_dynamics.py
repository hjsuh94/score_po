import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, TrainParams

from examples.light_dark.dynamics import SingleIntegrator

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    network = MLP(4, 2, cfg.nn_layers)
    dynamics = NNDynamicalSystem(2, 2, network)
    dataset_size = cfg.dataset_size
    
    dynamics_true = SingleIntegrator()
    x_batch = 2.0 * torch.rand(dataset_size, 2) - 1.0
    u_batch = 2.0 * torch.rand(dataset_size, 2) - 1.0
    xnext_batch = dynamics_true.dynamics_batch(x_batch, u_batch)
    dataset = TensorDataset(x_batch, u_batch, xnext_batch)
    
    params = TrainParams()
    params.load_from_config(cfg)

    dynamics.train_network(dataset, params)

if __name__ == "__main__":
    main()