import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from score_po.dynamical_system import (
    NNEnsembleDynamicalSystem,
    NNDynamicalSystem)
from score_po.nn import MLP, EnsembleNetwork, TrainParams, Normalizer

from examples.light_dark.dynamics import SingleIntegrator

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    dynamics_true = SingleIntegrator()
    u_normalizer = Normalizer(k =0.1 * torch.ones(2), b= torch.zeros(2))
    network = MLP(4, 2, [128, 128, 128])
    dynamics = NNDynamicalSystem(2, 2, network=network, u_normalizer=u_normalizer)
    dynamics_lst = [dynamics.to(cfg.train.device) for _ in range(5)]
    
    dynamics = NNEnsembleDynamicalSystem(dim_x=2, dim_u=2, ds_lst=dynamics_lst,
                                         u_normalizer=u_normalizer)
    
    x_batch = 2.0 * torch.rand(cfg.dataset_size, 2) - 1.0
    u_batch = 0.1 * (2.0 * torch.rand(cfg.dataset_size, 2) - 1.0)
    xnext_batch = dynamics_true.dynamics_batch(x_batch, u_batch)
    dataset = TensorDataset(x_batch, u_batch, xnext_batch)
    
    params = TrainParams()
    params.load_from_config(cfg)

    dynamics.train_network(dataset, params)

if __name__ == "__main__":
    main()