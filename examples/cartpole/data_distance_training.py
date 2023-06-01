"""
Learn the data distance estimator. 
"""
import os
from typing import Union, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from score_po.data_distance import DataDistanceEstimatorXu
from score_po.nn import (
    MLP,
    TrainParams,
    Normalizer,
)

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


def get_dde_network():
    return MLP(5, 1, [128, 128, 128, 128], activation=torch.nn.ReLU())


@hydra.main(config_path="./config", config_name="data_distance_training")
def main(cfg: DictConfig):
    """
    Train a data distance estimator for (x, u)
    """
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_dde_network()
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), params.save_best_model)

    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    xu_lo = torch.cat((x_lo, torch.tensor([-cfg.plant_param.u_max]))).to(device)
    xu_up = torch.cat((x_up, torch.tensor([cfg.plant_param.u_max]))).to(device)

    dde = DataDistanceEstimatorXu(
        dim_x=4, dim_u=1, network=network, domain_lb=xu_lo, domain_ub=xu_up
    )

    metric = torch.tensor(2 / (xu_up - xu_lo)).to(device)

    loss_lst = dde.train_network(dataset, params, metric)


if __name__ == "__main__":
    main()
