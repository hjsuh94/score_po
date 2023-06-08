from omegaconf import DictConfig, OmegaConf
import os
from typing import List, Tuple

import hydra
import numpy as np
import torch
import wandb

from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem
from score_po.nn import TrainParams, Normalizer
from score_po.dynamical_system import NNEnsembleDynamicalSystem

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


@hydra.main(config_path="./config", config_name="learn_model_ensemble")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)
    device = cfg.device
    x_lo = torch.tensor(cfg.plant_param.x_lo, device=device)
    x_up = torch.tensor(cfg.plant_param.x_up, device=device)
    u_lo = torch.tensor([-cfg.plant_param.u_max], device=device)
    u_up = torch.tensor([cfg.plant_param.u_max], device=device)
    x_normalizer = Normalizer(k=(x_up - x_lo) / 2, b=(x_up + x_lo) / 2)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)
    ds_lst = [None] * cfg.plant_param.num_models
    for i in range(cfg.plant_param.num_models):
        # Add normalizer for each individual model. As a result, we set the normalizer of the ensemble model to None.
        ds_lst[i] = CartpoleNNDynamicalSystem(
            hidden_layers=cfg.plant_param.hidden_layers[i],
            device=device,
            x_normalizer=x_normalizer,
            u_normalizer=u_normalizer,
        )
    ensemble = NNEnsembleDynamicalSystem(
        dim_x=4, dim_u=1, ds_lst=ds_lst, x_normalizer=None, u_normalizer=None
    )

    print(f"Load dataset {cfg.dataset.load_path}")
    dataset = torch.load(cfg.dataset.load_path)

    params = TrainParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), cfg.train.save_best_model)
    ensemble.train_network(dataset, params, sigma=0.0)


if __name__ == "__main__":
    main()
