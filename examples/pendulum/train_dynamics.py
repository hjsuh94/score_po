import numpy as np
import torch
import matplotlib.pyplot as plt
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import MLP, TrainParams, Normalizer

from examples.pendulum.pendulum_keypoint_plant import PendulumPlant

from examples.pendulum.train_ncsn_xux import generate_data

@hydra.main(config_path="./config", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.train.seed)
    device = cfg.train.device
    plant = PendulumPlant(dt=cfg.plant_param.dt)
    x_lo = torch.tensor( cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    x_kp_lo = torch.tensor([-plant.l, -plant.l, x_lo[1]])
    x_kp_up = torch.tensor([plant.l, plant.l, x_up[1]])
    u_lo = torch.tensor([-cfg.plant_param.u_max])
    u_up = torch.tensor([cfg.plant_param.u_max])
    x_normalizer = Normalizer(k=(x_kp_up - x_kp_lo) / 2, b=(x_kp_up + x_kp_lo) / 2)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)
    
    network = MLP(4, 3, cfg.nn_layers, layer_norm=True)
    dynamics = NNDynamicalSystem(3, 1, network, x_normalizer=x_normalizer, u_normalizer=u_normalizer)
        
    params = TrainParams()
    params.load_from_config(cfg)

    if cfg.train_score:
        dataset = generate_data(
            plant=plant,
            x_lo=x_lo,
            x_up=x_up,
            u_max=cfg.plant_param.u_max,
            sample_size=cfg.dataset_size,
            device=device,
        )
        dynamics.train_network(dataset, params)

if __name__ == "__main__":
    main()