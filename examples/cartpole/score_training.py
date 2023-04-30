import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams, save_module, Normalizer

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


def get_score_network():
    return MLP(5, 5, [128, 128, 128, 128])


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    """
    Train a score function estimator for log p(x, u)
    """
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_score_network()
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)

    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)

    x_normalizer = Normalizer(k=(x_up - x_lo) / 2, b=(x_up + x_lo) / 2)
    u_normalizer = Normalizer(
        k=torch.tensor([cfg.plant_param.u_max]), b=torch.tensor([0])
    )
    score_estimator = ScoreEstimatorXu(
        dim_x=4,
        dim_u=1,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    # TODO(hongkai.dai): do a sweep on sigma (try sigma = 0.1)
    sigma = torch.tensor([0.01], device=device)
    loss_lst = score_estimator.train_network(
        dataset, params, sigma, split=True
    )


if __name__ == "__main__":
    main()
