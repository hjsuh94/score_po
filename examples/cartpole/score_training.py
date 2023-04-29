import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams, save_module, Normalizer


def get_score_network():
    return MLP(5, 5, [128, 128, 128, 128])


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    """
    Train a score function estimator for log p(x, u)
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_score_network()
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)

    x_normalizer = Normalizer(k=torch.tensor([2, np.pi, 3, 12]), b=torch.zeros(4))
    u_normalizer = Normalizer(k=torch.tensor([80]), b=torch.tensor(0))
    score_estimator = ScoreEstimatorXu(
        dim_x=4,
        dim_u=1,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )
    # TODO(hongkai.dai): do a sweep on sigma (try sigma = 0.1)
    loss_lst = score_estimator.train_network(
        dataset, params, sigma=0.01, split=True
    )


if __name__ == "__main__":
    main()
