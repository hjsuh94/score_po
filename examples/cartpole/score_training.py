import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from score_po.score_matching import ScoreEstimator
from score_po.nn import MLP, TrainParams


def get_network():
    return MLP(6, 5, [64, 64, 64, 64])


@hydra.main(config_path="./config", config_name="score_training")
def main(cfg: DictConfig):
    """
    Train a score function estimator for log p(x, u)
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dataset = torch.load(cfg.dataset.load_path)

    network: MLP = get_network()
    network.to(device)

    params = TrainParams()
    params.load_from_config(cfg)
    save_ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    save_path = os.path.join(save_ckpt_dir, "score_network.pth")
    params.save_best_model = save_path

    # This dataset stores x and u separately. ScoreFunctionEstimator needs z = (x, u) so we process the dataset here.
    x_data = dataset.tensors[0]
    u_data = dataset.tensors[1]
    z_data = torch.cat((x_data, u_data), dim=-1)
    z_dataset = torch.utils.data.TensorDataset(z_data)

    score_estimator = ScoreEstimator(network, dim_x=4, dim_u=1)
    loss_lst = score_estimator.train_network(
        z_dataset, params, sigma_max=1, sigma_min=0.01, n_sigmas=10
    )


if __name__ == "__main__":
    main()
