import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from score_po.score_matching import ScoreEstimator 
from score_po.nn import MLP, TrainParams, save_module, Normalizer


def get_network():
    return MLP(5, 5, [128, 128, 128, 128])


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

    # This dataset stores x and u separately. ScoreEstimator needs z = (x, u) so we process the dataset here.
    x_data = dataset.tensors[0]
    u_data = dataset.tensors[1]
    z_data = torch.cat((x_data, u_data), dim=-1)
    z_dataset = torch.utils.data.TensorDataset(z_data)

    z_normalizer = Normalizer(k=torch.Tensor([2, np.pi, 3, 12, 80]), b=None)
    score_estimator = ScoreEstimator(
        dim_x=4, dim_u=1, network=network, z_normalizer=z_normalizer
    )
    loss_lst = score_estimator.train_network(
        z_dataset, params, sigma=0.01, mode="denoising", split=True
    )


if __name__ == "__main__":
    main()
