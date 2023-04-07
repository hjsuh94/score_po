import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from score_po.score_matching import ScoreFunctionEstimator
from score_po.nn import MLP, AdamOptimizerParams


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

    params = ScoreFunctionEstimator.TrainParams()
    params.adam_params.batch_size = cfg.train.batch_size
    params.adam_params.epochs = cfg.train.epochs
    params.adam_params.lr = cfg.train.lr
    params.data_split = [0.9, 0.1]
    params.enable_wandb = cfg.train.wandb.enabled
    save_ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    save_path = os.path.join(save_ckpt_dir, "score_network.pth")
    params.save_best_model = save_path

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            name=cfg.train.wandb.name,
            dir=os.getcwd(),
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.train.wandb.entity,
        )

    # This dataset stores x and u separately. ScoreFunctionEstimator needs z = (x, u) so we process the dataset here.
    x_data = dataset.tensors[0]
    u_data = dataset.tensors[1]
    z_data = torch.cat((x_data, u_data), dim=-1)
    z_dataset = torch.utils.data.TensorDataset(z_data)

    score_estimator = ScoreFunctionEstimator(network, dim_x=4, dim_u=1)
    loss_lst = score_estimator.train_network(
        z_dataset, params, sigma_max=1, sigma_min=0.01, n_sigmas=10
    )


if __name__ == "__main__":
    main()
