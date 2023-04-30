import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import hydra
from omegaconf import DictConfig

from score_po.score_matching import NoiseConditionedScoreEstimatorXux
from score_po.nn import (
    MLPwEmbedding, TrainParams, Normalizer, generate_cosine_schedule)

from examples.light_dark.dynamics import SingleIntegrator
from examples.light_dark.environment import Environment

@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    env = Environment()
    dynamics = SingleIntegrator()
    env.add_ellipse([0, 0], 0.4, 0.4)
    x_batch = env.sample_points(cfg.dataset_size)
    u_batch = 0.1 * 2.0 * (torch.rand(x_batch.shape[0], 2) - 0.5)
    u_normalizer = Normalizer(k =0.2 * torch.ones(2), b= torch.zeros(2))
    xnext_batch = dynamics.dynamics_batch(x_batch, u_batch)
    dataset = TensorDataset(x_batch, u_batch, xnext_batch)
    
    params = TrainParams()
    params.load_from_config(cfg)
    
    network = MLPwEmbedding(6, 6, cfg.nn_layers, 10)
    sf = NoiseConditionedScoreEstimatorXux(
        2, 2, network, x_normalizer=None, u_normalizer=u_normalizer)
    sf.train_network(dataset, params, sigma_lst=generate_cosine_schedule(
        0.3, 0.01, 10))


if __name__ == "__main__":
    main()