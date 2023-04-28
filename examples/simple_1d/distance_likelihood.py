import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from omegaconf import DictConfig, OmegaConf
import hydra

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm

from score_po.data_distance import DataDistance, DataDistanceEstimatorXu
from score_po.score_matching import ScoreEstimatorXu
from score_po.nn import MLP, TrainParams

# Does score matching actually give me gradients of the perturbed distribution?


def get_normalized_probs(x, data, sigma):
    prob = np.zeros(10000)
    for i, data_pt in enumerate(data):
        prob += norm.pdf(x, loc=data_pt, scale=sigma)
    normalized_prob = prob / len(data)
    return normalized_prob


def get_dist_and_grads(x, data, sigma, cfg):
    x_tensor = torch.Tensor(x).reshape(-1, 1)
    data_tensor = torch.Tensor(data).reshape(-1, 1)

    dataset = TensorDataset(data_tensor)
    metric = torch.ones(1) / (sigma**2.0)
    dst = DataDistance(dataset, metric)

    dists = dst.get_energy_to_data(x_tensor).cpu().numpy()
    grads = dst.get_energy_gradients(x_tensor).cpu().numpy()
    return dists, grads


def get_score(x, data, sigma, cfg):
    x_tensor = torch.Tensor(x).to(cfg.device)

    # Score match.
    network = MLP(1, 1, cfg.nn_layers)
    sf = ScoreEstimatorXu(1, 0, network)
    sf.to(cfg.device)
    params = TrainParams()
    params.load_from_config(cfg)

    data_x = torch.Tensor(data).reshape(-1, 1)
    data_u = torch.zeros(data_x.shape[0], 0)
    
    dataset = torch.utils.data.TensorDataset(data_x, data_u)
    sf.train_network(dataset, params, torch.Tensor([sigma]), split=False)

    return sf.get_score_z_given_z(x_tensor.reshape(-1, 1)).detach().cpu().numpy()


def get_learned_dist(x, data, sigma, cfg):
    x_tensor = torch.Tensor(x).reshape(-1, 1)
    data_tensor = torch.Tensor(data).reshape(-1, 1)

    dataset = TensorDataset(data_tensor)
    metric = torch.ones(1) / (sigma**2.0)

    network = MLP(1, 1, cfg.nn_layers)
    dde = DataDistanceEstimatorXu(
        1, 0, network, torch.ones(1) * -3, torch.ones(1) * 3
    )

    params = TrainParams()
    params.load_from_config(cfg)

    dde.train_network(dataset, params, metric)
    z = dde.get_energy_to_data(x_tensor).detach().numpy()
    grads = dde.get_energy_gradients(x_tensor).detach().numpy()
    return z, grads


@hydra.main(config_path="./config", config_name="distance_likelihood")
def main(cfg: DictConfig):

    plt.figure(figsize=(15, 5))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # -----------------------------------------------------------------------------
    # 1. Draw data.
    # -----------------------------------------------------------------------------

    N = cfg.N
    data = 2.0 * (np.random.rand(N) - 0.5)
    x_range = cfg.x_range
    x = np.linspace(-x_range, x_range, cfg.n_grid)

    # -----------------------------------------------------------------------------
    # 2. Plot the perturbed distribution.
    # -----------------------------------------------------------------------------

    # Get distribution.
    sigma = cfg.sigma
    normalized_prob = get_normalized_probs(x, data, sigma)

    plt.subplot(1, 3, 1)
    plt.title("Perturbed Data Distribution")
    plt.plot(x, normalized_prob, color="springgreen", label="Perturbed Distribution")
    plt.plot(data, np.zeros(N), "ro", label="Data")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("Log-Likelihood vs. Softmin")
    plt.plot(x, -np.log(normalized_prob), label="Log-Likelihood", color="springgreen")
    plt.plot(data, np.zeros(N), "ro")

    # -----------------------------------------------------------------------------
    # 2. Get Distance and Gradients to the perturbed distribution.
    # -----------------------------------------------------------------------------

    # Get gradients of the distribution.

    dists, grads = get_dist_and_grads(x, data, sigma, cfg)
    dists_learned, grads_learned = get_learned_dist(x, data, sigma, cfg)

    plt.plot(x, dists, label="Softmin", color="red", alpha=0.8)
    plt.plot(x, dists_learned, label="Learned Distance", linestyle="--", color="blue")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Autodiff vs. Score-Matching")
    plt.plot(x, grads, label="Autodiff on Distance", color="red")
    plt.plot(data, np.zeros(N), "ro")

    # -----------------------------------------------------------------------------
    # 3. Do Score Matching.
    # -----------------------------------------------------------------------------

    z = get_score(x, data, sigma, cfg)
    plt.plot(x, -z, label="Score-Matching", color="springgreen")
    plt.plot(
        x,
        grads_learned,
        label="Autodiff on Learned Distance",
        color="blue",
        linestyle="--",
    )
    plt.legend()

    plt.savefig("distance_likelihood.png")
    plt.close()


if __name__ == "__main__":
    main()
