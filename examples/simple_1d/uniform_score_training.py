"""
Given a uniform distribution in 1D
p(x) = 1/b if a <= x <= a+b
p(x) = 0   otherwise

We learn a score estimator that approximates the score function ∇ₓ log p(x), and then
generate the sample using this learned score function together with Langevin dynamics.
"""
from omegaconf import DictConfig
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

from score_po.nn import MLP, TrainParams, Normalizer
from score_po.score_matching import ScoreEstimator, langevin_dynamics


def generate_dataset(size: int, a: float, b: float, device: str):
    data = torch.rand((size, 1), device=device) * b + a
    return torch.utils.data.TensorDataset(data)


def train_score_estimator(
    dataset: torch.utils.data.TensorDataset,
    train_params: TrainParams,
    a: float,
    b: float,
):
    mlp = MLP(1, 1, [128, 128, 128])
    normalizer = Normalizer(k=torch.tensor(b / 2), b=torch.tensor(a + 0.5))
    sf = ScoreEstimator(dim_x=1, dim_u=0, network=mlp, z_normalizer=normalizer)
    sf.train_network(dataset, train_params, sigma=0.1, mode="denoising")
    return sf


def plot_score_function(sf: torch.nn.Module, a: float, b: float, device: str):
    x = torch.linspace(a - 2, a + b + 2, 100).to(device).reshape((-1, 1))
    score_val = sf(x)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x.cpu().detach().numpy(), score_val.cpu().detach().numpy())
    ax.set_xlabel("x")
    ax.set_ylabel("score estimate")
    fig.savefig(os.path.join(os.getcwd(), "1d_score_estimate.png"), format="png")


def generate_langevin_samples(sf: torch.nn.Module, a: float, b: float, device: str):
    x0 = torch.randn((10000, 1), device=device)
    with torch.no_grad():
        xT = langevin_dynamics(x0, sf, epsilon=2e-3, steps=5000)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hist(x0.cpu().detach().numpy())
    ax1.set_title("Histogram of initial samples")
    ax2 = fig.add_subplot(212)
    ax2.hist(xT.cpu().detach().numpy())
    ax2.set_title("Langevin dynamics result")
    ax2.set_xlim([a - 2, a + b + 2])
    fig.savefig(os.path.join(os.getcwd(), "1d_langevin_dynamics.png"), format="png")


@hydra.main(config_path="./config", config_name="uniform_score_training")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    device = cfg.device
    params = TrainParams()
    params.load_from_config(cfg)

    a = 3
    b = 4
    dataset = generate_dataset(size=20000, a=a, b=b, device=device)
    sf = train_score_estimator(dataset, params, a, b)
    plot_score_function(sf, a, b, device)
    generate_langevin_samples(sf, a, b, device)


if __name__ == "__main__":
    main()
