from omegaconf import OmegaConf, DictConfig
import os
from typing import Union

import hydra

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from score_po.nn import (
    MLPwEmbedding,
    MLP,
    Normalizer,
    generate_cosine_schedule,
    TrainParams,
)
from score_po.score_matching import (
    ScoreEstimatorXu,
    NoiseConditionedScoreEstimatorXu,
    langevin_dynamics,
    noise_conditioned_langevin_dynamics,
)


def generate_data(radius: float, sample_size: int, device: str) -> torch.Tensor:
    angle = 2 * np.pi * torch.rand((sample_size, 1), device=device)
    return radius * torch.cat((torch.cos(angle), torch.sin(angle)), dim=1)


def construct_score(radius: float, device: str, noise_conditioned: bool):
    if noise_conditioned:
        network = MLPwEmbedding(
            dim_in=2, dim_out=2, hidden_layers=[32, 32, 32], embedding_size=32
        ).to(device)
        sf_cls = NoiseConditionedScoreEstimatorXu
    else:
        network = MLP(dim_in=2, dim_out=2, hidden_layers=[32, 32, 32], layer_norm=True)
        sf_cls = ScoreEstimatorXu
    sf = sf_cls(
        dim_x=2,
        dim_u=0,
        network=network,
        x_normalizer=Normalizer(k=torch.ones(2) * 2 * radius, b=torch.zeros(2)),
        u_normalizer=None,
    ).to(device)
    return sf


def train_score(
    sf: Union[ScoreEstimatorXu, NoiseConditionedScoreEstimatorXu],
    x_samples: torch.Tensor,
    cfg: DictConfig,
):
    u_samples = torch.zeros((x_samples.shape[0], 0), device=x_samples.device)

    dataset = torch.utils.data.TensorDataset(x_samples, u_samples)

    params = TrainParams()
    params.load_from_config(cfg)
    params.save_best_model = os.path.join(os.getcwd(), "score", params.save_best_model)

    if cfg.noise_conditioned:
        sigma_lst = generate_cosine_schedule(
            sigma_max=cfg.sigma_max, sigma_min=cfg.sigma_min, steps=cfg.sigma_steps
        ).to(cfg.device)

        sf.train_network(dataset, params, sigma_lst, split=True)
    else:
        sigma = torch.tensor([cfg.sigma], device=cfg.device)
        sf.train_network(dataset, params, sigma, split=True)
    return sf


def plot_result(
    sf: Union[ScoreEstimatorXu, NoiseConditionedScoreEstimatorXu],
    radius: float,
    cfg: DictConfig,
    sample_size: int
):
    num_pts = 20
    x0 = torch.randn((num_pts, 2), device=cfg.device)
    if cfg.noise_conditioned:
        langevin_fn = noise_conditioned_langevin_dynamics
    else:
        langevin_fn = langevin_dynamics
    x_history = (
        langevin_fn(x0, sf, cfg.langevin_epsilon, cfg.langevin_steps, noise=False)
        .cpu()
        .detach()
        .numpy()
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta) * radius, np.sin(theta) * radius, linestyle="--")

    cmap = LinearSegmentedColormap.from_list("r_colormap", [(0.1, 0, 0), (1, 0, 0)])
    color_indices = np.linspace(0, 1, x_history.shape[0])
    colors = cmap(color_indices)
    for i in range(num_pts):
        ax.scatter(x_history[:, i, 0], x_history[:, i, 1], c=colors)
        ax.plot(x_history[:, i, 0], x_history[:, i, 1])

    ax.set_aspect("equal", "box")
    ax.set_title(f"Dataset size={sample_size}\nnoise conditioned={cfg.noise_conditioned}")

    fig.savefig(
        os.path.join(
            os.getcwd(),
            f"langevin_eps{cfg.langevin_epsilon}_steps{cfg.langevin_steps}.png",
        ),
        format="png",
    )


@hydra.main(config_path="./config", config_name="demo")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)

    device = cfg.device

    if cfg.generate_data:
        samples = generate_data(
            radius=cfg.radius, sample_size=cfg.sample_size, device=cfg.device
        )
        torch.save(samples, os.path.join(os.getcwd(), cfg.save_samples))
    else:
        samples = torch.load(cfg.load_samples)

    sf = construct_score(cfg.radius, device, cfg.noise_conditioned)
    if cfg.train_score:
        sf = train_score(sf=sf, x_samples=samples, cfg=cfg)
    else:
        sf.load_state_dict(torch.load(cfg.load_score))

    plot_result(sf, radius=cfg.radius, cfg=cfg, sample_size=samples.shape[0])


if __name__ == "__main__":
    main()
