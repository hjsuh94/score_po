import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import TensorDataset
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from score_po.score_matching import (NoiseConditionedScoreEstimatorXux, noise_conditioned_langevin_dynamics)
from score_po.nn import (
    MLPwEmbedding, TrainParams, Normalizer, generate_cosine_schedule)

from examples.pendulum.pendulum_keypoint_plant import PendulumPlant

def generate_data(
    plant,
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    u_max: float,
    sample_size: int,
    device: str,
) -> torch.utils.data.TensorDataset:
    """
    Generate many state/action/next_state tuple.

    Args:
        dt: the time step to compute the next state.
        x_lo: Lower bound of x
        x_up: Upper bound of x
        u_max: The input is within -u_max <= u <= u_max
        sample_size: The number of state/action/next_state samples.
        device: either "cpu" or "cuda"

    Returns:
        dataset: A dataset containing (state, action, next_state)
    """
    x_samples = torch.rand((sample_size, 2), device=device) * (x_up - x_lo).to(
        device
    ) + x_lo.to(device)

    u_samples = torch.rand((sample_size, 1), device=device) * 2 * u_max - u_max

    xnext_samples = plant.dynamics_batch(x_samples, u_samples)

    return TensorDataset(
        plant.state_to_keypoints(x_samples),
        u_samples,
        plant.state_to_keypoints(xnext_samples),
    )

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
    
    params = TrainParams()
    params.load_from_config(cfg)
    
    network = MLPwEmbedding(7, 7, cfg.nn_layers, 10)
    sf = NoiseConditionedScoreEstimatorXux(
        3, 1, network, x_normalizer=x_normalizer, u_normalizer=u_normalizer)
    
    if cfg.train_score:
        dataset = generate_data(
            plant=plant,
            x_lo=x_lo,
            x_up=x_up,
            u_max=cfg.plant_param.u_max,
            sample_size=cfg.dataset_size,
            device=device,
        )
        sf.train_network(dataset, params, sigma_lst=generate_cosine_schedule(
            0.2, 0.01, 10))
    else:
        sf.load_state_dict(torch.load(cfg.load_score))
        
    plot_result(sf, plant, x_lo, x_up, u_up, plant.l, cfg, cfg.dataset_size)

def plot_result(
    sf: NoiseConditionedScoreEstimatorXux,
    plant,
    x_lo, 
    x_up, 
    u_max,
    radius: float,
    cfg: DictConfig,
    sample_size: int
):
    num_pts = 20
    x0 = torch.randn((num_pts, 2)) * (x_up - x_lo) + x_lo
    x0 = plant.state_to_keypoints(x0)
    u0 = torch.rand((num_pts, 1)) * 2 * u_max - u_max
    xnext0 = torch.randn((num_pts, 2)) * (x_up - x_lo) + x_lo
    xnext0 = plant.state_to_keypoints(xnext0)
    z0 = torch.cat([x0, u0, xnext0], dim=1)
    langevin_fn = noise_conditioned_langevin_dynamics
    x_history = (
        langevin_fn(z0, sf, cfg.langevin_epsilon, cfg.langevin_steps, noise=False)
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
    ax.set_title(f"Dataset size={sample_size}\nnoise conditioned")

    fig.savefig(
        os.path.join(
            os.getcwd(),
            f"langevin_eps{cfg.langevin_epsilon}_steps{cfg.langevin_steps}.png",
        ),
        format="png",
    )
    
if __name__ == "__main__":
    main()