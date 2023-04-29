from omegaconf import DictConfig, OmegaConf
import os
from typing import List, Tuple

import hydra
import numpy as np
import torch
import wandb

from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem
from score_po.nn import TrainParams, Normalizer

OmegaConf.register_new_resolver("np.pi", lambda x: np.pi * x)


def generate_data(
    dt: float,
    x_lo: torch.Tensor,
    x_up: torch.Tensor,
    u_max: float,
    sample_size: int,
    device: str,
) -> torch.utils.data.TensorDataset:
    """
    Generate many state/action/next_state tuple.

    Make sure that in state and next_state, the cart and the pole are all within the
    left and right wall.

    Args:
        dt: the time step to compute the next state.
        x_lo: Lower bound of x
        x_up: Upper bound of x
        u_max: The input is within -u_max <= u <= u_max
        sample_size: The number of state/action/next_state samples. Note that the
        dataset might have a size smaller than `sample_size`, because some of the
        sampled state/next_state may fall outside of the wall.
        device: either "cpu" or "cuda"

    Returns:
        dataset: A dataset containing (state, action, next_state)
    """
    plant = CartpolePlant(dt=dt)

    def within_walls(state_batch: torch.Tensor) -> torch.Tensor:
        """
        Return the flag if each state in state_batch is within the wall.
        """
        cart_within_wall = torch.all(torch.logical_and(
            state_batch <= x_up, state_batch >= x_up
        ), dim=1)
        return cart_within_wall

    x_samples = torch.rand((sample_size, 4), device=device) * (x_up - x_lo).to(
        device
    ) + x_lo.to(device)

    u_samples = torch.rand((sample_size, 1), device=device) * 2 * u_max - u_max

    xnext_samples = plant.dynamics_batch(x_samples, u_samples)

    admissible_flag = torch.logical_and(
        within_walls(x_samples), within_walls(xnext_samples)
    )
    return torch.utils.data.TensorDataset(
        x_samples[admissible_flag],
        u_samples[admissible_flag],
        xnext_samples[admissible_flag],
    )


@hydra.main(config_path="./config", config_name="learn_model")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))
    torch.manual_seed(cfg.seed)
    device = cfg.device
    x_lo = torch.tensor(cfg.plant_param.x_lo)
    x_up = torch.tensor(cfg.plant_param.x_up)
    u_lo = torch.tensor([-cfg.plant_param.u_max])
    u_up = torch.tensor([cfg.plant_param.u_max])
    x_normalizer = Normalizer(k=(x_up - x_lo) / 2, b=(x_up + x_lo) / 2)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.plant_param.hidden_layers,
        device=device,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    )

    if cfg.dataset.load_path is None:
        dataset = generate_data(
            dt=cfg.plant_param.dt,
            x_lo=x_lo,
            x_up=x_up,
            u_max=cfg.plant_param.u_max,
            sample_size=cfg.dataset.sample_size,
            device=device,
        )
        dataset_dir = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_save_path = os.path.join(dataset_dir, "dataset.pth")
        print(f"Save dataset to {dataset_save_path}.")
        torch.save(dataset, dataset_save_path)
    else:
        print(f"Load dataset {cfg.dataset.load_path}")
        dataset = torch.load(cfg.dataset.load_path)

    params = TrainParams()
    params.load_from_config(cfg)
    if cfg.train.load_ckpt is not None:
        nn_plant.load_state_dict(torch.load(cfg.train.load_ckpt))
    nn_plant.train_network(dataset, params, sigma=0.0)


if __name__ == "__main__":
    main()
