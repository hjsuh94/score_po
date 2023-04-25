"""
The corridor has a given width.
| car |
|     |
|     |
|     |
|     -----------
|  .          goal
-----------------
The center of the two corridors intersect at the origin.
"""
from omegaconf import DictConfig
import os

import hydra
import numpy as np
import torch

from examples.corridor.car_plant import SingleIntegrator
from score_po.dynamical_system import NNDynamicalSystem
from score_po.nn import Normalizer, MLP, TrainParams


def draw_corridor(
    ax, corridor_width: float, horizontal_max: float, vertical_max: float, **kwargs
):
    ax.plot(
        np.array([-corridor_width / 2, horizontal_max]),
        np.array([-corridor_width / 2, -corridor_width / 2]),
        **kwargs,
    )
    ax.plot(
        np.array([corridor_width / 2, horizontal_max]),
        np.array([corridor_width / 2, corridor_width / 2]),
        **kwargs,
    )
    ax.plot(
        np.array([-corridor_width / 2, -corridor_width / 2]),
        np.array([-corridor_width / 2, vertical_max]),
        **kwargs,
    )
    ax.plot(
        np.array([corridor_width / 2, corridor_width / 2]),
        np.array([corridor_width / 2, vertical_max]),
        **kwargs,
    )


def in_corridor(corridor_width: float, x: torch.Tensor) -> torch.Tensor:
    """
    Returns if x[i] is within the corridor.
    """
    return torch.logical_and(
        torch.all(x[:, :2] >= -corridor_width / 2, dim=1),
        torch.any(x[:, :2] <= corridor_width / 2, dim=1),
    )


def uniform_sample_in_box(lo: torch.Tensor, up: torch.Tensor, batch: int, dim_sample):
    return torch.rand((batch, dim_sample)) * (up - lo) + lo


def generate_data(
    corridor_width: float,
    horizontal_max: float,
    vertical_max: float,
    u_lo: torch.Tensor,
    u_up: torch.Tensor,
    sample_size: int,
    dt: float,
    device: str,
) -> torch.utils.data.TensorDataset:
    """
    Create a dataset (x, u, x_next) for the car in the corridor. The state is always within the corridor.
    """
    dim_x = 2

    horizontal_corridor_states = uniform_sample_in_box(
        torch.tensor([-corridor_width / 2, -corridor_width / 2]),
        torch.tensor([horizontal_max, corridor_width / 2]),
        int(sample_size / 2),
        dim_x,
    ).to(device=device)
    vertical_corridor_states = uniform_sample_in_box(
        torch.tensor([-corridor_width / 2, -corridor_width / 2]),
        torch.tensor([corridor_width / 2, vertical_max]),
        int(sample_size / 2),
        dim_x,
    ).to(device=device)

    x = torch.cat((horizontal_corridor_states, vertical_corridor_states), dim=0)

    u = uniform_sample_in_box(u_lo, u_up, x.shape[0], dim_x).to(device)

    single_integrator = SingleIntegrator(dt=dt, dim_x=dim_x)

    x_next = single_integrator.dynamics_batch(x, u)

    in_corridor_mask = torch.logical_and(
        in_corridor(corridor_width, x_next), in_corridor(corridor_width, x)
    )

    return torch.utils.data.TensorDataset(
        x[in_corridor_mask], u[in_corridor_mask], x_next[in_corridor_mask]
    )


@hydra.main(config_path="./config", config_name="learn_model")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dim_x = 2
    corridor_width = cfg.corridor_width
    horizontal_max = cfg.horizontal_max
    vertical_max = cfg.vertical_max
    x_normalizer = Normalizer(
        k=torch.tensor(
            [horizontal_max + corridor_width / 2, vertical_max + corridor_width / 2]
        ),
        b=torch.tensor(
            [
                (horizontal_max - corridor_width / 2) / 2,
                (vertical_max - corridor_width / 2) / 2,
            ]
        ),
    )
    u_lo = torch.tensor(cfg.u_lo)
    u_up = torch.tensor(cfg.u_up)
    u_normalizer = Normalizer(k=(u_up - u_lo) / 2, b=(u_up + u_lo) / 2)

    network = MLP(
        dim_in=dim_x + dim_x, dim_out=dim_x, hidden_layers=cfg.nn_plant.hidden_layers
    )

    nn_plant = NNDynamicalSystem(
        dim_x=dim_x,
        dim_u=dim_x,
        network=network,
        x_normalizer=x_normalizer,
        u_normalizer=u_normalizer,
    ).to(device)

    if cfg.dataset.load_path is None:
        dataset = generate_data(
            corridor_width,
            horizontal_max,
            vertical_max,
            u_lo,
            u_up,
            sample_size=cfg.dataset.sample_size,
            dt=cfg.nn_plant.dt,
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
