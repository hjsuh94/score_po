from omegaconf import DictConfig, OmegaConf
import os
from typing import List, Tuple

import hydra
import numpy as np
import torch

from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem
from score_po.dynamical_system import NNDynamicalSystem


def generate_data(
    cart_length: float,
    left_wall: float,
    right_wall: float,
    theta_range: List[float],
    cart_vel_range: List[float],
    thetadot_range: List[float],
    dt: float,
    u_max: float,
    sample_size: int,
    device: str,
) -> torch.utils.data.TensorDataset:
    """
    Generate many state/action/next_state tuple.

    Make sure that in state and next_state, the cart and the pole are all within the
    left and right wall.

    Args:
        cart_length: The length of the cart.
        left_wall: The horizontal position of the left wall.
        right_wall: The horizontal position of the right wall.
        theta_range: The lower and upper bound of theta.
        cart_vel_range: The lower and upper bound of cart velocity.
        thetadot_range: The lower and upper bound of theta_dot.
        dt: the time step to compute the next state.
        u_max: The input is within -u_max <= u <= u_max
        sample_size: The number of state/action/next_state samples. Note that the
        dataset might have a size smaller than `sample_size`, because some of the
        sampled state/next_state may fall outside of the wall.
        device: either "cpu" or "cuda"

    Returns:
        dataset: A dataset containing (state, action, next_state)
    """
    assert right_wall - left_wall > cart_length

    plant = CartpolePlant(dt=dt)

    def scale_to_range(rand_tensor: torch.Tensor, lo: float, up: float) -> None:
        """
        rand_tensor is within [0, 1]. Scale rand_tensor to the range [lo, up]
        """
        rand_tensor *= up - lo
        rand_tensor += lo

    def within_walls(state_batch: torch.Tensor) -> torch.Tensor:
        """
        Return the flag if each state in state_batch is within the wall.
        """
        cart_within_wall = torch.logical_and(
            state_batch[:, 0] < right_wall - cart_length / 2,
            state_batch[:, 0] > left_wall + cart_length / 2,
        )
        pole_pos_x = state_batch[:, 0] + torch.sin(state_batch[:, 1]) * plant.l2
        pole_within_wall = torch.logical_and(
            pole_pos_x < right_wall, pole_pos_x > left_wall
        )
        return torch.logical_and(cart_within_wall, pole_within_wall)

    x_samples = torch.rand((sample_size, 4), device=device)
    scale_to_range(
        x_samples[:, 0], left_wall + cart_length / 2, right_wall - cart_length / 2
    )
    scale_to_range(x_samples[:, 1], theta_range[0], theta_range[1])
    scale_to_range(x_samples[:, 2], cart_vel_range[0], cart_vel_range[1])
    scale_to_range(x_samples[:, 3], thetadot_range[0], thetadot_range[1])

    u_samples = torch.rand((sample_size, 1), device=device)
    scale_to_range(u_samples, -u_max, u_max)

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
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dt = cfg.nn_plant.dt
    # The bounds on the state and control are loose outer bounds of a trajectory that
    # swings up the cart-pole, that swing-up trajectory is obtained from
    # Underactuated dircol.ipynb
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_layers=cfg.nn_plant.hidden_layers,
        x_lo=torch.tensor([-1, -np.pi, -3, -12]),
        x_up=torch.tensor([1, 1.5 * np.pi, 3, 12]),
        u_lo=torch.tensor([-cfg.nn_plant.u_max]),
        u_up=torch.tensor([cfg.nn_plant.u_max]),
        device=device,
    )

    if cfg.dataset.load_filename is None:
        dataset = generate_data(
            cart_length=0.5,
            left_wall=-2.0,
            right_wall=2.0,
            theta_range=[nn_plant.x_lo[1].item(), nn_plant.x_up[1].item()],
            cart_vel_range=[nn_plant.x_lo[2].item(), nn_plant.x_up[2].item()],
            thetadot_range=[nn_plant.x_lo[3].item(), nn_plant.x_up[3].item()],
            dt=cfg.nn_plant.dt,
            u_max=cfg.nn_plant.u_max,
            sample_size=cfg.dataset.sample_size,
            device=device,
        )
        if cfg.dataset.save_filename is not None:
            save_path = (
                os.path.dirname(os.path.abspath(__file__)) + cfg.dataset.save_filename
            )
            print(f"Save dataset to {save_path}.")
            torch.save(dataset, save_path)
    else:
        load_path = (
            os.path.dirname(os.path.abspath(__file__)) + cfg.dataset.load_filename
        )
        dataset = torch.load(load_path)

    params = NNDynamicalSystem.TrainParams()
    params.load_from_config(cfg)
    if cfg.train.save_ckpt is not None:
        save_path = os.path.dirname(os.path.abspath(__file__)) + cfg.train.save_ckpt
        print(f"Save dynamics network state dict to {save_path}")
        params.save_best_model = save_path
    if cfg.train.load_ckpt is not None:
        load_path = os.path.dirname(os.path.abspath(__file__)) + cfg.train.load_ckpt
        nn_plant.net.mlp.load_state_dict(torch.load(load_path))
    nn_plant.train_network(dataset, params, sigma=0.0)


if __name__ == "__main__":
    main()
