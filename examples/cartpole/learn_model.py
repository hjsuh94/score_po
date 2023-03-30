from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from examples.cartpole.cartpole_plant import CartpolePlant, CartpoleNNDynamicalSystem
from score_po.dynamical_system import AdamOptimizerParams


def generate_data(
    cart_length: float,
    left_wall: float,
    right_wall: float,
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
        dt: the time step to compute the next state.
        u_max: The input is within -u_max <= u <= u_max
        sample_size: The number of state/action/next_state samples. Note that the
        dataset might have a size smaller than `sample_size`, because some of the state/next_state may fall outside of the wall.
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
    scale_to_range(x_samples[:, 1], -1.5 * np.pi, 1.5 * np.pi)
    scale_to_range(x_samples[:, 2], -6, 6)
    scale_to_range(x_samples[:, 3], -6, 6)

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


def main():
    device = "cuda"
    dt = 0.05
    plant = CartpolePlant(dt)
    # The bounds on the state and control are loose outer bounds of a trajectory that
    # swings up the cart-pole, that trajectory is obtained from
    # Underactuated dircol.ipynb
    nn_plant = CartpoleNNDynamicalSystem(
        hidden_widths=[16, 16, 8],
        x_lo=torch.tensor([-2, -np.pi, -3, -12]),
        x_up=torch.tensor([2, 1.5 * np.pi, 3, 12]),
        u_lo=torch.tensor([-80.0]),
        u_up=torch.tensor([80.0]),
        device=device,
    )

    dataset = generate_data(
        cart_length=0.5,
        left_wall=-2.0,
        right_wall=2.0,
        dt=dt,
        u_max=80.0,
        sample_size=100000,
        device=device,
    )

    params = AdamOptimizerParams()
    params.iters = 100
    optimizer = torch.optim.Adam(nn_plant.net.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)

    data_loader_train = torch.utils.data.DataLoader(
        dataset, batch_size=params.batch_size
    )
    data_loader_eval = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    for epoch in tqdm(range(params.iters)):
        for x_batch, u_batch, xnext_batch in data_loader_train:
            loss = nn_plant.evaluate_dynamic_loss(
                torch.cat((x_batch, u_batch), dim=-1), xnext_batch, sigma=0
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Evaluate the loss on the whole dataset.
        with torch.no_grad():
            for x_all, u_all, xnext_all in data_loader_eval:
                loss_eval = nn_plant.evaluate_dynamic_loss(
                    torch.cat((x_all, u_all), dim=-1), xnext_all, sigma=0
                )
            print(f"epoch {epoch}, total loss {loss_eval.item()}")


if __name__ == "__main__":
    main()
