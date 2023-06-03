from typing import List, Optional

import numpy as np
import torch

from score_po.dynamical_system import (
    DynamicalSystem,
    NNDynamicalSystem,
    midpoint_integration,
)
import score_po.nn


class CartpolePlant(DynamicalSystem):
    """
    A discrete time cart-pole plant.
    """

    def __init__(self, dt: float):
        """
        Args:
            dt: time step.
        """
        super().__init__(dim_x=4, dim_u=1)
        assert dt > 0
        self.dt = dt
        self.m1 = 10
        self.m2 = 1
        self.l2 = 0.5
        self.g = 9.81

    def dynamics(self, x, u):
        return self.dynamics_batch(x.reshape((1, -1)), u.reshape((1, -1))).squeeze(0)

    def dynamics_batch(self, x_batch, u_batch):
        return midpoint_integration(self.calc_derivative, x_batch, u_batch, self.dt)

    def calc_derivative(self, x_batch, u_batch):
        """
        Compute the state derivative of the continuous-time system.
        """
        theta = x_batch[:, 1]
        pdot = x_batch[:, 2]
        thetadot = x_batch[:, 3]

        if isinstance(x_batch, torch.Tensor):
            s_theta = torch.sin(theta)
            c_theta = torch.cos(theta)
        elif isinstance(x_batch, np.ndarray):
            s_theta = np.sin(theta)
            c_theta = np.cos(theta)

        denom = self.m1 + self.m2 * s_theta**2
        pddot = (
            u_batch.squeeze(1)
            + self.m2 * s_theta * (self.l2 * thetadot**2 + self.g * c_theta) / denom
        )
        thetaddot = (
            -u_batch.squeeze(1) * c_theta
            - self.m2 * self.l2 * thetadot**2 * c_theta * s_theta
            - (self.m1 + self.m2) * self.g * s_theta
        ) / (denom * self.l2)
        if isinstance(x_batch, torch.Tensor):
            return torch.concat(
                (
                    pdot.reshape((-1, 1)),
                    thetadot.reshape((-1, 1)),
                    pddot.reshape((-1, 1)),
                    thetaddot.reshape((-1, 1)),
                ),
                dim=-1,
            )
        elif isinstance(x_batch, np.ndarray):
            return np.concatenate(
                (
                    pdot.reshape((-1, 1)),
                    thetadot.reshape((-1, 1)),
                    pddot.reshape((-1, 1)),
                    thetaddot.reshape((-1, 1)),
                ),
                axis=-1,
            )

    def visualize(self, ax, x: np.ndarray, **kwargs):
        assert x.shape == (4,)
        base_size = np.array([0.3, 0.1])
        wheel_radius = 0.03
        base_center = np.array([x[0], 2 * wheel_radius + base_size[1] / 2])
        # Draw the base
        base_vertices = base_center + (base_size / 2) * np.array(
            [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
        )

        ax.plot(base_vertices[:, 0], base_vertices[:, 1], **kwargs)
        theta = x[1]

        # Draw the pole
        pole_width = 0.01
        theta_shift = theta 
        pole = (
            base_center
            + (
                np.array([pole_width, self.l2])
                * np.array([[-0.5, 0], [0.5, 0], [0.5, -1], [-0.5, -1], [-0.5, 0]])
            )
            @ np.array(
                [
                    [np.cos(theta_shift), -np.sin(theta_shift)],
                    [np.sin(theta_shift), np.cos(theta_shift)],
                ]
            ).T
        )
        # pole = base_center + plant.l2 * np.array(
        #    [[0, 0], [np.cos(theta - np.pi / 2), np.sin(theta - np.pi / 2)]]
        # )
        ax.plot(pole[:, 0], pole[:, 1], **kwargs)

        # Draw the wheels.
        wheel_centers = [
            np.array([x[0] - base_size[0] / 4, wheel_radius]),
            np.array([x[0] + base_size[0] / 3, wheel_radius]),
        ]
        for wheel_center in wheel_centers:
            ax.plot(
                wheel_center[0] + wheel_radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
                wheel_center[1] + wheel_radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
                **kwargs
            )


class CartpoleNNDynamicalSystem(NNDynamicalSystem):
    """
    The neural network approximates the residual dynamics.
    """

    def __init__(
        self,
        hidden_layers: List[int],
        device: str,
        x_normalizer: Optional[score_po.nn.Normalizer] = None,
        u_normalizer: Optional[score_po.nn.Normalizer] = None,
    ):
        """
        Args:
            x_lo, x_up, u_lo, u_up: (suggestive) bounds on state and control. We don't
            clamp the state/control to lie within these range, but will use these
            bounds to normalize the input to the network.
        """
        residual_net = score_po.nn.MLP(
            dim_in=5,
            dim_out=4,
            hidden_layers=hidden_layers,
            activation=torch.nn.LeakyReLU(),
        ).to(device)
        self.device = device

        super().__init__(
            network=residual_net,
            dim_x=4,
            dim_u=1,
            x_normalizer=x_normalizer,
            u_normalizer=u_normalizer,
        )

