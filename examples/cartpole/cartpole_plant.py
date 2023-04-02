from typing import List

import numpy as np
import torch

from score_po.dynamical_system import DynamicalSystem, NNDynamicalSystem
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
        # Assume mid-point integration
        xdot = self.calc_derivative(x_batch, u_batch)
        x_mid = x_batch + xdot * self.dt / 2
        # Assume constant action u within the step.
        xdot_mid = self.calc_derivative(x_mid, u_batch)
        xnext = x_batch + xdot_mid * self.dt
        return xnext

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


class CartpoleNNDynamicalSystem(NNDynamicalSystem):
    """
    The neural network approximates the residual dynamics.
    """

    def __init__(
        self,
        hidden_widths: List[int],
        x_lo: torch.Tensor,
        x_up: torch.Tensor,
        u_lo: torch.Tensor,
        u_up: torch.Tensor,
        device: str,
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
            hidden_layers=hidden_widths,
            activation=torch.nn.LeakyReLU(),
        ).to(device)
        self.device = device
        self.x_lo = x_lo.to(self.device)
        self.x_up = x_up.to(self.device)
        self.u_lo = u_lo.to(self.device)
        self.u_up = u_up.to(self.device)

        super().__init__(network=residual_net, dim_x=4, dim_u=1)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor, eval: bool = True):
        return self.dynamics(x.unsqueeze(0), u.unsqueeze(0), eval).squeeze(0)

    def dynamics_batch(
        self, x_batch: torch.Tensor, u_batch: torch.Tensor, eval: bool = True
    ):
        if eval:
            self.net.eval()
        else:
            self.net.train()

        # We first normalize x and u.
        # Our state and control can have very different magnitude. We can learn a more
        # accurate model with normalized input.
        x_normalized = (2 * x_batch - (self.x_lo + self.x_up)) / (self.x_up - self.x_lo)
        u_normalized = (2 * u_batch - (self.u_lo + self.u_up)) / (self.u_up - self.u_lo)
        xu_batch = torch.concat((x_normalized, u_normalized), dim=1)

        # The network only predicts the residual dynamics
        delta_x_batch = self.net(xu_batch)
        return x_batch + delta_x_batch
