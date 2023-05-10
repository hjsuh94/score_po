from typing import List, Optional

import numpy as np
import torch

from score_po.dynamical_system import DynamicalSystem, NNDynamicalSystem, midpoint_integration
import score_po.nn


class PendulumPlant(DynamicalSystem):
    """
    A discrete time pendulum keypoint plant.
    x = [c, s, thetadot]
    """

    def __init__(self, dt: float):
        """
        Args:
            dt: time step.
        """
        super().__init__(dim_x=3, dim_u=1)
        assert dt > 0
        self.dt = dt
        self.m = 1.0
        self.l = 0.5
        self.g = 9.81
        self.b = 0.1

    def dynamics(self, x, u):
        """
        Return the next keypoint given the current keypoint and the control input.
        """
        state = self.keypoints_to_state(x.reshape((1, -1)))
        next_state =  self.dynamics_batch(state, u.reshape((1, -1)))
        return self.state_to_keypoints(next_state).squeeze(0)

    def dynamics_batch(self, x_batch, u_batch):
        return midpoint_integration(self.calc_derivative, x_batch, u_batch, self.dt)

    def calc_derivative(self, x_batch, u_batch):
        """
        Compute the state derivative of the continuous-time system.
        """
        theta = x_batch[:, 0]
        thetadot = x_batch[:, 1]

        if isinstance(x_batch, torch.Tensor):
            s_theta = torch.sin(theta)
        elif isinstance(x_batch, np.ndarray):
            s_theta = np.sin(theta)

        thetaddot = (u_batch.squeeze(1) - self.b * thetadot + self.m * self.g * self.l * s_theta) /(self.m*self.l**2)
        if isinstance(x_batch, torch.Tensor):
            return torch.concat(
                (
                    thetadot.reshape((-1, 1)),
                    thetaddot.reshape((-1, 1)),
                ),
                dim=-1,
            )
        elif isinstance(x_batch, np.ndarray):
            return np.concatenate(
                (
                    thetadot.reshape((-1, 1)),
                    thetaddot.reshape((-1, 1)),
                ),
                axis=-1,
            )
            
    def state_to_keypoints(self, x):
        theta = x[:, :1]
        return torch.cat((self.l * torch.cos(theta), self.l * torch.sin(theta), x[:, 1:]), dim=1)
    
    def keypoints_to_state(self, x):
        theta = torch.atan2(x[:, 1:2], x[:, 0:1])
        return torch.cat((theta, x[:, 2:]), dim=1)


class PendulumNNDynamicalSystem(NNDynamicalSystem):
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
            dim_in=3,
            dim_out=2,
            hidden_layers=hidden_layers,
            activation=torch.nn.LeakyReLU(),
        ).to(device)
        self.device = device

        super().__init__(
            network=residual_net,
            dim_x=2,
            dim_u=1,
            x_normalizer=x_normalizer,
            u_normalizer=u_normalizer,
        )