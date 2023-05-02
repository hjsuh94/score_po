from typing import List, Optional

import numpy as np
import torch

from score_po.dynamical_system import (
    DynamicalSystem,
)

import score_po.nn


class SingleIntegrator(DynamicalSystem):
    """
    A discrete-time version of xdot = u
    """

    def __init__(self, dt: float, dim_x: int = 2):
        """
        Args:
          dt: time step
        """
        super().__init__(dim_x=dim_x, dim_u=dim_x)
        assert dt > 0
        self.dt = dt

    def dynamics_batch(self, x_batch, u_batch):
        return x_batch + u_batch * self.dt


class DougleIntegrator(DynamicalSystem):
    """
    A discrete-time version of x = [q, qdot], qddot = u
    """

    def __init__(self, dt: float, dim_q: int = 2):
        """
        Args:
          dt: time step
        """
        super().__init__(dim_x=2 * dim_q, dim_u=dim_q)
        assert dt > 0
        self.dt = dt

    def dynamics_batch(self, x_batch, u_batch):
        dim_q = int(self.dim_x / 2)
        q_batch = x_batch[:, :dim_q]
        qdot_batch = x_batch[:, dim_q:]
        qdot_next = qdot_batch + u_batch * self.dt
        q_next = q_batch + (qdot_batch + qdot_next) / 2 * self.dt
        return torch.cat((q_next, qdot_next), dim=-1)

