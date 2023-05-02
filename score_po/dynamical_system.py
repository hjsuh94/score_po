import abc
from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

from score_po.policy import Policy
from score_po.nn import AdamOptimizerParams, WandbParams, TrainParams, Normalizer
import score_po.nn

"""
Classes for dynamical systems. 
"""


class DynamicalSystem(torch.nn.Module):
    def __init__(self, dim_x, dim_u):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.is_differentiable = False

    def dynamics(self, x, u):
        """
        Evaluate dynamics in state-space form.
        input:
            x of shape n
            u of shape m
        output:
            xnext of shape n
        """
        return self.dynamics_batch(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)

    @abc.abstractmethod
    def dynamics_batch(self, x_batch, u_batch):
        """
        Evaluate dynamics in state-space form.
        input:
            x of shape (batch_size, n)
            u of shape (batch_size, m)
        output:
            xnext of shape (batch_size, n)
        """

    def forward(self, x, u):
        return self.dynamics_batch(x, u)


class NNDynamicalSystem(DynamicalSystem):
    """
    Neural network dynamical system, where the network is of
    - input shape (n + m)
    - output shape (n)
    We train the network to predict the residual dynamics in the
    normalized space, such that net(xnow, u) = xnext - xnow
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        network,
        x_normalizer: Optional[Normalizer] = None,
        u_normalizer: Optional[Normalizer] = None,
    ):
        super().__init__(dim_x, dim_u)
        self.net = network
        self.is_differentiable = True
        self.check_input_consistency()
        self.x_normalizer: Normalizer = (
            Normalizer(k=torch.ones(dim_x), b=torch.zeros(dim_x))
            if x_normalizer is None
            else x_normalizer
        )
        self.u_normalizer: Normalizer = (
            Normalizer(k=torch.ones(dim_u), b=torch.zeros(dim_u))
            if u_normalizer is None
            else u_normalizer
        )

    def check_input_consistency(self):
        if hasattr(self.net, "dim_in") and (
            self.net.dim_in is not self.dim_x + self.dim_u
        ):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and (self.net.dim_out is not self.dim_x):
            raise ValueError("Inconsistent output size of neural network.")

    def dynamics(self, x, u, eval=True):
        return self.dynamics_batch(x.unsqueeze(0), u.unsqueeze(0), eval).squeeze(0)

    def dynamics_batch(self, x_batch, u_batch, eval=True):
        if eval:
            self.net.eval()

        x_normalized = self.x_normalizer(x_batch)
        u_normalized = self.u_normalizer(u_batch)
        normalized_input = torch.hstack((x_normalized, u_normalized))
        normalized_output = self.net(normalized_input)
        # we don't use self.x_normalizer.denormalize since in the
        # residual dynamics, the bias terms cancel out.
        return self.x_normalizer.k * normalized_output + x_batch

    def forward(self, x, u, eval):
        return self.dynamics_batch(x, u, eval)

    def evaluate_dynamic_loss(
        self, xu, x_next, sigma=0.0, normalize_loss: bool = False
    ):
        """
        Evaluate L2 loss.
        data_samples:
            xu: shape (B, dim_x + dim_u)
            x_next: shape (B, dim_x)
            sigma: vector of dim_x + dim_u used for data augmentation.
            normalize_loss: if set to true, then we normalize x_next and dynamics(xu)
            when computing the loss.
        """
        B = xu.shape[0]
        if sigma > 0:
            noise = torch.normal(0, sigma, size=xu.shape, device=xu.device)
            databar = xu + noise
        else:
            databar = xu
        pred = self.dynamics_batch(
            databar[:, : self.dim_x], databar[:, self.dim_x :], eval=False
        )  # B x dim_x
        if normalize_loss:
            x_next = self.x_normalizer(x_next)
            pred = self.x_normalizer(pred)
        loss = 0.5 * ((x_next - pred) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def train_network(
        self, dataset: TensorDataset, params: TrainParams, sigma: float = 0.0
    ):
        """
        Train a network given a dataset and optimization parameters.
        """

        def loss(tensors, placeholder):
            x, u, x_next = tensors
            return self.evaluate_dynamic_loss(
                torch.cat((x, u), dim=-1), x_next, sigma=sigma
            )

        return score_po.nn.train_network(self, params, dataset, loss, split=True)


def midpoint_integration(
    dyn_fun: callable, x: torch.Tensor, u: torch.Tensor, dt: float
):
    """
    Given a continuous-time dynamics xdot = f(x, u), compute the discrete-time dynamics via mid-point integration

    Args:
      dyn_fun: xdot = dyn_fun(x, u)
    """
    xdot = dyn_fun(x, u)
    x_mid = x + xdot * dt / 2
    xdot_mid = dyn_fun(x_mid, u)
    x_next = x + xdot_mid * dt
    return x_next


def sim_openloop_batch(
    dynamical_system: DynamicalSystem,
    x0_batch: torch.Tensor,
    u_trj_batch: torch.Tensor,
    noise_trj_batch: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Simulate the system with a batch open loop u trajectories.

    Args:
      u_trj: Size (batch_size, T, dim_u)
      x0_batch: Size (batch_size, dim_x)
      noise_trj: Size (batch_size, T, dim_x)
    Returns:
      x_trj_batch: Size (batch_size, T+1, dim_x)
    """
    batch_size = x0_batch.shape[0]
    T = u_trj_batch.shape[1]
    assert x0_batch.shape[1] == dynamical_system.dim_x
    assert u_trj_batch.shape[0] == batch_size
    assert u_trj_batch.shape[2] == dynamical_system.dim_u
    if noise_trj_batch is not None:
        assert noise_trj_batch.shape == (batch_size, T, dynamical_system.dim_x)
    x_trj_batch = torch.zeros(
        (batch_size, T + 1, dynamical_system.dim_x), device=x0_batch.device
    )
    x_trj_batch[:, 0, :] = x0_batch
    for i in range(T):
        x_trj_batch[:, i + 1, :] = dynamical_system.dynamics_batch(
            x_trj_batch[:, i, :], u_trj_batch[:, i, :]
        )
        if noise_trj_batch is not None:
            x_trj_batch[:, i + 1, :] = (
                x_trj_batch[:, i + 1, :] + noise_trj_batch[:, i, :]
            )
    return x_trj_batch


def sim_openloop(
    dynamical_system: DynamicalSystem,
    x0: torch.Tensor,
    u_trj: torch.Tensor,
    noise_trj: Optional[torch.Tensor],
) -> torch.Tensor:
    return sim_openloop_batch(
        dynamical_system,
        x0.unsqueeze(0),
        u_trj.unsqueeze(0),
        None if noise_trj is None else noise_trj.unsqueeze(0),
    ).squeeze(0)
