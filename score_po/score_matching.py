from dataclasses import dataclass
import os
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

import wandb

from score_po.nn import (
    AdamOptimizerParams,
    TrainParams,
    train_network,
    Normalizer,
    save_module,
    MLPwEmbedding,
)

"""
Classes for training score functions.
"""


class ScoreEstimatorXu(torch.nn.Module):
    """
    Score function estimator that stores the object
    ∇_z log p(z): R^(dim_x + dim_u) -> R^(dim_x + dim_u), where
    z = [x, u]^T. The class has functionalities for:
    1. Returning ∇_z log p(z)
    2. Training the estimator from existing data of (x,u) pairs.

    Note that this score estimator trains for a single noise-level
    without being conditioned on noise.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        network,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
    ):
        """
        We denote z̅ as z after normalization, namely
        z̅ = (z - b) / k
        where k is the normalization constant.
        The network estimate ∇_z̅ log p(z̅), the score of the normalized ̅z̅, based on
        which we compute ∇_z log p(z), the score of the un-normalized z.

        Args:
          network: A network ϕ that outputs ϕ(z̅) ≈ ∇_z̅ log p(z̅), where z̅ is the
          normalized data.
          z_normalizer: The normalizer for z.
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
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
        self.register_buffer("sigma", torch.ones(1))
        self.check_input_consistency()

    def check_input_consistency(self):
        if hasattr(self.net, "dim_in") and self.net.dim_in is not (
            self.dim_x + self.dim_u
        ):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and self.net.dim_out is not (
            self.dim_x + self.dim_u
        ):
            raise ValueError("Inconsistent output size of neural network.")

    def get_xu_from_z(self, z):
        x = z[:, : self.dim_x]
        u = z[:, self.dim_x :]
        return x, u

    def get_z_from_xu(self, x, u):
        return torch.cat((x, u), dim=1)

    def normalize_z(self, z):
        """
        Normalize z assuming z = [x,u]
        """
        x, u = self.get_xu_from_z(z)
        xbar = self.x_normalizer(x)
        ubar = self.u_normalizer(u)
        zbar = self.get_z_from_xu(xbar, ubar)
        return zbar

    def _get_score_zbar_given_zbar(self, zbar):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        self.net.to(zbar.device)
        return self.net(zbar)

    def get_score_z_given_z(self, z):
        """
        Compute ∇_z log p(z).

        We know that z̅ = (z−b)/k, hence p(z̅) = k*p(z) (based on transforming a
        continuous-valued random variable), hence log p(z̅) = log k + log p(z).
        As a result, ∇_z̅ log p(z̅) = ∇_z̅ log p(z) = k * ∇_z log p(z), namely
        ∇_z log p(z) = 1/k * ∇_z̅ log p(z̅)
        """
        zbar = self.normalize_z(z)
        return self._get_score_zbar_given_zbar(zbar) / torch.cat(
            (self.x_normalizer.k, self.u_normalizer.k)
        )

    def forward(self, z):
        return self.get_score_z_given_z(z)

    def evaluate_loss(self, x_batch, u_batch, sigma):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:
          sigma: The noise level in the NORMALIZED z̅ space.
        """

        # Normalize the data
        z_batch = self.get_z_from_xu(x_batch, u_batch)
        data_normalized = self.normalize_z(z_batch)
        databar = data_normalized + torch.randn_like(z_batch) * sigma

        target = -1 / (sigma**2) * (databar - data_normalized)
        scores = self._get_score_zbar_given_zbar(databar)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma: torch.Tensor,
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        # Retain memory of the noise level.
        self.sigma = sigma
        # We assume z_batch is (x_batch, u_batch)
        loss_fn = lambda z_batch, net: self.evaluate_loss(
            z_batch[0], z_batch[1], self.sigma
        )
        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst


class ScoreEstimatorXux(torch.nn.Module):
    """
    Score function estimator that stores the object
    ∇_z log p(z): R^(dim_x + dim_u + dim_x) -> R^(dim_x + dim_u + dim_x), where
    z = [x, u, xnext]^T. The class has functionalities for:
    1. Returning ∇_z log p(z)
    2. Training the estimator from existing data of (x,u, xnext) pairs.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        network,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
        alpha=1.0,
    ):
        """
        We denote z̅ as z after normalization, namely
        z̅ = (z - b) / k
        where k is the normalization constant.
        The network estimate ∇_z̅ log p(z̅), the score of the normalized ̅z̅, based on
        which we compute ∇_z log p(z), the score of the un-normalized z.

        Args:
          network: A network ϕ that outputs ϕ(z̅) ≈ ∇_z̅ log p(z̅), where z̅ is the
          normalized data.
          z_normalizer: The normalizer for z.

        We also use a scalar variable alpha to weight the normalization on xnext
        differently from normalization on x,u during the training process which
        gives us a handle to penalize dynamics error more in log p(x,u,xnext) formulation.
        A higher value of alpha will penalize dynamics error more.
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
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
        self.xnext_normalizer: Normalizer = Normalizer(
            k=self.x_normalizer.k / alpha, b=self.x_normalizer.b
        )
        self.register_buffer("sigma", torch.ones(1))
        self.check_input_consistency()

    def check_input_consistency(self):
        if hasattr(self.net, "dim_in") and self.net.dim_in is not (
            self.dim_x + self.dim_u + self.dim_x
        ):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and self.net.dim_out is not (
            self.dim_x + self.dim_u + self.dim_x
        ):
            raise ValueError("Inconsistent output size of neural network.")

    def get_xux_from_z(self, z):
        x = z[:, : self.dim_x]
        u = z[:, self.dim_x : self.dim_x + self.dim_u]
        xnext = z[:, self.dim_x + self.dim_u :]
        return x, u, xnext

    def get_z_from_xux(self, x, u, xnext):
        return torch.cat((x, u, xnext), dim=1)

    def normalize_z(self, z):
        """
        Normalize z assuming z = [x,u]
        """
        x, u, xnext = self.get_xux_from_z(z)
        xbar = self.x_normalizer(x)
        ubar = self.u_normalizer(u)
        xnextbar = self.xnext_normalizer(xnext)
        zbar = self.get_z_from_xux(xbar, ubar, xnextbar)
        return zbar

    def _get_score_zbar_given_zbar(self, zbar):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        self.net.to(zbar.device)
        if eval:
            self.net.eval()
        else:
            self.net.train()
        return self.net(zbar)

    def get_score_z_given_z(self, z):
        """
        Compute ∇_z log p(z).

        We know that z̅ = (z−b)/k, hence p(z̅) = k*p(z) (based on transforming a
        continuous-valued random variable), hence log p(z̅) = log k + log p(z).
        As a result, ∇_z̅ log p(z̅) = ∇_z̅ log p(z) = k * ∇_z log p(z), namely
        ∇_z log p(z) = 1/k * ∇_z̅ log p(z̅)
        """
        zbar = self.normalize_z(z)
        return self._get_score_zbar_given_zbar(zbar) / torch.cat(
            (self.x_normalizer.k, self.u_normalizer.k, self.xnext_normalizer.k)
        )

    def forward(self, z):
        return self.get_score_z_given_z(z)

    def evaluate_loss(self, x_batch, u_batch, xnext_batch, sigma):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:
          sigma: The noise level in the NORMALIZED z̅ space.
        """

        # Normalize the data
        z_batch = self.get_z_from_xux(x_batch, u_batch, xnext_batch)
        data_normalized = self.normalize_z(z_batch)
        databar = data_normalized + torch.randn_like(z_batch) * sigma

        target = -1 / (sigma**2) * (databar - data_normalized)
        scores = self._get_score_zbar_given_zbar(databar)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma: torch.Tensor,
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma = sigma
        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        loss_fn = lambda z_batch, net: self.evaluate_loss(
            z_batch[0], z_batch[1], z_batch[2], self.sigma
        )
        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst


class NoiseConditionedScoreEstimatorXu(ScoreEstimatorXu):
    """
    Train a noise conditioned score estimator.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        network: MLPwEmbedding,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
    ):
        super().__init__(dim_x, dim_u, network, x_normalizer, u_normalizer)
        self.register_buffer("sigma_lst", torch.ones(network.embedding_size))

    def _get_score_zbar_given_zbar(self, zbar, i):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        return self.net(zbar, i)

    def get_score_z_given_z(self, z, i):
        """
        Compute ∇_z log p(z, sigma) where sigma is the noise level.
        We enter an integer value here as a token for sigma, s.t.
        self.sigma_lst[i] = sigma.
        """
        zbar = self.normalize_z(z)
        return self._get_score_zbar_given_zbar(zbar, i) / torch.cat(
            (self.x_normalizer.k, self.u_normalizer.k)
        )

    def evaluate_loss(self, x_batch, u_batch, i):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:
          i: index of sigma_lst.
        """

        sigma = self.sigma_lst[i]

        # Normalize the data
        z_batch = self.get_z_from_xu(x_batch, u_batch)
        data_normalized = self.normalize_z(z_batch)
        databar = data_normalized + torch.randn_like(z_batch) * sigma

        target = -1 / (sigma**2) * (databar - data_normalized)
        scores = self._get_score_zbar_given_zbar(databar, i)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_lst: torch.Tensor,
        split=True,
        sample_sigma=False,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma_lst = sigma_lst

        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        def loss_fn(z_batch, net):
            loss = 0.0
            if sample_sigma:
                idx = torch.randint(0, len(sigma_lst), (1,)).item()
                loss_sigma = self.evaluate_loss(z_batch[0], z_batch[1], idx)
                loss = (sigma_lst[idx] ** 2.0) * loss_sigma
            else:
                for i, sigma in enumerate(sigma_lst):
                    loss_sigma = self.evaluate_loss(z_batch[0], z_batch[1], i)
                    loss += (sigma**2.0) * loss_sigma
            return loss

        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst


class NoiseConditionedScoreEstimatorXux(ScoreEstimatorXux):
    """
    Train a noise conditioned score estimator.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        network: MLPwEmbedding,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
        alpha=1.0,
    ):
        super().__init__(dim_x, dim_u, network, x_normalizer, u_normalizer, alpha)
        self.register_buffer("sigma_lst", torch.ones(network.embedding_size))

    def _get_score_zbar_given_zbar(self, zbar, i):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        return self.net(zbar, i)

    def get_score_z_given_z(self, z, i):
        """
        Compute ∇_z log p(z, sigma) where sigma is the noise level.
        We enter an integer value here as a token for sigma, s.t.
        self.sigma_lst[i] = sigma.
        """
        zbar = self.normalize_z(z)
        return self._get_score_zbar_given_zbar(zbar, i) / torch.cat(
            (self.x_normalizer.k, self.u_normalizer.k, self.xnext_normalizer.k)
        )

    def evaluate_loss(self, x_batch, u_batch, xnext_batch, i):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:
          i: index of sigma_lst.
        """

        sigma = self.sigma_lst[i]

        # Normalize the data
        z_batch = self.get_z_from_xux(x_batch, u_batch, xnext_batch)
        data_normalized = self.normalize_z(z_batch)
        databar = data_normalized + torch.randn_like(z_batch) * sigma

        target = -1 / (sigma**2) * (databar - data_normalized)
        scores = self._get_score_zbar_given_zbar(databar, i)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_lst: torch.Tensor,
        split=True,
        sample_sigma=False,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma_lst = sigma_lst

        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        def loss_fn(z_batch, net):
            loss = 0.0
            if sample_sigma:
                idx = torch.randint(0, len(sigma_lst), (1,)).item()
                loss_sigma = self.evaluate_loss(z_batch[0], z_batch[1], z_batch[2], idx)
                loss = (sigma_lst[idx] ** 2.0) * loss_sigma
            else:
                for i, sigma in enumerate(sigma_lst):
                    loss_sigma = self.evaluate_loss(
                        z_batch[0], z_batch[1], z_batch[2], i
                    )
                    loss += (sigma**2.0) * loss_sigma
            return loss

        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst


def langevin_dynamics(
    x0: torch.Tensor,
    score: torch.nn.Module,
    epsilon: float,
    steps: int,
    noise: bool = True,
) -> torch.Tensor:
    """
    Generate samples using Langevin dynamics
    xₜ₊₁ = xₜ + ε/2*∇ₓ log p(xₜ) + √ε * noise
    where noise ~ N(0, I)

    Args:
      x0: A batch of samples at the beginning. Size is (batch_size, x_size)
      score: a torch Module that outputs ∇ₓ log p(x)
      epsilon: ε in the documentation above. The step size.
      steps: The total number of steps in Langenvin dynamics.
    Returns:
      x_history: the history of all xₜ.
    """
    assert epsilon > 0
    sqrt_epsilon = np.sqrt(epsilon)
    x_history = x0.repeat((steps,) + (1,) * x0.ndim)
    for t in range(1, steps):
        x_history[t] = (
            x_history[t - 1]
            + score(x_history[t - 1]) * epsilon / 2
            + float(noise) * (sqrt_epsilon * torch.randn_like(x_history[t - 1]))
        )
    return x_history


def noise_conditioned_langevin_dynamics(
    x0: torch.Tensor,
    score_estimator: Union[
        NoiseConditionedScoreEstimatorXu, NoiseConditionedScoreEstimatorXux
    ],
    epsilon: float,
    steps: int,
    noise: bool = True,
):
    assert epsilon > 0
    sqrt_epsilon = np.sqrt(epsilon)
    x_history = x0.repeat((steps,) + (1,) * x0.ndim)
    for t in range(1, steps):
        idx = round(len(score_estimator.sigma_lst) * (t / (steps + 1)) - 0.5)
        x_history[t] = (
            x_history[t - 1]
            + score_estimator.get_score_z_given_z(x_history[t - 1], idx) * epsilon / 2
            + float(noise) * (sqrt_epsilon * torch.randn_like(x_history[t - 1]))
        )
    return x_history
