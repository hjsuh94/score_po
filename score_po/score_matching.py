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
        if hasattr(self.net, "dim_in") and self.net.dim_in != (
            self.dim_x + self.dim_u + self.dim_x
        ):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and self.net.dim_out != (
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

class NoiseConditionedScoreEstimatorXuxImage(ScoreEstimatorXux):
    """
    Train a noise conditioned score estimator.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        sigma_list,
        network: MLPwEmbedding,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
    ):
        super().__init__(dim_x, dim_u, network, x_normalizer, u_normalizer)
        self.register_buffer("sigma_lst", sigma_list)

    def _get_score_zbar_given_zbar(self, zbar, sigma):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        return self.net(zbar, sigma)

    def get_score_z_given_z(self, z, sigma):
        """
        Compute ∇_z log p(z, sigma) where sigma is the noise level.
        We enter an integer value here as a token for sigma, s.t.
        self.sigma_lst[i] = sigma.
        """
        zbar = z
        return self._get_score_zbar_given_zbar(zbar, sigma)

    def evaluate_loss(self, x_batch, u_batch, xnext_batch, i, anneal_power=2):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:

        """
        samples_x = x_batch.reshape(-1, 1, 32, 32)
        samples_xnext = xnext_batch.reshape(-1, 1, 32, 32)
        samples = torch.cat((samples_x, samples_xnext), dim=1)
        labels = torch.randint(0, len(self.sigma_lst), (samples.shape[0],), device=samples.device)
        used_sigmas = self.sigma_lst[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        used_sigmas_u = self.sigma_lst[labels].view(u_batch.shape[0], *([1] * len(u_batch.shape[1:])))
        noise = torch.randn_like(samples) * used_sigmas
        noise_u = torch.randn_like(u_batch) * used_sigmas_u
        perturbed_samples = samples + noise
        perturbed_samples_u = u_batch + noise_u
        target = - 1 / (used_sigmas ** 2) * noise
        target_u = - 1 / (used_sigmas_u ** 2) * noise_u
        scores, scores_u = self.net(perturbed_samples, perturbed_samples_u, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        loss_u = 1 / 2. * ((scores_u - target_u) ** 2).sum(dim=-1) * used_sigmas_u.squeeze() ** anneal_power
        loss += loss_u
        return loss.mean(dim=0)

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_lst: torch.Tensor,
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma_lst = sigma_lst

        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        def loss_fn(z_batch, net):
            loss = self.evaluate_loss(z_batch[0], z_batch[1], z_batch[2], 0)
            return loss

        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst

class NoiseConditionedScoreEstimatorXuxImageU(ScoreEstimatorXux):
    """
    Train a noise conditioned score estimator.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        sigma_list,
        network: MLPwEmbedding,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
    ):
        super().__init__(dim_x, dim_u, network, x_normalizer, u_normalizer)
        self.register_buffer("sigma_lst", sigma_list)

    def _get_score_zbar_given_zbar(self, zbar, sigma):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        return self.net(zbar, sigma)

    def get_score_z_given_z(self, z, sigma):
        """
        Compute ∇_z log p(z, sigma) where sigma is the noise level.
        We enter an integer value here as a token for sigma, s.t.
        self.sigma_lst[i] = sigma.
        """
        labels = sigma*torch.ones(z.shape[0], device=z.device)
        scores = self.net(z, labels.long())
        return scores

    def get_xux_from_z(self, z):
        x = z[:, 0, :, :]
        u = z[:, 1, :, :]
        xnext = z[:, 2, :, :]
        return x, u, xnext

    def evaluate_loss(self, x_batch, u_batch, xnext_batch, i, anneal_power=2):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:

        """
        samples_x = x_batch.reshape(-1, 1, 32, 32)
        samples_u = u_batch.reshape(-1, 1, 32, 32)
        samples_xnext = xnext_batch.reshape(-1, 1, 32, 32)
        samples = torch.cat((samples_x, samples_u, samples_xnext), dim=1)
        labels = torch.randint(0, len(self.sigma_lst), (samples.shape[0],), device=samples.device)
        used_sigmas = self.sigma_lst[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        noise = torch.randn_like(samples) * used_sigmas
        perturbed_samples = samples + noise
        target = - 1 / (used_sigmas ** 2) * noise
        scores = self.net(perturbed_samples, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        return loss.mean(dim=0)

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_lst: torch.Tensor,
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma_lst = sigma_lst

        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        def loss_fn(z_batch, net):
            loss = self.evaluate_loss(z_batch[0], z_batch[1], z_batch[2], 0)
            return loss

        loss_lst = train_network(self, params, dataset, loss_fn, split)
        return loss_lst

class NoiseConditionedScoreEstimatorXuImageU(ScoreEstimatorXu):
    """
    Train a noise conditioned score estimator.
    """

    def __init__(
        self,
        dim_x,
        dim_u,
        sigma_list,
        network: MLPwEmbedding,
        x_normalizer: Normalizer = None,
        u_normalizer: Normalizer = None,
    ):
        super().__init__(dim_x, dim_u, network, x_normalizer, u_normalizer)
        self.register_buffer("sigma_lst", sigma_list)

    def _get_score_zbar_given_zbar(self, zbar, sigma):
        """
        Compute the score ∇_z̅ log p(z̅) for the normalized z̅
        """
        return self.net(zbar, sigma)

    def get_score_z_given_z(self, z, sigma):
        """
        Compute ∇_z log p(z, sigma) where sigma is the noise level.
        We enter an integer value here as a token for sigma, s.t.
        self.sigma_lst[i] = sigma.
        """
        # samples = z
        labels = sigma*torch.ones(z.shape[0], device=z.device)
        scores = self.net(z, labels.long())
        return scores

    def get_xu_from_z(self, z):
        x = z[:, 0, :, :]
        u = z[:, 1, :, :]
        return x, u

    def evaluate_loss(self, x_batch, u_batch, i, anneal_power=2):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.

        Args:

        """
        samples_x = x_batch.reshape(-1, 1, 32, 32)
        samples_u = u_batch.reshape(-1, 1, 32, 32)
        samples = torch.cat((samples_x, samples_u), dim=1)
        labels = torch.randint(0, len(self.sigma_lst), (samples.shape[0],), device=samples.device)
        used_sigmas = self.sigma_lst[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        noise = torch.randn_like(samples) * used_sigmas
        perturbed_samples = samples + noise
        target = - 1 / (used_sigmas ** 2) * noise
        scores = self.net(perturbed_samples, labels)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        return loss.mean(dim=0)

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_lst: torch.Tensor,
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        self.sigma_lst = sigma_lst

        # We assume z_batch is (x_batch, u_batch, xnext_batch)
        def loss_fn(z_batch, net):
            loss = self.evaluate_loss(z_batch[0], z_batch[1], 0)
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


def noise_conditioned_langevin_dynamics_image(x_mod, u_mod, scorenet, sigmas,
                                              n_steps_each=5, step_lr=0.0000062,
                                              final_only=False, verbose=True, denoise=True):
    """
    Args:
        x_mod: Batch of samples of (x_t, x_{t+1}) to initialize sampling. Size: (batch_size, 2, num_pixels, num_pixels)
        u_mod: Batch of samples u_t to initialize sampling. Size: (batch_size, num_control)
        scorenet: a torch Module that outputs ∇ₓ log p(x)
        sigmas: noise levels
        n_steps_each: number of Langevin steps for each noise level
        step_lr: Langevin dynamics step size
        final_only: if True: returns only the final sample; else, return full history of samples
        verbose: whether to print out the progress
        denoise: whether to denoise the final samples
    """
    images, images_u = [], []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad, grad_u = scorenet(x_mod, u_mod, labels)
                noise = torch.randn_like(x_mod).to(x_mod.device)
                noise_u = torch.randn_like(u_mod).to(u_mod.device)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)
                u_mod = u_mod + step_size * grad_u + noise_u * torch.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                    images_u.append(u_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            grad_final, grad_u_final = scorenet(x_mod, u_mod, last_noise)
            x_mod = x_mod + sigmas[-1] ** 2 * grad_final
            u_mod = u_mod + sigmas[-1] ** 2 * grad_u_final
            images.append(x_mod.to('cpu'))
            images_u.append(u_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')], [u_mod.to('cpu')]
        else:
            return images, images_u

def noise_conditioned_langevin_dynamics_image_u(x_mod, scorenet, sigmas,
                                              n_steps_each=5, step_lr=0.0000062,
                                              final_only=False, verbose=True, denoise=True):
    """
    Args:
        x_mod: Batch of samples of (x_t, x_{t+1}) to initialize sampling. Size: (batch_size, 2, num_pixels, num_pixels)
        u_mod: Batch of samples u_t to initialize sampling. Size: (batch_size, num_control)
        scorenet: a torch Module that outputs ∇ₓ log p(x)
        sigmas: noise levels
        n_steps_each: number of Langevin steps for each noise level
        step_lr: Langevin dynamics step size
        final_only: if True: returns only the final sample; else, return full history of samples
        verbose: whether to print out the progress
        denoise: whether to denoise the final samples
    """
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)
                noise = torch.randn_like(x_mod).to(x_mod.device)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            grad_final = scorenet(x_mod, last_noise)
            x_mod = x_mod + sigmas[-1] ** 2 * grad_final
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images