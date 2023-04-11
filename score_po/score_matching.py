from dataclasses import dataclass
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

import wandb

from score_po.nn import AdamOptimizerParams, TrainParams, train_network

"""
Classes for training score functions.
"""


class ScoreEstimator:
    """
    Score function estimator that stores the object
    ∇_z log p(z): R^(dim_x + dim_u) -> R^(dim_x + dim_u), where
    z = [x, u]^T. The class has functionalities for:
    1. Returning ∇_z log p(z), ∇_x log p(x,u), ∇_u log p(x,u)
    2. Training the estimator from existing data of (x,u) pairs.

    Note that this score estimator trains for a single noise-level
    without being conditioned on noise.
    """

    def __init__(self, dim_x, dim_u, network):
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        self.sigma = 0.1
        self.check_input_consistency()

    def check_input_consistency(self):
        if self.net.dim_in is not (self.dim_x + self.dim_u):
            raise ValueError("Inconsistent input size of neural network.")
        if self.net.dim_out is not (self.dim_x + self.dim_u):
            raise ValueError("Inconsistent output size of neural network.")

    def get_score_z_given_z(self, input, eval=True):
        self.net.to(input.device)
        if eval:
            self.net.eval()
        else:
            self.net.train()

        return self.net(input)

    def get_score_x_given_z(self, z, eval=True):
        """Give ∇_x log p(z) part of the score function."""
        return self.get_score_z_given_z(z, eval)[:, : self.dim_x]

    def get_score_u_given_z(self, z, eval=True):
        """Give ∇_u log p(z) part of the score function."""
        return self.get_score_z_given_z(z, eval)[:, self.dim_x :]

    # The rest of the functions are same except they have x u as arguments.
    def get_score_z_given_xu(self, x, u, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_z_given_z(z, eval)

    def get_score_x_given_xu(self, x, u, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_x_given_z(z, eval)

    def get_score_u_given_xu(self, x, u, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_u_given_z(z, eval)

    def evaluate_denoising_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(2) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.
        """
        databar = data + torch.randn_like(data) * sigma

        target = -1 / (sigma**2) * (databar - data)
        scores = self.get_score_z_given_z(databar, eval=False)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def evaluate_slicing_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(3) from Song & Ermon 2019.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        Adopted from Song's codebase.
        """
        databar = data + torch.randn_like(data) * sigma
        databar.requires_grad = True

        vectors = torch.randn_like(databar)

        grad1 = self.get_score_z_given_z(databar)
        gradv = torch.sum(grad1 * vectors)
        grad2 = autograd.grad(gradv, databar, create_graph=True)[0]
        grad1 = grad1.view(databar.shape[0], -1)

        loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0
        loss2 = torch.sum((vectors * grad2).view(databar.shape[0], -1), dim=-1)

        loss1 = loss1.view(1, -1).mean(dim=0)
        loss2 = loss2.view(1, -1).mean(dim=0)
        loss = loss1 + loss2

        return loss.mean()

    def evaluate_loss(self, z_batch, sigma, mode):
        if mode == "denoising":
            loss = self.evaluate_denoising_loss_with_sigma(z_batch, sigma)
        elif mode == "slicing":
            loss = self.evaluate_slicing_loss_with_sigma(z_batch, sigma)
        else:
            raise ValueError("Invalid mode. Only supports denoising or slicing.")
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma,
        mode="denoising",
        split=True,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        # NOTE: We use x_batch[0] here since dataloader gives a tuple (x_batch,).
        loss_fn = lambda x_batch, net: self.evaluate_loss(x_batch[0], sigma, mode)
        loss_lst = train_network(self.net, params, dataset, loss_fn, split)
        return loss_lst

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))


class NoiseConditionedScoreEstimator(ScoreEstimator):
    """
    A noise-conditioned score estimator that accepts noise input.
    """

    def __init__(self, dim_x, dim_u, network):
        super().__init__(dim_x, dim_u, network)

    def check_input_consistency(self):
        if self.net.dim_in is not (self.dim_x + self.dim_u + 1):
            raise ValueError(
                "Inconsistent input size of neural network. "
                + "Did you forget to add 1 dimension for noise parameter?"
            )
        if self.net.dim_out is not (self.dim_x + self.dim_u):
            raise ValueError("Inconsistent output size of neural network")

    def get_score_z_given_z(self, z, sigma, eval=True):
        """
        input:
            z of shape (B, dim_x + dim_u)
            sigma, float
        output:
            ∇_z log p(z) of shape (B, dim_x + dim_u)
        """
        if eval:
            self.net.eval()

        input = torch.hstack((z, sigma * torch.ones(z.shape[0], 1).to(z.device)))
        return self.net(input)

    def get_score_x_given_z(self, z, sigma, eval=True):
        """Give ∇_x log p(z) part of the score function."""
        return self.get_score_z_given_z(z, sigma, eval)[:, : self.dim_x]

    def get_score_u_given_z(self, z, sigma, eval=True):
        """Give ∇_u log p(z) part of the score function."""
        return self.get_score_z_given_z(z, sigma, eval)[:, self.dim_x :]

    # The rest of the functions are same except they have x u as arguments.
    def get_score_z_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_z_given_z(z, sigma, eval)

    def get_score_x_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_x_given_z(z, sigma, eval)

    def get_score_x_given_xu(self, x, u, sigma, eval=True):
        z = torch.hstack((x, u))
        return self.get_score_u_given_z(z, sigma, eval)

    def evaluate_denoising_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(5) from Song & Ermon.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        """
        databar = data + torch.randn_like(data) * sigma
        target = -1 / (sigma**2) * (databar - data)
        scores = self.get_score_z_given_z(databar, sigma, eval=False)

        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        loss = 0.5 * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def evaluate_slicing_loss_with_sigma(self, data, sigma):
        """
        Evaluate denoising loss, Eq.(5) from Song & Ermon.
            data of shape (B, dim_x + dim_u)
            sigma, a scalar variable.
        """
        databar = data + torch.randn_like(data) * sigma
        databar.requires_grad_(True)

        vectors = torch.randn_like(databar)

        grad1 = self.get_score_z_given_z(databar, sigma)
        gradv = torch.sum(grad1 * vectors)
        grad2 = autograd.grad(gradv, databar, create_graph=True)[0]
        grad1 = grad1.view(databar.shape[0], -1)

        loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0
        loss2 = torch.sum((vectors * grad2).view(databar.shape[0], -1), dim=-1)

        loss1 = loss1.view(1, -1).mean(dim=0)
        loss2 = loss2.view(1, -1).mean(dim=0)
        loss = loss1 + loss2

        return loss.mean()

    def evaluate_denoising_loss(self, data, sigma_lst):
        """
        Evaluate loss given input:
            data: of shape (B, dim_x + dim_u)
            sigma_lst: a geometric sequence of sigmas to train on.
        """
        loss = torch.zeros(1, device=data.device)
        for sigma in sigma_lst:
            loss += sigma**2.0 * self.evaluate_denoising_loss_with_sigma(data, sigma)
        return loss / len(sigma_lst)

    def evaluate_slicing_loss(self, data, sigma_lst):
        """
        Evaluate loss given input:
            data: of shape (B, dim_x + dim_u)
            sigma_lst: a geometric sequence of sigmas to train on.
        """
        loss = torch.zeros(1, device=data.device)
        for sigma in sigma_lst:
            loss += sigma**2.0 * self.evaluate_slicing_loss_with_sigma(data, sigma)
        return loss / len(sigma_lst)

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        sigma_max=1,
        sigma_min=-3,
        n_sigmas=10,
    ):
        """
        Train a network given a dataset and optimization parameters.
        Following Song & Ermon, we train a noise-conditioned score function where
        the sequence of noise is provided with a geometric sequence of length
        n_sigmas, with max 10^log_sigma_max and min 10^log_sigma_min.
        """
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), params.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.epochs)

        data_loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=params.adam_params.batch_size
        )
        data_loader_eval = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

        sigma_lst = np.geomspace(sigma_min, sigma_max, n_sigmas)
        loss_lst = np.zeros(params.adam_params.epochs)

        best_loss = np.inf
        for epoch in tqdm(range(params.adam_params.epochs)):
            training_loss = 0.0
            for z_batch in data_loader_train:
                z_batch = z_batch[0]
                optimizer.zero_grad()
                loss = self.evaluate_denoising_loss(z_batch, sigma_lst)
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                for z_all in data_loader_eval:
                    z_all = z_all[0]
                    loss_eval = self.evaluate_denoising_loss(z_all, sigma_lst)
                    loss_lst[epoch] = loss_eval.item()
                if params.enabled:
                    wandb.log(
                        {"loss": loss_eval.item()},
                        step=epoch,
                    )
                else:
                    print(
                        f"epoch {epoch}, loss {loss_eval.item()}"
                    )
                if params.save_best_model is not None and loss_eval.item() < best_loss:
                    model_path = os.path.join(os.getcwd(), params.save_best_model)
                    torch.save(self.net.state_dict(), model_path)
                    best_loss = loss_eval.item()

        return loss_lst
