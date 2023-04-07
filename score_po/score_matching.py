from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

from score_po.nn import AdamOptimizerParams


class ScoreFunctionEstimator:
    """
    Score function estimator that stores the object
    ∇_z log p(z): R^(dim_x + dim_u) -> R^(dim_x + dim_u), where
    z = [x, u]^T. The class has functionalities for:
    1. Returning ∇_z log p(z), ∇_x log p(x,u), ∇_u log p(x,u)
    2. Training the estimator from existing data of (x,u) pairs.
    3. Training the
    """

    def __init__(self, network, dim_x, dim_u):
        self.net = network
        self.dim_x = dim_x
        self.dim_u = dim_u

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

        input = torch.hstack((z, sigma * torch.ones(z.shape[0], 1, device=z.device)))
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

    @dataclass
    class TrainParams:
        adam_params: AdamOptimizerParams
        dataset_split: Tuple[int] = (0.9, 0.1)
        # Save the best model (with the smallest validation error to this path)
        save_best_model: Optional[str] = None
        enable_wandb: bool = False

        def __init__(self):
            self.adam_params = AdamOptimizerParams()

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
        optimizer = optim.Adam(self.net.parameters(), params.adam_params.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, params.adam_params.epochs
        )

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, params.dataset_split
        )
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=params.adam_params.batch_size
        )
        data_loader_eval = torch.utils.data.DataLoader(
            val_dataset, batch_size=len(val_dataset)
        )

        sigma_lst = np.geomspace(sigma_min, sigma_max, n_sigmas)
        loss_lst = np.zeros(params.adam_params.epochs)

        best_loss = np.inf
        for epoch in tqdm(range(params.adam_params.epochs)):
            training_loss = 0.0
            for z_batch in data_loader_train:
                z_batch = z_batch[0]
                loss = self.evaluate_denoising_loss(z_batch, sigma_lst)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                training_loss += loss.item() * z_batch.shape[0]
            training_loss /= len(train_dataset)

            with torch.no_grad():
                for z_all in data_loader_eval:
                    z_all = z_all[0]
                    loss_eval = self.evaluate_denoising_loss(z_all, sigma_lst)
                    loss_lst[epoch] = loss_eval.item()
                if params.enable_wandb:
                    wandb.log(
                        {"training_loss": training_loss, "val_loss": loss_eval.item()},
                        step=epoch,
                    )
                else:
                    print(
                        f"epoch {epoch}, training loss {training_loss}, val_loss {loss_eval.item()}"
                    )
                if params.save_best_model is not None and loss_eval.item() < best_loss:
                    torch.save(self.net.state_dict(), params.save_best_model)
                    best_loss = loss_eval.item()

        return loss_lst

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
