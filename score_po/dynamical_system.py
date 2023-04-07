import abc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

from score_po.policy import Policy
from score_po.nn import AdamOptimizerParams 

"""
Classes for dynamical systems. 
"""


class DynamicalSystem(abc.ABC):
    def __init__(self, dim_x, dim_u):
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.is_differentiable = False

    @abc.abstractmethod
    def dynamics(self, x, u):
        """
        Evaluate dynamics in state-space form.
        input:
            x of shape n
            u of shape m
        output:
            xnext of shape n
        """

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


class NNDynamicalSystem(DynamicalSystem):
    """
    Neural network dynamical system, where the network is of
    - input shape (n + m)
    - output shape (n)
    """

    def __init__(self, network, dim_x, dim_u):
        super().__init__(dim_x, dim_u)
        self.net = network
        self.is_differentiable = True

    def dynamics(self, x, u, eval=True):
        if eval:
            self.net.eval()

        input = self.hstack((x, u))[None, :]
        return self.net(input)[0, :]

    def dynamics_batch(self, x_batch, u_batch, eval=True):
        if eval:
            self.net.eval()

        input = self.hstack((x_batch, u_batch))
        return self.net(input)

    def evaluate_dynamic_loss(self, data, labels, sigma=0.0):
        """
        Evaluate L2 loss.
        data_samples:
            data of shape (B, dim_x + dim_u + dim_x)
            sigma: vector of dim_x + dim_u used for data augmentation.
        """
        B = data.shape[0]
        if sigma > 0:
            noise = torch.normal(0, sigma, size=data.shape, device=data.device)
            databar = data + noise
        else:
            databar = data
        pred = self.dynamics_batch(
            databar[:, : self.dim_x], databar[:, self.dim_x :], eval=False
        )  # B x dim_x
        loss = 0.5 * ((labels - pred) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    @dataclass
    class TrainParams:
        adam_params: AdamOptimizerParams
        dataset_split: Tuple[int] = (0.9, 0.1)
        # Save the best model (with the smallest validation error to this path)
        save_best_model: Optional[str] = None
        enable_wandb: bool = False

        def __init__(self):
            self.adam_params = AdamOptimizerParams()

    def train_network(self, dataset: TensorDataset, params: TrainParams, sigma=0.0):
        """
        Train a network given a dataset and optimization parameters.
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

        loss_lst = torch.zeros(params.adam_params.epochs)

        best_loss = np.inf

        for epoch in tqdm(range(params.adam_params.epochs)):
            training_loss = 0.0
            for x_batch, u_batch, xnext_batch in data_loader_train:
                loss = self.evaluate_dynamic_loss(
                    torch.cat((x_batch, u_batch), dim=-1), xnext_batch, sigma=sigma
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                training_loss += (loss.item() * x_batch.shape[0])

            training_loss /= len(train_dataset)

            with torch.no_grad():
                for x_all, u_all, xnext_all in data_loader_eval:
                    loss_eval = self.evaluate_dynamic_loss(
                        torch.cat((x_all, u_all), dim=-1), xnext_all, sigma=0
                    )
                    loss_lst[epoch] = loss_eval.item()
                if params.enable_wandb:
                    wandb.log({"training_loss": training_loss, "val loss": loss_eval.item()}, step=epoch)
                else:
                    print(f"epoch={epoch}, training_loss={training_loss}, val_loss={loss_eval.item()}")
                if params.save_best_model is not None and loss_eval.item() < best_loss:
                    torch.save(self.net.mlp.state_dict(), params.save_best_model)
                    best_loss = loss_eval.item()

        return loss_lst

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
