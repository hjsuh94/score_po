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
from score_po.nn import AdamOptimizerParams, WandbParams, TrainParams
import score_po.nn

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

    def __init__(self, dim_x, dim_u, network):
        super().__init__(dim_x, dim_u)
        self.net = network
        self.is_differentiable = True
        self.check_input_consistency()

    def check_input_consistency(self):
        if hasattr(self.net, "dim_in") and (self.net.dim_in is not self.dim_x + self.dim_u):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and (self.net.dim_out is not self.dim_x):
            raise ValueError("Inconsistent output size of neural network.")

    def dynamics(self, x, u, eval=True):
        if eval:
            self.net.eval()
        self.net = self.net.to(x.device)

        input = torch.hstack((x, u))[None, :]
        return self.net(input)[0, :]

    def dynamics_batch(self, x_batch, u_batch, eval=True):
        if eval:
            self.net.eval()
        self.net = self.net.to(x_batch.device)

        input = torch.hstack((x_batch, u_batch))
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

    def train_network(self, dataset: TensorDataset, params: TrainParams, sigma=0.0):
        """
        Train a network given a dataset and optimization parameters.
        """

        def loss(tensors, placeholder):
            x, u, x_next = tensors
            return self.evaluate_dynamic_loss(
                torch.cat((x, u), dim=-1), x_next, sigma=0.0
            )

        return score_po.nn.train_network(self.net, params, dataset, loss, split=True)

    def save_network_parameters(self, filename):
        file_dir = os.path.dirname(filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
