import abc
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from score_po.policy import Policy
from score_po.nn import AdamOptimizerParams
from score_po.dataset import Dataset, DynamicsDataset

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

    def evaluate_dynamic_loss(self, data, labels, sigma):
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

    def train_network(
        self, dataset: DynamicsDataset, params: AdamOptimizerParams, sigma=0.0
    ):
        """
        Train a network given a dataset and optimization parameters.
        Provides a
        """
        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), params.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, params.iters)

        loss_lst = torch.zeros(params.iters)

        for iter in tqdm(range(params.iters)):
            optimizer.zero_grad()
            (data, label) = dataset.draw_from_dataset(params.batch_size)
            loss = self.evaluate_dynamic_loss(data, label, sigma)
            loss_lst[iter] = torch.clone(loss)[0].detach()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return loss_lst

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
