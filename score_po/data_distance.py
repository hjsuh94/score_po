import os

import torch
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

from score_po.nn import TrainParams, train_network_sampling, Normalizer


class DataDistance:
    def __init__(self, dataset: TensorDataset, metric: torch.Tensor):
        """
        This class contains functionalities to compute data from some queried
        point x to all the coordinates within the dataset.

        datset: TensorDataset consisted of tensors.
        metric: a metric to enforce on two data, with size dim_x. The distance
        is evaluated based on the metric using the following formula:
        \|x - y\| = \sum_i metric_i * (x_i - y_i)^2.
        """
        self.dataset = dataset
        self.metric = metric
        self.dim_z = self.metric.shape  # torch.size
        self.data_tensor = self.dataset_to_tensor(self.dataset)
        self.data_size = len(dataset)
        self.dim_data = self.data_tensor.shape[1]

    def dataset_to_tensor(self, dataset: TensorDataset):
        """Return tensorized form of data as B x dim_data"""
        return dataset.tensors[0]

    def get_einsum_string(self, length):
        """Get einsum string of specific length."""
        string = "ijklmnopqrstuvwxyz"
        if length > len(string):
            raise ValueError("dimension is larger than supported.")
        return string[:length]

    def get_pairwise_energy(self, z_batch):
        """
        Given x_batch of shape (B, dim_data) and dataset tensor (D, dim_data),
        return a (B, D) array where each element computes:
        0.5 * (z_b - z_d)^T @ metric @ (x_b - x_d)
        """
        # B x D x n
        pairwise_dist = z_batch.unsqueeze(1) - self.data_tensor.unsqueeze(0)
        self.metric = self.metric.to(z_batch.device)
        e_str = self.get_einsum_string(len(self.dim_z))

        summation_string = "bd" + e_str + "," + e_str + "," + "bd" + e_str + "->bd"
        pairwise_quadratic = 0.5 * torch.einsum(
            summation_string, pairwise_dist, self.metric, pairwise_dist
        )
        return pairwise_quadratic

    def get_energy_to_data(self, x_batch):
        """
        Given x_batch, compute softmin distance to data.
        """
        B = x_batch.shape[0]
        pairwise_energy = self.get_pairwise_energy(x_batch)
        return -torch.logsumexp(-pairwise_energy, 1)

    def get_energy_gradients(self, z_batch):
        """
        Given x, compute distance to data.
        """
        z_batch = z_batch.clone()
        z_batch.requires_grad = True

        loss = self.get_energy_to_data(z_batch).sum()
        loss.backward()
        return z_batch.grad


class DataDistanceEstimatorXu(torch.nn.Module):
    def __init__(self, dim_x, dim_u, network, domain_lb, domain_ub):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        self.register_buffer("domain_lb", domain_lb)
        self.register_buffer("domain_ub", domain_ub)
        self.check_input_consistency()

    def check_input_consistency(self):
        if self.net.dim_in is not self.dim_x + self.dim_u:
            raise ValueError("Inconsistent input size of neural network.")
        if self.net.dim_out != 1:
            raise ValueError("Inconsistent output size of neural network.")

    def sample_from_domain(self, batch_size):
        scale = self.domain_ub - self.domain_lb
        return scale * torch.rand(
            batch_size, self.dim_x + self.dim_u) + self.domain_lb
    
    def get_xu_from_z(self, z):
        x = z[:, :self.dim_x]
        u = z[:, self.dim_x:]
        return x, u
    
    def get_z_from_xu(self, x, u):
        return torch.cat((x, u), dim=1)
    
    def get_energy_to_data(self, z_batch):
        return self.net(z_batch)

    def forward(self, z_batch):
        return self.get_energy_to_data(z_batch)

    def get_energy_gradients(self, z_batch):
        """
        Given x, compute distance to data.
        """
        z_batch = z_batch.clone()
        z_batch.requires_grad = True

        loss = self.get_energy_to_data(z_batch).sum()
        loss.backward()
        return z_batch.grad

    def evaluate_loss(self, z_batch, dst: DataDistance):
        target = dst.get_energy_to_data(z_batch)
        values = self.get_energy_to_data(z_batch)

        target = target.view(target.shape[0], -1)
        values = values.view(target.shape[0], -1)

        loss = torch.abs(target - values).mean()
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        metric: torch.Tensor,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """                
        # Retain memory of the metric.
        self.register_buffer("metric", metric)
        cat_dataset = TensorDataset(torch.cat(dataset.tensors[0:2], dim=1))
        dst = DataDistance(cat_dataset, self.metric)
        loss_fn = lambda z_batch, net: self.evaluate_loss(
            z_batch, dst)
        loss_lst = train_network_sampling(
            self, params, self.sample_from_domain, loss_fn)
        return loss_lst


class DataDistanceEstimatorXux(torch.nn.Module):
    def __init__(self, dim_x, dim_u, network, domain_lb, domain_ub):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        self.register_buffer("domain_lb", domain_lb)
        self.register_buffer("domain_ub", domain_ub)
        self.check_input_consistency()

    def check_input_consistency(self):
        if self.net.dim_in is not self.dim_x + self.dim_u + self.dim_x:
            raise ValueError("Inconsistent input size of neural network.")
        if self.net.dim_out != 1:
            raise ValueError("Inconsistent output size of neural network.")

    def sample_from_domain(self, batch_size):
        scale = self.domain_ub - self.domain_lb
        return scale * torch.rand(batch_size, 
                                  self.dim_x + self.dim_u + self.dim_x) + self.domain_lb
    
    def get_xux_from_z(self, z):
        x = z[:, :self.dim_x]
        u = z[:, self.dim_x:self.dim_x + self.dim_u]
        xnext = z[:, self.dim_x+self.dim_u:]
        return x, u, xnext
    
    def get_z_from_xux(self, x, u, xnext):
        return torch.cat((x, u, xnext), dim=1)

    def get_energy_to_data(self, z_batch):
        return self.net(z_batch)

    def forward(self, z_batch):
        return self.get_energy_to_data(z_batch)

    def get_energy_gradients(self, z_batch):
        """
        Given x, compute distance to data.
        """
        z_batch = z_batch.clone()
        z_batch.requires_grad = True

        loss = self.get_energy_to_data(z_batch).sum()
        loss.backward()
        return z_batch.grad

    def evaluate_loss(self, z_batch, dst: DataDistance):
        target = dst.get_energy_to_data(z_batch)
        values = self.get_energy_to_data(z_batch)

        target = target.view(target.shape[0], -1)
        values = values.view(target.shape[0], -1)

        loss = torch.abs(target - values).mean()
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        metric: torch.Tensor,
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        # Retain memory of the metric.
        self.register_buffer("metric", metric)
        # assume dataset is x_batch, u_batch, xnext_batch
        cat_dataset = TensorDataset(torch.cat(dataset.tensors, dim=1))
        dst = DataDistance(cat_dataset, self.metric)
        loss_fn = lambda z_batch, net: self.evaluate_loss(
            z_batch, dst)
        loss_lst = train_network_sampling(
            self, params, self.sample_from_domain, loss_fn)
        return loss_lst        
