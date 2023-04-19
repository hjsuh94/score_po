import os

import torch
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
import wandb

from score_po.nn import TrainParams, train_network_sampling


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
        self.dim_x = self.metric.shape  # torch.size
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

    def get_pairwise_energy(self, x_batch):
        """
        Given x_batch of shape (B, dim_data) and dataset tensor (D, dim_data),
        return a (B, D) array where each element computes:
        0.5 * (x_b - x_d)^T @ metric @ (x_b - x_d)
        """
        # B x D x n
        pairwise_dist = x_batch.unsqueeze(1) - self.data_tensor.unsqueeze(0)
        self.metric = self.metric.to(x_batch.device)
        e_str = self.get_einsum_string(len(self.dim_x))

        summation_string = "bd" + e_str + "," + e_str + "," + "bd" + e_str + "->bd"
        pairwise_quadratic = 0.5 * torch.einsum(
            summation_string, pairwise_dist, self.metric, pairwise_dist
        )
        return pairwise_quadratic

    def get_energy_to_data(self, x_batch, mode="softmin"):
        """
        Given x_batch, compute softmin distance to data.
        """
        B = x_batch.shape[0]
        pairwise_energy = self.get_pairwise_energy(x_batch)
        if mode == "softmin":
            return -torch.logsumexp(-pairwise_energy, 1)
        elif mode == "min":
            truemin, _ = torch.min(pairwise_energy, dim=1)
            return truemin
        else:
            raise ValueError(
                "Unsupported mode. get_energy_to_data supports softmin or min."
            )

    def get_energy_gradients(self, x_batch):
        """
        Given x, compute distance to data.
        """
        x_batch = x_batch.clone()
        x_batch.requires_grad = True

        loss = self.get_energy_to_data(x_batch).sum()
        loss.backward()
        return x_batch.grad


class DataDistanceEstimator(torch.nn.Module):
    def __init__(self, dim_x, network, metric, domain_lb, domain_ub):
        super().__init__()
        self.dim_x = dim_x
        self.net = network
        self.register_buffer("metric", metric)
        self.register_buffer("domain_lb", domain_lb)
        self.register_buffer("domain_ub", domain_ub)
        self.check_input_consistency()

    def check_input_consistency(self):
        if self.net.dim_in is not self.dim_x:
            raise ValueError("Inconsistent input size of neural network.")
        if self.net.dim_out != 1:
            raise ValueError("Inconsistent output size of neural network.")

    def sample_from_domain(self, batch_size):
        scale = self.domain_ub - self.domain_lb
        return scale * torch.rand(batch_size, self.dim_x) + self.domain_lb

    def get_energy_to_data(self, x_batch, eval=True):
        self.net.to(x_batch.device)
        if eval:
            self.net.eval()
        else:
            self.net.train()

        return self.net(x_batch)

    def forward(self, x_batch, eval=True):
        return self.get_energy_to_data(x_batch, eval)

    def get_energy_gradients(self, x_batch):
        """
        Given x, compute distance to data.
        """
        x_batch = x_batch.clone()
        x_batch.requires_grad = True

        loss = self.get_energy_to_data(x_batch).sum()
        loss.backward()
        return x_batch.grad

    def evaluate_loss(self, x_batch, dst: DataDistance, mode):
        target = dst.get_energy_to_data(x_batch, mode=mode)
        values = self.get_energy_to_data(x_batch)

        target = target.view(target.shape[0], -1)
        values = values.view(target.shape[0], -1)

        loss = torch.abs(target - values).mean()
        return loss

    def train_network(
        self,
        dataset: TensorDataset,
        params: TrainParams,
        mode="softmin",
    ):
        """
        Train a network given a dataset and optimization parameters.
        """
        dst = DataDistance(dataset, self.metric)
        loss_fn = lambda x_batch, net: self.evaluate_loss(x_batch, dst, mode)
        loss_lst = train_network_sampling(
            self.net, params, self.sample_from_domain, loss_fn, split=False
        )
        return loss_lst
