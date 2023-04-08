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
        """
        self.dataset = dataset
        self.metric = metric
        self.metric_mat = torch.diag(metric)
        self.data_tensor = self.dataset_to_tensor(self.dataset)
        self.data_size = len(dataset)
        self.dim_data = self.data_tensor.shape[1]

    def dataset_to_tensor(self, dataset: TensorDataset):
        """Return tensorized form of data as B x dim_data"""
        return dataset.tensors[0]

    def get_pairwise_energy(self, x_batch):
        """
        Given x_batch of shape (B, dim_data) and dataset tensor (D, dim_data),
        return a (B, D) array where each element computes:
        0.5 * (x_b - x_d)^T @ metric @ (x_b - x_d)
        """
        # B x D x n
        pairwise_dist = x_batch[:, None, :] - self.data_tensor[None, :, :]

        pairwise_quadratic = 0.5 * torch.einsum(
            "bdi,ii,bdi->bd", pairwise_dist, self.metric_mat, pairwise_dist
        )
        return pairwise_quadratic

    def get_energy_to_data(self, x_batch, mode="softmin"):
        """
        Given x_batch, compute softmin distance to data.
        """
        B = x_batch.shape[0]
        device = x_batch.device
        self.metric_mat = self.metric_mat.to(device)
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


class DataDistanceEstimator:
    def __init__(self, dim_x, network, metric, domain_lb, domain_ub):
        self.dim_x = dim_x
        self.net = network
        self.metric = metric
        self.domain_lb = domain_lb
        self.domain_ub = domain_ub
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

    def save_network_parameters(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.net.load_state_dict(torch.load(filename))
