from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset
from omegaconf import DictConfig, OmegaConf

from score_po.nn import Normalizer, TrainParams
import score_po.nn


class Cost(torch.nn.Module):
    """
    Cost class.
    """

    def __init__(self):
        super().__init__()

    def get_running_cost(self, x, u):
        """
        Get running cost, c(x_t, u_t) that will be summed over horizon.
        input:
            x of shape (n,)
            u of shape (m,)
        output:
            cost of scalar.
        """
        return self.get_running_cost_batch(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)

    def get_running_cost_batch(self, x_batch, u_batch):
        """
        input:
            x_batch of shape (B, n)
            u_batch of shape (B, m)
        output:
            cost of shape (B,)
        """
        raise ValueError("virtual function.")

    def get_terminal_cost(self, x):
        """
        Get terminal cost c(x_T) that will be evaluated on the final step.
        input:
            x of shape (n,)
        output:
            cost of shape scalar.
        """
        return self.get_terminal_cost_batch(x.unsqueeze(0)).squeeze(0)

    def get_terminal_cost_batch(self, x_batch):
        """
        input:
            x of shape (B, n)
        output:
            cost of shape scalar (B,)
        """
        raise ValueError("virtual function.")

    def forward(self, terminal: bool, *input_tensors):
        if terminal:
            return self.get_terminal_cost_batch(*input_tensors)
        else:
            return self.get_running_cost_batch(*input_tensors)


class QuadraticCost(Cost):
    def __init__(
        self,
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        Qd: torch.Tensor = None,
        xd: torch.Tensor = None,
    ):
        super().__init__()
        self.register_buffer("Q", Q)  # quadratic state weight
        self.register_buffer("R", R)  # quadratic input weight
        self.register_buffer("Qd", Qd)  # quadratic terminal weight
        self.register_buffer("xd", xd)  # desired state

    def get_running_cost(self, x, u):
        xQx = torch.einsum("i,ij,j", x - self.xd, self.Q, x - self.xd)
        uRu = torch.einsum("i,ij,j", u, self.R, u)
        return xQx + uRu

    def get_running_cost_batch(self, x_batch, u_batch):
        xQx = torch.einsum("bi,ij,bj->b", x_batch - self.xd, self.Q, x_batch - self.xd)
        uRu = torch.einsum("bi,ij,bj->b", u_batch, self.R, u_batch)
        return xQx + uRu

    def get_terminal_cost(self, x):
        return torch.einsum("i,ij,j", x - self.xd, self.Qd, x - self.xd)

    def get_terminal_cost_batch(self, x_batch):
        return torch.einsum(
            "bi,ij,bj->b", x_batch - self.xd, self.Qd, x_batch - self.xd
        )

    def load_from_config(self, cfg: DictConfig):
        self.Q = torch.diag(torch.Tensor(cfg.cost.Q))
        self.R = torch.diag(torch.Tensor(cfg.cost.R))
        self.Qd = torch.diag(torch.Tensor(cfg.cost.Qd))
        self.xd = torch.Tensor(cfg.cost.xd)


class NNCost(Cost):
    def __init__(
        self,
        dim_x,
        dim_u,
        network,
        x_normalizer: Optional[Normalizer] = None,
        u_normalizer: Optional[Normalizer] = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        self.is_differentiable = True
        self.check_input_consistency()
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

    def check_input_consistency(self):
        if hasattr(self.net, "dim_in") and (
            self.net.dim_in is not self.dim_x + self.dim_u
        ):
            raise ValueError("Inconsistent input size of neural network.")
        if hasattr(self.net, "dim_out") and (self.net.dim_out != 1):
            raise ValueError("Inconsistent output size of neural network.")

    def get_running_cost(self, x, u):
        return self.get_running_cost_batch(x.unsqueeze(0), u.unsqueeze(0)).squeeze(0)

    def get_running_cost_batch(self, x_batch, u_batch):
        x_normalized = self.x_normalizer(x_batch)
        u_normalized = self.u_normalizer(u_batch)
        normalized_input = torch.hstack((x_normalized, u_normalized))
        return self.net(normalized_input).squeeze(1)

    def get_terminal_cost(self, x):
        """We assume the learned cost does not have any terminal costs."""
        return torch.zeros(1).to(x.device)

    def get_terminal_cost_batch(self, x_batch):
        """We assume the learned cost does not have any terminal costs."""
        return torch.zeros((x_batch.shape[0])).to(x_batch.device)

    def train_network(
        self, dataset: TensorDataset, params: TrainParams, sigma: float = 0.0
    ):
        """
        Train a network given a dataset and optimization parameters.
        """

        def loss(tensors, placeholder):
            x, u, c_true = tensors
            cost_pred = self.get_running_cost_batch(x, u)
            return 0.5 * ((cost_pred - c_true) ** 2).mean(dim=0)

        return score_po.nn.train_network(self, params, dataset, loss, split=True)
