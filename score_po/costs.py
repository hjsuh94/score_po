from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset
from omegaconf import DictConfig, OmegaConf

from score_po.nn import Normalizer, TrainParams
import score_po.nn
import torch.nn as nn


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

class QuadraticCostImage(Cost):
    """
    Cost class for images. Does an extra preprocessing step to convert the predicted images to weighted average.
    """
    def __init__(
        self,
        Q: torch.Tensor = None,
        R: torch.Tensor = None,
        Qd: torch.Tensor = None,
        xd: torch.Tensor = None,
        nx = 32
    ):
        super().__init__()
        self.register_buffer("Q", Q)  # quadratic state weight
        self.register_buffer("R", R)  # quadratic input weight
        self.register_buffer("Qd", Qd)  # quadratic terminal weight
        self.register_buffer("xd", xd)  # desired state
        # self.y_mesh, self.x_mesh = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32))
        self.y_mesh, self.x_mesh = torch.meshgrid(torch.linspace(-1.2, 1.2, 32), torch.linspace(-1.2, 1.2, 32))
        # self.x_mesh = self.x_mesh.T
        self.y_mesh_u, self.x_mesh_u = torch.meshgrid(torch.linspace(-0.3, 0.3, 32), torch.linspace(-0.3, 0.3, 32))
        # self.x_mesh_u = self.x_mesh_u.T

    def get_running_cost(self, x, u):
        x_norm = x / x.sum()
        pos_x = (x_norm * self.x_mesh.to(x_norm.device)).sum()
        pos_y = (x_norm * self.y_mesh.to(x_norm.device)).sum()
        x = torch.hstack([pos_x, pos_y])
        xQx = torch.einsum("i,ij,j", x - self.xd, self.Q, x - self.xd)

        u_norm = u / u.sum()
        pos_x = (u_norm * self.x_mesh_u.to(u_norm.device)).sum()
        pos_y = (u_norm * self.y_mesh_u.to(u_norm.device)).sum()
        u = torch.hstack([pos_x, pos_y])
        uRu = torch.einsum("i,ij,j", u, self.R, u)
        return xQx + uRu

    def get_running_cost_batch(self, x_batch, u_batch, eps=1e-6):
        m = nn.Threshold(0.1, 0.)
        x_batch = m(x_batch)
        u_batch = m(u_batch)
        x_norm = x_batch.clamp(min=0.0, max=1.) / (x_batch.clamp(min=0.0, max=1.) + eps).sum(dim=(1,2))[:,None,None].repeat(1, x_batch.shape[-2], x_batch.shape[-1])
        pos_x = (x_norm * self.x_mesh.to(x_norm.device)).sum(dim=(1,2))
        pos_y = (x_norm * self.y_mesh.to(x_norm.device)).sum(dim=(1,2))
        # N = x_batch.shape[0]
        # temp_x = (x_batch.view(N, -1) == x_batch.view(N, -1).max(dim=1, keepdim=True)[0]).view_as(
        #     x_batch) * self.x_mesh.to(x_batch.device)
        # temp_y = (x_batch.view(N, -1) == x_batch.view(N, -1).max(dim=1, keepdim=True)[0]).view_as(
        #     x_batch) * self.y_mesh.to(x_batch.device)
        # pos_x = temp_x.sum(dim=[1,2])
        # pos_y = temp_y.sum(dim=[1, 2])
        x_batch = torch.hstack([pos_x[:,None], pos_y[:,None]])
        xQx = torch.einsum("bi,ij,bj->b", x_batch - self.xd, self.Q, x_batch - self.xd)
        xQx_pathlen = torch.einsum("bi,ij,bj->b", x_batch[1:] - x_batch[:-1], self.Q, x_batch[1:] - x_batch[:-1])
        xQx_pathlen = torch.hstack([xQx_pathlen, torch.tensor(0.).to(xQx_pathlen.device)])

        u_batch = u_batch.reshape(u_batch.shape[0], int(np.sqrt(u_batch.shape[1])), int(np.sqrt(u_batch.shape[1])))
        u_norm = u_batch.clamp(min=0.0, max=1.) / (u_batch.clamp(min=0.0, max=1.) + eps).sum(dim=(1, 2))[:, None, None].repeat(1, u_batch.shape[-2], u_batch.shape[-1])
        pos_x = (u_norm * self.x_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        pos_y = (u_norm * self.y_mesh_u.to(u_norm.device)).sum(dim=(1, 2))
        u_batch = torch.hstack([pos_x[:, None], pos_y[:, None]])
        uRu = torch.einsum("bi,ij,bj->b", u_batch, self.R, u_batch)
        uRu_pathlen = torch.einsum("bi,ij,bj->b", u_batch[1:] - u_batch[:-1], self.R, u_batch[1:] - u_batch[:-1])
        uRu_pathlen = torch.hstack([uRu_pathlen, torch.tensor(0.).to(uRu_pathlen.device)])

        return xQx_pathlen + uRu_pathlen# +xQx + uRu #

    def get_terminal_cost(self, x, eps=1e-6):
        m = nn.Threshold(0.1, 0.)
        x = m(x)
        x_norm = x.clamp(min=0.,max=1.) / (x.clamp(min=0.,max=1.) + eps).sum()
        pos_x = (x_norm * self.x_mesh.to(x_norm.device)).sum()
        pos_y = (x_norm * self.y_mesh.to(x_norm.device)).sum()

        # temp_x = (x.view(1, -1) == x.view(1, -1).max(dim=1, keepdim=True)[0]).view_as(
        #     x) * self.x_mesh.to(x.device)
        # temp_y = (x.view(1, -1) == x.view(1, -1).max(dim=1, keepdim=True)[0]).view_as(
        #     x) * self.y_mesh.to(x.device)
        # pos_x = temp_x.sum()
        # pos_y = temp_y.sum()
        x = torch.hstack([pos_x, pos_y])
        return torch.einsum("i,ij,j", x - self.xfinal.to(x.device), self.Qd, x - self.xfinal.to(x.device))

    def get_terminal_cost_batch(self, x_batch):
        # x_norm = x_batch.clamp(min=0.05, max=1.) / x_batch.clamp(min=0.05, max=1.).sum(dim=(1,2))[:,None,None].repeat(1, x_batch.shape[-2], x_batch.shape[-1])
        # pos_x = (x_norm * self.x_mesh.to(x_norm.device)).sum(dim=(1, 2))
        # pos_y = (x_norm * self.y_mesh.to(x_norm.device)).sum(dim=(1, 2))
        N = x_batch.shape[0]
        temp_x = (x_batch.view(N, -1) == x_batch.view(N, -1).max(dim=1, keepdim=True)[0]).view_as(
            x_batch) * self.x_mesh.to(x_batch.device)
        temp_y = (x_batch.view(N, -1) == x_batch.view(N, -1).max(dim=1, keepdim=True)[0]).view_as(
            x_batch) * self.y_mesh.to(x_batch.device)
        pos_x = temp_x.sum(dim=[1, 2])
        pos_y = temp_y.sum(dim=[1, 2])

        x_batch = torch.hstack([pos_x[:,None], pos_y[:,None]])
        return torch.einsum(
            "bi,ij,bj->b", x_batch - self.xd, self.Qd, x_batch - self.xd
        )

    def load_from_config(self, cfg: DictConfig, xd=None):
        self.Q = torch.diag(torch.Tensor(cfg.cost.Q))
        self.R = torch.diag(torch.Tensor(cfg.cost.R))
        self.Qd = torch.diag(torch.Tensor(cfg.cost.Qd))
        if xd is None:
            self.xd = torch.Tensor(cfg.cost.xd)
        else:
            self.xd = xd
        self.xfinal = torch.Tensor(cfg.trj.xT)
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
