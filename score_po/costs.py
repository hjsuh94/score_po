from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


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
        raise ValueError("virtual function.")

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
        raise ValueError("virtual function.")

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
        self.params_to_device(x.device)
        xQx = torch.einsum("i,ij,j", x - self.xd, self.Q, x - self.xd)
        uRu = torch.einsum("i,ij,j", u, self.R, u)
        return xQx + uRu

    def get_running_cost_batch(self, x_batch, u_batch):
        xQx = torch.einsum("bi,ij,bj->b", x_batch - self.xd, self.Q, x_batch - self.xd)
        uRu = torch.einsum("bi,ij,bj->b", u_batch, self.R, u_batch)
        return xQx + uRu

    def get_terminal_cost(self, x):
        return torch.einsum("i,ij,j", x, self.Qd, x)

    def get_terminal_cost_batch(self, x_batch):
        return torch.einsum(
            "bi,ij,bj->b", x_batch - self.xd, self.Qd, x_batch - self.xd
        )

    def load_from_config(self, cfg: DictConfig):
        self.Q = torch.diag(torch.Tensor(cfg.cost.Q))
        self.R = torch.diag(torch.Tensor(cfg.cost.R))
        self.Qd = torch.diag(torch.Tensor(cfg.cost.Qd))
        self.xd = torch.Tensor(cfg.cost.xd)
