import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

class Cost:
    """
    Cost class.
    """

    def __init__(self):
        pass

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


class QuadraticCost(Cost):
    def __init__(self, Q=None, R=None, Qd=None, xd=None):
        super().__init__()
        self.Q = Q  # quadratic state weight
        self.R = R  # quadratic input weight
        self.Qd = Qd  # quadratic terminal weight
        self.xd = xd  # desired state
        
    def params_to_device(self, device):
        self.Q = self.Q.to(device)
        self.R = self.R.to(device)
        self.Qd = self.Qd.to(device)
        self.xd = self.xd.to(device)

    def get_running_cost(self, x, u):
        self.params_to_device(x.device)
        xQx = torch.einsum("i,ij,j", x - self.xd, self.Q, x - self.xd)
        uRu = torch.einsum("i,ij,j", u, self.R, u)
        return xQx + uRu

    def get_running_cost_batch(self, x_batch, u_batch):
        self.params_to_device(x_batch.device)
        xQx = torch.einsum("bi,ij,bj->b", x_batch - self.xd, self.Q, x_batch - self.xd)
        uRu = torch.einsum("bi,ij,bj->b", u_batch, self.R, u_batch)
        return xQx + uRu

    def get_terminal_cost(self, x):
        self.params_to_device(x.device)
        return torch.einsum("i,ij,j", x, self.Qd, x)

    def get_terminal_cost_batch(self, x_batch):
        self.params_to_device(x_batch.device)
        return torch.einsum(
            "bi,ij,bj->b", x_batch - self.xd, self.Qd, x_batch - self.xd
        )
        
    def load_from_config(self, cfg: DictConfig):
        self.Q = torch.diag(torch.Tensor(cfg.cost.Q))
        self.R = torch.diag(torch.Tensor(cfg.cost.R))
        self.Qd = torch.diag(torch.Tensor(cfg.cost.Qd))
        self.xd = torch.Tensor(cfg.cost.xd)
