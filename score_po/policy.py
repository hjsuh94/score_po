import abc
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, dim_x, dim_u):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_params = 0
        self.params = 0

    def forward(self, x_batch, t):
        raise ValueError("Virtual class.")

    def save_parameters(self, filename):
        file_dir = os.path.dirname(filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exists_ok=True)
        torch.save(self.state_dict(), filename)


class TimeVaryingOpenLoopPolicy(Policy):
    """
    Implement policy of the form
        u_t = pi_t.
    """

    def __init__(self, dim_x, dim_u, T):
        super().__init__(dim_x, dim_u)
        self.T = T
        self.dim_params = self.dim_u * self.T
        self.params = nn.Parameter(torch.zeros((T, self.dim_u)))

    def forward(self, x_batch, t):
        return self.params[t, :][None, :].to(x_batch.device)


class TimeVaryingStateFeedbackPolicy(Policy):
    """
    Implement policy of the form
        u_t = K_t * x_t + mu_t.
    """

    def __init__(self, dim_x, dim_u, T):
        super().__init__(dim_x, dim_u)
        self.T = T
        self.dim_params = self.dim_u * (self.dim_x + 1) * self.T
        self.params = nn.ParameterDict(
            {
                "gain": nn.Parameter(torch.zeros(T, self.dim_u, self.dim_x)),
                "bias": nn.Parameter(torch.zeros(T, self.dim_u)),
            }
        )

    def forward(self, x_batch, t):
        self.params = self.params.to(x_batch.device)
        gain = self.params["gain"][t, :, :]
        bias = self.params["bias"][t, :]
        return torch.einsum("ij,bj->bi", gain, x_batch) + bias


class NNPolicy(Policy):
    """
    Implement policy of the form
        u_t = pi(x_t, theta) where pi is NN parametrized by theta.
    """

    def __init__(self, dim_x, dim_u, network):
        super().__init__(dim_x, dim_u)
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        assert(isinstance(self.net, nn.Module))
        self.dim_params = len(self.net.get_vectorized_parameters())

    def forward(self, x_batch, t):
        self.net = self.net.to(x_batch.device)
        return self.net(x_batch)
