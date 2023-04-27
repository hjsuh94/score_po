import abc
from enum import Enum
import os
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class Clamper(nn.Module):
    """
    Clamp an input within the bound, if the bound is not None.

    Depending on the method, we can either do
      a hard clamp out = clamp(in, lower, upper)
      a soft clamp using tanh function.
    """

    class Method(Enum):
        HARD = (1,)
        TANH = (2,)

    def __init__(
        self,
        lower: Optional[torch.Tensor],
        upper: Optional[torch.Tensor],
        method: Method = Method.HARD,
    ):
        super().__init__()
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)
        self.method = method
        if self.lower is not None and self.upper is not None:
            assert self.lower.shape == self.upper.shape
            assert torch.all(self.lower <= self.upper)

    def forward(self, x):
        if self.lower is None and self.upper is None:
            return x
        if self.method == Clamper.Method.HARD:
            return torch.clamp(x, min=self.lower, max=self.upper)
        elif self.method == Clamper.Method.TANH:
            assert self.lower is not None
            assert self.upper is not None
            return (
                torch.tanh(x) * (self.upper - self.lower) / 2
                + (self.upper + self.lower) / 2
            )


class Policy(nn.Module):
    def __init__(self, dim_x, dim_u, u_clip: Optional[Clamper] = None):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.dim_params = 0
        self.params = 0
        # flag for whether the policy is time-varying, used for 
        # whether to batch-compute or not.
        self.tv = True 
        self.u_clip: Clamper = (
            Clamper(lower=None, upper=None, method=Clamper.Method.HARD)
            if u_clip is None
            else u_clip
        )

    def _unclipped_u(self, x_batch, t):
        """
        Compute u before clipping it within the bound.
        """
        raise NotImplementedError

    def forward(self, x_batch, t):
        return self.u_clip(self._unclipped_u(x_batch, t))


class TimeVaryingOpenLoopPolicy(Policy):
    """
    Implement policy of the form
        u_t = clamp(pi_t).
    """

    def __init__(self, dim_x, dim_u, T: int, u_clip: Optional[Clamper] = None):
        super().__init__(dim_x, dim_u, u_clip)
        self.T = T
        self.dim_params = self.dim_u * self.T
        self.params = nn.Parameter(torch.zeros((T, self.dim_u)))
        self.tv = True        

    def _unclipped_u(self, x_batch, t):
        return self.params[t, :].repeat([x_batch.shape[0], 1]).to(x_batch.device)


class TimeVaryingStateFeedbackPolicy(Policy):
    """
    Implement policy of the form
        u_t = K_t * x_t + mu_t.
    """

    def __init__(self, dim_x, dim_u, T, u_clip: Optional[Clamper] = None):
        super().__init__(dim_x, dim_u, u_clip)
        self.T = T
        self.dim_params = self.dim_u * (self.dim_x + 1) * self.T
        self.params = nn.ParameterDict(
            {
                "gain": nn.Parameter(torch.zeros(T, self.dim_u, self.dim_x)),
                "bias": nn.Parameter(torch.zeros(T, self.dim_u)),
            }
        )
        self.tv = True

    def _unclipped_u(self, x_batch, t):
        self.params = self.params.to(x_batch.device)
        gain = self.params["gain"][t, :, :]
        bias = self.params["bias"][t, :]
        return torch.einsum("ij,bj->bi", gain, x_batch) + bias


class NNPolicy(Policy):
    """
    Implement policy of the form
        u_t = pi(x_t, theta) where pi is NN parametrized by theta.
    """

    def __init__(self, dim_x, dim_u, network, u_clip: Optional[Clamper] = None):
        super().__init__(dim_x, dim_u, u_clip)
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.net = network
        assert isinstance(self.net, nn.Module)
        self.dim_params = len(self.net.get_vectorized_parameters())
        self.tv = False

    def _unclipped_u(self, x_batch, t):
        self.net = self.net.to(x_batch.device)
        return self.net(x_batch)
