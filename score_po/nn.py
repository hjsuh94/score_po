import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
List of architectures and parameters for NN training.
"""
class AdamOptimizerParams:
    def __init__(self):
        self.lr = 1e-3
        self.iters = 1000
        self.batch_size = 512

class MLP(nn.Module):
    """
    Vanilla MLP with ReLU nonlinearity.
    hidden_layers takes a list of hidden layers.
    For example,

    MLP(3, 5, [128, 128])

    makes MLP with two hidden layers with 128 width.
    """

    def __init__(self, dim_in, dim_out, hidden_layers):
        super().__init__()

        layers = []
        layers.append(nn.Linear(dim_in, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], dim_out))

        self.mlp = nn.Sequential(*layers)

    def get_vectorized_parameters(self):
        """
        Get a vectorized representation of the parameters.
        """
        params_vec = torch.zeros(0)
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) == nn.Linear:
                params_vec = torch.hstack((params_vec, torch.ravel(self.mlp[i].weight)))
                params_vec = torch.hstack((params_vec, self.mlp[i].bias))
        return params_vec

    def set_vectorized_parameters(self, params_vector):
        """
        Get a vectorized representation of the parameters.
        """
        idx = 0
        for i in range(len(self.mlp)):
            layer = self.mlp[i]
            if type(layer) == nn.Linear:
                dim_in = layer.in_features
                dim_out = layer.out_features

                # Recover linear and bias parameters.
                linear_params = params_vector[idx : idx + dim_in * dim_out]
                idx += dim_in * dim_out
                bias_params = params_vector[idx : idx + dim_out]
                idx += dim_out

                # Set the NN parameters.
                layer.weight = nn.parameter.Parameter(
                    linear_params.view(dim_out, dim_in)
                )
                layer.bias = nn.parameter.Parameter(bias_params)

    def get_vectorized_gradients(self):
        """
        If the layers have registered gradients, extract them out to a vector
        representation.
        """
        params_vec = torch.zeros(0)
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) == nn.Linear:
                params_vec = torch.hstack(
                    (params_vec, torch.ravel(self.mlp[i].weight.grad))
                )
                params_vec = torch.hstack((params_vec, self.mlp[i].bias.grad))
        return params_vec

    def forward(self, x):
        return self.mlp(x)
