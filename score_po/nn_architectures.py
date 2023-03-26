import numpy as np
import torch
import torch.nn as nn

"""
List of NN architectures used for the repo.
"""


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

    def forward(self, x):
        return self.mlp(x)


def test():
    net = MLP(3, 5, [128, 128, 128])
    print(net(torch.zeros(3)))
