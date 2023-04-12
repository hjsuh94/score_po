import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Literal
from hydra import initialize, compose
from torch.utils.data import TensorDataset

import score_po.dynamical_system as mut
from score_po.nn import MLP, TrainParams


class TestNNDynamicalSystem:
    def test_constructor(self):
        with np.testing.assert_raises(ValueError):
            # throws since input does not equal dim_x + dim_u.
            network = MLP(4, 2, [128, 128])
            mut.NNDynamicalSystem(4, 2, network)

        with np.testing.assert_raises(ValueError):
            # throws since output does not equal number of states.
            network = MLP(4, 3, [128, 128])
            mut.NNDynamicalSystem(2, 2, network)

        # Doesn't throw since 3 + 2 = 5.
        network = MLP(5, 3, [128, 128])
        sf = mut.NNDynamicalSystem(3, 2, network)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_dynamics_eval(self, device: Literal["cpu", "cuda"]):
        network = MLP(3, 2, [128, 128])
        dynamics = mut.NNDynamicalSystem(2, 1, network)

        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)

        np.testing.assert_equal(
            dynamics.dynamics_batch(x_tensor, u_tensor).shape, torch.Size([100, 2])
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device: Literal["cpu", "cuda"]):
        network = MLP(4, 2, [128, 128])
        dynamics = mut.NNDynamicalSystem(2, 2, network)
        dataset_size = 1000

        x_batch = 2.0 * torch.rand(dataset_size, 2) - 1.0
        u_batch = 2.0 * torch.rand(dataset_size, 2) - 1.0
        xnext_batch = x_batch + u_batch

        dataset = TensorDataset(x_batch, u_batch, xnext_batch)
        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)
        dynamics.train_network(dataset, params)
