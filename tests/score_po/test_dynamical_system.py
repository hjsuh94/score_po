import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Literal
from hydra import initialize, compose
from torch.utils.data import TensorDataset

import score_po.dynamical_system as mut
from score_po.nn import MLP, TrainParams, Normalizer


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
        sf = mut.NNDynamicalSystem(
            3,
            2,
            network,
            x_normalizer=Normalizer(k=torch.tensor(2.0), b=torch.tensor(1.0)),
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_dynamics_eval(self, device: Literal["cpu", "cuda"]):
        network = MLP(3, 2, [128, 128])
        x_normalizer = Normalizer(
            k=torch.tensor([1.0, 2.0]), b=torch.tensor([0.1, 0.2])
        )
        u_normalizer = Normalizer(k=torch.tensor(3.0), b=torch.tensor(0.3))
        dynamics = mut.NNDynamicalSystem(2, 1, network, x_normalizer, u_normalizer).to(
            device
        )

        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)

        np.testing.assert_equal(
            dynamics.dynamics_batch(x_tensor, u_tensor).shape, torch.Size([100, 2])
        )

        x_normalized = x_normalizer(x_tensor)
        u_normalized = u_normalizer(u_tensor)
        normalized_input = torch.cat((x_normalized, u_normalized), dim=-1)
        output_expected = x_normalizer.denormalize(network(normalized_input))
        np.testing.assert_allclose(
            output_expected.cpu().detach().numpy(),
            dynamics.dynamics_batch(x_tensor, u_tensor).cpu().detach().numpy(),
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device: Literal["cpu", "cuda"]):
        torch.manual_seed(10)
        network = MLP(4, 2, [32, 16])
        dynamics = mut.NNDynamicalSystem(2, 2, network).to(device)
        dataset_size = 10000

        x_batch = 2.0 * torch.rand(dataset_size, 2, device=device) - 1.0
        u_batch = 2.0 * torch.rand(dataset_size, 2, device=device) - 1.0
        xnext_batch = x_batch + u_batch

        dataset = TensorDataset(x_batch, u_batch, xnext_batch)
        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)
            params.device = device
        loss_lst = dynamics.train_network(dataset, params)
        # We train many epochs, such that the best loss is obtained in the middle of the training, not at the end.
        assert np.min(loss_lst) < loss_lst[-1]

        # We reload the saved model. Since we save the best
        network_reload = MLP(4, 2, [32, 16])
        dynamics_reload = mut.NNDynamicalSystem(2, 2, network_reload).to(device)
        dynamics_reload.load_state_dict(torch.load(params.save_best_model))
        assert torch.any(network_reload.mlp[0].weight.data != network.mlp[0].weight.data)
