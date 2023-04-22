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
        dynamics = mut.NNDynamicalSystem(
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

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_evaluate_dynamic_loss(self, device):
        network = MLP(4, 2, [32, 16])
        x_normalizer = Normalizer(k=torch.tensor([2, 3.0]), b=torch.tensor([0, 1.0]))
        u_normalizer = Normalizer(k=torch.tensor([3, 4.0]), b=torch.tensor([1, 2.0]))
        dut = mut.NNDynamicalSystem(2, 2, network, x_normalizer, u_normalizer).to(
            device
        )

        batch = 1000
        xu = torch.randn((batch, 4)).to(device) * 3 + 1
        x_next = torch.randn(batch, 2).to(device) * 2
        loss_unnormalized = dut.evaluate_dynamic_loss(
            xu, x_next, sigma=0, normalize_loss=False
        )
        loss_unnormalized_expected = 0.5 * (
            (dut(xu[:, :2], xu[:, 2:], eval=True) - x_next) ** 2
        ).sum(dim=-1).mean(dim=0)
        np.testing.assert_allclose(
            loss_unnormalized.cpu().detach(), loss_unnormalized_expected.cpu().detach()
        )

        loss_normalized = dut.evaluate_dynamic_loss(
            xu, x_next, sigma=0, normalize_loss=True
        )
        loss_normalized_expected = 0.5 * (
            (x_normalizer(dut(xu[:, :2], xu[:, 2:], eval=True)) - x_normalizer(x_next))
            ** 2
        ).sum(dim=-1).mean(dim=0)
        np.testing.assert_allclose(
            loss_unnormalized.cpu().detach(), loss_unnormalized_expected.cpu().detach()
        )
