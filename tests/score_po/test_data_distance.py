import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from typing import Literal
from hydra import initialize, compose
from scipy.special import logsumexp

import score_po.data_distance as mut
from score_po.nn import MLP, TrainParams


class TestDataDistance:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_pairwise_energy(self, device):
        # Generate some arbitary data.
        data = torch.rand(5, 3).to(device)
        sigma = 0.2
        x_batch = torch.rand(20, 3).to(device)

        # Manually compute quadratic energy.
        quadratic = torch.zeros((20, 5)).to(device)
        for i, x in enumerate(x_batch):
            for j, data_pt in enumerate(data):
                quadratic[i, j] = 0.5 * torch.linalg.norm((x - data_pt) / sigma) ** 2
        quadratic = quadratic.detach().cpu().numpy()

        # Compute quadratic by mut.
        dataset = TensorDataset(data)
        metric = torch.ones(3).to(device) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        quadratic_mut = dst.get_pairwise_energy(x_batch).detach().cpu().numpy()

        np.testing.assert_allclose(quadratic, quadratic_mut, rtol=1e-3)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_multidimensional_pairwise_energy(self, device):
        data = torch.rand(5, 3, 2, 5).to(device)
        sigma = 0.2
        x_batch = torch.rand(3, 3, 2, 5).to(device)

        # Manually compute quadratic energy.
        quadratic = torch.zeros((3, 5)).to(device)
        for b in range(x_batch.shape[0]):
            for d in range(data.shape[0]):
                quadratic[b, d] = (
                    0.5 * torch.linalg.norm((data[d] - x_batch[b]) / sigma) ** 2
                )
        quadratic = quadratic.detach().cpu().numpy()

        # Compute quadratic by mut.
        dataset = TensorDataset(data)
        metric = torch.ones(3, 2, 5).to(device) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        quadratic_mut = dst.get_pairwise_energy(x_batch).detach().cpu().numpy()

        np.testing.assert_allclose(quadratic, quadratic_mut, rtol=1e-3)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_energy_to_data(self, device):
        # Generate some arbitary data.
        data = torch.rand(5, 3).to(device)
        sigma = 0.2
        x_batch = torch.rand(20, 3).to(device)

        # Manually compute minimum
        quadratic = torch.zeros((20, 5)).to(device)
        for i, x in enumerate(x_batch):
            for j, data_pt in enumerate(data):
                quadratic[i, j] = 0.5 * torch.linalg.norm((x - data_pt) / sigma) ** 2
        softmin = -torch.logsumexp(-quadratic, 1).cpu().numpy()

        # Compute minimum by mut.
        dataset = TensorDataset(data)
        metric = torch.ones(3) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        softmin_mut = dst.get_energy_to_data(x_batch).cpu().numpy()

        np.testing.assert_allclose(softmin, softmin_mut, rtol=1e-3)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_multidimensional_energy_to_data(self, device):
        data = torch.rand(5, 3, 2, 5).to(device)
        sigma = 0.2
        x_batch = torch.rand(3, 3, 2, 5).to(device)

        # Manually compute quadratic energy.
        quadratic = torch.zeros((3, 5)).to(device)
        for b in range(x_batch.shape[0]):
            for d in range(data.shape[0]):
                quadratic[b, d] = (
                    0.5 * torch.linalg.norm((data[d] - x_batch[b]) / sigma) ** 2
                )
        softmin = -torch.logsumexp(-quadratic, 1).cpu().numpy()

        # Compute minimum by mut.
        dataset = TensorDataset(data)
        metric = torch.ones(3, 2, 5) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        softmin_mut = dst.get_energy_to_data(x_batch).cpu().numpy()
        np.testing.assert_allclose(softmin, softmin_mut, rtol=1e-3)

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_energy_gradients(self, device):
        # TODO(terry-suh): it's desirable to check against manual computations.
        # Generate some arbitary data.
        data = torch.rand(5, 3).to(device)
        sigma = 0.2
        x_batch = torch.rand(20, 3).to(device)

        dataset = TensorDataset(data)
        metric = torch.ones(3) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        grads = dst.get_energy_gradients(x_batch).cpu()
        np.testing.assert_equal(grads.shape, torch.Size([20, 3]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_multidimensional_energy_to_data(self, device):
        data = torch.rand(5, 3, 2, 5).to(device)
        sigma = 0.2
        x_batch = torch.rand(3, 3, 2, 5).to(device)

        dataset = TensorDataset(data)
        metric = torch.ones(3, 2, 5) / (sigma**2.0)
        dst = mut.DataDistance(dataset, metric)
        grads = dst.get_energy_gradients(x_batch).cpu()
        np.testing.assert_equal(grads.shape, torch.Size([3, 3, 2, 5]))


class TestDataDistanceEstimatoXu:
    def initialize_dde(self):
        network = MLP(5, 1, [128, 128])
        domain_lb = -torch.ones(5)
        domain_ub = torch.ones(5)
        dde = mut.DataDistanceEstimatorXu(
            3, 2, network, domain_lb, domain_ub)
        return dde

    def test_constructor(self):
        network = MLP(5, 1, [128, 128])
        domain_lb = -torch.ones(5)
        domain_ub = torch.ones(5)

        with np.testing.assert_raises(ValueError):
            mut.DataDistanceEstimatorXu(3, 1, network, domain_lb, domain_ub)

        # Doesn't throw since input is 5.
        dde = mut.DataDistanceEstimatorXu(3, 2, network, domain_lb, domain_ub)
        assert ("domain_lb" in dde.state_dict())
        assert ("domain_ub" in dde.state_dict())

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_distance_eval(self, device: Literal["cpu", "cuda"]):
        x_tensor = torch.rand(100, 5).to(device)
        dde = self.initialize_dde().to(device)
        distance = dde.get_energy_to_data(x_tensor)
        np.testing.assert_equal(distance.shape, torch.Size([100, 1]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_gradients_eval(self, device: Literal["cpu", "cuda"]):
        x_tensor = torch.rand(100, 5).to(device)
        dde = self.initialize_dde().to(device)
        gradients = dde.get_energy_gradients(x_tensor)
        np.testing.assert_equal(gradients.shape, torch.Size([100, 5]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device):
        dde = self.initialize_dde()

        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        dataset = torch.utils.data.TensorDataset(
            torch.rand(100,3), torch.rand(100,2), 
            torch.rand(100,3))            
        metric = torch.rand(5)
        dde.train_network(dataset, params, metric)
        assert ("metric" in dde.state_dict())

class TestDataDistanceEstimatoXux:
    def initialize_dde(self):
        network = MLP(8, 1, [128, 128])
        domain_lb = -torch.ones(8)
        domain_ub = torch.ones(8)
        dde = mut.DataDistanceEstimatorXux(
            3, 2, network, domain_lb, domain_ub)
        return dde

    def test_constructor(self):
        network = MLP(8, 1, [128, 128])
        domain_lb = -torch.ones(8)
        domain_ub = torch.ones(8)

        with np.testing.assert_raises(ValueError):
            mut.DataDistanceEstimatorXux(2, 2, network, domain_lb, domain_ub)

        # Doesn't throw since input is 3 + 2 + 3 = 8.
        dde = mut.DataDistanceEstimatorXux(3, 2, network, domain_lb, domain_ub)
        assert ("domain_lb" in dde.state_dict())
        assert ("domain_ub" in dde.state_dict())

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_distance_eval(self, device: Literal["cpu", "cuda"]):
        x_tensor = torch.rand(100, 8).to(device)
        dde = self.initialize_dde().to(device)
        distance = dde.get_energy_to_data(x_tensor)
        np.testing.assert_equal(distance.shape, torch.Size([100, 1]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_gradients_eval(self, device: Literal["cpu", "cuda"]):
        x_tensor = torch.rand(100, 8).to(device)
        dde = self.initialize_dde().to(device)
        gradients = dde.get_energy_gradients(x_tensor)
        np.testing.assert_equal(gradients.shape, torch.Size([100, 8]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device):
        dde = self.initialize_dde()

        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        dataset = torch.utils.data.TensorDataset(
            torch.rand(100,3), torch.rand(100,2), 
            torch.rand(100,3))
        metric = torch.rand(8)
        dde.train_network(dataset, params, metric)
        assert ("metric" in dde.state_dict())
