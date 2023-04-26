import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Literal
from hydra import initialize, compose

import score_po.score_matching as mut
from score_po.nn import MLP, TrainParams, Normalizer


class TestScoreEstimatorXu:
    def test_constructor(self):
        network = MLP(5, 5, [128, 128])

        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimatorXu(2, 2, network)
        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimatorXu(4, 2, network)

        # Doesn't throw since 3 + 2 = 5.
        sf = mut.ScoreEstimatorXu(3, 2, network)

    def test_constructor_with_normalizer(self):
        network = MLP(5, 5, [128, 128])

        sf = mut.ScoreEstimatorXu(
            3,
            2,
            network,
            x_normalizer=Normalizer(
                k=torch.tensor([1, 2.0, 3.0]),
                b=torch.tensor([0, 0.5, 1]),
            ),
            u_normalizer=Normalizer(
                k=torch.tensor([4.0, 5]),
                b=torch.Tensor([1.5, 2])
            )
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize(
        "x_normalizer",
        (
            None,
            Normalizer(k=torch.tensor([2.0, 3.0]), b=torch.tensor([-1, -2])),
        )
    )
    @pytest.mark.parametrize(
        "u_normalizer",
        (
            None,
            Normalizer(k=torch.tensor([4.0]), b=torch.tensor([-3.0])),
        )
    )    
    def test_score_eval(self, device: Literal["cpu", "cuda"], x_normalizer, u_normalizer):
        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)
        z_tensor = torch.cat((x_tensor, u_tensor), dim=1)

        network = MLP(3, 3, [128, 128])
        sf = mut.ScoreEstimatorXu(2, 1, network, x_normalizer, u_normalizer)
        
        z_normalized = torch.cat((
            sf.x_normalizer.to(device)(x_tensor),
            sf.u_normalizer.to(device)(u_tensor)
        ), dim=1)
        
        score_z_expected = (
            network.to(device)(z_normalized)
            / torch.cat((sf.x_normalizer.to(device).k, sf.u_normalizer.to(device).k))
        )
        np.testing.assert_allclose(
            sf.get_score_z_given_z(z_tensor).cpu().detach().numpy(),
            score_z_expected.cpu().detach().numpy(),
        )

        np.testing.assert_equal(
            sf.get_score_z_given_z(z_tensor).shape, torch.Size([100, 3])
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device: Literal["cpu", "cuda"]):
        # Generate random data.

        network = MLP(5, 5, [128, 128])
        sf = mut.ScoreEstimatorXu(3, 2, network)

        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        x_tensor = torch.rand(100, 3)
        u_tensor = torch.rand(100, 2)
        dataset = torch.utils.data.TensorDataset(x_tensor, u_tensor)
        sf.train_network(dataset, params, 0.1, split=False)
        sf.train_network(dataset, params, 0.3, split=True)


class TestScoreEstimatorXux:
    def test_constructor(self):
        network = MLP(8, 8, [128, 128])

        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimatorXux(2, 2, network)
        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimatorXux(2, 3, network)

        # Doesn't throw since 3 + 2 + 3 = 8.
        sf = mut.ScoreEstimatorXux(3, 2, network)

    def test_constructor_with_normalizer(self):
        network = MLP(8, 8, [128, 128])

        sf = mut.ScoreEstimatorXux(
            3,
            2,
            network,
            x_normalizer=Normalizer(
                k=torch.tensor([1, 2.0, 3.0]),
                b=torch.tensor([0, 0.5, 1]),
            ),
            u_normalizer=Normalizer(
                k=torch.tensor([4.0, 5]),
                b=torch.Tensor([1.5, 2])
            )
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize(
        "x_normalizer",
        (
            None,
            Normalizer(k=torch.tensor([2.0, 3.0]), b=torch.tensor([-1, -2])),
        )
    )
    @pytest.mark.parametrize(
        "u_normalizer",
        (
            None,
            Normalizer(k=torch.tensor([4.0]), b=torch.tensor([-3.0])),
        )
    )    
    def test_score_eval(self, device: Literal["cpu", "cuda"], x_normalizer, u_normalizer):
        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)
        xnext_tensor = torch.rand(100, 2).to(device)
        
        z_tensor = torch.cat((x_tensor, u_tensor, xnext_tensor), dim=1)

        network = MLP(5, 5, [128, 128])
        sf = mut.ScoreEstimatorXux(2, 1, network, x_normalizer, u_normalizer)
        
        z_normalized = torch.cat((
            sf.x_normalizer.to(device)(x_tensor),
            sf.u_normalizer.to(device)(u_tensor),
            sf.x_normalizer.to(device)(xnext_tensor),            
        ), dim=1)
        
        score_z_expected = (
            network.to(device)(z_normalized)
            / torch.cat(
                (sf.x_normalizer.to(device).k, sf.u_normalizer.to(device).k,
                 sf.x_normalizer.to(device).k))
        )
        np.testing.assert_allclose(
            sf.get_score_z_given_z(z_tensor).cpu().detach().numpy(),
            score_z_expected.cpu().detach().numpy(),
        )

        np.testing.assert_equal(
            sf.get_score_z_given_z(z_tensor).shape, torch.Size([100, 5])
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device: Literal["cpu", "cuda"]):
        # Generate random data.

        network = MLP(8, 8, [128, 128])
        sf = mut.ScoreEstimatorXux(3, 2, network)

        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        x_tensor = torch.rand(100, 3)
        u_tensor = torch.rand(100, 2)
        xnext_tensor = torch.rand(100, 3)
        dataset = torch.utils.data.TensorDataset(x_tensor, u_tensor, xnext_tensor)
        sf.train_network(dataset, params, 0.1, split=False)
        sf.train_network(dataset, params, 0.3, split=True)


class GaussianScore(torch.nn.Module):
    """
    Compute the score of a Gaussian distribution N(x, mu, sigma)
    """

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("variance", variance)
        self.register_buffer("variance_inv", torch.linalg.inv(self.variance))

    def forward(self, x):
        return -(x - self.mean) @ self.variance_inv


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_langevin_dynamics(device):
    # Test to generate many samples (through Langevin dynamics), starting from a
    # Gaussian distribution with mean 0 and variance 1, we want to generate samples
    # from a Gaussian distribution with mean (10, 20) and variance 0.1.
    torch.manual_seed(123)
    mean = torch.tensor([10, 20.0], device=device)
    variance = 0.1 * torch.eye(2, device=device)
    gaussian_score = GaussianScore(mean, variance).to(device)
    gaussian_score.eval()

    x0 = torch.randn((1000, 2), device=device)
    epsilon = 2e-3
    steps = 1000
    x_history = mut.langevin_dynamics(x0, gaussian_score, epsilon, steps)
    assert x_history.shape == (steps,) + tuple(x0.shape)
    np.testing.assert_allclose(
        torch.mean(x_history[-1], dim=0).cpu().detach().numpy(),
        mean.cpu().detach().numpy(),
        atol=0.2,
    )
