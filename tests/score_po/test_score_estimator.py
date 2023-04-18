import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Literal
from hydra import initialize, compose

import score_po.score_matching as mut
from score_po.nn import MLP, TrainParams, Normalizer


class TestScoreEstimator:
    def test_constructor(self):
        network = MLP(5, 5, [128, 128])

        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimator(2, 2, network)
        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimator(4, 2, network)

        # Doesn't throw since 3 + 2 = 5.
        sf = mut.ScoreEstimator(3, 2, network)

    def test_constructor_with_normalizer(self):
        network = MLP(5, 5, [128, 128])

        sf = mut.ScoreEstimator(
            3,
            2,
            network,
            z_normalizer=Normalizer(
                k=torch.tensor([1, 2.0, 3.0, 4.0, 5]),
                b=torch.tensor([0, 0.5, 1, 1.5, 2]),
            ),
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize(
        "z_normalizer",
        (
            None,
            Normalizer(k=torch.tensor([2.0, 3.0, 4.0]), b=torch.tensor([-1, -2, -3.0])),
        ),
    )
    def test_score_eval(self, device: Literal["cpu", "cuda"], z_normalizer):
        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)
        z_tensor = torch.cat((x_tensor, u_tensor), dim=1)

        network = MLP(3, 3, [128, 128])
        sf = mut.ScoreEstimator(2, 1, network, z_normalizer)

        score_z_expected = (
            network.to(device)(sf.z_normalizer.to(device)(z_tensor))
            / sf.z_normalizer.to(device).k
        )
        np.testing.assert_allclose(
            sf.get_score_z_given_z(z_tensor).cpu().detach().numpy(),
            score_z_expected.cpu().detach().numpy(),
        )

        np.testing.assert_equal(
            sf.get_score_z_given_z(z_tensor).shape, torch.Size([100, 3])
        )
        np.testing.assert_equal(
            sf.get_score_x_given_z(z_tensor).shape, torch.Size([100, 2])
        )
        np.testing.assert_equal(
            sf.get_score_u_given_z(z_tensor).shape, torch.Size([100, 1])
        )
        np.testing.assert_equal(
            sf.get_score_z_given_xu(x_tensor, u_tensor).shape, torch.Size([100, 3])
        )
        print(sf.get_score_x_given_xu(x_tensor, u_tensor).shape)
        np.testing.assert_equal(
            sf.get_score_x_given_xu(x_tensor, u_tensor).shape, torch.Size([100, 2])
        )
        np.testing.assert_equal(
            sf.get_score_u_given_xu(x_tensor, u_tensor).shape, torch.Size([100, 1])
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_train(self, device: Literal["cpu", "cuda"]):
        # Generate random data.

        network = MLP(5, 5, [128, 128])
        sf = mut.ScoreEstimator(3, 2, network)

        params = TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        data_tensor = torch.rand(100, 5)
        dataset = torch.utils.data.TensorDataset(data_tensor)
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
    torch.manual_seed(123)
    mean = torch.tensor([10, 20.0], device=device)
    variance = 0.1 * torch.eye(2, device=device)
    gaussian_score = GaussianScore(mean, variance).to(device)

    x0 = torch.randn((1000, 2), device=device)
    epsilon = 1e-4
    steps = 10000
    xT = mut.langevin_dynamics(x0, gaussian_score, epsilon, steps)
    assert xT.shape == x0.shape
    np.testing.assert_allclose(torch.mean(xT, dim=0).cpu().detach().numpy(), mean.cpu().detach().numpy(), atol=0.2)
