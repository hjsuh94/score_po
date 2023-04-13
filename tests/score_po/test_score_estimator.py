import numpy as np
import pytest
import torch
import torch.nn as nn
from typing import Literal
from hydra import initialize, compose

import score_po.score_matching as mut
from score_po.nn import MLP, TrainParams


class TestScoreEstimator:
    def test_constructor(self):
        network = MLP(5, 5, [128, 128])

        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimator(2, 2, network)
        with np.testing.assert_raises(ValueError):
            mut.ScoreEstimator(4, 2, network)

        # Doesn't throw since 3 + 2 = 5.
        sf = mut.ScoreEstimator(3, 2, network)

        np.testing.assert_allclose(sf.x_normalizer.cpu(), np.ones((3,)))
        np.testing.assert_allclose(sf.u_normalizer.cpu(), np.ones((2,)))

    def test_constructor_with_normalizer(self):
        network = MLP(5, 5, [128, 128])

        sf = mut.ScoreEstimator(
            3,
            2,
            network,
            x_normalizer=torch.tensor([1, 2.0, 3]),
            u_normalizer=torch.tensor([4, 5]),
        )
        np.testing.assert_allclose(sf.x_normalizer.cpu(), np.array([1.0, 2.0, 3]))
        np.testing.assert_allclose(sf.u_normalizer.cpu(), np.array([4.0, 5]))

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("x_normalizer", (None, torch.tensor([2.0, 3.0])))
    @pytest.mark.parametrize("u_normalizer", (None, torch.tensor([4.0])))
    def test_score_eval(
        self, device: Literal["cpu", "cuda"], x_normalizer, u_normalizer
    ):
        x_tensor = torch.rand(100, 2).to(device)
        u_tensor = torch.rand(100, 1).to(device)
        z_tensor = torch.cat((x_tensor, u_tensor), dim=1)

        network = MLP(3, 3, [128, 128])
        sf = mut.ScoreEstimator(2, 1, network, x_normalizer, u_normalizer)

        z_normalizer = torch.cat((sf.x_normalizer, sf.u_normalizer), dim=-1).to(device)
        score_z_expected = network.to(device)(z_tensor / z_normalizer) / z_normalizer
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
