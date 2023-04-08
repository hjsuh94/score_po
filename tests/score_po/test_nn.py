import numpy as np
import pytest
import torch
import torch.nn as nn

import score_po.nn as mut
from typing import Literal
from hydra import compose, initialize


class TestMLP:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_mlp_device(self, device: Literal["cpu", "cuda"]):
        # Test if MLP can evaluate expressions.
        net = mut.MLP(3, 3, [128, 128, 128]).to(device)
        random_vector = torch.rand(100, 3).to(device)
        assert net(random_vector).shape == random_vector.shape

        # Should throw if MLP doesn't get correct input.
        with np.testing.assert_raises(RuntimeError):
            net(torch.rand(100, 5).to(device))

    def test_mlp_vectorize_parameters(self):
        # Test vectorization of MLP parameters.
        net = mut.MLP(5, 5, [128, 128, 128, 128])
        net.eval()

        params = net.get_vectorized_parameters()

        # test if conversion is same with torch convention.
        np.testing.assert_allclose(
            params.detach().numpy(),
            nn.utils.parameters_to_vector(net.parameters()).detach().numpy(),
        )

        # test if setting and getting gives us same values.
        random_params = torch.rand(len(params))
        net.set_vectorized_parameters(random_params)
        ret_params = net.get_vectorized_parameters()
        np.testing.assert_allclose(random_params, ret_params.detach())

    def test_mlp_gradients(self):
        # Test differentiability w.r.t. parameters.
        net = mut.MLP(5, 5, [128, 128])

        batch_data = torch.rand(100, 5)
        net.zero_grad()
        output = net(batch_data)
        loss = ((output - batch_data)).sum(dim=-1).mean(dim=0)
        loss.backward()

        grad = net.get_vectorized_gradients()
        np.testing.assert_equal(len(grad), len(net.get_vectorized_parameters()))


class TestTrainConfig:
    def test_constructor(self):
        params = mut.TrainParams()

        # These lines test if we can call on these parameters.
        assert hasattr(params.adam_params, "lr")
        assert hasattr(params.adam_params, "epochs")
        assert hasattr(params.adam_params, "batch_size")

        assert hasattr(params.wandb_params, "enabled")
        assert hasattr(params.wandb_params, "project")
        assert hasattr(params.wandb_params, "entity")

        assert hasattr(params, "dataset_split")
        assert hasattr(params, "save_best_model")
        assert hasattr(params, "device")

    def test_cfg_load(self):
        params = mut.TrainParams()
        with initialize(config_path="./config"):
            cfg = compose(config_name="train_params")
            params.load_from_config(cfg)

        np.testing.assert_equal(params.adam_params.lr, 1e-5)
        np.testing.assert_equal(params.adam_params.epochs, 3)
        np.testing.assert_equal(params.adam_params.batch_size, 256)

        np.testing.assert_equal(params.wandb_params.enabled, True)
        np.testing.assert_equal(params.device, "cuda")
        np.testing.assert_equal(params.save_best_model, "best")

        np.testing.assert_allclose(params.dataset_split, np.array([0.6, 0.4]))