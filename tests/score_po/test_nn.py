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
        np.testing.assert_equal(params.save_best_model, "best.pth")

        np.testing.assert_allclose(params.dataset_split, np.array([0.6, 0.4]))


class TestEnsemble:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_ensemble(self, device: Literal["cpu", "cuda"]):
        network_lst = []
        x_batch = torch.rand(100, 3).to(device)
        mean = torch.zeros(100, 5).to(device)

        for k in range(4):
            # Test if MLP can evaluate expressions.
            net = mut.MLP(3, 5, [128, 128, 128]).to(device)
            network_lst.append(net)
            mean += net(x_batch)

        ensemble = mut.EnsembleNetwork(
            (3,),
            (5,),
            network_lst,
        )
        mean = mean / 4

        np.testing.assert_allclose(
            ensemble(x_batch).detach().cpu().numpy(), mean.detach().cpu().numpy()
        )
        # Test if ensemble can be applied without metric.
        np.testing.assert_equal(ensemble.get_variance(x_batch).shape, torch.Size([100]))

        # Test if ensemble can be applied with metric.
        metric = torch.ones(5)
        np.testing.assert_allclose(
            ensemble.get_variance(x_batch).detach().cpu().numpy(),
            ensemble.get_variance(x_batch, metric=metric).detach().cpu().numpy(),
        )
        np.testing.assert_allclose(
            2.0 * ensemble.get_variance(x_batch).detach().cpu().numpy(),
            ensemble.get_variance(x_batch, metric=2.0 * metric).detach().cpu().numpy(),
        )

        # Test if we can get gradients of ensemlbe.
        np.testing.assert_equal(
            ensemble.get_variance_gradients(x_batch).shape, torch.Size([100, 3])
        )

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_multidimensional_ensemble(self, device: Literal["cpu", "cuda"]):
        network_lst = []
        x_batch = torch.rand(100, 3, 5, 2).to(device)
        mean = torch.zeros(100, 3, 2).to(device)
        weights = torch.rand(4)

        for k in range(4):
            # TODO(terry-suh): replace this with e.g. ConvNets.
            fun = lambda x, weights=weights[k]: weights * x.sum(dim=2)
            network_lst.append(fun)
            mean += fun(x_batch)

        ensemble = mut.EnsembleNetwork(
            (3, 5, 2),
            (3, 2),
            network_lst,
        )
        mean = mean / 4

        # Test if the manual calculation leads to same value.
        np.testing.assert_allclose(
            ensemble(x_batch).detach().cpu().numpy(), mean.detach().cpu().numpy()
        )

        # Test if ensemble can be applied without metric.
        np.testing.assert_equal(ensemble.get_variance(x_batch).shape, torch.Size([100]))

        # Test if ensemble can be applied with metric
        metric = torch.ones((3, 2))
        np.testing.assert_allclose(
            ensemble.get_variance(x_batch).detach().cpu().numpy(),
            ensemble.get_variance(x_batch, metric=metric).detach().cpu().numpy(),
        )

        np.testing.assert_allclose(
            2.0 * ensemble.get_variance(x_batch).detach().cpu().numpy(),
            ensemble.get_variance(x_batch, metric=2.0 * metric).detach().cpu().numpy(),
        )


class TestNormalizer:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_constructor(self, device):
        dut1 = mut.Normalizer(k=None, b=None)
        dut1.to(device)
        assert dut1.k.device.type == device
        assert dut1.b.device.type == device
        assert dut1.k.item() == 1
        assert dut1.b.item() == 0
        assert len(dut1.state_dict()) == 2
        assert len(list(dut1.parameters())) == 0

        dut2 = mut.Normalizer(k=torch.tensor([1.0, 3.0]), b=torch.tensor([2.0, 4]))
        dut2.to(device)
        assert dut2.k.device.type == device
        assert dut2.b.device.type == device
        np.testing.assert_allclose(dut2.k.cpu(), np.array([1.0, 3.0]))
        np.testing.assert_allclose(dut2.b.cpu(), np.array([2.0, 4.0]))
        assert len(dut2.state_dict()) == 2
        assert len(list(dut2.parameters())) == 0

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_forward(self, device):
        dut = mut.Normalizer(k=torch.tensor([1.0, 3.0]), b=torch.tensor([2.0, 4.0])).to(
            device
        )
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6]]).to(device)
        xbar = dut(x)
        assert (x.shape == xbar.shape)

        np.testing.assert_allclose(xbar.cpu(), ((x - dut.b) / dut.k).cpu())

        x_denormalized = dut.denormalize(xbar)
        np.testing.assert_allclose(x_denormalized.cpu(), x.cpu())