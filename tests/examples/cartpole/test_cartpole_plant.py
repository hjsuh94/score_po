import examples.cartpole.cartpole_plant as mut

import numpy as np
import pytest
import torch
from scipy import integrate

from score_po.nn import Normalizer


class TestCartpolePlant:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_dynamics_batch(self, device):
        dt = 0.01
        dut = mut.CartpolePlant(dt)
        x_batch = np.array([[0.5, 0.2, -1, 2], [-0.3, 0.1, 0.5, 0.4]])
        u_batch = np.array([[2], [-1.0]])
        xdot_batch = dut.dynamics_batch(x_batch, u_batch)
        assert xdot_batch.shape == x_batch.shape

        x_batch_torch = torch.from_numpy(x_batch).to(device)
        u_batch_torch = torch.from_numpy(u_batch).to(device)
        xdot_batch_torch = dut.dynamics_batch(x_batch_torch, u_batch_torch)
        np.testing.assert_allclose(xdot_batch, xdot_batch_torch.cpu().detach().numpy())

    def test_dynamics(self):
        dt = 0.01
        dut = mut.CartpolePlant(dt)
        x = np.array([0.5, 0.4, -1, -2])
        u = np.array([3])
        xnext = dut.dynamics(x, u)
        assert xnext.shape == (4,)

        def dyn(t, x_val):
            return dut.calc_derivative(
                x_val.reshape((1, -1)), u.reshape((1, -1))
            ).squeeze(0)

        sol = integrate.solve_ivp(dyn, [0, dt], x)
        np.testing.assert_allclose(sol.y[:, -1], xnext, atol=0.01)


class TestCartpoleNNDynamicalSystem:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_constructor(self, device):
        hidden_layers = [8, 4]
        x_normalizer = Normalizer(k=torch.tensor([2, np.pi, 3, 10]), b=None)
        u_normalizer = Normalizer(k=torch.tensor([80]), b=None)
        dut = mut.CartpoleNNDynamicalSystem(
            hidden_layers, device, x_normalizer, u_normalizer
        )
        assert dut.net.mlp[0].weight.data.device.type == device


    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("eval", (False, True))
    def test_dynamics_batch(self, device, eval):
        torch.manual_seed(123)
        hidden_layers = [8, 4]
        x_normalizer = Normalizer(k=torch.tensor([2, np.pi, 3, 10]), b=None)
        u_normalizer = Normalizer(k=torch.tensor([80]), b=None)
        dut = mut.CartpoleNNDynamicalSystem(
            hidden_layers, device, x_normalizer, u_normalizer
        ).to(device)

        batch_size = 20
        x_batch = torch.rand((batch_size, 4), device=device)
        u_batch = (torch.rand((batch_size, 1), device=device) - 0.5)  * 80
        xnext = dut.dynamics_batch(x_batch, u_batch, eval)
        assert xnext.shape == x_batch.shape

        x_normalized = x_normalizer(x_batch) 
        u_normalized = u_normalizer(u_batch) 
        xnext_expected = (
            x_normalizer.denormalize(dut.net(torch.concat((x_normalized, u_normalized), dim=1))) + x_batch
        )
        np.testing.assert_allclose(
            xnext.cpu().detach().numpy(), xnext_expected.cpu().detach().numpy()
        )
