import examples.cartpole.cartpole_plant as mut

import numpy as np
import pytest
import scipy
import torch


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

        sol = scipy.integrate.solve_ivp(dyn, [0, dt], x)
        np.testing.assert_allclose(sol.y[:, -1], xnext, atol=0.01)


class TestCartpoleNNDynamicalSystem:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_constructor(self, device):
        hidden_widths = [8, 4]
        x_lo = torch.tensor([-2, -1.0 * np.pi, -3, -10])
        x_up = torch.tensor([2, 1.5 * np.pi, 3, 10])
        u_lo = torch.tensor([-80.0])
        u_up = torch.tensor([80.0])
        dut = mut.CartpoleNNDynamicalSystem(
            hidden_widths, x_lo, x_up, u_lo, u_up, device
        )

        assert len(dut.net) == 2 * len(hidden_widths) + 1

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    @pytest.mark.parametrize("eval", (False, True))
    def test_dynamics_batch(self, device, eval):
        torch.manual_seed(123)
        hidden_widths = [8, 4]
        x_lo = torch.tensor([-2, -1.0 * np.pi, -3, -10])
        x_up = torch.tensor([2, 1.5 * np.pi, 3, 10])
        u_lo = torch.tensor([-80.0])
        u_up = torch.tensor([80.0])
        dut = mut.CartpoleNNDynamicalSystem(
            hidden_widths, x_lo, x_up, u_lo, u_up, device
        )

        batch_size = 20
        x_batch = torch.rand((batch_size, 4), device=device)
        u_batch = (torch.rand((batch_size, 1), device=device) - 0.5) * dut.u_up * 2
        xnext = dut.dynamics_batch(x_batch, u_batch, eval)
        assert xnext.shape == x_batch.shape

        x_normalized = (x_batch - (dut.x_lo + dut.x_up) / 2) / (
            (dut.x_up - dut.x_lo) / 2
        )
        u_normalized = (u_batch - (dut.u_lo + dut.u_up) / 2) / (
            (dut.u_up - dut.u_lo) / 2
        )
        xnext_expected = (
            dut.net(torch.concat((x_normalized, u_normalized), dim=1)) + x_batch
        )
        np.testing.assert_allclose(
            xnext.cpu().detach().numpy(), xnext_expected.cpu().detach().numpy()
        )
