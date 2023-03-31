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
