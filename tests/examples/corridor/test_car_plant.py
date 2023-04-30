import examples.corridor.car_plant as mut

import pytest
import torch
import numpy as np

from score_po.dynamical_system import midpoint_integration


def test_single_integrator_dynamics():
    dut = mut.SingleIntegrator(dt=0.1, dim_x=2)
    x = torch.tensor([[1, 2], [3, 4.0], [5, 6.0]])
    u = torch.tensor([[0.5, 0.6], [0.1, -0.2], [-0.3, 0.4]])
    x_next = dut.dynamics_batch(x, u)
    x_next_expected = midpoint_integration(
        lambda x_batch, u_batch: u_batch, x, u, dut.dt
    )
    np.testing.assert_allclose(x_next.detach(), x_next_expected.detach())


def test_double_integrator_dynamics():
    dut = mut.DougleIntegrator(dt=0.1, dim_q=2)
    x = torch.tensor([[1, 2, 3, 4.0], [5, 6, 7, 8.0]])
    u = torch.tensor([[0.1, -1], [0.5, -2]])
    x_next = dut.dynamics_batch(x, u)
    x_next_expected = midpoint_integration(
        lambda x_batch, u_batch: torch.cat((x_batch[:, 2:], u_batch), dim=-1),
        x,
        u,
        dut.dt,
    )
    np.testing.assert_allclose(x_next.detach(), x_next_expected.detach())
