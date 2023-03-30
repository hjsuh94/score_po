import examples.cartpole.learn_model as mut

import numpy as np
import pytest
import scipy
import torch

from examples.cartpole.cartpole_plant import CartpolePlant


class TestGenerateData:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test(self, device):
        torch.manual_seed(123)

        dt = 0.01
        plant = CartpolePlant(dt)

        left_wall = -3.0
        right_wall = 4.0
        cart_length = 0.2
        u_max = 10
        sample_size = 100
        dataset = mut.generate_data(
            cart_length, left_wall, right_wall, dt, u_max, sample_size, device
        )
        assert len(dataset) <= sample_size

        loader = torch.utils.data.DataLoader(dataset)

        def check_within_all(x):
            assert torch.all(x[:, 0] >= left_wall + cart_length / 2)
            assert torch.all(x[:, 0] <= right_wall - cart_length / 2)
            pole_pos_x = x[:, 0] + plant.l2 * torch.sin(x[:, 1])
            assert torch.all(pole_pos_x <= right_wall)
            assert torch.all(pole_pos_x >= left_wall)

        # Check that the states are within the
        for batch in loader:
            x_samples, u_samples, xnext_samples = batch
            assert torch.all(u_samples <= u_max)
            assert torch.all(u_samples >= -u_max)
            check_within_all(x_samples)
            check_within_all(xnext_samples)
            # Now simulate the dynamics with scipy for dt
            for i in range(x_samples.shape[0]):

                def dyn(t, x):
                    return plant.calc_derivative(
                        x.reshape((1, -1)),
                        u_samples[i].cpu().detach().numpy().reshape((1, -1)),
                    )

                sol = scipy.integrate.solve_ivp(
                    dyn, [0, dt], x_samples[i].cpu().detach().numpy()
                )
                np.testing.assert_allclose(
                    sol.y[:, -1], xnext_samples[i].cpu().detach().numpy(), atol=1e-1
                )
