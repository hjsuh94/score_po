import numpy as np
import pytest 
import torch 
import torch.nn as nn

import score_po.nn as mut

class TestMLP:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_mlp_device(self, device):
        # Test if MLP can evaluate expressions.
        net = mut.MLP(3, 3, [128, 128, 128]).to(device)
        random_vector = torch.rand(100, 3).to(device)
        assert net(random_vector).shape == random_vector.shape
        
        # Should throw if MLP doesn't get correct input.
        with np.testing.assert_raises(RuntimeError):
            net(torch.rand(100,5).to(device))
        
    def test_mlp_vectorize_parameters(self):
        net = mut.MLP(5, 5, [128, 128, 128, 128])
        net.eval()
        
        params = net.get_vectorized_parameters()
        
        # test if conversion is same with torch convention.
        np.testing.assert_allclose(params.detach().numpy(),
            nn.utils.parameters_to_vector(net.parameters()).detach().numpy()
        )
        
        # test if setting and getting gives us same values.
        random_params = torch.rand(len(params))
        net.set_vectorized_parameters(random_params)
        ret_params = net.get_vectorized_parameters()
        np.testing.assert_allclose(random_params, ret_params.detach())
        
    def test_mlp_gradients(self):
        net = mut.MLP(5, 5, [128, 128])
        
        batch_data = torch.rand(100, 5)
        net.zero_grad()
        output = net(batch_data)
        loss = ((output - batch_data)).sum(dim=-1).mean(dim=0)
        loss.backward()
        
        grad = net.get_vectorized_gradients()
        np.testing.assert_equal(
            len(grad), len(net.get_vectorized_parameters()))
