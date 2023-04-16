import score_po.costs as mut

import pytest
import torch


class TestQuadraticCost:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_constructor(self, device):
        Q = torch.eye(2)
        R = torch.eye(3)
        Qd = torch.eye(2) * 10
        xd = torch.tensor([1.0, 2.0])
        dut = mut.QuadraticCost(Q, R, Qd, xd).to(device)
        assert len(dut.state_dict()) > 0
        assert len(list(dut.parameters())) == 0

    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_forward(self, device):
        Q = torch.eye(2)
        R = torch.eye(3)
        Qd = torch.eye(2) * 10
        xd = torch.tensor([1.0, 2.0])
        dut = mut.QuadraticCost(Q, R, Qd, xd).to(device)
        x = torch.tensor([[1, 3], [0, 5], [2, 4.0]], device=device)
        u = torch.tensor([[1, 3, 5], [2, 4, 6], [-1, -2, -3.0]], device=device)

        running_cost = dut(False, x, u)
        assert running_cost.shape == (x.shape[0],)

        terminal_cost = dut(True, x)
        assert terminal_cost.shape == (x.shape[0],)
