import score_po.policy as mut

import pytest
import torch


class TestClamper:
    @pytest.mark.parametrize(
        "method", (mut.Clamper.Method.HARD, mut.Clamper.Method.TANH)
    )
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_constructor(self, method, device):
        dut1 = mut.Clamper(
            lower=torch.Tensor([1, 3.0]), upper=torch.Tensor([2.0, 4]), method=method
        ).to(device)
        assert dut1.lower.device.type == device
        assert dut1.upper.device.type == device
        assert len(list(dut1.parameters())) == 0
        assert len(dut1.state_dict()) > 0

    @pytest.mark.parametrize(
        "method", (mut.Clamper.Method.HARD, mut.Clamper.Method.TANH)
    )
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_forward(self, method, device):
        dut = mut.Clamper(
            lower=torch.Tensor([1, 3.0]), upper=torch.Tensor([2.0, 4]), method=method
        ).to(device)
        x = torch.tensor([[2, 5], [0.5, 3], [1, 3.5], [-2, 4]], device=device)
        x_clamp = dut(x)
        assert x_clamp.shape == x.shape
        assert torch.all(x_clamp >= dut.lower)
        assert torch.all(x_clamp <= dut.upper)


class TestTimeVaryingOpenLoopPolicy:
    @pytest.mark.parametrize("device", ("cpu", "cuda"))
    def test_forward(self, device):
        dut = mut.TimeVaryingOpenLoopPolicy(
            dim_x=3,
            dim_u=2,
            T=5,
            u_clip=mut.Clamper(
                lower=torch.tensor([0.0, -1]),
                upper=torch.tensor([3, 4.0]),
                method=mut.Clamper.Method.HARD,
            ),
        )
        dut.params.data = torch.cos(torch.rand(dut.T, dut.dim_u))
        dut.to(device)
        batch_size = 5
        x_batch = torch.empty((batch_size, dut.dim_x), device=device)
        u_batch = dut.forward(x_batch, t=0)
        assert u_batch.shape == (batch_size, dut.dim_u)
        assert torch.all(u_batch <= dut.u_clip.upper)
        assert torch.all(u_batch >= dut.u_clip.lower)
