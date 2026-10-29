import pytest
import torch

from policybench.kernels.fused_euler import HAS_TRITON, fused_euler_denorm, reference_euler_denorm

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not HAS_TRITON, reason="needs triton")
def test_fused_euler_matches_reference():
    scale = torch.tensor([250.0, 243.0], device="cuda")
    bias = torch.tensor([262.0, 268.0], device="cuda")
    x = torch.randn(4, 16, 2, device="cuda")
    v = torch.randn_like(x)
    out = fused_euler_denorm(x, v, 0.1, scale, bias)
    expected = reference_euler_denorm(x, v, 0.1, scale, bias)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)
