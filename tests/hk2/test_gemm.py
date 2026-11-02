import pytest
import torch

from hk2.kernels import gemm


@pytest.mark.parametrize("m,n,k", [(256, 512, 256), (100, 200, 96), (7, 4096, 512)])
@pytest.mark.parametrize("act", [None, "silu", "gelu", "relu"])
def test_linear(m, n, k, act):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = x.float() @ w.float().T + b.float()
    ref = {None: y, "silu": torch.nn.functional.silu(y), "relu": torch.relu(y),
           "gelu": torch.nn.functional.gelu(y, approximate="tanh")}[act]
    torch.testing.assert_close(gemm.linear(x, w, b, act=act).float(), ref, atol=6e-2, rtol=2e-2)


def test_matmul_batched_lead():
    a = torch.randn(2, 3, 64, 128, device="cuda", dtype=torch.float16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(gemm.matmul(a, b).float(), a.float() @ b.float(), atol=0.1, rtol=2e-2)
