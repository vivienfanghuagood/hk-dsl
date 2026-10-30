import pytest
import torch

from hk2.kernels import rowwise

SHAPES = [(7, 4096), (128, 5120), (33, 1000), (1024, 128), (5, 8192)]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dt", DTYPES)
def test_rmsnorm(shape, dt):
    x = torch.randn(shape, device="cuda", dtype=dt)
    w = torch.randn(shape[-1], device="cuda", dtype=dt)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(dt)
    tol = 2e-2 if dt != torch.float32 else 1e-5
    torch.testing.assert_close(rowwise.rmsnorm(x, w), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dt", DTYPES)
def test_layernorm(shape, dt):
    x = torch.randn(shape, device="cuda", dtype=dt) * 3 + 1
    w = torch.randn(shape[-1], device="cuda", dtype=dt)
    b = torch.randn(shape[-1], device="cuda", dtype=dt)
    ref = torch.nn.functional.layer_norm(x.float(), (shape[-1],), w.float(), b.float(), 1e-5).to(dt)
    tol = 3e-2 if dt != torch.float32 else 1e-4
    torch.testing.assert_close(rowwise.layernorm(x, w, b), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dt", DTYPES)
def test_softmax(shape, dt):
    x = torch.randn(shape, device="cuda", dtype=dt) * 4
    ref = torch.softmax(x.float(), -1).to(dt)
    tol = 1e-2 if dt != torch.float32 else 1e-6
    torch.testing.assert_close(rowwise.softmax(x), ref, atol=tol, rtol=tol)
