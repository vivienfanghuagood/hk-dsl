"""hk2 end to end on the GPU this runs on: the same kernel sources on CDNA and RDNA."""
import pytest
import torch

import hk2 as hk

dev = "cuda"


@hk.jit
def vadd(x, y, out, BLOCK: hk.constexpr = 1024):
    i = hk.program_id(0) * BLOCK
    hk.store(out, i, hk.load(x, i, BLOCK) + hk.load(y, i, BLOCK))


@pytest.mark.parametrize("n", [1024, 100003, 7])
def test_vadd(n):
    x, y = torch.randn(n, device=dev), torch.randn(n, device=dev)
    out = torch.empty_like(x)
    vadd[(hk.cdiv(n, 1024),)](x, y, out)
    torch.testing.assert_close(out, x + y)


@hk.jit
def scale_bias(x, b, out, alpha, BM: hk.constexpr = 32, BN: hk.constexpr = 64):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    t = hk.load(x, (m0, n0), (BM, BN)).to(hk.f32)
    bias = hk.load(b, n0, BN).to(hk.f32)
    hk.store(out, (m0, n0), t * alpha + bias[None, :])


@pytest.mark.parametrize("m,n", [(64, 128), (100, 72), (33, 200)])
def test_broadcast_2d(m, n):
    x = torch.randn(m, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, device=dev, dtype=torch.bfloat16)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    scale_bias[(hk.cdiv(m, 32), hk.cdiv(n, 64))](x, b, out, 0.5)
    ref = (x.float() * 0.5 + b.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@hk.jit
def softmax(x, out, BM: hk.constexpr = 4, BN: hk.constexpr = 1024):
    m0 = hk.program_id(0) * BM
    t = hk.load(x, (m0, 0), (BM, BN), other=float("-inf")).to(hk.f32)
    m = t.max(1)
    e = hk.exp(t - m[:, None])
    s = e.sum(1)
    hk.store(out, (m0, 0), e / s[:, None])


@pytest.mark.parametrize("m,n", [(64, 1024), (37, 1000), (8, 512)])
def test_softmax(m, n):
    x = torch.randn(m, n, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    softmax[(hk.cdiv(m, 4),)](x, out, BN=1 << (n - 1).bit_length())
    torch.testing.assert_close(out, torch.softmax(x, -1), atol=1e-5, rtol=1e-5)


@hk.jit
def matmul(a, b, c, BM: hk.constexpr = 64, BN: hk.constexpr = 64, BK: hk.constexpr = 32,
           TRANS_B: hk.constexpr = False):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN), hk.f32)
    for k in range(0, a.shape[1], BK):
        x = hk.load(a, (m0, k), (BM, BK))
        if TRANS_B:
            y = hk.load(b, (n0, k), (BN, BK)).T
        else:
            y = hk.load(b, (k, n0), (BK, BN))
        acc = hk.dot(x, y, acc)
    hk.store(c, (m0, n0), acc.to(c.dtype))


@pytest.mark.parametrize("m,n,k", [(64, 64, 32), (256, 128, 512), (100, 72, 96), (1, 256, 64)])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_matmul(m, n, k, trans_b, dt):
    a = torch.randn(m, k, device=dev, dtype=dt)
    b = torch.randn(n, k, device=dev, dtype=dt) if trans_b else torch.randn(k, n, device=dev, dtype=dt)
    c = torch.empty(m, n, device=dev, dtype=dt)
    matmul[(hk.cdiv(m, 64), hk.cdiv(n, 64))](a, b, c, TRANS_B=trans_b)
    ref = a.float() @ (b.float().T if trans_b else b.float())
    torch.testing.assert_close(c.float(), ref, atol=0.1, rtol=2e-2)


@hk.jit
def causal_fill(out, BM: hk.constexpr = 32, BN: hk.constexpr = 32):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    rows = m0 + hk.arange(BM)[:, None]
    cols = n0 + hk.arange(BN)[None, :]
    hk.store(out, (m0, n0), hk.where(cols <= rows, 1.0, 0.0))


def test_arange_where():
    out = torch.empty(96, 96, device=dev)
    causal_fill[(3, 3)](out)
    torch.testing.assert_close(out, torch.ones(96, 96, device=dev).tril())


@hk.jit
def cond_scale(x, out, flag, BLOCK: hk.constexpr = 256):
    i = hk.program_id(0) * BLOCK
    t = hk.load(x, i, BLOCK)
    if flag > 0:
        t = t * 2.0
    else:
        t = t - 1.0
    hk.store(out, i, t)


def test_runtime_if():
    x = torch.randn(1000, device=dev)
    out = torch.empty_like(x)
    cond_scale[(4,)](x, out, 1)
    torch.testing.assert_close(out, x * 2)
    cond_scale[(4,)](x, out, 0)
    torch.testing.assert_close(out, x - 1)
