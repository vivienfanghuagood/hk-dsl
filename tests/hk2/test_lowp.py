"""int8 and fp8 dots: int8 on every GPU with an int8 matrix instruction, fp8 on gfx950."""
import pytest
import torch

import hk2 as hk

arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


@hk.jit
def mm(a, b, c, BM: hk.constexpr = 64, BN: hk.constexpr = 64, BK: hk.constexpr = 64):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN))
    for k in range(0, a.shape[1], BK):
        acc = hk.dot(hk.load(a, (m0, k), (BM, BK)), hk.load(b, (n0, k), (BN, BK)).T, acc)
    hk.store(c, (m0, n0), acc)


@hk.jit
def mm8(a, b, c, BM: hk.constexpr = 64, BN: hk.constexpr = 64, BK: hk.constexpr = 64):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN), hk.i32)
    for k in range(0, a.shape[1], BK):
        acc = hk.dot(hk.load(a, (m0, k), (BM, BK)), hk.load(b, (n0, k), (BN, BK)).T, acc)
    hk.store(c, (m0, n0), acc)


@pytest.mark.skipif(arch not in ("gfx950", "gfx1100", "gfx1101", "gfx1102"), reason="int8 matrix instructions")
@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (200, 136, 320), (256, 512, 1024)])
def test_int8(m, n, k):
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    c = torch.empty(m, n, device="cuda", dtype=torch.int32)
    mm8[(hk.cdiv(m, 64), hk.cdiv(n, 64))](a, b, c)
    ref = (a.double() @ b.double().T).to(torch.int32)
    assert torch.equal(c, ref)


@pytest.mark.skipif(arch != "gfx950", reason="fp8 matrix instructions (OCP) on gfx950")
@pytest.mark.parametrize("dt", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (200, 136, 320)])
def test_fp8(m, n, k, dt):
    a = (torch.randn(m, k, device="cuda") * 2).to(dt)
    b = (torch.randn(n, k, device="cuda") * 2).to(dt)
    c = torch.empty(m, n, device="cuda", dtype=torch.float32)
    mm[(hk.cdiv(m, 64), hk.cdiv(n, 64))](a, b, c)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=1e-2)  # the matrix core accumulates fp8 products its own way


@hk.jit
def quant(x, q, back, BLOCK: hk.constexpr = 256):
    i = hk.program_id(0) * BLOCK
    t = hk.load(x, i, BLOCK)
    f8 = t.to(q.dtype)
    hk.store(q, i, f8)
    hk.store(back, i, f8.to(hk.f32))


@pytest.mark.skipif(arch != "gfx950", reason="fp8 conversions on gfx950")
@pytest.mark.parametrize("dt", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_casts(dt):
    x = torch.randn(1000, device="cuda") * 8
    q = torch.empty(1000, device="cuda", dtype=dt)
    back = torch.empty_like(x)
    quant[(4,)](x, q, back)
    assert torch.equal(q.view(torch.uint8), x.to(dt).view(torch.uint8))
    torch.testing.assert_close(back, x.to(dt).float())
