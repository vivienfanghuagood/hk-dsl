"""hk.ops.gemm.scaled_mm, the fp8 GEMM on gfx1201 (hk.ops.gemm's fp8 WMMA, op_dtype=fp8e4m3/fp8e5m2):
against the fp32 product of the same fp8 values, per-tensor and per-row x per-column scales, with
and without a bias, a ragged M; repeatable bit for bit; the refusals."""

from __future__ import annotations

import pytest

from hk.target import current_arch, get_target

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if "fp8e4m3" not in get_target(current_arch()).mma or get_target(current_arch()).wave != 32:
    pytest.skip("an fp8 WMMA (gfx1201)", allow_module_level=True)

from hk.ops.gemm import FP8_DEFAULT, scaled_mm  # noqa: E402

FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _operands(m, n, k, dt, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(FP8[dt])
    b = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(FP8[dt])
    return a, b


def _check(y, ref):
    # The fp8 values are exact in fp32 and the accumulation is fp32: what is
    # left is bf16's rounding of C and the order of the sum.
    assert (y.float() - ref).abs().max() / ref.abs().max() < 8e-3


@pytest.mark.parametrize("dt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("m,n,k", [(256, 512, 1024), (77, 256, 512), (4096, 1024, 2048)])
def test_per_tensor_scales(dt, m, n, k):
    a, b = _operands(m, n, k, dt)
    sa, sb = torch.tensor(0.75, device="cuda"), torch.tensor(1.5, device="cuda")
    y = scaled_mm(a, b, sa, sb)
    _check(y, (a.float() @ b.float().T) * 0.75 * 1.5)


@pytest.mark.parametrize("dt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("bias", [False, True])
def test_per_row_and_column_scales(dt, bias):
    m, n, k = 333, 768, 1024
    a, b = _operands(m, n, k, dt, seed=1)
    sa = torch.rand(m, 1, device="cuda") + 0.5
    sb = torch.rand(n, device="cuda") + 0.5
    bb = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None
    y = scaled_mm(a, b, sa, sb, bias=bb)
    ref = (a.float() @ b.float().T) * sa * sb + (bb.float() if bias else 0)
    _check(y, ref)
    assert all(torch.equal(scaled_mm(a, b, sa, sb, bias=bb), y) for _ in range(3))


def test_matches_torch_scaled_mm():
    m, n, k = 512, 1024, 2048
    a, b = _operands(m, n, k, "e4m3", seed=2)
    sa, sb = torch.rand(m, 1, device="cuda") + 0.5, torch.rand(1, n, device="cuda") + 0.5
    y = scaled_mm(a, b, sa, sb)
    t = torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    assert (y.float() - t.float()).abs().max() / t.float().abs().max() < 8e-3


def test_refusals():
    a, b = _operands(128, 256, 512, "e4m3")
    one = torch.ones((), device="cuda")
    with pytest.raises(TypeError):
        scaled_mm(a, b.to(torch.float8_e5m2), one, one)
    with pytest.raises(TypeError):
        scaled_mm(a.to(torch.bfloat16), b.to(torch.bfloat16), one, one)
    with pytest.raises(ValueError):
        scaled_mm(a[:, : FP8_DEFAULT["k_step"] // 2].contiguous(),
                  b[:, : FP8_DEFAULT["k_step"] // 2].contiguous(), one, one)
    with pytest.raises(ValueError):
        scaled_mm(a, b, torch.ones(7, device="cuda"), one)
