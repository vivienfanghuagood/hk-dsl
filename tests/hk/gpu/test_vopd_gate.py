"""The VOPD gate on the GPU: the strip RMSNorm it exists for, launched 200 times.

On gfx1100, the bounded strip kernel with the fused add (strip_addrmsw, 16
warps, EXACT=0: 1024 x 5120 bf16 rows), built by AMD clang 23 (TheRock 7.14),
had a VOPD with gfx11's interlock hazard: X's and Y's VSRC1 were distinct VGPRs
of the same parity (llvm/llvm-project#220348). It got element 1 of every
lane's run wrong in 25-65% of launches. hk.runtime.compile's VOPD gate rebuilds
such a kernel without dual issue (tests/hk/codegen/test_vopd_gate.py has the
build side). With the gate off (HK_VOPD_GATE=0) and that compiler, both bf16
cases here fail (132 of 199 launches differ).

Every launch writes into NaN-filled outputs. Each case must give the same
bits on every launch, no NaN, and be within 0.51 ulp of float64. The fp16
cases and weight_offset 0 / 1 (#166's (1 + w) fold) are the other
instantiations of the same plan.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

LAUNCHES = 200


def _ulps(got, want, dtype):
    mant = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    e = torch.floor(torch.log2(want.to(dtype).double().abs().clamp_min(2.0 ** -14)))
    return ((got.double() - want).abs() / torch.pow(2.0, e - mant)).max().item()


@pytest.mark.parametrize("offset", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_strip_fused_add_is_the_same_on_every_launch(dtype, offset, monkeypatch):
    from hk.ops import norm

    monkeypatch.setattr(norm, "LAYOUT", "strip")
    rows, cols = 1024, 5120
    g = torch.Generator(device="cuda").manual_seed(rows * 31 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, generator=g)
    r = (torch.randn(rows, cols, device="cuda", generator=g) * 4).to(dtype)
    w = (torch.randn(cols, device="cuda", generator=g) * 0.1).to(dtype)
    z = x.double() + r.double()
    want = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * (offset + w.double())
    first, differ, nan, worst = None, 0, 0, 0.0
    for _ in range(LAUNCHES):
        xi = torch.full_like(x, float("nan"))
        ri = torch.full_like(r, float("nan"))
        xi.copy_(x)
        ri.copy_(r)
        hk.ops.fused_add_rms_norm(xi, ri, w, weight_offset=offset)
        torch.cuda.synchronize()
        nan += bool(xi.isnan().any() or ri.isnan().any())
        if first is None:
            first = (xi.clone(), ri.clone())
            worst = _ulps(xi, want, dtype)
            assert torch.equal(ri, (x.float() + r.float()).to(dtype)), "residual is not x + residual"
        elif not (torch.equal(xi, first[0]) and torch.equal(ri, first[1])):
            differ += 1
    assert (differ, nan) == (0, 0), f"{differ} of {LAUNCHES - 1} launches differ, {nan} with NaN"
    assert worst <= 0.51, worst
