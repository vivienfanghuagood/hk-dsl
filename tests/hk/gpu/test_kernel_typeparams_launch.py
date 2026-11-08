"""One kernel for every dtype and warp count, launched: `hk.Tensor[hk.T]` read from the tensors (or
given), `warps` per call, each specialization computing what its factory twin did
(docs/language.md, "One kernel for every dtype and warp count")."""

from __future__ import annotations

import pytest

import hk
from hk.ops import fused, quant

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


def _ref(a, b):
    return (torch.nn.functional.silu(a.float()) * b.float()).to(a.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("warps", [1, 4])
def test_silu_mul_picks_T_from_the_tensors(dtype, warps):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(64, 512, device="cuda", dtype=dtype, generator=g)
    b = torch.randn(64, 512, device="cuda", dtype=dtype, generator=g)
    o = torch.empty_like(a)
    fused.silu_mul_kernel(a, b, o, warps=warps)
    torch.testing.assert_close(o, _ref(a, b), rtol=1e-2, atol=1e-2)
    # the same launch as the table's specialization, bit for bit
    o2 = torch.empty_like(a)
    s = {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]
    fused.KERNELS[f"silu_mul_{s}_w{warps}"](a, b, o2)
    assert torch.equal(o, o2)


def test_explicit_T_and_constexprs():
    a = torch.randn(32, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(32, 1024, device="cuda", dtype=torch.float16)
    o = torch.empty_like(a)
    fused.silu_mul_kernel(a, b, o, T=hk.fp16, warps=2, TILES=2)
    torch.testing.assert_close(o, _ref(a, b), rtol=1e-2, atol=1e-2)


def test_disagreeing_tensors_are_refused():
    a = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    o = torch.empty_like(a)
    with pytest.raises(TypeError, match="disagree on T"):
        fused.silu_mul_kernel(a, b, o)


def test_a_cached_launch_still_checks_every_tensor():
    a = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(a)
    fused.silu_mul_kernel(a, a, o)                      # caches the bf16 launch, keyed on `a`
    bad = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    with pytest.raises((RuntimeError, TypeError)):
        fused.silu_mul_kernel(a, bad, o)                # same key; the launcher refuses `b`


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantize_round_trip_on_the_generic_kernels(dtype):
    x = torch.randn(48, 640, device="cuda", dtype=dtype)
    q, s = quant.quantize(x)                                   # quantize_kernel, T read from x
    err = (x.float() - q.float() * s.unsqueeze(-1)).abs()
    assert bool((err <= s.unsqueeze(-1) * 0.5 * (1 + 1e-5) + 1e-6).all())   # half a step at most
    y = quant.dequantize(q, s, dtype=x.dtype)                  # dequantize_kernel, T read from o
    assert y.dtype == x.dtype
    torch.testing.assert_close(y, (q.float() * s.unsqueeze(-1)).to(dtype))
