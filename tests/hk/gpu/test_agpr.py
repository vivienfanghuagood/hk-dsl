"""hk.arch.agpr on gfx950: an MMA accumulator held in AGPRs computes bitwise what the VGPR form
does -- the same MFMAs on the same operands in the same order (include/hk/cdna_agpr.cuh)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="AGPRs are gfx950's here")]


@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,h,hkv,n", [(1, 8, 2, 1024), (2, 4, 4, 777), (1, 16, 2, 4096)])
def test_attention_with_o_in_agprs_is_bitwise_the_vgpr_form(d, causal, b, h, hkv, n):
    from hk.ops import attn_cdna
    from hk.runtime.resources import ResourceError

    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(b, x, n, d, device="cuda", dtype=torch.bfloat16, generator=g)
               for x in (h, hkv, hkv))
    o_v, o_a = torch.empty_like(q), torch.empty_like(q)
    attn_cdna.kernel_for(d, causal)(q, k, v, o_v)
    try:
        attn_cdna.kernel_for(d, causal, acc="agpr")(q, k, v, o_a)
    except ResourceError as e:
        # LLVM 22 (rocm 7.2.3) splits the registers 128 V / 128 A once AGPRs
        # are used, and the d128 kernel needs more VGPRs than that; LLVM 23
        # gives V = 256 - A. The gate refuses the spill; nothing to compare.
        if "spill" in str(e):
            pytest.skip("this toolchain spills the AGPR form at this shape")
        raise
    torch.cuda.synchronize()
    assert torch.equal(o_a, o_v)
    assert torch.isfinite(o_a.float()).all()
