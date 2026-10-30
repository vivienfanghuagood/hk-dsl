"""gfx1100's decode GEMM, built: every shipped schedule builds without spilling and runs at the
occupancy it carries (hk.ops.decode_gemm.SCHEDULES).

Under the TheRock ROCm 7.14 compiler (AMD clang 23) the four-tile kernel spilled (vgpr 256 -> 264,
388 B of scratch a lane) and the spill gate refused it, so the production vLLM images routed no
GEMM to hk. The cause was include/rdna3's fp32 LDS reads through generic pointers -- a FLAT load and
a 64-bit address an element in the warp-0 combine; through LDS pointers the kernel is 194 VGPRs
there. This test holds the build under whichever hipcc runs it (CI's clang 22; the gfx1100 images'
clang 23 when run there), and the occupancy caps the schedules were measured at.
Compile-only: hipcc, no GPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.ops import decode_gemm as D  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.target import get_target  # noqa: E402

#: K of the production models' decode GEMMs (MiniCPM5-2B's 2048 and 6144, Qwen3-8B's 4096, 12288)
KS = (2048, 4096, 6144, 12288)


def _build(tokens: int, k: int):
    s = D.schedule_for(16 * tokens, k)
    kern = D.decode_gemm_kernel(
        f"dg_build_t{tokens}_k{k}", tokens=tokens, slices=s["slices"], warps=s["warps"],
        combine_depth=s["combine_depth"], resident=s["resident"], splits=s["splits"],
        occupancy=s.get("occupancy", 0), arch="gfx1100")
    tr = kern.trace()
    b = hc.build(kern.with_defines(scaffold.device(tr)), "gfx1100", name=tr.name, kind="co",
                 max_vgprs=256)
    return s, b.kernels[0]


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("tokens", sorted(D.SCHEDULES))
def test_every_shipped_schedule_builds_without_spilling(tokens, k):
    s, r = _build(tokens, k)
    assert r.scratch == 0 and not r.spills, r.line()
    cap = s.get("occupancy", 0)
    if cap:
        # the cap is a floor on the registers: at most `cap` waves a SIMD, and no fewer than
        # the cap allows (it is not an accident of a bigger register need)
        assert get_target("gfx1100").occupancy(r.vgpr) == cap, r.line()


def test_an_occupancy_cap_is_a_vgpr_floor_in_the_source():
    k = D.decode_gemm_kernel("dg_cap_src", tokens=2, slices=2, warps=16, combine_depth=8,
                             occupancy=6, arch="gfx1100")
    src = k.with_defines(scaffold.device(k.trace()))
    # 6 waves of 1536 VGPRs at a 24-register granule: 217 registers or more (v216)
    assert 'asm volatile("" ::: "v216");' in src
    with pytest.raises(ValueError, match="max_occupancy=4"):
        D.decode_gemm_kernel("dg_cap_low", tokens=2, slices=2, warps=16, occupancy=4,
                             arch="gfx1100").with_defines(src)
