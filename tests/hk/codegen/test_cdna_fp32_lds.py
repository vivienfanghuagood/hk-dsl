"""fp32 register tiles through LDS on gfx950 compile to ds_read / ds_write, not FLAT.

include/hk/cdna.cuh's store(st<float>, rt<float>) and load(rt<float>, st<float>) (the cross-warp
combine of fp32 partial sums: hk.ops.decode_gemm, w4a16, moe) address the tile element by element.
Through a generic pointer the compiler cannot tell the LDS from global memory and emits FLAT
accesses:
- they are slower to issue;
- they count in vmcnt as well as lgkmcnt, so each wait for an LDS value also waited for every
  global load in flight -- in a megakernel, its weight prefetch.

DeepSeek-V4.1's K1 port had 271 flat loads and 171 flat stores and 9 ds ops, and ran 0.4 us
slower. Compile-only: hipcc, no GPU.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.target import gfx950  # noqa: E402


def _isa(k) -> str:
    with tempfile.TemporaryDirectory() as d:
        cpp, asm = Path(d) / "k.cpp", Path(d) / "k.s"
        cpp.write_text(k.source(scaffold="bare"))
        cmd = [hc.hipcc(), *[f for f in hc.flags(gfx950.GFX950, "probe", k.build_flags(k.trace()))
                             if f not in ("-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage")],
               "--offload-device-only", "-S", str(cpp), "-o", str(asm)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        text = asm.read_text()
    at = re.search(r"^_Z\w*_kernel\w*:", text, re.M).start()
    # the whole function: clang 24 places an early-exit block (and its s_endpgm) before the
    # block that reads LDS, so the first s_endpgm is not its end
    return text[at:text.index(".Lfunc_end", at)]


@pytest.mark.hipcc
@pytest.mark.parametrize("layout", ["row", "col"])
def test_an_fp32_tile_through_lds_is_ds_reads_and_writes(layout):
    st = hk.ST(hk.fp32, 16, 16)
    rt = hk.RT(hk.fp32, 16, 16, layout)

    @hk.kernel(arch="gfx950", warps=2, grid=lambda p: (1, 1, 1), name=f"fp32_lds_{layout}")
    def k(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        s = hk.lds.alloc(st, name="s", count=2)
        w = hk.arch.warp_id()
        t = x.at[0, 0].load(rt)
        for i in range(2):
            if w == i:
                s[i].store(t)
        hk.arch.barrier(drain=True)
        if w == 0:
            o.at[0, 0] = s[0].load(rt) + s[1].load(rt)

    text = _isa(k)
    assert not re.findall(r"\bflat_\w+", text), re.findall(r"\bflat_\w+", text)[:4]
    assert re.findall(r"\bds_write\w*", text) and re.findall(r"\bds_read\w*", text)


@pytest.mark.hipcc
@pytest.mark.parametrize("frag", [None, (32, 32)])
def test_an_fp32_vector_through_lds_is_ds_reads_and_writes(frag):
    """hk.SharedVec (hk.collective.cross_warp, a vector read back in another tile's layout): include/cdna4's
    load(rv, sv) / store(sv, rv) indexed sv::data through a generic pointer -- FLAT, counted in vmcnt, so
    an attention kernel's per-block max exchange waited for its next KV block's DMA."""
    rt = hk.RT(hk.fp32, 32, 32, "col", **({"frag": frag} if frag else {}))
    vt = hk.row_vec(rt)

    @hk.kernel(arch="gfx950", warps=4, grid=lambda p: (1, 1, 1), name=f"fp32_sv_{frag is None}")
    def k(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        t = x.at[0, 0].load(rt)
        m = hk.collective.cross_warp(t.max(axis=0), "max", 4)
        t *= m
        o.at[0, 0].store(t)

    text = _isa(k)
    assert not re.findall(r"\bflat_\w+", text), re.findall(r"\bflat_\w+", text)[:4]
    assert re.findall(r"\bds_write\w*", text) and re.findall(r"\bds_read\w*", text)


@pytest.mark.hipcc
def test_a_bf16_col_tile_stored_to_lds_is_ds_writes():
    """A 16-bit col-layout tile (an accumulator converted to bf16: attention's P handed to the other warps)
    -> LDS: include/cdna4's store wrote it element by element through `U *` into st::data -- FLAT."""
    rt = hk.RT(hk.fp32, 16, 32, "col")

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1), name="bf16_col_lds")
    def k(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 16, 32), name="s")
        s.store(x.at[0, 0].load(rt).to(hk.bf16))
        hk.arch.barrier(drain=True)
        o.at[0, 0].store(s.load(hk.RT(hk.bf16, 16, 32, "col", frag=(16, 32))))

    text = _isa(k)
    assert not re.findall(r"\bflat_\w+", text), re.findall(r"\bflat_\w+", text)[:4]
    assert re.findall(r"\bds_write_b16\w*", text)
