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
    return text[at:text.index("s_endpgm", at)]


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
