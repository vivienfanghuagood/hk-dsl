"""gfx950's attention, built: the gate (256 VGPRs, two waves a SIMD, no
scratch), the code-object waits gate, and the copy count hk-lds-pipeline
derives its vmcnts from, held to the ISA.

A 64-row K or V block of D bf16 columns over 512 threads is D / 64 16-byte
copies a thread; the code object has to issue exactly that many
buffer_load_dwordx4 ... lds per copy, or every derived wait is off.
Compile-only: hipcc and llvm-objdump, no GPU.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.ops import attn_cdna  # noqa: E402
from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def _objdump() -> str:
    for c in (shutil.which("llvm-objdump"), str(hc.rocm_path() / "llvm" / "bin" / "llvm-objdump")):
        if c and Path(c).exists():
            return c
    pytest.skip("no llvm-objdump")


@pytest.mark.parametrize("d", attn_cdna.HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True])
def test_it_builds_within_two_waves_and_its_copies_are_counted(d, causal):
    k = attn_cdna.attn_kernel(f"attn_build_d{d}_{int(causal)}", head_dim=d, causal=causal,
                              arch="gfx950")
    tr = k.trace()
    src = scaffold.device(tr)
    copies = src.count("kittens::group_load_async<8>(")
    b = hc.build(src, "gfx950", name=tr.name, kind="co", max_vgprs=256, min_occupancy=2)
    (kr,) = b.kernels
    assert kr.scratch == 0 and not kr.spills and kr.occupancy >= 2
    asm = subprocess.run([_objdump(), "-d", str(b.so_path)], capture_output=True,
                         text=True, check=True).stdout
    dma = len(re.findall(r"buffer_load_dwordx4 .* lds", asm))
    assert dma == copies * (d // 64)
    (ck,) = cocheck.check(b.so_path)
    assert ck.ok, ck.line()
