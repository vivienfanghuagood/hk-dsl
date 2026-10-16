"""gfx950's GEMM, built: the gate, the code-object waits gate, and the copy
count hk-lds-pipeline derives its vmcnts from, held to the ISA.

A derived vmcnt is only right if the count is: a copy counted high makes
every wait issued while it is in flight too shallow. hk counts a 128x64 bf16
group load over 512 threads as two 16-byte copies a thread; the code object
has to issue exactly that many buffer_load_dwordx4 ... lds per copy.
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
from hk.ops import gemm_cdna  # noqa: E402
from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def _objdump() -> str:
    for c in (shutil.which("llvm-objdump"), str(hc.rocm_path() / "llvm" / "bin" / "llvm-objdump")):
        if c and Path(c).exists():
            return c
    pytest.skip("no llvm-objdump")


@pytest.mark.parametrize("block_n", [256, 128])
def test_the_gemm_builds_within_two_waves_and_its_copies_are_counted(block_n):
    k = gemm_cdna.gemm_kernel(f"gemm_cdna_build_{block_n}",
                              **{**gemm_cdna.DEFAULT, "block_n": block_n}, arch="gfx950")
    tr = k.trace()
    copies = len(re.findall(r"hk.group_load .*async", tr.dump()))
    b = hc.build(scaffold.device(tr), "gfx950", name=tr.name, kind="co",
                 max_vgprs=256, min_occupancy=2)
    (kr,) = b.kernels
    assert kr.scratch == 0 and kr.occupancy >= 2
    asm = subprocess.run([_objdump(), "-d", str(b.so_path)], capture_output=True,
                         text=True, check=True).stdout
    dma = len(re.findall(r"buffer_load_dwordx4 .* lds", asm))
    # A half of A is always 128x64 (two copies a thread); a half of B is
    # 128x64 or, at block_n 128, 64x64 (one).
    per_b = 2 if block_n == 256 else 1
    a_copies, b_copies = copies // 2, copies - copies // 2
    assert dma == 2 * a_copies + per_b * b_copies
    (ck,) = cocheck.check(b.so_path)
    assert ck.ok, ck.line()
