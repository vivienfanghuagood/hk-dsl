"""gfx950's paged attention, built: the gate (no scratch, the occupancy the
LDS allows), the code-object waits gate, and the copies counted against the
ISA, as tests/hk/codegen/test_cdna_attn_builds.py does for attention.

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
from hk.ops import attn_cdna, paged_cdna  # noqa: E402
from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def _objdump() -> str:
    for c in (shutil.which("llvm-objdump"), str(hc.rocm_path() / "llvm" / "bin" / "llvm-objdump")):
        if c and Path(c).exists():
            return c
    pytest.skip("no llvm-objdump")


def _build(k, **gate):
    tr = k.trace()
    src = scaffold.device(tr)
    b = hc.build(src, "gfx950", name=tr.name, kind="co", **gate)
    (kr,) = b.kernels
    assert kr.scratch == 0 and not kr.spills
    asm = subprocess.run([_objdump(), "-d", str(b.so_path)], capture_output=True,
                         text=True, check=True).stdout
    (ck,) = cocheck.check(b.so_path)
    assert ck.ok, ck.line()
    return src, asm, kr


@pytest.mark.parametrize("d", paged_cdna.HEAD_DIMS)
@pytest.mark.parametrize("mode,splits", [("plain", 1), ("fused", 8)])
def test_the_decode_builds_and_its_page_copies_are_counted(d, mode, splits):
    src, asm, kr = _build(paged_cdna.decode_kernel(
        f"pc_build_{mode}_{d}", head_dim=d, splits=splits, mode=mode, arch="gfx950"))
    # A page of K or V is 32 x D bf16 over one warp: D / 16 16-byte copies.
    copies = src.count("kittens::group_load_async<1, true>(")
    assert len(re.findall(r"buffer_load_dwordx4 .* sc0 nt lds", asm)) == copies * (d // 16)
    # The page table and the length through the scalar cache.
    assert re.search(r"s_load_dword\b", asm)


@pytest.mark.parametrize("d", attn_cdna.HEAD_DIMS)
@pytest.mark.parametrize("kv_split", [False, True])
def test_the_paged_prefill_builds_within_two_waves(d, kv_split):
    src, asm, kr = _build(attn_cdna.attn_kernel(
        f"pp_build_{d}_{int(kv_split)}", head_dim=d, causal=True, paged=True,
        kv_split=kv_split, arch="gfx950"),
        max_vgprs=256, min_occupancy=2)
    assert kr.occupancy >= 2
    warps = 8 if d == 128 else 4
    copies = src.count(f"kittens::group_load_async<{warps}>(")
    # A 32 x D page over `warps` warps: D * 64 / (1024 * warps) copies each.
    assert len(re.findall(r"buffer_load_dwordx4 .* lds", asm)) == copies * d * 64 // (1024 * warps)


def test_the_prefill_merge_builds():
    _build(attn_cdna.prefill_merge_kernel("pp_merge_build", arch="gfx950"))
