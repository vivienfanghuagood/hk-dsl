"""HipKittens' CDNA4 GEMM and GQA forward build for gfx950 through hk's build,
spill gate included. Two builds, a few seconds each.

A clean build is not a right answer. Built by ROCm 7.1 the GEMM passes this
gate and computes the wrong rows on an MI355X, so the GPU tier and the
on-device check in hk.ops.hipkittens hold that. And ROCm 7.2's clang 22 does
not compile the attention kernels at all. That is upstream's, and it is
declared below rather than skipped.
"""

from __future__ import annotations

import re

import pytest

from hk.ops import hipkittens as hkk
from hk.runtime.compile import CompileError


def _hip_version():
    m = re.search(r"HIP version: (\d+)\.(\d+)", hkk.hipcc_version())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def test_the_gemm_builds_without_spilling():
    b = hkk.gemm_build()
    (r,) = b.kernels
    assert "micro_tk" in r.name
    assert not r.spills
    # 512 threads, 160 KB of LDS: one workgroup a CU, 2 waves a SIMD.
    assert r.occ_by_reg >= 2


def test_gqa_causal_builds_without_spilling():
    try:
        b = hkk.gqa_build(64, 8, 1024, 128, causal=True)
    except CompileError as e:
        if _hip_version() >= (7, 2) and "ambiguous" in str(e):
            pytest.xfail("HipKittens' CDNA4 attention does not compile with ROCm 7.2's "
                         "clang 22 (its rt/art/rv overloads are ambiguous); it does with 7.1")
        raise
    (r,) = b.kernels
    assert "attend_ker<128>" in r.name
    assert not r.spills
    assert r.occ_by_reg >= 2
