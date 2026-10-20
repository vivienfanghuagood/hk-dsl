"""hk.templates.gemm's kernels, built: an epilogue costs the GEMM no wave.

The template's promise is the plain kernel's speed with the epilogue on
top; the first way to lose it is to lose occupancy -- an epilogue's
registers pushing the kernel past a granule, so a SIMD holds fewer waves
through the whole K loop. Each kernel here is built with the ready-made
epilogues and held to the plain kernel's occupancy, and to no spill (which
the build refuses anyway). Compile-only: hipcc, no GPU.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.ops import gemm as rdna  # noqa: E402
from hk.ops import gemm_cdna, gemm_cdna_dp  # noqa: E402

T = hk.templates.gemm
TG = importlib.import_module("hk.templates.gemm")


def _ep(fn, kinds):
    return TG.Epilogue(fn, kinds, {n: hk.bf16 for n in kinds}, {}, hk.bf16)


EPILOGUES = {
    "bias_gelu": lambda: _ep(T.bias_gelu, {"bias": "row"}),
    "residual": lambda: _ep(T.residual, {"residual": "tile"}),
}

STEPS = {
    "gfx1100-wmma": ("gfx1100", lambda: T.plan(4096, 4096, 4096, arch="gfx1100")[0]),
    "gfx950-pingpong": ("gfx950", lambda: TG.Step(
        "pp", gemm_cdna.schedule_for(4096, 4096, 4096), (0, 4096), True)),
    "gfx950-dp128": ("gfx950", lambda: TG.Step(
        "dp", {**gemm_cdna_dp.DEFAULT, **gemm_cdna.CANDIDATES["dp128x0"][1]},
        (0, 4096), True)),
}


def _occupancy(kernel):
    (k,) = kernel.build().kernels
    assert k.scratch == 0
    return k.occupancy


@pytest.mark.parametrize("step", list(STEPS))
@pytest.mark.parametrize("epilogue", list(EPILOGUES))
def test_an_epilogue_keeps_the_kernels_occupancy(step, epilogue):
    arch, make = STEPS[step]
    st = make()
    plain = T.kernel_for(st, None, arch=arch)
    fused = T.kernel_for(st, EPILOGUES[epilogue](), arch=arch)
    assert _occupancy(fused) >= _occupancy(plain)
