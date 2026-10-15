"""The CDNA compile gate: the kernels that run on an MI355X build for gfx950 and
gfx942 through hk's own build, spill gate included.

What ran, and against what, is in docs/design/cdna.md; this holds the build so
a change to the translation or the shim that breaks it fails here, with hipcc
and no GPU. gfx942 is compile-only -- there is no MI300 here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"hk_cdna_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _kernels():
    import _catalog  # noqa: PLC0415

    add = _catalog.load("01_vector_add").add
    rms = _catalog.load("02_rmsnorm").rmsnorm_kernel()
    sm = _catalog.load("03_softmax").softmax_kernel()
    probes = REPO / "tools" / "cdna-probes"
    mma = _load(probes / "mma_reg.py").mma_reg
    lds = _load(probes / "gemm_lds.py").gemm_lds
    return [("vector_add", add), ("rmsnorm", rms), ("softmax", sm),
            ("mma_reg", mma), ("gemm_lds", lds)]


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
@pytest.mark.parametrize("name", ["vector_add", "rmsnorm", "softmax", "mma_reg", "gemm_lds"])
def test_it_builds_without_spilling(arch, name):
    k = dict(_kernels())[name]
    k = hk.kernel(k.fn, arch=arch, warps=k.warps, grid=k.grid_fn, name=k.name)
    b = hc.build(scaffold.bare(k.trace()), arch, name=k.name, kind="co")
    (r,) = b.kernels
    assert not r.spills
    assert r.occ_by_reg == 8
