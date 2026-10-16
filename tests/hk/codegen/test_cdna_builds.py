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
    from hk.ops import norm, quant  # noqa: PLC0415

    return [("vector_add", add), ("rmsnorm", rms), ("softmax", sm),
            ("mma_reg", mma), ("gemm_lds", lds),
            # include/hk/cdna.cuh's memory ops: bounded rows and a vector
            # weight; int8 and store_scalar; Philox; rows by index.
            ("addrmsw", norm.KERNELS["addrmsw_bf16_w4"]),
            ("quantize", quant.KERNELS["quantize_bf16_w1"]),
            ("dropout", dict(_catalog.kernels())["dropout"]),
            ("embedding", dict(_catalog.kernels())["embedding"]),
            ("kv_scatter", dict(_catalog.kernels())["kv_scatter"])]


NAMES = ["vector_add", "rmsnorm", "softmax", "mma_reg", "gemm_lds",
         "addrmsw", "quantize", "dropout", "embedding", "kv_scatter"]


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
@pytest.mark.parametrize("name", NAMES)
def test_it_builds_without_spilling(arch, name):
    # The device scaffold: the code-object launcher's, whose layout table
    # names every field of the strided gl.
    k = dict(_kernels())[name]
    k = hk.kernel(k.fn, arch=arch, warps=k.warps, grid=k.grid_fn, name=k.name,
                  defines=getattr(k, "defines", None))
    b = hc.build(k.with_defines(scaffold.device(k.trace())), arch, name=k.name, kind="co")
    (r,) = b.kernels
    assert not r.spills
    assert r.occ_by_reg >= 7
