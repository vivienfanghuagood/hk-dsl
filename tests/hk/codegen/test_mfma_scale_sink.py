"""A scaled MFMA whose tile is scatter-added must stay out of the scatter's
lane-divergent `if`, and the code-object checker sees when it does not.

tools/cdna-probes/mfma_scale_sink.py builds K1's stage-2 shape twice: once as
hk generates it, and once without hk_pinned (HK_UNPINNED). roc-7.2.3, the
hipcc in CI's rocm/pytorch 7.2.3 image, sinks every V_MFMA_SCALE of the
unpinned build into the scatter's `if`. The MFMA then runs under a partial
EXEC. In K1 the copies that assemble its operands were moved in with it, and
stage 2 came out 0.28 off. hk.runtime.cocheck warns about those MFMAs. With
the pin it warns about none, on every compiler. Compile-only: hipcc and the
ROCm LLVM tools.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402

PROBE = Path(__file__).resolve().parents[3] / "tools" / "cdna-probes" / "mfma_scale_sink.py"


def _probe():
    spec = importlib.util.spec_from_file_location("hk_mfma_scale_sink", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exec(k):
    return [w for w in k.warnings if "partial EXEC" in w]


def _check(pinned: bool, steps: int):
    p = _probe()
    b = hc.build(p.source(steps, pinned), "gfx950",
                 name=f"mx_sink_{steps}_{'pinned' if pinned else 'unpinned'}", kind="co")
    (k,) = cocheck.check(Path(b.so_path))
    return k


@pytest.mark.parametrize("steps", [2, 6])
def test_the_pinned_scatter_add_keeps_every_mfma_under_the_full_exec(steps):
    k = _check(True, steps)
    assert k.ok and not _exec(k), k.line()


def test_roc_7_2_sinks_the_unpinned_mfmas_into_the_scatters_if():
    if "roc-7.2" not in hc._hipcc_version():
        pytest.skip("the sink is roc-7.2.3's (rocm/pytorch 7.2.3, CI's image); ROCm 10 keeps "
                    "the MFMAs out of the `if` without the pin")
    k = _check(False, 2)
    assert _exec(k) and all("v_mfma_scale" in w for w in _exec(k))
