"""The bind after hk's LDS waits is load-bearing on gfx950, and the
code-object checker sees why.

tools/cdna-probes/bind_hazard.py builds an LDS GEMM step loop as hk
generates it and with every bind removed. Without the bind LLVM hoists the
MFMAs above the s_waitcnt that guards their operands -- on the MI355X every
such build returned NaN -- and hk.runtime.cocheck finds those MFMAs in the
disassembly; with it, none. Compile-only: hipcc and the ROCm LLVM tools.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.runtime import cocheck  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402
from hk.runtime.resources import ResourceError  # noqa: E402

PROBE = Path(__file__).resolve().parents[3] / "tools" / "cdna-probes" / "bind_hazard.py"


def _probe():
    spec = importlib.util.spec_from_file_location("hk_bind_hazard", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("bk,nb", [(32, 2), (64, 4)])
def test_without_the_bind_mfmas_read_registers_the_lds_has_not_filled(bk, nb, monkeypatch):
    bh = _probe()
    tr = bh.gemm_steps(bk, nb).trace()
    src = scaffold.device(tr)
    # build's wait gate refuses the unbound kernel outright ...
    with pytest.raises(ResourceError, match="LDS load is\\s+still filling"):
        hc.build(bh.unbound(src), "gfx950", name=f"{tr.name}_unbound_gated", kind="co")
    # ... and with the gate off (for looking at it), the checker says why
    monkeypatch.setenv("HK_WAIT_GATE", "0")
    found = {}
    for variant, s in (("bound", src), ("unbound", bh.unbound(src))):
        b = hc.build(s, "gfx950", name=f"{tr.name}_{variant}", kind="co")
        (k,) = cocheck.check(Path(b.so_path))
        found[variant] = k
    assert found["bound"].ok and not found["bound"].hazards
    assert found["unbound"].hazards
    assert all("v_mfma" in h for h in found["unbound"].hazards)
