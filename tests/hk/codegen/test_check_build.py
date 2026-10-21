"""hk.check's build step: hipcc, no GPU. The resources per instantiation
come from the compiler's remarks, as `k.build().kernels` reports them."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]


def test_build_only_report():
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul, gpu=False)
    assert r.status == "built", str(r)
    (b,) = r.builds
    (i,) = b["instantiations"]
    assert i["vgpr"] > 0 and i["spill"] == 0 and i["scratch"] == 0
    assert 1 <= i["occupancy"] <= i["max_waves_per_simd"]
    assert i["occupancy"] == min(i["occupancy_by_regs"], i["occupancy_by_lds"])
    assert Path(b["source"]).exists()
    assert "vgpr=" in str(r)


def test_cli_build_only_for_two_archs():
    fixture = HERE / "check_fixture.py"
    proc = subprocess.run(
        [sys.executable, "-m", "hk.check", f"{fixture}:gelu_mul", "--build-only",
         "--arch", "gfx1100,gfx950", "--json", "-", "--quiet"],
        capture_output=True, text=True, cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "python")})
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout)
    assert [r["arch"] for r in out["reports"]] == ["gfx1100", "gfx950"]
    assert all(r["status"] == "built" for r in out["reports"])
    waves = {r["arch"]: r["builds"][0]["threads"] for r in out["reports"]}
    assert waves == {"gfx1100": 32, "gfx950": 64}
