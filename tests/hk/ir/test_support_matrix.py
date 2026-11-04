"""docs/support-matrix.md is generated from CI's files and the recorded engine facts
(scripts/gen_support_matrix.py): it must be regenerated when either changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_support_matrix.py"


def test_support_matrix_is_current():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, (
        "docs/support-matrix.md is stale; run `python3 scripts/gen_support_matrix.py` and commit "
        "the result:\n" + r.stdout[-4000:] + r.stderr[-2000:])


def test_every_arch_has_a_gpu_tier_and_an_engine_row():
    text = (REPO / "docs" / "support-matrix.md").read_text(encoding="utf-8")
    tests, engines = text.split("## Serving engines", 1)
    for arch in ("gfx1100", "gfx1201", "gfx950"):
        assert f"| {arch} |" in tests, arch
        assert f"### {arch} (" in engines, arch
