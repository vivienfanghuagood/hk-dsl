"""scripts/ci/summarize.py: a GPU tier whose tests saw no GPU is a failure, whatever pytest's exit code.

A container that could not open /dev/kfd (the host's video/render groups not added) once made every GPU
test skip, and the tier reported "success: 1 passed, 1776 skipped"."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _summarize(tmp_path, groups):
    suite = tmp_path / "gpu"
    suite.mkdir()
    lines = []
    for name, summary in groups.items():
        (suite / f"{name}.log").write_text(f"....\n{summary}\n")
        lines.append(f"{name} exit 0")
    (suite / "status").write_text("\n".join(lines + ["done"]) + "\n")
    out = tmp_path / "run"
    out.mkdir()
    subprocess.run([sys.executable, str(REPO / "scripts" / "ci" / "summarize.py"), str(suite),
                    "--sha", "0" * 40, "--tier", "gfx950", "--out", str(out)],
                   check=False, capture_output=True, text=True)
    return json.loads((out / "result.json").read_text())


def test_a_tier_that_skipped_nearly_everything_fails(tmp_path):
    r = _summarize(tmp_path, {"torch": "1 passed, 1776 skipped in 5.00s"})
    assert r["state"] == "failure"
    assert "saw no GPU" in r["description"]


def test_a_tier_where_nothing_passed_fails(tmp_path):
    r = _summarize(tmp_path, {"torch": "40 skipped in 1.00s"})
    assert r["state"] == "failure" and "no test passed" in r["description"]


def test_a_normal_tier_stays_green(tmp_path):
    r = _summarize(tmp_path, {"attn": "70 passed, 3 skipped in 99.38s",
                              "torch": "58 passed, 21 skipped, 15 warnings in 10.85s"})
    assert r["state"] == "success", r["description"]
