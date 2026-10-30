"""docs/supported-hardware.md is generated from the target table, the route table and the GPU
tier's runners: it must be regenerated when any of them changes, and the three must agree with
each other for the page to be true."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_hardware_docs.py"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_hardware_docs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_page_is_current():
    r = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0, (
        "docs/supported-hardware.md is stale; run `python3 scripts/gen_hardware_docs.py` and "
        "commit the result:\n" + r.stdout[-4000:] + r.stderr[-2000:])


def test_every_target_names_its_gpus():
    target, _, mods = _gen()._load()
    for arch, t in target.TARGETS.items():
        m = mods[arch]
        assert m.PARTS, arch
        # measured on a part exactly when the target says its numbers are measured
        assert bool(m.MEASURED_ON) == t.verified, arch
        assert not m.MEASURED_ON or m.MEASURED_ON in m.PARTS, arch


def test_ci_runs_on_every_measured_target():
    gen = _gen()
    target, _, _ = gen._load()
    ci = gen.ci_gpu_arches(gen.GPU_WORKFLOW.read_text(encoding="utf-8"))
    assert ci, "no GPU-tier runner found in gpu.yml: the pattern no longer matches"
    assert set(ci) <= set(target.TARGETS), ci
    assert set(ci) == {a for a, t in target.TARGETS.items() if t.verified}, ci


def test_routes_name_known_arches_and_engines():
    gen = _gen()
    target, switches, _ = gen._load()
    for r in switches.ROUTES.values():
        assert set(r.arches) <= set(target.TARGETS), r.name
        assert r.engine in gen.ENGINES, r.name
        assert r.default or not r.arches, f"{r.name}: arches only matter to a default-on route"


def test_ci_label_pattern():
    gen = _gen()
    text = ("jobs:\n  a:\n    runs-on: [self-hosted, hk-gpu, gfx950]\n"
            "  b:\n    runs-on: [self-hosted, hk-gpu, gfx1100-x4]\n"
            "  c:\n    runs-on: ubuntu-latest\n")
    assert gen.ci_gpu_arches(text) == ["gfx1100", "gfx950"]
