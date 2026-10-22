"""hk.profile on a GPU: a small kernel profiled end to end on this arch.

Needs rocprofv3 (skipped without it). The thread trace runs where its
decoder is installed (docs/profile.md#the-thread-trace) and is a note
otherwise; on a machine whose counters read zero (the W7900D pod) the
report says so and keeps the timing and the ISA -- either way the call is
timed, its kernel found, its code disassembled and a verdict written.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
FIXTURE = HERE / "check_fixture.py"


def _rocprofv3():
    sys.path.insert(0, str(HERE.parents[1] / "python"))
    from hk.profile.run import rocprofv3  # noqa: PLC0415

    return rocprofv3()


@pytest.fixture(scope="module")
def strip_profile():
    if _rocprofv3() is None:
        pytest.skip("rocprofv3 is not installed")
    import hk

    return hk.profile(f"{FIXTURE}:gelu_mul_strip",
                      "bf16[256,4096], bf16[256,4096], bf16[256,4096]",
                      iters=5, timeout=240)


def test_a_kernel_is_timed_counted_and_judged(strip_profile):
    p = strip_profile
    assert p.ok, str(p)
    (k,) = p.kernels
    assert k["name"].startswith("gelu_mul_strip") and k["hk"]
    assert k["time"]["us"] > 0 and k["time"]["samples"] == 5
    assert k["grid"]["workgroups"] == 256 * (4096 // 512)
    assert k["isa"]["total"]["classes"]["vmem_load"] >= 2
    assert k["verdict"]["line"]
    assert p.sources["timing"].startswith("rocprofv3")
    if p.sources["counters"].startswith("rocprofv3"):
        # The counters ran: the waves are the grid's, the bytes the tensors'.
        assert k["counters"]["SQ_WAVES"] == k["grid"]["workgroups"] * k["grid"][
            "waves_per_workgroup"]
        assert k["roofline"]["bytes"] > 0
    else:
        assert any("counter" in n for n in p.notes), p.notes
    json.dumps(p.json())


def test_the_cli_writes_json(tmp_path):
    if _rocprofv3() is None:
        pytest.skip("rocprofv3 is not installed")
    import os

    out = tmp_path / "p.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(HERE.parents[1] / "python"), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        [sys.executable, "-m", "hk.profile", f"{FIXTURE}:gelu_mul",
         "--inputs", "bf16[64,1024], bf16[64,1024], bf16[64,1024]",
         "--no-counters", "--no-att", "--iters", "3", "--json", str(out), "-q"],
        capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(out.read_text())
    assert doc["schema"] == 1 and doc["tool"] == "hk.profile"
    (k,) = doc["kernels"]
    assert k["time"]["us"] > 0 and doc["verdict"]["line"]
    assert doc["sources"]["counters"] == "off"


def test_without_rocprofv3_it_times_the_call_and_reads_the_code(monkeypatch):
    """No profiler: the call timed in-process and its kernel disassembled,
    with a note that says what is missing."""
    import hk
    import hk.profile.run as run

    monkeypatch.setattr(run, "rocprofv3", lambda: None)
    p = hk.profile(f"{FIXTURE}:gelu_mul_strip",
                   "bf16[256,4096], bf16[256,4096], bf16[256,4096]")
    assert p.ok, str(p)
    (k,) = p.kernels
    assert k["time"]["us"] > 0 and k["isa"]["total"]["classes"]["vmem_load"] >= 2
    assert any("rocprofv3 not found" in n for n in p.notes)
    assert p.sources["counters"].startswith("off")
