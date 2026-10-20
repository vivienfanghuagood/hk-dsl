"""scoreboard.py --rotate-buffers: the per-arm buffer dicts are found in the
arms' closures and passed one timed arm along each round, and put back."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "hk-bench"))
import scoreboard as sb  # noqa: E402


def _arms():
    outs = {a: f"buf_{a}" for a in sb.ARMS}
    q = {a: f"q_{a}" for a in sb.ARMS}
    res = {}
    seen = {}

    def run(a):
        return lambda: seen.__setitem__(a, (outs[a], q[a]))

    arms = {a: (run(a), None, None) for a in ("hk", "torch", "triton")}
    arms["inductor"] = (lambda: res.__setitem__("inductor", 1), None, None)
    arms["flydsl"] = "no kernel"
    return arms, outs, q, seen


def test_the_buffers_are_found_in_the_closures():
    arms, outs, q, _ = _arms()
    found = sb._arm_buffers(arms)
    assert any(d is outs for d in found) and any(d is q for d in found)


def test_each_round_every_timed_arm_takes_the_next_ones_buffer():
    arms, outs, q, seen = _arms()
    timed = ["hk", "torch", "triton"]
    rotate = sb.buffer_rotation(arms, timed)
    rotate(1)
    for a in timed:
        arms[a][0]()
    assert seen == {"hk": ("buf_torch", "q_torch"), "torch": ("buf_triton", "q_triton"),
                    "triton": ("buf_hk", "q_hk")}
    rotate(0)
    assert outs["hk"] == "buf_hk" and q["triton"] == "q_triton"
    assert outs["flydsl"] == "buf_flydsl"


def test_one_owner_is_nothing_to_rotate():
    arms, _, _, _ = _arms()
    assert sb.buffer_rotation(arms, ["hk"]) is None
