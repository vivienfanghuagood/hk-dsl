"""hk.check on a GPU: a right kernel passes, a wrong one fails with the
first mismatching index, a function's kernels are found, and the timing has
every contender."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def test_raw_kernel_passes_and_is_timed():
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul, fx.reference, fx.inputs, rounds=3, burn=0)
    assert r.status == "pass", str(r)
    (c,) = r.cases
    (o,) = c["correct"]["outputs"]
    assert o["ok"] and o["mismatches"] == 0 and o["max_abs"] < 0.05
    assert set(c["time"]["ms"]) == {"hk", "eager", "compile"}
    for k in ("eager", "compile"):
        assert c["time"]["vs"][k]["verdict"] in ("win", "loss", "tie", "noise")
    assert c["first_call_s"]["hk"] > 0
    assert r.builds[0]["instantiations"][0]["spill"] == 0


def test_wrong_kernel_fails_at_the_first_unwritten_element():
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul_wrong, fx.reference, fx.inputs, time=False)
    assert r.status == "fail" and r.exit_code == 1
    (o,) = r.cases[0]["correct"]["outputs"]
    assert not o["ok"]
    # 64x1024: the last column block (columns 960..1023) is never written.
    assert o["first_mismatch"]["index"] == [0, 960]
    assert o["mismatches"] == 64 * 64
    assert "first at (0, 960)" in str(r)


def test_function_and_shapes():
    import hk

    import check_fixture as fx

    cases = [lambda device, s=s: fx.fn_inputs(device, s) for s in fx.SHAPES]
    r = hk.check(fx.gelu_mul_fn, fx.reference, cases=cases, baselines=("eager",),
                 extra={"ops.gelu_then_mul": lambda a, b: hk.ops.mul(
                     hk.ops.gelu(a.float()).to(a.dtype), b)},
                 rounds=3, burn=0)
    assert r.status == "pass", str(r)
    assert [b["kernel"] for b in r.builds] == ["gelu_mul"]
    assert len(r.cases) == 2
    assert set(r.cases[0]["time"]["ms"]) == {"hk", "eager", "ops.gelu_then_mul"}


def test_cli_json(capsys):
    from hk.check.__main__ import main

    code = main([f"{HERE / 'check_fixture.py'}:gelu_mul", "--rounds", "3", "--burn", "0",
                 "--json", "-", "--quiet"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["status"] == "pass"
    assert out["cases"][0]["correct"]["ok"]


def test_cli_ref_expression_over_an_op(capsys):
    from hk.check.__main__ import main

    code = main(["hk.ops.silu_mul", "--ref", "F.silu(a) * b", "--inputs",
                 "bf16[64,4096], bf16[64,4096]", "--no-time", "--json", "-", "--quiet"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    assert out["builds"] and out["cases"][0]["correct"]["ok"]


def test_inplace_arguments():
    """vLLM's fused_add_rms_norm updates x and residual in place: --inplace
    gives the reference their values before the call and compares both."""
    import torch

    import hk

    def reference(x, r, w):
        s = x + r
        return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * w, s

    def inputs(device):
        x = torch.randn(64, 4096, device=device, dtype=torch.bfloat16)
        return x, torch.randn_like(x), torch.randn(4096, device=device, dtype=torch.bfloat16)

    r = hk.check(hk.ops.fused_add_rms_norm, reference, inputs, inplace=[0, 1],
                 time=False)
    assert r.status == "pass", str(r)
    assert len(r.cases[0]["correct"]["outputs"]) == 2


def test_sweep_times_each_variant():
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul_strip, fx.reference, fx.inputs, sweep={"CW": [1024]},
                 rounds=3, burn=0)
    assert r.status == "pass", str(r)
    t = r.cases[0]["time"]
    assert "hk[CW=1024]" in t["ms"] and t["best"] in ("hk", "hk[CW=1024]")
    assert r.cases[0]["correct"]["variants"]["hk[CW=1024]"]["ok"]
