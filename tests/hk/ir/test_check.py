"""hk.check without a GPU or hipcc: the input specs, the comparison and
verdict rules, and a trace-only report. The build tier is
tests/hk/codegen/test_check_build.py, the GPU tier tests/hk/gpu/test_check.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_spec_parser():
    from hk.check.__main__ import parse_spec

    items = parse_spec("bf16[4096,14336], i32[64]@randint(0,128), fp32[2,3]@randn*0.5, 1e-6")
    assert items[0] == {"dtype": "bfloat16", "shape": [4096, 14336], "fill": None,
                        "args": [], "scale": None}
    assert items[1]["fill"] == "randint" and items[1]["args"] == [0, 128]
    assert items[2]["scale"] == 0.5
    assert items[3] == {"scalar": 1e-6}
    with pytest.raises(ValueError, match="dtype"):
        parse_spec("bf17[4]")
    with pytest.raises(ValueError, match="expected dtype"):
        parse_spec("bf16(4)")


def test_split_top_ignores_nested_commas():
    from hk.check.__main__ import split_top

    assert split_top("a[1,2], b@f(1,2),3") == ["a[1,2]", "b@f(1,2)", "3"]


def test_verdict_rules():
    from hk.check import verdict

    assert verdict(1.10, 0.02) == "win"
    assert verdict(0.90, 0.02) == "loss"
    assert verdict(1.01, 0.02) == "tie"
    assert verdict(1.30, 0.20) == "noise"     # a band past 15% tells nothing


def test_tolerance_scales_atol():
    from hk.check import TOLERANCES, tolerance

    rtol, atol = tolerance("torch.bfloat16", scale=4.0)
    assert rtol == TOLERANCES["bfloat16"][0]
    assert atol == pytest.approx(TOLERANCES["bfloat16"][1] * 4.0)
    assert tolerance("torch.float32", rtol=0.5, atol=0.25) == (0.5, 0.25)


def test_hk_check_is_callable():
    import hk

    assert callable(hk.check) and hk.check.check is not None


def test_trace_only_report_of_a_raw_kernel():
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul, gpu=False, build=False)
    assert r.status == "built" and r.mode == "trace-only"
    (b,) = r.builds
    assert b["kernel"] == "gelu_mul" and b["warps"] == 1 and b["lds_bytes"] == 0
    assert "instantiations" not in b
    js = json.loads(r.dumps())
    assert js["schema"] == 1 and js["status"] == "built"
    assert "gelu_mul" in str(r)


def test_trace_error_is_reported_not_raised():
    import hk

    @hk.kernel(grid=lambda p: (1, 1, 1))
    def bad(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        x = a[hk.tile_coord(0, 0, 0, 0)].load(hk.RT(hk.bf16, 16, 64))
        o[hk.tile_coord(0, 0, 0, 0)] = x.gelu()       # gelu is fp32-only

    r = hk.check(bad, gpu=False, build=False)
    assert r.status == "error" and r.exit_code == 2
    (e,) = r.errors
    assert e["stage"] == "trace" and "fp32" in e["message"]


def test_cpu_discovery_of_a_function():
    torch = pytest.importorskip("torch")
    del torch
    import hk

    import check_fixture as fx

    r = hk.check(fx.gelu_mul_fn, fx.reference, fx.fn_inputs, gpu=False, build=False)
    assert [b["kernel"] for b in r.builds] == ["gelu_mul"]


def test_cli_trace_only_json(capsys):
    from hk.check.__main__ import main

    fixture = Path(__file__).resolve().parents[1] / "check_fixture.py"
    code = main([f"{fixture}:gelu_mul", "--trace-only", "--json", "-", "--quiet"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["builds"][0]["kernel"] == "gelu_mul"


def test_cpu_discovery_through_a_kept_launcher():
    """A template that keeps its own launch cache calls Kernel._launcher, not
    Kernel.__call__ (hk.templates.elementwise/rowwise do): found too."""
    pytest.importorskip("torch")
    import hk

    import check_fixture as fx

    def fn(a, b):
        o = a.new_empty(a.shape)
        fx.gelu_mul._launcher()(a, b, o)
        return o

    r = hk.check(fn, fx.reference, fx.fn_inputs, gpu=False, build=False)
    assert [b["kernel"] for b in r.builds] == ["gelu_mul"]


@pytest.mark.arch("gfx1100")  # gfx1100's strip chunk
def test_sweep_drops_a_variant_that_does_not_trace():
    import hk

    import check_fixture as fx

    # gfx1100's strip chunk at run 8 is 32 lanes x 8 = 256 columns: 128 is not one.
    r = hk.check(fx.gelu_mul_strip, gpu=False, build=False, sweep={"CW": [128, 1024]})
    assert r.status == "built"
    assert [b.get("variant") for b in r.builds] == [None, "hk[CW=1024]"]
    assert any("hk[CW=128] does not build" in n for n in r.notes)
