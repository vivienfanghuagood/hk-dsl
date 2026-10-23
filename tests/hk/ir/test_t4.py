"""bench/t4 without a GPU: the scoring rules on synthetic timings, the pass
rule, the kernel and import audits, the routing generator, and the public
shape file's invariants.

score.py's statistics and verdict are the functions a sealed run uses, so a
rule that is wrong here is wrong there.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench.t4 import reference as R  # noqa: E402
from bench.t4 import score, shapes, stats  # noqa: E402

np = pytest.importorskip("numpy")


def _sets(t, noise, n_sets=3, rounds=7, rng=None):
    rng = rng or random.Random(0)
    return [[t * (1 + rng.gauss(0, noise)) for _ in range(rounds)] for _ in range(n_sets)]


# ---------------------------------------------------------------- statistics


def test_best_baseline_and_ratio():
    rng = random.Random(1)
    sets = {"base:a": _sets(0.100, 0.01, rng=rng), "base:b": _sets(0.090, 0.01, rng=rng),
            "sol:x": _sets(0.092, 0.01, rng=rng), "proxy:dense": _sets(0.010, 0.01, rng=rng)}
    got = stats.score_shape(sets, ["base:a", "base:b"], "sol:x")
    assert got["best"] == "base:b"          # the proxy is never a baseline
    assert 0.95 < got["ratio"] < 1.0
    lo, hi = got["band"]
    assert lo <= got["ratio"] <= hi


def test_rounds_faster_than_the_card_are_dropped():
    floor = stats.floor_ms(1e12, "gfx950")            # ~0.35 ms
    sets = [[0.5, 0.5, 0.5, 0.01, 0.5]]
    med, _, kept = stats.summary(sets[0], floor)
    assert med == 0.5 and kept == 4
    assert stats.arm_time([[0.01, 0.02]], floor) == (None, 0.0)


def test_ratio_uses_sets_both_arms_have():
    assert stats.ratio_band([[1.0]], [[None]]) == (None, None, None)
    r, lo, hi = stats.ratio_band([[1.0], [1.0], [1.0]], [[0.5], [1.0], [2.0]])
    assert (r, lo, hi) == (1.0, 0.5, 2.0)


# ---------------------------------------------------------------- the pass rule


def _rows(ratios, correct=True, audit=True):
    return [dict(name=f"s{i}", correct=correct, audit_ok=audit, ratio=r) for i, r in enumerate(ratios)]


def test_verdict_pass_and_each_way_to_fail():
    assert stats.verdict(_rows([1.0, 0.97, 0.96, 1.02]))["passed"]
    v = stats.verdict(_rows([1.2, 1.2, 0.84]))           # geomean fine, one shape too low
    assert not v["passed"] and "min" in v["why"]
    v = stats.verdict(_rows([0.94, 0.94, 0.94]))
    assert not v["passed"] and "geomean" in v["why"]
    rows = _rows([1.0, 1.0])
    rows[1]["correct"] = False
    v = stats.verdict(rows)
    assert not v["passed"] and "wrong" in v["why"] and v["min"] == 0.0
    v = stats.verdict(_rows([1.0, 1.0], audit=False))
    assert not v["passed"] and "audit" in v["why"]
    v = stats.verdict(_rows([1.0, None]))                 # not timed counts as 0
    assert not v["passed"]
    assert not stats.verdict([])["passed"]


def test_verdict_counts_shapes_at_target():
    v = stats.verdict(_rows([0.94, 0.95, 1.10, 0.99]))
    assert v["reached"] == 3 and v["passed"]


def test_noise_alone_does_not_decide_a_tie():
    """A solution exactly as fast as the best baseline, timed with 1-2% noise
    in 3 sets x 7 rounds, lands within 3% of 1.0 every seed."""
    for seed in range(50):
        rng = random.Random(seed)
        b, s = _sets(0.2, 0.015, rng=rng), _sets(0.2, 0.015, rng=rng)
        r, _, _ = stats.ratio_band(b, s)
        assert abs(r - 1) < 0.03


# ---------------------------------------------------------------- audits


@pytest.mark.parametrize("name,kind", [
    ("_Z20grouped_gemm_kernel7globals", "hk"),
    ("grouped_gemm_kernel(globals)", "hk"),
    ("void hk_flat::moe_kernel(int*, float*)", "hk"),
    ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1", "foreign"),
    ("_body", "foreign"),
    ("_gmm_kernel", "foreign"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int> >(int, ...)", "torch"),
    ("void at::native::tensor_kernel_scan_innermost_dim<int, std::plus<int> >(...)", "torch"),
    ("void at::native::(anonymous namespace)::addmm_gemm_kernel(...)", "foreign"),
    ("Memset (Device)", "torch"),
    ("ck::kernel_grouped_gemm_xdl<...>", "foreign"),
])
def test_kernel_classes(name, kind):
    assert score.classify_kernel(name) == kind


def test_kernel_audit():
    ok = score.kernel_audit([dict(name="_Z9gg_kernel7globals", us=90.0),
                             dict(name="void at::native::reduce_kernel<...>", us=10.0)])
    assert ok["ok"] and ok["hk_share"] == 0.9
    bad = score.kernel_audit([dict(name="_Z9gg_kernel7globals", us=90.0),
                              dict(name="Cijk_Alik_Bljk_BBS", us=1.0)])
    assert not bad["ok"] and bad["foreign"]
    assert not score.kernel_audit([dict(name="void at::native::fill", us=1.0)])["ok"]  # no hk work


def test_import_audit(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("import torch\nimport hk\nfrom hk.ops import gemm\n\ndef solve(x, w, offsets, out):\n    pass\n")
    assert score.import_audit(good) == []
    bad = tmp_path / "bad.py"
    bad.write_text("import triton.language as tl\nfrom aiter.ops.triton.gmm import gmm\n"
                   "import importlib\nm = importlib.import_module('vllm')\n"
                   "from bench.t4 import baselines\n")
    assert score.import_audit(bad) == ["aiter.ops.triton.gmm", "bench.t4", "triton.language", "vllm"]


def test_the_trivial_solution_is_flagged_by_nothing_static():
    # torch_loop fails the *kernel* audit (hipBLASLt), not the import audit.
    assert score.import_audit(REPO / "bench/t4/solutions/torch_loop.py") == []


# ---------------------------------------------------------------- routing and shapes


def test_routing_is_deterministic_and_sums():
    s = dict(name="t", model="m", proj="gate_up", E=64, N=256, K=128, tokens=1000, top_k=6,
             skew=0.8, seed=5)
    a, b = R.group_sizes(s, 1), R.group_sizes(s, 1)
    assert a == b and sum(a) == 6000 and len(a) == 64
    assert R.group_sizes(s, 2) != a
    assert max(a) > 2 * min(a)                # skew makes popular experts
    flat = R.group_sizes(dict(s, skew=0.0), 1)
    assert max(flat) < 1.5 * (6000 / 64)


def test_edge_routing():
    for e_n in (8, 16, 64, 160):
        s = dict(name="t", model="m", proj="down", E=e_n, N=256, K=128, tokens=4096, top_k=2,
                 skew=0.5, seed=3)
        z = R.edge_sizes(s)
        assert sum(z) == 8192 and len(z) == e_n
        assert z[0] == 0 and z[e_n - 1] == 0 and z[1] == 1 and z[2] >= 4096
    assert R.offsets_of([0, 3, 0, 2]) == [0, 0, 3, 3, 5]


def test_public_shapes_file_is_what_the_seed_draws():
    assert shapes.main(["--check"]) == 0
    doc = json.loads(R.PUBLIC.read_text())
    for arch in ("gfx950", "gfx1100"):
        got = R.load_shapes(R.PUBLIC, arch)
        assert len(got) >= 10
        for s in got:
            assert s["N"] % 128 == 0 and s["K"] % 64 == 0
            assert 8 <= s["E"] <= 256 and 1 <= R.rows(s) <= 65536
            assert R.flops(s) <= shapes.FLOP_CAP[arch]
    assert set(doc) == {"note", "gfx950", "gfx1100"}


def test_validate_refuses_outside_the_guarantees():
    s = dict(name="t", model="m", proj="down", E=8, N=100, K=128, tokens=1, top_k=2, skew=0, seed=1)
    with pytest.raises(ValueError):
        R.validate(s)
    with pytest.raises(ValueError):
        R.validate(dict(s, N=128, proj="up"))


def test_no_sealed_material_in_the_repository():
    """The hidden set lives outside the repo ($HK_T4_SEALED); nothing here may
    name a shapes file other than the public one."""
    here = REPO / "bench/t4"
    assert not list(here.rglob("shapes.json"))
    assert not list(here.rglob("models.json"))
    task = (here / "task.md").read_text()
    assert "solve(x, w, offsets, out)" in task
    assert "HK_T4_SEALED" not in task


def _doc(names, ratio, arch="gfx950"):
    meta = dict(arch=arch, gpu="g", label="x", date="d", commit="c", versions={}, missing={},
                sets=3, rounds=7, shapes="public", solutions=["sol:x"], command="score")
    rows = [dict(name=n, shape=dict(E=8, N=128, K=128), rows=10, ms={"sol:x": 0.1}, errors={},
                 best_baseline="base:triton",
                 solutions={"sol:x": dict(correct=True, audit_ok=True, ratio=ratio, band=[ratio, ratio])})
            for n in names]
    return dict(meta=meta, rows=rows, verdict={})


def test_merge(tmp_path):
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(_doc(["s1", "s2"], 1.0)))
    b.write_text(json.dumps(_doc(["s2", "s3"], 0.8)))
    assert score.main(["--merge", str(a), str(b), "--out", str(tmp_path), "--label", "m"]) == 0
    m = json.loads((tmp_path / "gfx950-m.json").read_text())
    assert [r["name"] for r in m["rows"]] == ["s1", "s2", "s3"]
    assert m["rows"][1]["solutions"]["sol:x"]["ratio"] == 0.8          # the later run wins
    assert not m["verdict"]["sol:x"]["passed"] and len(m["meta"]["merged"]) == 2
    c.write_text(json.dumps(_doc(["s4"], 1.0, arch="gfx1100")))
    with pytest.raises(SystemExit):
        score.main(["--merge", str(a), str(c), "--out", str(tmp_path)])


def test_markdown_renders():
    doc = dict(meta=dict(arch="gfx950", gpu="MI355X", label="t", date="d", commit="c",
                         versions={"torch": "x"}, missing={"aiter_ck": "no"}, sets=3, rounds=7,
                         shapes="public", solutions=["sol:x"]),
               rows=[dict(name="s", shape=dict(E=8, N=128, K=128), rows=10,
                          ms={"base:triton": 0.1, "sol:x": 0.1}, errors={"base:hk_ops_moe": "no"},
                          best_baseline="base:triton",
                          solutions={"sol:x": dict(correct=True, audit_ok=True, ratio=1.0,
                                                   band=[0.99, 1.01], audit=None)})],
               verdict={"sol:x": stats.verdict(_rows([1.0]))})
    md = score.markdown(doc)
    assert "PASS" in md and "| s |" in md and "1.000 (0.99-1.01)" in md


def test_roofline_counts_only_experts_with_rows():
    s = dict(E=4, N=256, K=128, top_k=1, tokens=8, skew=1.0)
    nbytes = R.min_bytes(s, sizes=[0, 8, 0, 0])
    assert nbytes == 2 * (8 * (128 + 256) + 1 * 256 * 128)
    assert R.min_bytes(s, sizes=[2, 2, 2, 2]) - nbytes == 2 * 3 * 256 * 128
    # A big GEMM is compute-bound, a one-row call bound by the weights' bytes.
    assert stats.roofline_us(2.52e15, 1, "gfx950") == pytest.approx(1e6)
    assert stats.roofline_us(1, 8.0e12, "gfx950") == pytest.approx(1e6)
    assert stats.roofline_us(1, 1, "gfx942") is None
