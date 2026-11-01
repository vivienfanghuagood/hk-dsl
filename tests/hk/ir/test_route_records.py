"""Routes measured on this machine, without a GPU: records over the shipped
table, a budget that measures decode first and resumes, and the vLLM patch's
tuning at load deciding what to measure."""

from __future__ import annotations

import json

import pytest

from hk.autotune import routes as R
from hk.integration import engine, vllm_autotune, vllm_gemm

# This machine's records and the shipped table, both gfx1100's.
pytestmark = pytest.mark.arch("gfx1100")

QKV = (6144, 4096)                         # Qwen3-8B's, in the shipped table
NEW = (4608, 3584)                         # Qwen2.5-7B's qkv: in no table


@pytest.fixture(autouse=True)
def records(tmp_path, monkeypatch):
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.delenv("HK_VLLM_ROUTES", raising=False)
    monkeypatch.delenv("HK_AUTOTUNE", raising=False)
    monkeypatch.delenv(vllm_autotune.BUDGET_ENV, raising=False)
    monkeypatch.setattr(vllm_gemm, "OPTIONAL", {})
    vllm_gemm.set_routes(None)
    yield tmp_path
    vllm_gemm.set_routes(None)


def _pt(m, engine_ms, **paths):
    return [m, {"engine": engine_ms, **paths}, {}, {p: 0.001 for p in ("engine", *paths)}, 1]


def _ent(name, points, **meta):
    e = R.entry(name, points)
    e["meta"] = {"engine": "vllm 0.31.0 rocm_unquantized_gemm", **meta}
    return e


def _save(shapes):
    with R.records_lock("vllm_linear", "gfx1100"):
        return R.save_user("vllm_linear", "gfx1100",
                           {f"{n}x{k}": e for (n, k), e in shapes.items()},
                           {"kind": "vllm_linear", "arch": "gfx1100"})


DECODE_WIN = [_pt(m, 1.0, hk_decode=0.6) for m in (1, 8, 16, 64)]


def test_a_measured_shape_joins_the_shipped_table():
    assert vllm_gemm.route_for(16, *NEW) == "fallback"
    p = _save({NEW: _ent("qkv", DECODE_WIN)})
    vllm_gemm.set_routes(None)
    t = vllm_gemm.routes()
    assert vllm_gemm.route_for(16, *NEW) == "hk_decode"
    assert t.origin[NEW] == str(p) and t.origin[QKV].endswith("routes_vllm_linear_gfx1100.json")
    assert vllm_gemm.route_for(16, *QKV) == "hk_decode"          # the shipped rows stand


def test_this_machines_record_replaces_the_shipped_shape_whole():
    _save({QKV: _ent("qkv", [_pt(m, 1.0, hk_decode=1.2) for m in (1, 16, 64)])})
    vllm_gemm.set_routes(None)
    assert vllm_gemm.route_for(16, *QKV) == "fallback"
    assert vllm_gemm.route_for(4096, *QKV) == "fallback"         # nothing of the old entry


def test_autotune_off_ignores_the_records(monkeypatch):
    _save({NEW: _ent("qkv", DECODE_WIN)})
    monkeypatch.setenv("HK_AUTOTUNE", "0")
    vllm_gemm.set_routes(None)
    assert vllm_gemm.route_for(16, *NEW) == "fallback"


def test_a_pinned_table_is_the_table(monkeypatch, tmp_path):
    _save({NEW: _ent("qkv", DECODE_WIN)})
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"arch": "gfx1100", "shapes": {}}))
    monkeypatch.setenv("HK_VLLM_ROUTES", str(pinned))
    vllm_gemm.set_routes(None)
    assert vllm_gemm.route_for(16, *NEW) == "fallback"


def test_a_record_against_another_engine_is_stale():
    _save({NEW: _ent("qkv", DECODE_WIN, engine="vllm 0.16.1 F.linear")})
    t = engine.shipped_table("vllm_linear", "gfx1100", "HK_VLLM_ROUTES",
                             engine="vllm 0.31.0 rocm_unquantized_gemm")
    assert NEW not in t.shapes() and t.meta["stale"] == [NEW]
    t = engine.shipped_table("vllm_linear", "gfx1100", "HK_VLLM_ROUTES")
    assert NEW in t.shapes()                                       # no engine named: kept


def test_records_alone_make_a_table_for_an_arch_with_none():
    with R.records_lock("vllm_linear", "gfx1201"):
        R.save_user("vllm_linear", "gfx1201", {"4608x3584": _ent("qkv", DECODE_WIN)},
                    {"kind": "vllm_linear", "arch": "gfx1201"})
    t = engine.shipped_table("vllm_linear", "gfx1201", "HK_VLLM_ROUTES")
    assert t is not None and t.lookup(16, *NEW) == "hk_decode"
    assert engine.shipped_table("vllm_linear", "gfx9999", "HK_VLLM_ROUTES") is None


def test_a_partial_entry_answers_nothing_above_what_it_measured():
    e = R.entry("o", [_pt(m, 1.0, hk_decode=0.6) for m in (8, 64)]
                + [_pt(m, 1.0, hk=0.8) for m in (80, 128)], partial=True)
    assert e["partial"] and e["above"] == "fallback"
    t = engine.RouteTable({NEW: e})
    assert t.lookup(128, *NEW) == "hk" and t.lookup(129, *NEW) == "fallback"
    assert t.lookup(4096, *NEW) == "fallback"
    assert t.complete() == set() and t.partial == {NEW}
    t.overlay({NEW: R.entry("o", e["points"])}, "later")
    assert t.complete() == {NEW} and t.lookup(4096, *NEW) == "hk"


def test_the_records_file_is_a_table_the_route_tools_read(records):
    p = _save({NEW: _ent("qkv", DECODE_WIN)})
    d = json.loads(p.read_text())
    assert d["arch"] == "gfx1100" and d["shapes"]["4608x3584"]["meta"]["engine"]
    assert R.decisions(d)[("4608x3584", 16)] == ("hk_decode", False)
    assert engine.RouteTable.from_json(p).lookup(16, *NEW) == "hk_decode"
    assert not list(records.glob("*.tmp"))


QWEN25 = {"architectures": ["Qwen2ForCausalLM"], "hidden_size": 3584,
          "intermediate_size": 18944, "num_attention_heads": 28, "num_key_value_heads": 4,
          "num_hidden_layers": 28, "vocab_size": 152064}
LLAMA3 = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
          "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8,
          "num_hidden_layers": 32, "vocab_size": 128256, "attention_bias": False,
          "mlp_bias": False}


def test_a_configs_linear_shapes_and_their_biases():
    assert R.linear_shapes(QWEN25) == [
        ("qkv", 4608, 3584, 28), ("o", 3584, 3584, 28), ("gate_up", 37888, 3584, 28),
        ("down", 3584, 18944, 28), ("lm_head", 152064, 3584, 1)]
    assert R.biased(QWEN25) == {"qkv"}
    assert [s[1:3] for s in R.linear_shapes(LLAMA3, tp=2)] == [
        (3072, 4096), (4096, 2048), (14336, 4096), (4096, 7168), (64128, 4096)]
    assert R.biased(LLAMA3) == set()


#: Qwen3.8-27B's text config, the keys linear_shapes reads (16 full-attention
#: layers among 64; the rest Gated-DeltaNet).
QWEN38 = {"hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4,
          "head_dim": 256, "intermediate_size": 17408, "num_hidden_layers": 64,
          "vocab_size": 248320, "attn_output_gate": True,
          "linear_num_key_heads": 16, "linear_num_value_heads": 48,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128,
          "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}


def test_a_hybrid_decoders_shapes_are_its_layers_own():
    """The per-rank shapes vLLM and SGLang run Qwen3.8-27B's TP2 linears at:
    attention (gated q) on 16 layers, the Gated-DeltaNet projections on 48."""
    got = R.linear_shapes({"text_config": QWEN38}, tp=2)
    assert got == [
        ("qkv", 7168, 5120, 16), ("o", 5120, 3072, 16),
        ("qkvz", 8192, 5120, 48), ("ba", 48, 5120, 48), ("out", 5120, 3072, 48),
        ("gate_up", 17408, 5120, 64), ("down", 5120, 8704, 64), ("lm_head", 124160, 5120, 1)]
    # A dense decoder is what it was.
    assert R.linear_shapes(LLAMA3)[0] == ("qkv", 6144, 4096, 32)


def test_model_config_reads_a_file_or_a_directory(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(LLAMA3))
    assert R.model_config(str(tmp_path))["hidden_size"] == 4096
    assert R.model_config(str(tmp_path / "config.json"))["vocab_size"] == 128256


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def fake_measure(monkeypatch):
    """measure_points that takes 1 s a decode M and 10 s any other, on a
    clock of its own, and in which hk's GEMV wins every decode M by 40%."""
    clock = _Clock()
    monkeypatch.setattr(R.time, "monotonic", clock)
    calls = []

    def measure(n, k, ms, *, engine, deadline=None, **how):
        pts, done = [], True
        for m in ms:
            if deadline is not None and clock() > deadline:
                done = False
                break
            clock.t += 1.0 if m <= 64 else 10.0
            pts.append(_pt(m, 1.0, hk_decode=0.6) if m <= 64 else _pt(m, 1.0, hk=0.9))
            calls.append((n, k, m))
        return pts, 0, done

    monkeypatch.setattr(R, "measure_points", measure)
    return calls


SHAPES = [("o", 3584, 3584, 28), ("gate_up", 37888, 3584, 28), ("lm_head", 152064, 3584, 1)]


def _tune(**kw):
    kw.setdefault("timing", "graph")         # the budget's order, without context timing's re-check
    return R.tune_missing("vllm_linear", "gfx1100", SHAPES, engine=None,
                          engine_name="vllm 0.31.0 rocm_unquantized_gemm",
                          grid=list(range(1, 65)) + [80, 96, 112], **kw)


def test_a_budget_measures_every_shapes_decode_first(fake_measure):
    rep = _tune(budget=200)
    # decode Ms of every shape, largest weight stream first, before any prefill M
    first = [(n, k) for n, k, m in fake_measure if m > 64][:1]
    assert [nk for nk in dict.fromkeys((n, k) for n, k, m in fake_measure[:192])] == [
        (37888, 3584), (152064, 3584), (3584, 3584)]
    assert first == [(37888, 3584)]
    assert rep.partial and not rep.left
    t = engine.shipped_table("vllm_linear", "gfx1100", "HK_VLLM_ROUTES")
    for _, n, k, _ in SHAPES:
        assert t.lookup(16, n, k) == "hk_decode"
    assert "only up to M=" in rep.line()


def test_a_cut_short_shape_resumes_where_it_stopped(fake_measure):
    _tune(budget=200)
    before = len(fake_measure)
    rep = _tune(budget=None)
    again = fake_measure[before:]
    assert all(m > 64 for _, _, m in again)                         # decode not re-measured
    assert sorted(s[0] for s in rep.tuned) == ["gate_up", "lm_head", "o"]
    t = engine.shipped_table("vllm_linear", "gfx1100", "HK_VLLM_ROUTES")
    assert t.complete() >= {(n, k) for _, n, k, _ in SHAPES}
    assert t.lookup(100, 3584, 3584) == "hk"


def test_a_shape_measured_already_is_not_measured_again(fake_measure):
    _tune(budget=None)
    n0 = len(fake_measure)
    rep = _tune(budget=None)
    assert len(fake_measure) == n0 and len(rep.known) == 3
    assert rep.line().startswith("nothing to measure")
    rep = _tune(budget=None, force=True)
    assert len(fake_measure) > n0 and len(rep.tuned) == 3


def test_a_shipped_shape_is_known_without_a_record(fake_measure):
    rep = _tune(budget=None, complete={(3584, 3584)})
    assert ("o", 3584, 3584) in rep.known and all(n != 3584 or k != 3584
                                                  for n, k, _ in fake_measure)


def test_a_second_process_waits_for_the_first_then_reads_its_records(records):
    # Two tensor-parallel ranks, or two servers: the one that takes the lock
    # measures; the other waits, then finds the shape and measures nothing.
    import subprocess
    import sys
    import textwrap
    import time

    code = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(R.__file__.rsplit('/hk/', 1)[0])!r})
        from hk.autotune import routes as R
        with R.records_lock("vllm_linear", "gfx1100"):
            print("held", flush=True)
            time.sleep(1.5)
            R.save_user("vllm_linear", "gfx1100",
                        {{"3584x3584": R.entry("o", [[m, {{"engine": 1.0, "hk_decode": 0.6}}]
                                                    for m in range(1, 65)])}},
                        {{"kind": "vllm_linear", "arch": "gfx1100"}})
    """)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                         env={**__import__("os").environ, "HK_TUNE_DIR": str(records)})
    assert p.stdout.readline().strip() == "held"
    t0 = time.monotonic()
    rep = R.tune_missing("vllm_linear", "gfx1100", [("o", 3584, 3584, 28)], engine=None,
                         engine_name="vllm 0.31.0 rocm_unquantized_gemm",
                         grid=list(range(1, 65)), budget=None)
    assert p.wait(timeout=30) == 0
    assert time.monotonic() - t0 >= 1.0 and rep.waited >= 1.0
    assert rep.known == [("o", 3584, 3584)] and not rep.tuned


def test_context_timing_re_times_a_decode_m_once_the_context_has_moved(monkeypatch):
    # qkv is measured first, among o still on the engine's path; o then takes hk's GEMV at every
    # decode M, so qkv's decode Ms are measured once more, among o on hk's GEMV. o was measured
    # among qkv's final routes already. The LM head is timed alone, and is in no context.
    clock = _Clock()
    monkeypatch.setattr(R.time, "monotonic", clock)
    monkeypatch.setattr(R, "tunableop_state", lambda: "off")
    seen = []

    def measure(n, k, ms, *, engine, timing, context, deadline=None, **how):
        assert timing == ("graph" if n == 73472 else "context")         # the LM head: a graph alone
        seen.append(((n, k), [(cn, ck) for cn, ck, _ in context],
                     {m: tuple(at(m) for _, _, at in context) for m in ms}))
        return [_pt(m, 1.0, hk_decode=0.6) for m in ms], 0, True

    monkeypatch.setattr(R, "measure_points", measure)
    shapes = [("qkv", 2560, 2048, 40), ("o", 2048, 2048, 40), ("lm_head", 73472, 2048, 1)]
    R.tune_missing("vllm_linear", "gfx1100", shapes, engine=None, engine_name="vllm test",
                   grid=list(range(1, 65)) + [80])
    qkv_, o_, lm_ = (2560, 2048), (2048, 2048), (73472, 2048)
    assert [(nk, ctx) for nk, ctx, _ in seen] == [
        (qkv_, [o_]), (o_, [qkv_]), (lm_, []),                  # decode Ms, largest stream first
        (qkv_, [o_]),                                           # qkv's decode Ms once more
        (qkv_, [o_]), (o_, [qkv_]), (lm_, [])]                  # then the rest of the grid
    assert set(seen[0][2][8]) == {"engine"} and seen[1][2][8] == ("hk_decode",)
    assert sorted(seen[3][2]) == list(range(1, 65))
    assert all(at == ("hk_decode",) for at in seen[3][2].values())
    rec = R.load_user("vllm_linear", "gfx1100")["shapes"]
    assert rec["2560x2048"]["meta"]["timing"] == "context"
    assert rec["2560x2048"]["meta"]["tunableop"] == "off"
    assert rec["2560x2048"]["meta"]["context_routes"]["8"] == "d"         # o on hk's GEMV
    assert rec["2048x2048"]["meta"]["context_routes"]["8"] == "d"


def test_a_later_call_re_times_what_an_earlier_one_measured_in_another_context(monkeypatch):
    # An earlier start measured qkv's decode Ms among o on the engine's path ("e") and ran out of
    # time; this one measures o, which takes hk's GEMV, and so re-times qkv's decode Ms.
    monkeypatch.setattr(R, "tunableop_state", lambda: "off")
    qkv = R.entry("qkv", [_pt(m, 1.0, hk_decode=0.6) for m in range(1, 65)], partial=True)
    qkv["meta"] = {"engine": "vllm test", "timing": "context", "tunableop": "off",
                   "context_routes": {str(m): "e" for m in range(1, 65)}}
    R.save_user("vllm_linear", "gfx1100", {"2560x2048": qkv}, {"kind": "vllm_linear", "arch": "gfx1100"})
    measured = []

    def measure(n, k, ms, *, engine, deadline=None, **how):
        measured.append(((n, k), list(ms)))
        return [_pt(m, 1.0, hk_decode=0.6) for m in ms], 0, True

    monkeypatch.setattr(R, "measure_points", measure)
    R.tune_missing("vllm_linear", "gfx1100", [("qkv", 2560, 2048, 40), ("o", 2048, 2048, 40)],
                   engine=None, engine_name="vllm test", grid=list(range(1, 65)) + [80])
    assert measured[0] == ((2048, 2048), list(range(1, 65)))           # o's decode Ms
    assert measured[1] == ((2560, 2048), list(range(1, 65)))           # qkv's, once more
    rec = R.load_user("vllm_linear", "gfx1100")["shapes"]["2560x2048"]
    assert {rec["meta"]["context_routes"][str(m)] for m in range(1, 65)} == {"d"}


def test_a_record_under_another_tunableop_state_is_measured_again(fake_measure, monkeypatch):
    on = {"file": "mcpm5%d.csv", "sha": "0123456789abcdef", "tuning": False}
    monkeypatch.setattr(R, "tunableop_state", lambda: on)
    _tune(budget=None)
    n0 = len(fake_measure)
    assert len(_tune(budget=None).known) == 3 and len(fake_measure) == n0
    monkeypatch.setattr(R, "tunableop_state", lambda: "off")
    rep = _tune(budget=None)
    assert len(rep.tuned) == 3 and len(fake_measure) > n0
    assert R.load_user("vllm_linear", "gfx1100")["shapes"]["3584x3584"]["meta"]["tunableop"] == "off"


def test_no_budget_left_measures_nothing_and_says_so(fake_measure):
    rep = _tune(budget=-1)
    assert not fake_measure and len(rep.left) == 3
    assert "stay on the engine's path" in rep.line()


# ------------------------------------------------------------------ at load


class _Table:
    def __init__(self, have):
        self.have = set(have)

    def complete(self):
        return self.have


@pytest.fixture
def at_load(monkeypatch):
    seen = {}
    monkeypatch.setattr(vllm_autotune, "model_shapes", lambda model: (
        [("qkv_proj", *QKV, 36), ("o_proj", 3584, 3584, 28)], [("qkv_proj", *NEW)]))
    monkeypatch.setattr(vllm_gemm, "routes", lambda: _Table({QKV}))
    monkeypatch.setattr(R, "engine_path", lambda name="vllm": (None, "vllm test"))

    def tune(kind, arch, shapes, **kw):
        seen.update(shapes=shapes, **kw)
        return R.Report()

    monkeypatch.setattr(R, "tune_missing", tune)
    return seen


def test_at_load_measures_only_what_the_table_lacks(at_load):
    rep = vllm_autotune.at_load(object(), log=lambda s: None)
    assert at_load["shapes"] == [("o_proj", 3584, 3584, 28)]
    assert at_load["budget"] == vllm_autotune.DEFAULT_BUDGET
    assert at_load["engine_name"] == "vllm test"
    assert "qkv_proj 4608x3584 carries a bias" in rep["line"]


@pytest.mark.parametrize("env, value, why", [
    (vllm_autotune.BUDGET_ENV, "0", "HK_AUTOTUNE_BUDGET=0"),
    ("HK_AUTOTUNE", "0", "HK_AUTOTUNE=0"),
    ("HK_VLLM_ROUTES", "/some/table.json", "pinned"),
])
def test_at_load_measures_nothing_when_told_not_to(at_load, monkeypatch, env, value, why):
    monkeypatch.setenv(env, value)
    rep = vllm_autotune.at_load(object(), log=lambda s: None)
    assert "shapes" not in at_load and why in rep["line"]


def test_at_load_with_every_shape_known_says_so(at_load, monkeypatch):
    monkeypatch.setattr(vllm_gemm, "routes", lambda: _Table({QKV, (3584, 3584)}))
    rep = vllm_autotune.at_load(object(), log=lambda s: None)
    assert "shapes" not in at_load and rep["line"].startswith("all 2 shapes in the table")


def test_the_worker_is_wrapped_when_its_module_is_imported_not_before(tmp_path, monkeypatch):
    # vLLM's API server loads the plugin too and never builds a worker, so
    # the worker's module (seconds to import) is not imported for it.
    import importlib
    import sys

    pkg = tmp_path / "hkfake_vllm_worker"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "gpu_worker.py").write_text(
        "class Worker:\n    def load_model(self):\n        return 'loaded'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(vllm_autotune, "WORKER_MODULE", "hkfake_vllm_worker.gpu_worker")
    seen = []
    monkeypatch.setattr(vllm_autotune, "at_load",
                        lambda model, device=None: seen.append(model) or {"line": "x"})
    try:
        assert "measured at load" in vllm_autotune.install()
        assert "hkfake_vllm_worker.gpu_worker" not in sys.modules
        assert vllm_autotune.install() and sum(
            isinstance(f, vllm_autotune._OnImport) for f in sys.meta_path) == 1
        mod = importlib.import_module("hkfake_vllm_worker.gpu_worker")
        w = mod.Worker()
        w.model_runner = type("R", (), {"get_model": lambda self: "the model"})()
        assert w.load_model() == "loaded" and seen == ["the model"]
        assert not any(isinstance(f, vllm_autotune._OnImport) for f in sys.meta_path)
    finally:
        vllm_autotune.uninstall()
        sys.modules.pop("hkfake_vllm_worker.gpu_worker", None)
        sys.modules.pop("hkfake_vllm_worker", None)
    assert mod.Worker.load_model(w) == "loaded" and seen == ["the model"]


def test_python_m_hk_integration_tune_dispatches(monkeypatch):
    from hk.integration import __main__ as cli
    from hk.integration import tune

    seen = []
    monkeypatch.setattr(tune, "main", lambda argv: seen.append(argv) or 0)
    assert cli.main(["tune", "--model", "x"]) == 0
    assert seen == [["--model", "x"]]


@pytest.mark.parametrize("raw, want", [
    ("", 60.0), ("15", 15.0), ("0", 0.0), ("off", 0.0), ("inf", None), ("junk", 60.0)])
def test_the_budget_variable(monkeypatch, raw, want):
    monkeypatch.setenv(vllm_autotune.BUDGET_ENV, raw)
    assert vllm_autotune.budget() == want
