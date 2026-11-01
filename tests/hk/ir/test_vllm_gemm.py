"""The linear-layer routing, without a GPU or vLLM.

Routing is a measured table (hk/tuned/routes_vllm_linear_gfx1100.json, from
tools/hk-bench/route_tune.py). What is pinned here is what the table has to
keep saying for the measurement to have been worth taking: the decode kernel
owns every batch from 6 to 64 (below that, vLLM 0.31's skinny GEMMs), the
generated GEMM's wins are a comb in M rather than a threshold, and no route
sends a shape to a kernel that cannot take it.
"""

import os

import pytest

from hk.integration import engine, vllm_gemm

# The table under test is the one measured on gfx1100.
pytestmark = pytest.mark.arch("gfx1100")

QKV, O, GATE_UP, DOWN, LM_HEAD = ((6144, 4096), (4096, 4096), (24576, 4096),
                                  (4096, 12288), (151936, 4096))
SHAPES = (QKV, O, GATE_UP, DOWN, LM_HEAD)


@pytest.fixture(autouse=True)
def shipped(monkeypatch):
    # The table's own runs, whatever is installed: with FlyDSL importable its
    # rows would answer some of these Ms (tests/hk/gpu/test_flydsl_gemm.py).
    monkeypatch.setattr(vllm_gemm, "OPTIONAL", {})
    vllm_gemm.set_routes(None)
    yield
    vllm_gemm.set_routes(None)


def test_the_shipped_table_covers_qwen3_8b():
    t = vllm_gemm.routes()
    assert t is not None and t.meta["arch"] == "gfx1100"
    assert set(SHAPES) <= set(t.shapes())
    assert t.paths() <= set(vllm_gemm.PATCH.paths)


@pytest.mark.parametrize("nk", SHAPES)
@pytest.mark.parametrize("m", [6, 8, 15, 16, 17, 24, 32, 33, 48, 64])
def test_decode_batches_take_the_decode_kernel(m, nk):
    assert vllm_gemm.route_for(m, *nk) == "hk_decode"


@pytest.mark.parametrize("nk", SHAPES)
def test_one_token_stays_on_vllms_skinny_gemm(nk):
    # Since 0.2x vLLM runs wvSplitK / LLMM1 at M <= 5 on gfx11, and the table
    # is measured against what the installed vLLM runs. Timed as a captured step
    # runs it (graph timing), its one-token GEMV is level with or ahead of hk's
    # on these shapes; at M = 2-4 the decode GEMV can be ahead (6144x4096: 80.6
    # vs 74.0 us at M = 4), so only M = 1 is pinned here.
    assert vllm_gemm.route_for(1, *nk) == "fallback"


@pytest.mark.parametrize("nk", (QKV, O, GATE_UP, DOWN))
@pytest.mark.parametrize("m", [1024, 2048, 4096, 8192, 16384])
def test_prefill_sized_steps_take_the_gemm(m, nk):
    assert vllm_gemm.route_for(m, *nk) == "hk"


def test_the_gemm_wins_are_a_comb_not_a_threshold():
    # The last 128-row block is computed whole: just past a multiple of 128
    # the kernel loses, and just before the next one it wins again.
    r = [vllm_gemm.route_for(m, *QKV) for m in (128, 144, 192, 208, 256, 288, 304)]
    assert r == ["hk", "fallback", "fallback", "hk", "hk", "fallback", "hk"]
    hk_runs = [ru for ru in vllm_gemm.routes()._runs[QKV] if ru[2] == "hk" and ru[0] < 1024]
    assert len(hk_runs) >= 3


def test_an_m_between_two_disagreeing_measurements_falls_back():
    # 129 lies between 128 (a win) and 144 (a loss): not guessed.
    assert vllm_gemm.route_for(129, *QKV) == "fallback"
    # 65..79: the decode kernel stops at 64 and the GEMM was measured from 80.
    assert vllm_gemm.route_for(70, *QKV) == "fallback"


@pytest.mark.parametrize("n,k", [(6100, 4096), (6144, 4000), (100, 64)])
def test_a_shape_the_table_never_measured_falls_back(n, k):
    assert vllm_gemm.route_for(4096, n, k) == "fallback"
    assert vllm_gemm.route_for(4, n, k) == "fallback"


def test_a_table_cannot_send_a_shape_to_a_kernel_that_refuses_it():
    vllm_gemm.set_routes(engine.RouteTable({
        (6100, 4096): {"runs": [[1, 64, "hk_decode"], [80, 99999, "hk"]]},
        (6144, 4000): {"runs": [[1, 64, "hk_decode"], [80, 99999, "hk"]]},
    }))
    for nk in ((6100, 4096), (6144, 4000)):
        assert vllm_gemm.route_for(4, *nk) == "fallback"
        assert vllm_gemm.route_for(4096, *nk) == "fallback"


def test_an_unknown_path_in_a_table_falls_back():
    vllm_gemm.set_routes(engine.RouteTable({QKV: {"runs": [[1, 99999, "cutlass"]]}}))
    assert vllm_gemm.route_for(4096, *QKV) == "fallback"


class _Fly:
    """Stands in for flydsl_gemm: the tile its dispatcher would pick, per M."""

    def __init__(self, tiles):
        self.tiles = tiles

    def variant(self, m, n, k):
        return self.tiles.get(m)

    def run(self, x, weight, out):
        raise AssertionError("routing tests launch nothing")


FLY = engine.RouteTable({QKV: {
    "runs": [[1, 64, "hk_decode"], [80, 99999, "hk"]],
    "prefer": {"flydsl": [[1024, 4096, "128x128x32w4"], [96, 96, "32x32x64w1"]]}}})


def test_flydsl_takes_an_m_only_at_the_tile_that_was_measured(monkeypatch):
    vllm_gemm.set_routes(FLY)
    monkeypatch.setitem(vllm_gemm.OPTIONAL, "flydsl", _Fly({
        96: "32x32x64w1", 1024: "128x128x32w4", 2048: "128x128x32w4",
        1056: "32x64x64w2", 8192: "128x128x32w4"}))
    assert vllm_gemm.route_for(1024, *QKV) == "flydsl"
    assert vllm_gemm.route_for(2048, *QKV) == "flydsl"
    assert vllm_gemm.route_for(96, *QKV) == "flydsl"
    assert vllm_gemm.route_for(1056, *QKV) == "hk"      # a tile not measured here
    assert vllm_gemm.route_for(1030, *QKV) == "hk"      # no FlyDSL tile divides it
    assert vllm_gemm.route_for(8192, *QKV) == "hk"      # outside every range
    assert vllm_gemm.route_for(1024, 4096, 4096) == "fallback"   # shape not in table
    assert vllm_gemm.route_for(1024, *QKV, bias=True) == "fallback"
    assert vllm_gemm.route_for(1024, *QKV, bf16=False) == "fallback"


def test_without_flydsl_its_rows_route_as_if_absent(monkeypatch):
    vllm_gemm.set_routes(FLY)
    monkeypatch.setitem(vllm_gemm.OPTIONAL, "flydsl", _Fly({}))
    for m in (4, 96, 1024, 2048, 4096):
        assert vllm_gemm.route_for(m, *QKV) == FLY.lookup(m, *QKV)


def test_an_optional_path_inside_the_runs_is_refused():
    # The runs are what every install routes by; FlyDSL there would bypass
    # the check that it runs the tile that was measured.
    vllm_gemm.set_routes(engine.RouteTable({QKV: {"runs": [[1, 99999, "flydsl"]]}}))
    assert vllm_gemm.route_for(4096, *QKV) == "fallback"


def test_the_op_counts_the_flydsl_path():
    from hk.integration import flydsl_gemm

    assert flydsl_gemm.NAME == "flydsl" in vllm_gemm.PATCH.paths


def test_bias_dtype_and_layout_fall_back():
    for m in (4, 4096):
        assert vllm_gemm.route_for(m, *QKV, bias=True) == "fallback"
        assert vllm_gemm.route_for(m, *QKV, bf16=False) == "fallback"
        assert vllm_gemm.route_for(m, *QKV, contiguous_weight=False) == "fallback"
    assert vllm_gemm.route_for(0, *QKV) == "fallback"


def test_route_and_route_decode_are_views_of_one_answer():
    for m in range(0, 2049, 7):
        p = vllm_gemm.route_for(m, *QKV)
        assert vllm_gemm.route(m, *QKV) == (p == "hk")
        assert vllm_gemm.route_decode(m, *QKV) == (p == "hk_decode")


def test_the_op_declares_itself_graph_safe():
    assert vllm_gemm.PATCH.graph == "safe"
    assert vllm_gemm.STATS is vllm_gemm.PATCH.stats


def test_the_attention_backend_declares_its_decode_capturable():
    from hk.integration import vllm_backend

    assert vllm_backend.PATCH.graph == "safe"


class _Layer:
    pass


@pytest.fixture
def dispatch(monkeypatch):
    """vllm_gemm's dispatcher over a stand-in for vLLM's: (dispatch, calls) where calls records
    which GEMM each call went to."""
    calls = []
    monkeypatch.setattr(vllm_gemm, "_ORIG_DISPATCH",
                        lambda *a, **k: (lambda layer, x, w, b=None: calls.append(("vllm", a)) or "vllm"))
    monkeypatch.setattr(vllm_gemm, "_gemm",
                        lambda layer, x, w, b=None: calls.append(("hk",)) or "hk")
    table = engine.RouteTable({(6144, 4096): dict(runs=[[1, 64, "hk_decode"]], above="fallback"),
                               (4096, 4096): dict(runs=[[1, 8192, "fallback"]])})
    monkeypatch.setattr(vllm_gemm, "routes", lambda: table)
    monkeypatch.setattr(vllm_gemm, "_GEMMS", {})
    monkeypatch.setattr(vllm_gemm, "UNROUTED", vllm_gemm.Counter())
    return vllm_gemm.dispatch_unquantized_gemm, calls


def test_the_patched_dispatcher_passes_its_arguments_to_vllms(dispatch):
    # Newer vLLM passes a linear-backend name; vLLM's own dispatch gets it, for the layers that
    # keep vLLM's GEMM.
    torch = pytest.importorskip("torch")
    d, calls = dispatch
    assert d("auto") is d("auto")               # one stand-in per argument tuple
    d("auto")(_Layer(), None, torch.empty(4096, 4096, dtype=torch.float16))
    assert calls == [("vllm", ("auto",))]


def test_a_layer_the_table_never_routes_keeps_vllms_gemm_unwrapped(dispatch):
    """Decided once per layer: an fp16 weight, a bias, or a shape whose every M falls back
    goes straight to vLLM's GEMM -- no hk op in its path -- and is counted once."""
    torch = pytest.importorskip("torch")
    d, calls = dispatch
    bf16, fp16 = torch.bfloat16, torch.float16
    routed, fp16_layer, fallback_only, unmeasured = _Layer(), _Layer(), _Layer(), _Layer()
    for _ in range(3):
        d()(routed, None, torch.empty(6144, 4096, dtype=bf16))
        d()(fp16_layer, None, torch.empty(6144, 4096, dtype=fp16))
        d()(fallback_only, None, torch.empty(4096, 4096, dtype=bf16))
        d()(unmeasured, None, torch.empty(1024, 4096, dtype=bf16))
    assert [c[0] for c in calls] == ["hk", "vllm", "vllm", "vllm"] * 3
    assert vllm_gemm.UNROUTED == {"weight float16": 1, "no route at any M": 2}
    assert "unrouted at load: 2 (no route at any M), 1 (weight float16)" == vllm_gemm.unrouted_status()


def test_a_bias_or_a_strided_weight_is_left_to_vllm(dispatch):
    torch = pytest.importorskip("torch")
    d, calls = dispatch
    w = torch.empty(6144, 4096, dtype=torch.bfloat16)
    d()(_Layer(), None, w, torch.empty(6144, dtype=torch.bfloat16))
    d()(_Layer(), None, torch.empty(4096, 6144, dtype=torch.bfloat16).t())
    assert [c[0] for c in calls] == ["vllm", "vllm"]
    assert set(vllm_gemm.UNROUTED) == {"bias", "weight not a contiguous matrix"}


def test_a_new_table_decides_the_layer_again(dispatch, monkeypatch):
    torch = pytest.importorskip("torch")
    d, calls = dispatch
    layer, w = _Layer(), torch.empty(1024, 4096, dtype=torch.bfloat16)
    d()(layer, None, w)
    monkeypatch.setattr(vllm_gemm, "routes", lambda: engine.RouteTable(
        {(1024, 4096): dict(runs=[[1, 16, "hk_decode"]])}))      # e.g. measured at load
    d()(layer, None, w)
    assert [c[0] for c in calls] == ["vllm", "hk"]


@pytest.mark.parametrize("ent, want", [
    (dict(runs=[[1, 64, "hk_decode"]], above="fallback"), True),
    (dict(runs=[[1, 64, "fallback"]], above="hk"), True),
    (dict(runs=[[1, 8192, "fallback"]]), False),
    (dict(runs=[[1, 64, "fallback"]], prefer={"flydsl": [[128, None, "t"]]}), True),
])
def test_routes_any(ent, want):
    t = engine.RouteTable({(6144, 4096): ent})
    assert t.routes_any(6144, 4096) is want
    assert t.routes_any(1, 1) is False


@pytest.mark.parametrize("given, want", [
    (None, "/home/u/.cache/vllm/hk-linear"),
    ("/data/vllm", "/data/vllm/hk-linear"),
    ("/data/vllm/hk-linear", "/data/vllm/hk-linear"),
])
def test_the_patched_engine_compiles_into_a_cache_of_its_own(monkeypatch, given, want):
    # A graph compiled with vLLM's linear layers, loaded back, never calls ours.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/u")
    if given is None:
        monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    else:
        monkeypatch.setenv("VLLM_CACHE_ROOT", given)
    assert vllm_gemm._own_compile_cache() == want
    assert os.environ["VLLM_CACHE_ROOT"] == want


def test_several_routes_share_one_cache_named_for_all_of_them_in_every_process(monkeypatch):
    # hk is on by default, so one engine applies several routes, in any order; and a worker
    # inherits the root its parent set and applies them again. Neither may nest the path.
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm")
    for name in ("hk-int4", "hk-mxfp8", "hk-int4", "hk-mxfp8"):    # parent, then a worker
        got = vllm_gemm._own_compile_cache(name)
    assert got == os.environ["VLLM_CACHE_ROOT"] == "/data/vllm/hk-int4+mxfp8"
    monkeypatch.setenv("VLLM_CACHE_ROOT", "/data/vllm")
    for name in ("hk-mxfp8", "hk-int4"):
        got = vllm_gemm._own_compile_cache(name)
    assert got == "/data/vllm/hk-int4+mxfp8"


def test_apply_without_vllm_reports_rather_than_raises():
    from hk.integration import _common

    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply()")
    assert "not importable" in vllm_gemm.apply()
    assert vllm_gemm.status() == "not patched"


def test_layers_are_decided_at_load_and_counted(dispatch):
    """decide_layers walks the loaded model once: unquantized linears and the LM head are
    decided there (the embedding lookup is not a GEMM), and their first calls reuse it."""
    torch = pytest.importorskip("torch")
    d, calls = dispatch

    class UnquantizedLinearMethod:
        pass

    class UnquantizedEmbeddingMethod:
        pass

    def mod(n, k, dtype, qm, bias=None):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(torch.empty(n, k, dtype=dtype), requires_grad=False)
        m.quant_method, m.bias = qm(), bias
        return m

    model = torch.nn.Module()
    model.qkv = mod(6144, 4096, torch.bfloat16, UnquantizedLinearMethod)
    model.o = mod(4096, 4096, torch.float16, UnquantizedLinearMethod)
    model.embed = mod(151936, 4096, torch.bfloat16, UnquantizedEmbeddingMethod)
    model.lm_head = mod(151936, 4096, torch.bfloat16, UnquantizedEmbeddingMethod)
    line = vllm_gemm.decide_layers(model)
    assert line == ("1 layer through hk's op; unrouted at load: "
                    "1 (weight float16), 1 (no route at any M)")
    d()(model.qkv, None, model.qkv.weight)
    d()(model.o, None, model.o.weight)
    assert [c[0] for c in calls] == ["hk", "vllm"]
    assert sum(vllm_gemm.UNROUTED.values()) == 2           # not counted again on the call
