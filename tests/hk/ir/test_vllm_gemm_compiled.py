"""The vllm_gemm route inside a forward torch.compile traces whole, as vLLM compiles its models, held
to vLLM's CUDA-graph guard: the transformed code Dynamo produces must not name `update` -- vLLM
(compilation/wrapper.py, bytecode_hook) refuses to start when it does, since a write the graph
has to replay afterwards is lost on a CUDA-graph replay.

vLLM calls `dispatch_unquantized_gemm()` inside the forward, on every linear call. #177 cached its
stand-in in a module dict on the first call, which was the profile run's traced forward: Dynamo
replayed the write as `_GEMMS.update(...)`, and MiniCPM5 did not start with CUDA graphs and
torch.compile on. These run the same shape of forward on the CPU, with hk's op replaced by
`F.linear`: no GPU, no vLLM.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.integration import engine, vllm_gemm  # noqa: E402

N, K = 256, 128


def _vllm_dispatch(*args, **kwargs):
    return lambda layer, x, weight, bias=None: torch.nn.functional.linear(x, weight, bias) * 2


class _Linear(torch.nn.Module):
    """vLLM's UnquantizedLinearMethod.apply, in short: dispatch, then call the GEMM."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(N, K, dtype=dtype), requires_grad=False)

    def forward(self, x):
        return vllm_gemm.dispatch_unquantized_gemm()(self, x, self.weight, None)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Linear(), _Linear(torch.float16), _Linear()])

    def forward(self, x):
        return sum(layer(x.to(layer.weight.dtype)).float() for layer in self.layers)


def _decide(model):
    """What decide_layers does at load, for these stand-in layers."""
    for layer in model.layers:
        vllm_gemm._takes_layer(layer, layer.weight, None)


@pytest.fixture
def route(monkeypatch):
    monkeypatch.setattr(vllm_gemm, "_ORIG_DISPATCH", _vllm_dispatch)
    monkeypatch.setattr(vllm_gemm, "_gemm", lambda layer, x, weight, bias=None:
                        torch.nn.functional.linear(x, weight, bias))
    table = engine.RouteTable({(N, K): dict(runs=[[1, 64, "hk_decode"]], above="hk")})
    monkeypatch.setattr(vllm_gemm, "_TABLE", [table])
    monkeypatch.setattr(vllm_gemm, "_GEN", [0], raising=False)   # (absent before the fix)
    monkeypatch.setattr(vllm_gemm, "_GEMMS", {})
    monkeypatch.setattr(vllm_gemm, "UNROUTED", vllm_gemm.Counter())


def _compile_under_vllms_guard(model, x):
    """Compile `model` whole (fullgraph), as vLLM does, and return (out, the transformed code's
    names that vLLM's guard refuses)."""
    refused = []

    def hook(old, new):
        if "update" in new.co_names:
            refused.append(old.co_name)

    torch._dynamo.reset()
    handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
    try:
        out = torch.compile(model, fullgraph=True, backend="eager")(x)
    finally:
        handle.remove()
        torch._dynamo.reset()
    return out, refused


@pytest.mark.parametrize("decided", ["at load", "not decided", "for an older table"])
def test_the_compiled_forward_writes_nothing_vllms_guard_refuses(route, decided):
    model = _Model()
    if decided != "not decided":
        _decide(model)
    if decided == "for an older table":
        vllm_gemm.set_routes(vllm_gemm.routes())        # a new generation: every memo is stale
    memos = [getattr(layer, "_hk_gemm", None) for layer in model.layers]
    counted = dict(vllm_gemm.UNROUTED)
    x = torch.randn(4, K)
    out, refused = _compile_under_vllms_guard(model, x)
    assert refused == [], f"the traced forward writes state vLLM's CUDA-graph guard refuses: {refused}"
    # the trace kept nothing: the dispatcher's cache, the layers' decisions, the counts
    assert vllm_gemm._GEMMS == {}
    assert [getattr(layer, "_hk_gemm", None) for layer in model.layers] == memos
    assert dict(vllm_gemm.UNROUTED) == counted
    torch.testing.assert_close(out, model(x))


def test_the_compiled_forward_routes_as_eager_does(route):
    """Traced, a layer goes where eager sends it: the bf16 layers to hk's op, the fp16 one to
    vLLM's GEMM (which doubles, in this stand-in)."""
    model = _Model()
    _decide(model)
    x = torch.randn(4, K)
    out, _ = _compile_under_vllms_guard(model, x)
    want = sum(torch.nn.functional.linear(x.to(layer.weight.dtype), layer.weight).float()
               * (2 if layer.weight.dtype == torch.float16 else 1) for layer in model.layers)
    torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-1)


def test_eager_calls_still_cache_and_decide(route):
    model = _Model()
    model(torch.randn(4, K))
    assert len(vllm_gemm._GEMMS) == 1
    assert [layer._hk_gemm for layer in model.layers] == [(0, True), (0, False), (0, True)]
    assert vllm_gemm.UNROUTED == {"weight float16": 1}


@pytest.mark.parametrize("route", ["vllm_int4", "vllm_gptq"])
def test_the_4bit_routes_apply_writes_nothing_vllms_guard_refuses(monkeypatch, route):
    """The other routes whose Python runs inside vLLM's compiled forward: the AWQ and GPTQ
    linear methods' `apply`, on a layer hk holds weights for. (vllm_prefill and the MoE routes
    run inside vLLM's own custom ops, which Dynamo does not trace into.)"""
    import importlib
    import types

    mod = importlib.import_module(f"hk.integration.{route}")
    monkeypatch.setattr(mod, "PATCH", lambda x, *rest: x @ rest[-4].T)   # x @ hk_qw.T, a stand-in
    layer = torch.nn.Module()
    for name, t in (("qweight", torch.zeros(1)), ("scales", torch.zeros(1)),
                    ("qzeros", torch.zeros(1)), ("hk_qw", torch.randn(N, K)),
                    ("hk_s", torch.zeros(1)), ("hk_zs", torch.zeros(1))):
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    method = types.SimpleNamespace(quant_config=types.SimpleNamespace(group_size=128),
                                   kernel=types.SimpleNamespace(w_q_name="qweight",
                                                                w_s_name="scales",
                                                                w_zp_name="qzeros",
                                                                w_gidx_name=None))

    class _Quantized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return mod.apply(method, self.layer, x)

    x = torch.randn(2, 3, K)
    out, refused = _compile_under_vllms_guard(_Quantized(), x)
    assert refused == []
    torch.testing.assert_close(out, x @ layer.hk_qw.T)
