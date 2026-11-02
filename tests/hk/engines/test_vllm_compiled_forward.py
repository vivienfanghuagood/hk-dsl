"""hk's patched linear layers inside the forward torch.compile traces, through the installed vLLM's own
`UnquantizedLinearMethod.apply`, held to vLLM's CUDA-graph guard: Dynamo's transformed code must not
name `update` (vllm/compilation/wrapper.py, bytecode_hook), or vLLM refuses to start with CUDA graphs.

Whether a vLLM build reaches the patched dispatcher inside the compiled forward depends on the build:
some call `dispatch_unquantized_gemm()` in `apply`, on every forward (the MiniCPM5 image's vLLM,
g3775d5fca), others once when the method is built (the MiniCPM-V image's, g10cc2f6ae). #177's
dispatcher cached its GEMM on its first call; on the first kind that call was traced, and MiniCPM5
did not start. So this runs vLLM's own `apply`, whichever kind is installed. The tensors are on the
CPU, where the route calls `F.linear`, so nothing is launched; vLLM's ROCm platform is detected,
which needs a GPU visible. Skipped without vLLM.

    HIP_VISIBLE_DEVICES=3 python3 -m pytest tests/hk/engines/test_vllm_compiled_forward.py -q
"""

from __future__ import annotations

import inspect

import pytest

vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from hk.integration import _common, engine, vllm_gemm  # noqa: E402

N = K = 512


def test_vllms_guard_is_the_rule_held_here():
    from vllm.compilation import wrapper

    assert '"update" in new_code.co_names' in inspect.getsource(wrapper), (
        "vLLM's CUDA-graph guard changed: update _compiled_under_vllms_guard to its new rule")


@pytest.fixture
def patched(monkeypatch):
    """vllm_gemm's dispatcher rebound into the installed vLLM, as apply() does, over a table that
    routes (N, K); undone after."""
    from vllm.model_executor.layers import utils as vu

    orig = vu.dispatch_unquantized_gemm
    monkeypatch.setattr(vllm_gemm, "_ORIG_DISPATCH", orig)
    monkeypatch.setattr(vllm_gemm, "_TABLE",
                        [engine.RouteTable({(N, K): dict(runs=[[1, 64, "hk_decode"]], above="hk")})])
    monkeypatch.setattr(vllm_gemm, "_GEN", [0], raising=False)   # (absent before the fix)
    monkeypatch.setattr(vllm_gemm, "_GEMMS", {})
    monkeypatch.setattr(vllm_gemm, "UNROUTED", vllm_gemm.Counter())
    names = _common.rebind(vllm_gemm.PREFIXES, orig, vllm_gemm.dispatch_unquantized_gemm)
    assert names, "the rebind reached nothing in this vLLM"
    try:
        yield
    finally:
        _common.rebind(vllm_gemm.PREFIXES, vllm_gemm.dispatch_unquantized_gemm, orig)


def _model():
    """Three bf16 linear layers, each computed by vLLM's UnquantizedLinearMethod.apply, decided at
    load as vllm_autotune.decide_layers does."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(N, K, dtype=torch.bfloat16) / K ** 0.5,
                                             requires_grad=False)
            self.method = UnquantizedLinearMethod()     # built after the patch, as vLLM's are

        def forward(self, x):
            return self.method.apply(self, x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = Model()
    for layer in model.layers:
        vllm_gemm._takes_layer(layer, layer.weight, None)
    return model


def _compiled_under_vllms_guard(model, x):
    refused = []

    def hook(old, new):
        if "update" in new.co_names:
            refused.append(f"{old.co_filename}:{old.co_firstlineno} {old.co_name}")

    torch._dynamo.reset()
    handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
    try:
        out = torch.compile(model, fullgraph=True, backend="eager")(x)
    finally:
        handle.remove()
        torch._dynamo.reset()
    return out, refused


def test_the_patched_layers_compile_under_vllms_guard(patched):
    model = _model()
    x = torch.randn(4, K, dtype=torch.bfloat16)
    out, refused = _compiled_under_vllms_guard(model, x)
    assert refused == [], (
        f"vLLM {vllm.__version__} would refuse this forward with CUDA graphs on: Dynamo replays a "
        f"write after the graph in {refused}")
    torch.testing.assert_close(out, model(x))
