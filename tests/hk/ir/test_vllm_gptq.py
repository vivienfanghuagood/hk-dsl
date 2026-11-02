"""The GPTQ patch, without a GPU or vLLM."""

from types import SimpleNamespace

import pytest

from hk.integration import _common, vllm_gptq


def test_the_op_declares_itself_graph_safe():
    assert vllm_gptq.PATCH.graph == "safe"
    assert set(vllm_gptq.PATCH.paths) == {"hk", "fallback"}


def test_the_fake_is_tokens_by_output_features():
    torch = pytest.importorskip("torch")
    x = torch.empty(5, 5120, device="meta", dtype=torch.float16)
    qw = torch.empty(17408, 640, device="meta", dtype=torch.int32)
    out = vllm_gptq._fake(x, None, None, None, None, qw, None, None, 32)
    assert out.shape == (5, 17408)


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply_patch()")
    assert "not importable" in vllm_gptq.apply_patch()
    assert vllm_gptq.status() == "not patched"


def test_it_is_on_for_gfx1100_only_and_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract, switches

    assert switches.ROUTES["vllm_gptq"].default and switches.ROUTES["vllm_gptq"].arches == ("gfx1100",)
    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setattr(contract, "check", lambda routes: [])   # no vLLM here
    monkeypatch.setenv("HK_PATCH_VLLM", "1")
    monkeypatch.setenv("HK_VLLM_GPTQ", "1")
    monkeypatch.setattr(vllm_gptq, "apply_patch", lambda: seen.append("gptq") or "gptq: test")
    _vllm_plugin.register()
    assert "gptq" in seen


class RDNA3W4A16LinearKernel:          # vLLM's kernel, by name
    w_q_name, w_s_name, w_zp_name = "qweight", "scales", "qzeros"


class TritonW4A16LinearKernel:
    pass


def _method(kernel=RDNA3W4A16LinearKernel, desc_act=False, qt="uint4b8", group=32):
    return SimpleNamespace(kernel=kernel(), quant_config=SimpleNamespace(
        desc_act=desc_act, quant_type=qt, group_size=group))


@pytest.fixture(autouse=True)
def _no_vllm_signature(monkeypatch):
    """The plan for vLLM's gptq_gemm_rdna3 as the TheRock image's (no b_g_idx), without vLLM."""
    monkeypatch.setattr(vllm_gptq, "_GEMM_PLAN", [vllm_gptq.plan_call(_gemm_therock)])


@pytest.mark.parametrize("method,why", [
    (_method(), ""),
    (_method(desc_act=True), "act-order (desc_act)"),
    (_method(kernel=TritonW4A16LinearKernel), "kernel TritonW4A16LinearKernel"),
    (_method(qt="uint8b128"), "quant type uint8b128"),
    (_method(group=-1), "group -1"),
    (_method(kernel=type("RDNA3W4A16LinearKernel", (), {})), "kernel names no weight parameters "
     "(w_q_name / w_s_name)"),
])
def test_what_it_refuses_and_why(method, why):
    assert vllm_gptq._why_not(method) == why


def test_batches_past_64_stay_on_vllms_kernel(monkeypatch):
    """hk's kernel takes 1 (MIN_M) to 64 tokens; a bigger batch is vLLM's."""
    torch = pytest.importorskip("torch")
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers the fallback")
    from hk.ops import w4a16

    assert vllm_gptq.MIN_M == 1
    calls = []
    monkeypatch.setattr(w4a16, "plan", lambda *a: calls.append(a) or None)
    x = torch.empty(65, 5120, dtype=torch.float16)
    s = torch.empty(1, dtype=torch.float16)
    with pytest.raises(ImportError):          # the fallback is vLLM's op; no vLLM here
        vllm_gptq.gptq_w4a16(x, None, None, None, None, torch.empty(17408, 640), s, s, 32)
    assert calls == []                        # hk's plan is not even asked


# -- vLLM's gptq_gemm_rdna3 differs between builds; the fallback binds it by name -------------


def _gemm_therock(a, b_q_weight, b_qzeros, b_scales, use_v2_format):
    """vLLM 0.1.dev1+g10cc2f6ae (TheRock ROCm 7.14 images)."""


def _gemm_with_g_idx(a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format):
    """vLLM 0.1.dev1+g3775d5fca (the MiniCPM5 serving image): the act-order index added."""


def _gemm_unknown(a, b_q_weight, b_qzeros, b_scales, b_perm, use_v2_format):
    """A build with a parameter this route has no value for."""


@pytest.mark.parametrize("fn,want", [
    (_gemm_therock, {"a": "x", "b_q_weight": "w_q", "b_qzeros": "w_zp", "b_scales": "w_s",
                     "use_v2_format": False}),
    (_gemm_with_g_idx, {"a": "x", "b_q_weight": "w_q", "b_qzeros": "w_zp", "b_scales": "w_s",
                        "b_g_idx": "g_idx", "use_v2_format": False}),
])
def test_vllms_gemm_is_bound_by_name_on_both_builds(fn, want, monkeypatch):
    plan, problem = vllm_gptq.plan_call(fn)
    assert problem == "" and dict(plan) == want
    seen = {}
    monkeypatch.setattr(vllm_gptq, "_GEMM_PLAN", [(plan, problem)])
    ops = SimpleNamespace(gptq_gemm_rdna3=lambda **kw: seen.update(kw) or "vllm")
    monkeypatch.setitem(__import__("sys").modules, "vllm", SimpleNamespace(_custom_ops=ops))
    monkeypatch.setitem(__import__("sys").modules, "vllm._custom_ops", ops)
    assert vllm_gptq._vllm_gptq("X", "Q", "Z", "S", "G") == "vllm"
    assert seen == {n: {"x": "X", "w_q": "Q", "w_zp": "Z", "w_s": "S", "g_idx": "G"}.get(v, v)
                    for n, v in want.items()}


def test_a_build_with_an_unknown_parameter_refuses_its_layers_not_the_engine(monkeypatch):
    plan, problem = vllm_gptq.plan_call(_gemm_unknown)
    assert "b_perm" in problem
    monkeypatch.setattr(vllm_gptq, "_GEMM_PLAN", [(plan, problem)])
    assert vllm_gptq._why_not(_method()) == problem


def test_the_contract_reads_only_the_entry_points_it_wraps(monkeypatch):
    """The engine contract no longer counts gptq_gemm_rdna3's arguments: either build passes."""
    import sys

    from hk.integration import contract

    class AutoGPTQLinearMethod:
        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            pass

    for gemm in (_gemm_therock, _gemm_with_g_idx):
        ops = SimpleNamespace(gptq_gemm_rdna3=gemm)
        auto = SimpleNamespace(AutoGPTQLinearMethod=AutoGPTQLinearMethod)
        quant = SimpleNamespace(auto_gptq=auto)
        for name, mod in (("vllm", SimpleNamespace(_custom_ops=ops)), ("vllm._custom_ops", ops),
                          ("vllm.model_executor", SimpleNamespace()),
                          ("vllm.model_executor.layers", SimpleNamespace()),
                          ("vllm.model_executor.layers.quantization", quant),
                          ("vllm.model_executor.layers.quantization.auto_gptq", auto)):
            monkeypatch.setitem(sys.modules, name, mod)
        assert contract.CHECKS["vllm_gptq"]() == []
