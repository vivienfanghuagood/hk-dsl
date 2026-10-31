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
    out = vllm_gptq._fake(x, None, None, None, qw, None, None, 32)
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
    pass


class TritonW4A16LinearKernel:
    pass


def _method(kernel=RDNA3W4A16LinearKernel, desc_act=False, qt="uint4b8", group=32):
    return SimpleNamespace(kernel=kernel(), quant_config=SimpleNamespace(
        desc_act=desc_act, quant_type=qt, group_size=group))


@pytest.mark.parametrize("method,why", [
    (_method(), ""),
    (_method(desc_act=True), "act-order (desc_act)"),
    (_method(kernel=TritonW4A16LinearKernel), "kernel TritonW4A16LinearKernel"),
    (_method(qt="uint8b128"), "quant type uint8b128"),
    (_method(group=-1), "group -1"),
])
def test_what_it_refuses_and_why(method, why):
    assert vllm_gptq._why_not(method) == why


def test_batches_below_min_m_stay_on_vllms_kernel(monkeypatch):
    """At GPTQ's group 32, below MIN_M tokens vLLM's kernel is the faster."""
    torch = pytest.importorskip("torch")
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers the fallback")
    from hk.ops import w4a16

    calls = []
    monkeypatch.setattr(w4a16, "plan", lambda *a: calls.append(a) or None)
    x = torch.empty(vllm_gptq.MIN_M - 1, 5120, dtype=torch.float16)
    s = torch.empty(1, dtype=torch.float16)
    with pytest.raises(ImportError):          # the fallback is vLLM's op; no vLLM here
        vllm_gptq.gptq_w4a16(x, None, None, None, torch.empty(17408, 640), s, s, 32)
    assert calls == []                        # hk's plan is not even asked
