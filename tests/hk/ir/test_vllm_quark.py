"""The Quark W4A16 patch, without a GPU or vLLM."""

from collections import Counter
from types import SimpleNamespace

import pytest

from hk.integration import _common, vllm_gptq, vllm_quark


def test_the_op_declares_itself_graph_safe():
    assert vllm_quark.PATCH.graph == "safe"
    assert set(vllm_quark.PATCH.paths) == {"hk", "hk_large", "fallback"}


def test_the_fake_is_tokens_by_output_features():
    torch = pytest.importorskip("torch")
    x = torch.empty(5, 5120, device="meta", dtype=torch.float16)
    w_q = torch.empty(17408, 2560, device="meta", dtype=torch.int8)
    assert vllm_quark._fake(x, w_q, None, 128, 64).shape == (5, 17408)


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply_patch()")
    assert "not importable" in vllm_quark.apply_patch()
    assert vllm_quark.status() == "not patched"


def test_it_is_opt_in_and_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract, switches

    assert not switches.ROUTES["vllm_quark"].default
    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setattr(contract, "check_each", lambda routes: {})   # no vLLM here
    monkeypatch.setenv("HK_PATCH_VLLM", "1")
    monkeypatch.setenv("HK_VLLM_QUARK", "1")
    monkeypatch.setattr(vllm_quark, "apply_patch", lambda: seen.append("quark") or "quark: test")
    _vllm_plugin.register()
    assert "quark" in seen


class RDNAHybridW4A16LinearKernel:     # vLLM's gfx1x kernel, by name, as the Quark scheme names it
    w_q_name, w_s_name, w_zp_name = "weight", "weight_scale", None

    def __init__(self, zero_points=False, weight_type="uint4b8"):
        self.config = SimpleNamespace(zero_points=zero_points, weight_type=weight_type)


class TritonW4A16LinearKernel:
    pass


def _scheme(kernel=RDNAHybridW4A16LinearKernel, symmetric=True, group=128):
    return SimpleNamespace(kernel=kernel(), is_symmetric=symmetric, group_size=group)


@pytest.fixture
def _gfx1201(monkeypatch):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx1201")
    monkeypatch.setattr(vllm_gptq, "_SKINNY", {"max_m": 5, "limit": int(32768 * 1.2),
                                               "gfx115x": False})


@pytest.mark.parametrize("scheme,why", [
    (_scheme(), ""),
    (_scheme(kernel=TritonW4A16LinearKernel), "kernel TritonW4A16LinearKernel"),
    (_scheme(symmetric=False), "zero points (asymmetric)"),
    (_scheme(group=16), "group 16"),
])
def test_what_it_refuses_and_why(_gfx1201, scheme, why):
    assert vllm_quark._why_not(scheme) == why


def test_only_on_gfx1201(monkeypatch, _gfx1201):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    assert vllm_quark._why_not(_scheme()) == "kernel RDNAHybridW4A16LinearKernel on gfx1100"


@pytest.mark.parametrize("w_q,w_s,why", [
    ((5120, 8704), (5120, 136), ""),                     # down_proj: K 17408, group 128
    ((5120, 8704), (5120, 544), "scales (5120, 544)"),   # scales of another group
    ((5120, 8704 - 64), (5120, 136), "shape 5120x17280"),
])
def test_a_layer_is_checked_after_vllms_repack(monkeypatch, _gfx1201, w_q, w_s, why):
    torch = pytest.importorskip("torch")
    layer = SimpleNamespace(weight=torch.empty(*w_q, dtype=torch.int8),
                            weight_scale=torch.empty(*w_s, dtype=torch.float16))
    ran = []
    monkeypatch.setattr(vllm_quark, "_ORIG_PWAL", lambda self, layer: ran.append(1))
    monkeypatch.setattr(vllm_quark, "REFUSED", Counter())
    vllm_quark.process_weights_after_loading(_scheme(), layer)
    assert ran == [1]                                       # vLLM's repack always runs
    assert getattr(layer, "hk_quark", False) == (why == "")
    assert dict(vllm_quark.REFUSED) == ({why: 1} if why else {})


def test_a_refused_layer_keeps_vllms_apply(monkeypatch):
    layer = SimpleNamespace()
    monkeypatch.setattr(vllm_quark, "_ORIG_APPLY", lambda self, layer, x, bias: "vllm")
    assert vllm_quark.apply_weights(_scheme(), layer, None) == "vllm"


@pytest.mark.parametrize("m,k,path", [
    (1, 5120, "fallback"), (5, 5120, "fallback"),         # vLLM's HIP kernel: left to it
    (6, 5120, "hk"), (64, 5120, "hk"),                     # vLLM's Triton, to 64 tokens: hk's decode
    (65, 5120, "hk_large"), (2048, 5120, "hk_large"),      # above: hk's large-batch form
    (2, 17408, "fallback"), (3, 17408, "hk"),              # deep K: vLLM leaves HIP at 3 tokens
])
def test_hk_takes_every_batch_vllm_sends_to_triton(monkeypatch, _gfx1201, m, k, path):
    torch = pytest.importorskip("torch")
    from hk.ops import w4a16

    seen = []
    monkeypatch.setattr(w4a16, "exllama_matmul", lambda x, w_q, s, g: seen.append(("hk", g)) or "hk")
    monkeypatch.setattr(w4a16, "exllama_large_matmul",
                        lambda x, w_q, s, g: seen.append(("hk_large", g)) or "hk")
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda x, w_q, w_s, w_zp, bias, cu, g: seen.append((w_zp, g)) or "v"),
        raising=False)
    x = torch.empty(m, k, dtype=torch.float16)
    w_q = torch.empty(128, k // 2, dtype=torch.int8)
    w_s = torch.empty(128, k // 128, dtype=torch.float16)
    before = dict(vllm_quark.PATCH.stats)
    vllm_quark.quark_w4a16_hybrid(x, w_q, w_s, 128, 64)
    assert vllm_quark.PATCH.stats[path] == before[path] + 1
    # the group is the checkpoint's (128); vLLM's op is called without zero points
    assert seen == [(path if path != "fallback" else None, 128)]


def test_bf16_batches_stay_on_vllm(monkeypatch, _gfx1201):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda *a: "vllm"), raising=False)
    before = vllm_quark.PATCH.stats["fallback"]
    vllm_quark.quark_w4a16_hybrid(torch.empty(8, 5120, dtype=torch.bfloat16),
                                  torch.empty(64, 2560, dtype=torch.int8),
                                  torch.empty(64, 40, dtype=torch.bfloat16), 128, 64)
    assert vllm_quark.PATCH.stats["fallback"] == before + 1


def test_the_contract_reads_the_scheme_it_wraps():
    from hk.integration import contract

    good = {"pwal": ["self", "layer"], "apply": ["self", "layer", "x", "bias"],
            "class_has": {"self.is_symmetric": True, "self.group_size": True, "self.kernel": True}}
    assert contract.judge_vllm_quark(good) == []
    bad = dict(good, apply=["self", "layer", "x"], class_has=dict(good["class_has"],
                                                                  **{"self.kernel": False}))
    assert contract.judge_vllm_quark(bad) == [
        "QuarkW4A16Int4.apply_weights takes ['self', 'layer', 'x']",
        "QuarkW4A16Int4 no longer sets self.kernel"]
