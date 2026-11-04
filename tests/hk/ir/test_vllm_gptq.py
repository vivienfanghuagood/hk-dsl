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


def test_it_is_on_for_gfx1100_and_gfx1201_and_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract, switches

    assert switches.ROUTES["vllm_gptq"].default
    assert switches.ROUTES["vllm_gptq"].arches == ("gfx1100", "gfx1201")
    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setattr(contract, "check_each", lambda routes: {})   # no vLLM here
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


# -- RDNAHybridW4A16LinearKernel (gfx1201): hk reads vLLM's tensors in place ------------------


class RDNAHybridW4A16LinearKernel:     # vLLM's gfx1x kernel, by name
    w_q_name, w_s_name, w_zp_name = "qweight", "scales", "qzeros"


@pytest.fixture
def _vllm_switch(monkeypatch):
    """vLLM's own HIP-to-Triton switch, as rdna_hybrid_w4a16 has it (5 tokens, 1.2 x 32K elements)."""
    monkeypatch.setattr(vllm_gptq, "_SKINNY", {"max_m": 5, "limit": int(32768 * 1.2),
                                               "gfx115x": False})


@pytest.mark.parametrize("arch,why", [("gfx1201", ""),
                                      ("gfx1100", "kernel RDNAHybridW4A16LinearKernel on gfx1100")])
def test_hybrid_layers_are_taken_where_measured(monkeypatch, _vllm_switch, arch, why):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: arch)
    assert vllm_gptq._why_not(_method(kernel=RDNAHybridW4A16LinearKernel)) == why


def test_hybrid_layers_need_vllms_switch(monkeypatch):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx1201")
    monkeypatch.setattr(vllm_gptq, "_SKINNY", {})
    assert "MAX_SKINNY_BATCH_SIZE" in vllm_gptq._why_not(_method(kernel=RDNAHybridW4A16LinearKernel))


@pytest.mark.parametrize("m,k,path", [
    (1, 5120, "fallback"), (5, 5120, "fallback"),         # vLLM's HIP kernel: left to it
    (6, 5120, "hk"), (64, 5120, "hk"),                     # vLLM's Triton, to 64 tokens: hk's
    (65, 5120, "fallback"), (512, 5120, "fallback"),
    (2, 17408, "fallback"), (3, 17408, "hk"),              # deep K: vLLM leaves HIP at 3 tokens
])
def test_hk_takes_the_batches_vllm_sends_to_triton_up_to_64(monkeypatch, _vllm_switch, m, k, path):
    torch = pytest.importorskip("torch")
    from hk.ops import w4a16

    seen = []
    monkeypatch.setattr(w4a16, "exllama_matmul", lambda *a: seen.append("hk") or "hk")
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda x, w_q, w_s, w_zp, bias, cu, g: seen.append(w_zp) or "vllm"),
        raising=False)
    x = torch.empty(m, k, dtype=torch.float16)
    w_q = torch.empty(64, k // 2, dtype=torch.int8)
    w_s = torch.empty(64, k // 32, dtype=torch.float16)
    before = dict(vllm_gptq.HYBRID.stats)
    vllm_gptq.gptq_w4a16_hybrid(x, w_q, w_s, 32, 64)
    assert vllm_gptq.HYBRID.stats[path] == before[path] + 1
    # vLLM's op is called without zero points: these layers are symmetric
    assert seen == (["hk"] if path == "hk" else [None])


def test_bf16_batches_stay_on_vllm(monkeypatch, _vllm_switch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda *a: "vllm"), raising=False)
    x = torch.empty(8, 5120, dtype=torch.bfloat16)
    before = vllm_gptq.HYBRID.stats["fallback"]
    vllm_gptq.gptq_w4a16_hybrid(x, torch.empty(64, 2560, dtype=torch.int8),
                                torch.empty(64, 160, dtype=torch.bfloat16), 32, 64)
    assert vllm_gptq.HYBRID.stats["fallback"] == before + 1


def _hybrid_method(zero_points=False, weight_type="uint4b8"):
    m = _method(kernel=RDNAHybridW4A16LinearKernel)
    m.kernel.config = SimpleNamespace(zero_points=zero_points, weight_type=weight_type)
    return m


@pytest.mark.parametrize("w_q,w_s,cfg,why", [
    ((64, 2560), (64, 160), {}, ""),
    ((64, 2560), (64, 160), {"zero_points": True}, "zero points (not uint4b8)"),
    ((64, 2304), (64, 144), {}, "shape 64x4608"),           # K not a multiple of 1024
    ((64, 2560), (64, 80), {}, "scales (64, 80)"),
])
def test_a_hybrid_layer_is_checked_after_vllms_repack(monkeypatch, w_q, w_s, cfg, why):
    torch = pytest.importorskip("torch")
    layer = SimpleNamespace(qweight=torch.empty(*w_q, dtype=torch.int8),
                            scales=torch.empty(*w_s, dtype=torch.float16), qzeros=None)
    ran = []
    monkeypatch.setattr(vllm_gptq, "_ORIG_PWAL", lambda self, layer: ran.append(1))
    monkeypatch.setattr(vllm_gptq, "REFUSED", __import__("collections").Counter())
    vllm_gptq._hybrid_pwal(_hybrid_method(**cfg), layer)
    assert ran == [1]                                       # vLLM's repack always runs
    assert getattr(layer, "hk_hybrid", False) == (why == "")
    assert dict(vllm_gptq.REFUSED) == ({why: 1} if why else {})


@pytest.mark.parametrize("m,n,k,path", [
    (1024, 34816, 5120, "fallback"),                       # under PREFILL_MIN_M: vLLM's Triton
    (2048, 34816, 5120, "dequant"),                        # a big W from 2048 tokens
    (2048, 5120, 6144, "fallback"), (4096, 5120, 6144, "dequant"),   # a small W from 4096
])
def test_prefill_chunks_take_the_dequantize_path(monkeypatch, _vllm_switch, m, n, k, path):
    torch = pytest.importorskip("torch")
    from hk.ops import w4a16

    seen = []
    monkeypatch.setattr(w4a16, "dequant_into", lambda qw, s, g, out, exllama: seen.append(
        (tuple(out.shape), exllama)) or out)
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda *a: seen.append("vllm")), raising=False)
    monkeypatch.setattr(torch.nn.functional, "linear", lambda x, w: seen.append("linear"))
    monkeypatch.setattr(vllm_gptq, "_SCRATCH", {})
    x = torch.empty(m, k, dtype=torch.float16)
    before = vllm_gptq.HYBRID.stats[path]
    vllm_gptq.gptq_w4a16_hybrid(x, torch.empty(n, k // 2, dtype=torch.int8),
                                torch.empty(n, k // 32, dtype=torch.float16), 32, 64)
    assert vllm_gptq.HYBRID.stats[path] == before + 1
    assert seen == ([((n, k), True), "linear"] if path == "dequant" else ["vllm"])
