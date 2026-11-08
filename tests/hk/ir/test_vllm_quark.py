"""The Quark W4A16 patch, without a GPU or vLLM."""

from collections import Counter
from types import SimpleNamespace

import pytest

from hk.integration import _common, vllm_gptq, vllm_quark


def test_the_ops_declare_themselves_graph_safe():
    assert vllm_quark.HYBRID.graph == vllm_quark.PATCH.graph == "safe"
    assert set(vllm_quark.HYBRID.paths) == {"hk", "hk_large", "fallback"}
    assert set(vllm_quark.PATCH.paths) == {"hk", "fallback"}


def test_the_fakes_are_tokens_by_output_features():
    torch = pytest.importorskip("torch")
    x = torch.empty(5, 5120, device="meta", dtype=torch.float16)
    w_q = torch.empty(17408, 2560, device="meta", dtype=torch.int8)
    assert vllm_quark._fake_hybrid(x, w_q, None, 128, 64).shape == (5, 17408)
    hk_qw = torch.empty(17408, 640, device="meta", dtype=torch.int32)
    assert vllm_quark._fake(x, None, None, None, hk_qw, None, 128).shape == (5, 17408)


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply_patch()")
    assert "not importable" in vllm_quark.apply_patch()
    assert vllm_quark.status() == "not patched"


def test_it_is_on_for_gfx1201_and_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract, switches

    assert switches.ROUTES["vllm_quark"].default
    assert switches.ROUTES["vllm_quark"].arches == ("gfx1201",)
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


class RDNA3W4A16LinearKernel:          # gfx1100's; its repack names the zero points it synthesizes
    w_q_name, w_s_name, w_zp_name = "weight", "weight_scale", "qzeros"


class TritonW4A16LinearKernel:
    pass


def _scheme(kernel=RDNAHybridW4A16LinearKernel, symmetric=True, group=128):
    return SimpleNamespace(kernel=kernel(), is_symmetric=symmetric, group_size=group,
                           pack_reorder=True)


def _therock_gemm(a, b_q_weight, b_qzeros, b_scales, use_v2_format):
    """gptq_gemm_rdna3 as the TheRock images have it (no act-order index)."""


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


def test_the_hybrid_kernel_only_on_gfx1201(monkeypatch, _gfx1201):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    assert vllm_quark._why_not(_scheme()) == "kernel RDNAHybridW4A16LinearKernel on gfx1100"


@pytest.mark.parametrize("gemm,why", [
    (_therock_gemm, ""),
    (lambda a, b_q_weight, b_qzeros, b_scales, b_g_idx, use_v2_format: None,
     "this vLLM's gptq_gemm_rdna3 takes an act-order index the Quark layers do not hold"),
])
def test_the_rdna3_kernel_on_a_build_without_act_order(monkeypatch, gemm, why):
    monkeypatch.setattr(vllm_gptq, "_GEMM_PLAN", [vllm_gptq.plan_call(gemm)])
    assert vllm_quark._why_not(_scheme(kernel=RDNA3W4A16LinearKernel)) == why


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
    assert getattr(layer, "hk_hybrid", False) == (why == "")
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
    before = dict(vllm_quark.HYBRID.stats)
    vllm_quark.quark_w4a16_hybrid(x, w_q, w_s, 128, 64)
    assert vllm_quark.HYBRID.stats[path] == before[path] + 1
    # the group is the checkpoint's (128); vLLM's op is called without zero points
    assert seen == [(path if path != "fallback" else None, 128)]


def test_bf16_batches_stay_on_vllm(monkeypatch, _gfx1201):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.ops, "vllm", SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda *a: "vllm"), raising=False)
    before = vllm_quark.HYBRID.stats["fallback"]
    vllm_quark.quark_w4a16_hybrid(torch.empty(8, 5120, dtype=torch.bfloat16),
                                  torch.empty(64, 2560, dtype=torch.int8),
                                  torch.empty(64, 40, dtype=torch.bfloat16), 128, 64)
    assert vllm_quark.HYBRID.stats["fallback"] == before + 1


def _quark_checkpoint(k, n, group, g):
    """A symmetric Quark int4 checkpoint (pack_method reorder: signed codes in AWQ's nibble order)
    and the same weights as a GPTQ v1 checkpoint."""
    import torch

    from hk.ops.w4a16 import AWQ_ORDER

    q = torch.randint(0, 16, (k, n), generator=g, dtype=torch.int32)        # unsigned, zero 8
    signed = q ^ 8                                                           # Quark's nibble
    quark = torch.zeros(k, n // 8, dtype=torch.int64)
    for c in range(8):
        quark |= (signed[:, c::8].to(torch.int64) & 0xF) << (4 * AWQ_ORDER[c])
    quark = torch.where(quark >= 2 ** 31, quark - 2 ** 32, quark).to(torch.int32)
    gptq = torch.zeros(k // 8, n, dtype=torch.int64)
    for t in range(8):
        gptq |= q[t::8].to(torch.int64) << (4 * t)
    gptq = torch.where(gptq >= 2 ** 31, gptq - 2 ** 32, gptq).to(torch.int32)
    qz = torch.full((k // group, n // 8), 0x77777777, dtype=torch.int32)
    scales = (torch.rand(k // group, n, generator=g) * 0.02 + 0.005).half()
    return quark, gptq, qz, scales


def test_repack_quark_is_repack_gptq_of_the_same_weights():
    torch = pytest.importorskip("torch")
    from hk.ops import w4a16

    quark, gptq, qz, scales = _quark_checkpoint(256, 64, 128, torch.Generator().manual_seed(0))
    got = w4a16.repack_quark(quark, scales, 128)
    want = w4a16.repack_gptq(gptq, qz, scales, 128, symmetric=True)
    assert got[2] is None and want[2] is None
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_an_rdna3_layer_gets_hks_copy_before_vllms_repack(monkeypatch):
    torch = pytest.importorskip("torch")
    quark, gptq, qz, scales = _quark_checkpoint(256, 64, 128, torch.Generator().manual_seed(1))
    layer = SimpleNamespace(weight=torch.nn.Parameter(quark.clone(), requires_grad=False),
                            weight_scale=torch.nn.Parameter(scales, requires_grad=False))
    order = []

    def vllm_pwal(self, layer):                 # shuffles in place, adds its zero points
        order.append(layer.weight.data.clone())
        layer.weight.data.zero_()
        layer.qzeros = torch.zeros(2, 8, dtype=torch.int32)

    monkeypatch.setattr(vllm_gptq, "_GEMM_PLAN", [vllm_gptq.plan_call(_therock_gemm)])
    monkeypatch.setattr(vllm_quark, "_ORIG_PWAL", vllm_pwal)
    monkeypatch.setattr(vllm_quark, "REFUSED", Counter())
    vllm_quark.process_weights_after_loading(_scheme(kernel=RDNA3W4A16LinearKernel), layer)
    from hk.ops import w4a16
    want = w4a16.repack_gptq(gptq, qz, scales, 128, symmetric=True)
    assert torch.equal(order[0], quark)         # vLLM got the checkpoint's codes
    assert torch.equal(layer.hk_qw, want[0]) and torch.equal(layer.hk_s, want[1])
    assert not vllm_quark.REFUSED


@pytest.mark.parametrize("m,path", [(1, "hk"), (64, "hk"), (65, "fallback")])
def test_rdna3_batches(monkeypatch, m, path):
    torch = pytest.importorskip("torch")
    from hk.ops import w4a16

    monkeypatch.setattr(w4a16, "plan", lambda m, n, k, g, dt, sym: (
        (lambda x, qw, s, out: out.fill_(1)) if m <= 64 else None))
    monkeypatch.setattr(vllm_gptq, "_vllm_gptq", lambda x, w_q, w_zp, w_s, g_idx: "vllm")
    before = dict(vllm_quark.PATCH.stats)
    out = vllm_quark.quark_w4a16(torch.empty(m, 256, dtype=torch.float16), None, None, None,
                                 torch.empty(64, 32, dtype=torch.int32),
                                 torch.empty(4, 128, dtype=torch.float16), 128)
    assert vllm_quark.PATCH.stats[path] == before[path] + 1
    assert (out == "vllm") if path == "fallback" else out.shape == (m, 64)


def test_the_contract_reads_the_scheme_it_wraps():
    from hk.integration import contract

    good = {"pwal": ["self", "layer"], "apply": ["self", "layer", "x", "bias"],
            "class_has": {"self.is_symmetric": True, "self.group_size": True, "self.kernel": True,
                          "self.pack_reorder": True}}
    assert contract.judge_vllm_quark(good) == []
    bad = dict(good, apply=["self", "layer", "x"], class_has=dict(good["class_has"],
                                                                  **{"self.kernel": False}))
    assert contract.judge_vllm_quark(bad) == [
        "QuarkW4A16Int4.apply_weights takes ['self', 'layer', 'x']",
        "QuarkW4A16Int4 no longer sets self.kernel"]
