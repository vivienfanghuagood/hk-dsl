"""hk.integration.vllm_moe_a8w4: aiter.fused_moe patched -- hk takes prefill calls, AITER the
rest; the patched call agrees with AITER's within the K1 criterion (needs aiter: the vLLM image)."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytest.importorskip("aiter")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled MFMA", allow_module_level=True)

from hk.integration import vllm_moe_a8w4 as P  # noqa: E402


def _inputs(m, e=384, inter=1152, k=5120, topk=6, seed=0):
    # DeepSeek-V4.1-Flash's TP2 shapes: AITER has tuned kernels for them (others JIT-build CK)
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randint(0, 256, (e, 2 * inter, k // 2), device="cuda", dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (e, k, inter // 2), device="cuda", dtype=torch.uint8, generator=g)
    s1 = torch.randint(122, 130, (e * 2 * inter, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    s2 = torch.randint(122, 130, (e * k, inter // 32), device="cuda", dtype=torch.uint8, generator=g)
    f4 = torch.float4_e2m1fn_x2
    w1s = shuffle_weight(w1.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=True)
    w2s = shuffle_weight(w2.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=False)
    sc1 = shuffle_scale(s1, experts_cnt=e, is_guinterleave=True, gate_up=True)
    sc2 = shuffle_scale(s2, experts_cnt=e, is_guinterleave=True, gate_up=False)
    x = (torch.randn(m, k, device="cuda", generator=g) * 2).to(torch.bfloat16)
    ids = torch.rand(m, e, device="cuda", generator=g).topk(topk, 1).indices.to(torch.int32)
    wt = torch.rand(m, topk, device="cuda", generator=g)
    wt = (wt / wt.sum(1, keepdim=True)).float()
    return x, w1s, w2s, wt, ids, sc1, sc2


def test_patched_fused_moe_routes_and_agrees():
    import aiter.fused_moe as afm
    from aiter import ActivationType, QuantType

    x, w1, w2, wt, ids, s1, s2 = _inputs(2048)
    kw = dict(expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
              doweight_stage1=False, w1_scale=s1, w2_scale=s2, swiglu_limit=10.0,
              gate_mode="interleave")
    ref = afm.fused_moe(x, w1, w2, wt, ids, **kw)
    P.apply_patch()
    try:
        before = dict(P.STATS)
        out = afm.fused_moe(x, w1, w2, wt, ids, **kw)
        small = afm.fused_moe(x[:512], w1, w2, wt[:512], ids[:512], **kw)   # under MIN_M: AITER
        took = {k: P.STATS[k] - before.get(k, 0) for k in P.STATS}
    finally:
        P.revert()
    assert took == {"hk": 1, "fallback": 1}
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert err < 3e-2, err
    assert torch.isfinite(small).all()


def test_the_patched_fused_moe_keeps_aiters_signature():
    """vLLM passes gate_mode / quant_dtype_a only if inspect.signature(fused_moe) lists them;
    a bare (*args, **kwargs) wrapper made it drop gate_mode, and AITER then read the interleaved
    weights as separated ones (a4w4, wrong) on the fallback path too."""
    import inspect

    import aiter.fused_moe as afm

    want = inspect.signature(afm.fused_moe).parameters
    P.apply_patch()
    try:
        got = inspect.signature(afm.fused_moe).parameters
    finally:
        P.revert()
    assert list(got) == list(want)
    assert "gate_mode" in got
