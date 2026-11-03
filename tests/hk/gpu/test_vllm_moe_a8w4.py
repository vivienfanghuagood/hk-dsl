"""hk.integration.vllm_moe_a8w4: aiter.fused_moe patched -- hk takes calls of M >= MIN_M (2560,
where it wins), AITER the rest. The patched call is checked against an fp64 MoE on the same
dequantized inputs, not against AITER's fused_moe: on gfx950 that a8w4 path is now and then wrong in
a few whole token rows (on DeepSeek-V4.1-Flash's TP2 shape, 3 calls in 2200 on an idle MI355X were
0.04-0.056 off, against its usual 0.007-0.008), which failed this test on main while hk's output
was bit-identical in every call (needs aiter: the vLLM image)."""

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
    """(fused_moe's arguments in AITER's layouts, the raw weights and scales they were made from)."""
    # DeepSeek-V4.1-Flash's shapes (inter 1152 at TP2, 2304 at TP1): AITER has kernels for them
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
    return (x, w1s, w2s, wt, ids, sc1, sc2), (w1, w2, s1, s2)


_E2M1 = (0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6)


def _fp4(w, s):
    """Raw MXFP4 rows [n, k/2] (low nibble first) with E8M0 scales [n, k/32] -> fp64 [n, k]."""
    t = torch.tensor(_E2M1, dtype=torch.float64, device=w.device)
    b = w.long()
    v = torch.stack([t[b & 15], t[b >> 4]], -1).reshape(w.shape[0], -1)
    return v * torch.pow(2.0, s.double() - 127).repeat_interleave(32, -1)


def _mx(v):
    """Rows of v MX-quantized and back, in fp64, by AITER's per-1x32 rule (both the activations
    and stage 1's output go through it in AITER's and hk's a8w4 MoE): a 32-column block's scale
    is 2^e, e = ceil(log2(amax / 448)), and its values are rounded to FP8 e4m3."""
    m, k = v.shape
    b = v.view(m, k // 32, 32)
    amax = b.abs().amax(-1).clamp_min(2.0 ** -126)
    e = torch.floor(torch.log2(amax)) - 8
    sc = torch.pow(2.0, e + (amax > 448.0 * torch.pow(2.0, e)).double())[..., None]
    return ((b / sc).clamp(-448, 448).to(torch.float8_e4m3fn).double() * sc).view(m, k)


def _reference(x, raw, ids, wt, limit=10.0):
    """The routed experts' output [M, hidden] in fp64: x and h MX-quantized as the kernels do,
    SiLU-mul with AITER's swiglu clamp, every product and sum exact to fp64."""
    w1, w2, s1, s2 = raw
    e, two_i, _ = w1.shape
    inter, d = two_i // 2, w2.shape[1]
    m, topk = ids.shape
    xd = _mx(x.double())
    y = torch.zeros(m, d, dtype=torch.float64, device=x.device)
    tok = torch.arange(m, device=x.device).repeat_interleave(topk)
    fe, fw = ids.reshape(-1).long(), wt.reshape(-1).double()
    for j in range(e):
        sel = (fe == j).nonzero().flatten()
        if sel.numel() == 0:
            continue
        t = tok[sel]
        gu = xd[t] @ _fp4(w1[j], s1[j * two_i:(j + 1) * two_i]).T
        g, u = gu[:, :inter].clamp(max=limit), gu[:, inter:].clamp(-limit, limit)
        h = _mx(g * torch.sigmoid(g) * u)
        y.index_add_(0, t, (h @ _fp4(w2[j], s2[j * d:(j + 1) * d]).T) * fw[sel, None])
    return y


def _rel(a, ref):
    return ((a.double() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("inter", [1152, 2304], ids=["tp2", "tp1"])
def test_patched_fused_moe_routes_and_agrees(inter):
    import aiter.fused_moe as afm
    from aiter import ActivationType, QuantType

    (x, w1, w2, wt, ids, s1, s2), raw = _inputs(P.MIN_M, inter=inter)     # 2560: where hk wins
    kw = dict(expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
              doweight_stage1=False, w1_scale=s1, w2_scale=s2, swiglu_limit=10.0,
              gate_mode="interleave")
    aiter_out = afm.fused_moe(x, w1, w2, wt, ids, **kw)
    P.apply_patch()
    try:
        before = dict(P.STATS)
        out = afm.fused_moe(x, w1, w2, wt, ids, **kw)
        again = afm.fused_moe(x, w1, w2, wt, ids, **kw)
        under = P.MIN_M - 1                                   # one row short: AITER's
        small = afm.fused_moe(x[:under], w1, w2, wt[:under], ids[:under], **kw)
        took = {k: P.STATS[k] - before.get(k, 0) for k in P.STATS}
    finally:
        P.revert()
    assert took == {"hk": 2, "fallback": 1}
    assert torch.equal(out, again)        # hk's route is deterministic: no atomics, no races
    ref = _reference(x, raw, ids, wt)
    err = _rel(out, ref)
    # Informational only: AITER's own fused_moe against the same reference (usually 0.006-0.009;
    # its occasional bad rows are what this test no longer compares against).
    print(f"inter {inter}: hk {err:.4f}, AITER {_rel(aiter_out, ref):.4f} max-rel vs fp64")
    # hk is at 0.0047 (TP2) and 0.0055 (TP1), the same every run (fixed inputs, a deterministic
    # route): its fp32 accumulation and h's FP8 rounding against fp64's. AITER's correct calls are
    # at 0.006-0.009 here, so 1e-2 still passes any kernel of that accuracy; a few corrupt h values
    # of one (token, expert) pair -- AITER's failure -- cost 0.016-0.056 and do not.
    assert err < 1e-2, err
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
