"""The default-on routes' own code on the GPU, without the engine: each route's stand-in -- the
function or method it puts in the engine's place -- called the way the engine calls it, on every
arch the route is on by default (switches.ROUTES), against the same product in torch. The kernels
are tested on their own elsewhere; this is the branch each route takes, its per-arch limits and
the in-place writes the engine relies on. The engine is a stand-in module where the patch has to
find one (SGLang, SGLang-Diffusion), and the engine's own fallback is a stub that records the
call. tests/hk/ir/test_route_coverage.py says which test covers which (route, arch)."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from hk.integration import sglang as hks  # noqa: E402
from hk.integration import sglang_diffusion as sgld  # noqa: E402
from hk.integration import sglang_moe, vllm_gemm, vllm_gptq, vllm_int4, vllm_moe  # noqa: E402
from hk.integration import vllm_moe_wna16, vllm_norm  # noqa: E402
from hk.ops import moe, w4a16  # noqa: E402
from hk.target import current_arch, get_target  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")]

B = torch.bfloat16


def _on_here(route):
    from hk.integration import switches  # noqa: PLC0415

    if current_arch() not in switches.ROUTES[route].arches:
        pytest.skip(f"{route} is not on by default on {current_arch()}")


def _close(got, want, tol=2e-2):
    err = (got.float() - want.float()).abs().max().item()
    assert err <= tol * want.float().abs().max().item(), err


def _never(*a, **k):
    raise AssertionError("the route fell back to the engine's kernel")


def _stand_in(monkeypatch, home, **attrs):
    """A module `home` (and its parents) holding `attrs`, as the engine would have it."""
    parts = home.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)
    for k, v in attrs.items():
        setattr(sys.modules[home], k, v)
    return sys.modules[home]


# ---------------------------------------------------------------- MoE (vLLM, SGLang)


def _moe_case(tokens, g, experts=8, top_k=2, n=256, k=1024):
    a = torch.randn(tokens, k, device="cuda", dtype=B, generator=g)
    w = torch.randn(experts, n, k, device="cuda", dtype=B, generator=g) * k ** -0.5
    scores = torch.rand(tokens, experts, device="cuda", generator=g)
    topk_w, topk_ids = scores.topk(top_k, dim=1)
    topk_w = topk_w.softmax(dim=1).float()
    sorted_ids, expert_ids, ntp = moe.moe_align(topk_ids.to(torch.int32), 16, experts)
    c = torch.full((tokens, top_k, n), float("nan"), device="cuda", dtype=B)
    want = torch.stack([a[t].float() @ w[topk_ids[t, j]].float().t() * topk_w[t, j]
                        for t in range(tokens) for j in range(top_k)])
    return a, w, c, topk_w, topk_ids, sorted_ids, expert_ids, ntp, want


def test_vllm_moe_takes_a_decode_batch_and_writes_c_in_place(monkeypatch):
    _on_here("vllm_moe")
    monkeypatch.setattr(vllm_moe, "_ORIG", _never)
    tokens = min(2, vllm_moe.block16_max())          # gfx1201: 2 tokens a step
    g = torch.Generator("cuda").manual_seed(1)
    a, w, c, tw, _, sid, eid, ntp, want = _moe_case(tokens, g)
    before = vllm_moe.STATS["hk"]
    vllm_moe.invoke_fused_moe_triton_kernel(
        a, w, c, None, None, tw, sid, eid, ntp, True, 2, {"BLOCK_SIZE_M": 16}, None,
        False, False, False, False, False)
    assert vllm_moe.STATS["hk"] == before + 1
    _close(c.view(-1, c.shape[-1]), want)


def test_vllm_moe_leaves_a_batch_past_this_archs_limit_to_vllm(monkeypatch):
    _on_here("vllm_moe")
    seen = []
    monkeypatch.setattr(vllm_moe, "_ORIG", lambda *a: seen.append(a[0].shape[0]))
    tokens = vllm_moe.block16_max() + 1
    g = torch.Generator("cuda").manual_seed(2)
    a, w, c, tw, _, sid, eid, ntp, _ = _moe_case(tokens, g)
    vllm_moe.invoke_fused_moe_triton_kernel(
        a, w, c, None, None, tw, sid, eid, ntp, True, 2, {"BLOCK_SIZE_M": 16}, None,
        False, False, False, False, False)
    assert seen == [tokens]


def _wna16_case(tokens, second, g, experts=16, top_k=4, n=512, k=1024, group=32):
    """vLLM's moe_wna16 call for one of an AWQ layer's GEMMs (codes uint8 [E, N, K/2], scales
    [E, N, K/G], zero codes uint8 [E, N/2, K/G]) and its float64 result per (token, choice)."""
    q = torch.randint(0, 16, (experts, n, k), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (experts, n, k // group), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(experts, n, k // group, device="cuda", generator=g) * 0.02 + 0.001).half()
    qb = (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8).contiguous()
    zb = (z[:, 0::2] | (z[:, 1::2] << 4)).to(torch.uint8).contiguous()
    w = (q - z.repeat_interleave(group, dim=2)).double() * s.double().repeat_interleave(group, dim=2)
    scores = torch.rand(tokens, experts, device="cuda", generator=g)
    topk_w, topk_ids = scores.topk(top_k, dim=1)
    topk_w = topk_w.softmax(dim=1).float()
    sid, eid, ntp = moe.moe_align(topk_ids.to(torch.int32), 16, experts)
    rows = tokens * (top_k if second else 1)
    a = (torch.randn(rows, k, device="cuda", generator=g) * 0.5).half()
    want = torch.empty(tokens, top_k, n, device="cuda", dtype=torch.float64)
    rows = a.double().view(tokens, top_k if second else 1, k).expand(tokens, top_k, k)
    for e in range(experts):
        hit = topk_ids == e
        if hit.any():
            want[hit] = rows[hit] @ w[e].t()
    if second:
        want *= topk_w.double().unsqueeze(-1)
    want = want.view(tokens * top_k, n)
    c = torch.full((tokens, top_k, n), float("nan"), device="cuda", dtype=torch.float16)
    return a, qb, c, s, zb, topk_w, sid, eid, ntp, want, top_k


@pytest.mark.parametrize("second", [False, True], ids=["w13", "w2"])
def test_vllm_moe_wna16_takes_a_decode_batch_in_vllms_call_and_leaves_past_32(second, monkeypatch):
    """The stand-in for vLLM's invoke_fused_moe_wna16_triton_kernel: up to max_tokens() it runs hk's
    4-bit MoE GEMM on vLLM's own tensors and writes C in place (w2 with the routing weights), past
    it vLLM's kernel."""
    _on_here("vllm_moe_wna16")
    monkeypatch.setattr(vllm_moe_wna16, "_FORCE", True)      # a small layer, not one of VALIDATED
    cfg = {"BLOCK_SIZE_M": 16}
    g = torch.Generator("cuda").manual_seed(31 + second)
    for tokens in sorted({1, 5, 33, 300, vllm_moe_wna16.max_tokens()}):
        a, qb, c, s, zb, tw, sid, eid, ntp, want, top_k = _wna16_case(tokens, second, g)
        monkeypatch.setattr(vllm_moe_wna16, "_ORIG", _never)
        before = vllm_moe_wna16.STATS["hk"]
        vllm_moe_wna16.invoke_fused_moe_wna16_triton_kernel(
            a, qb, c, s, zb, tw, sid, eid, ntp, second, 1 if second else top_k, cfg, None,
            False, True, [0, 32])
        assert vllm_moe_wna16.STATS["hk"] == before + 1
        _close(c.view(-1, c.shape[-1]), want.float())
    seen = []
    monkeypatch.setattr(vllm_moe_wna16, "_ORIG", lambda *a: seen.append(a[2].shape[0]))
    tokens = vllm_moe_wna16.max_tokens() + 1
    a, qb, c, s, zb, tw, sid, eid, ntp, _, top_k = _wna16_case(tokens, second, g)
    vllm_moe_wna16.invoke_fused_moe_wna16_triton_kernel(
        a, qb, c, s, zb, tw, sid, eid, ntp, second, 1 if second else top_k, cfg, None,
        False, True, [0, 32])
    assert seen == [tokens]


def test_vllm_moe_asks_for_16_row_blocks_at_a_decode_batch(monkeypatch):
    _on_here("vllm_moe")
    monkeypatch.setattr(vllm_moe, "_ORIG_CONFIG", lambda *a: {"BLOCK_SIZE_M": 64})
    cfg = vllm_moe.try_get_optimal_moe_config((8, 512, 1024), (8, 1024, 256), 2, None, 1)
    assert cfg["BLOCK_SIZE_M"] == 16


def test_sglang_moe_takes_a_decode_batch_and_its_top_k_matches_torch(monkeypatch):
    _on_here("sglang_moe")
    monkeypatch.setattr(sglang_moe, "_ORIG", _never)
    g = torch.Generator("cuda").manual_seed(3)
    a, w, c, tw, ids, sid, eid, ntp, want = _moe_case(5, g)
    before = sglang_moe.STATS["hk"]
    sglang_moe.invoke_fused_moe_kernel(
        a, w, None, c, None, None, None, tw, ids, sid, eid, ntp, True, 2,
        {"BLOCK_SIZE_M": 16}, None, False, False, False, False, False)
    assert sglang_moe.STATS["hk"] == before + 1
    _close(c.view(-1, c.shape[-1]), want)
    logits = torch.randn(5, 8, device="cuda")
    got_w = torch.empty(5, 2, device="cuda")
    got_i = torch.empty(5, 2, device="cuda", dtype=torch.int32)
    sglang_moe.topk_softmax(got_w, got_i, logits, renormalize=True)
    p = logits.softmax(-1)
    ref_w, ref_i = p.topk(2, -1)
    assert torch.equal(got_i.long(), ref_i)
    _close(got_w, ref_w / ref_w.sum(-1, keepdim=True), 1e-5)


# ---------------------------------------------------------------- AWQ (vLLM)


def _awq_checkpoint(n, k, group, g):
    """A random AutoAWQ checkpoint and the W it means (as test_w4a16's)."""
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()

    def pack(v):
        r, c = v.shape
        v = v.reshape(r, c // 8, 8).to(torch.int64)
        shift = torch.tensor([4 * o for o in w4a16.AWQ_ORDER], device=v.device)
        words = (v << shift).sum(-1)
        return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)

    w = ((q - z.repeat_interleave(group, 0)).float()
         * s.float().repeat_interleave(group, 0)).t()
    return pack(q), pack(z), s, w


def test_vllm_int4_takes_up_to_this_archs_limit_and_leaves_the_rest(monkeypatch):
    _on_here("vllm_int4")
    seen = []
    monkeypatch.setattr(vllm_int4, "_vllm_awq", lambda x, *a: seen.append(x.shape[0]))
    n, k, group = 512, 1024, 128
    g = torch.Generator("cuda").manual_seed(4)
    qweight, qzeros, scales, w = _awq_checkpoint(n, k, group, g)
    qw, s, zs = w4a16.repack_awq(qweight, qzeros, scales, group, dtype=torch.float16)
    top = vllm_int4._max_m()                         # 64, 16 on gfx950
    for m in (1, top):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_int4.STATS["hk"]
        y = vllm_int4.w4a16(x, qweight, scales, qzeros, qw, s, zs, group)
        assert vllm_int4.STATS["hk"] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    large = vllm_int4._large_max_m()                 # gfx1201: the large-batch form above top
    for m in ((top + 1, 300, large) if large else ()):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_int4.STATS["hk_large"]
        y = vllm_int4.w4a16(x, qweight, scales, qzeros, qw, s, zs, group)
        assert vllm_int4.STATS["hk_large"] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    past = max(top, large) + 1
    vllm_int4.w4a16(torch.randn(past, k, device="cuda", dtype=torch.float16),
                    qweight, scales, qzeros, qw, s, zs, group)
    assert seen == [past]


def test_vllm_gptq_takes_min_m_to_64_and_leaves_the_rest(monkeypatch):
    _on_here("vllm_gptq")
    seen = []
    monkeypatch.setattr(vllm_gptq, "_vllm_gptq", lambda x, *a: seen.append(x.shape[0]))
    n, k, group = 512, 1024, 32
    g = torch.Generator("cuda").manual_seed(5)
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device="cuda", dtype=torch.int32)
    # the route's layout: symmetric, scales in 16-row blocks, an empty zs slot
    qw, sc, _ = w4a16.repack_gptq(qweight, qzeros, s, group, dtype=torch.float16, symmetric=True)
    zs = sc.new_empty(0)
    w = ((q - 8).float() * s.float().repeat_interleave(group, 0)).t()
    for m in sorted({vllm_gptq.MIN_M, 2, 4, 64}):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_gptq.STATS["hk"]
        y = vllm_gptq.gptq_w4a16(x, None, None, None, None, qw, sc, zs, group)
        assert vllm_gptq.STATS["hk"] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    for m in (vllm_gptq.MIN_M - 1, 65):
        if m > 0:
            vllm_gptq.gptq_w4a16(torch.randn(m, k, device="cuda", dtype=torch.float16),
                                 None, None, None, None, qw, sc, zs, group)
    assert seen == [m for m in (vllm_gptq.MIN_M - 1, 65) if m > 0]


def test_vllm_gptq_hybrid_takes_every_batch_vllm_sends_to_triton_in_place(monkeypatch):
    """RDNAHybridW4A16LinearKernel's layers (gfx1201): hk reads vLLM's own skinny-layout tensors and
    takes the batches vLLM would run on Triton, up to HYBRID_MAX_M tokens; vLLM's op the rest,
    called without zero points."""
    _on_here("vllm_gptq")
    seen = []
    monkeypatch.setattr(vllm_gptq, "_SKINNY", {"max_m": 5, "limit": int(32768 * 1.2), "gfx115x": False})
    monkeypatch.setattr(torch.ops, "vllm", types.SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda x, w_q, w_s, w_zp, bias, cu, g: seen.append((x.shape[0], w_zp))),
        raising=False)
    n, k, group = 512, 7168, 32                # K * 6 > vLLM's LDS limit: Triton from 5 tokens
    g = torch.Generator("cuda").manual_seed(6)
    q = torch.randint(0, 16, (n, k), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(n, k // group, device="cuda", generator=g) * 0.02 + 0.005).half()
    words = sum(q.to(torch.int64).reshape(n, k // 8, 8)[:, :, c] << (4 * i)
                for i, c in enumerate((0, 2, 4, 6, 1, 3, 5, 7)))     # ExLlama's nibble order
    w_q = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).view(torch.int8)
    w = (q - 8).float() * s.float().repeat_interleave(group, 1)
    for m in (6, 13, 40, vllm_gptq.HYBRID_MAX_M, vllm_gptq.HYBRID_MAX_M + 1, 300):
        path = "hk" if m <= vllm_gptq.HYBRID_MAX_M else "hk_large"
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_gptq.HYBRID.stats[path]
        y = vllm_gptq.gptq_w4a16_hybrid(x, w_q, s, group, 64)
        assert vllm_gptq.HYBRID.stats[path] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    for m in (1, 4):
        vllm_gptq.gptq_w4a16_hybrid(torch.randn(m, k, device="cuda", dtype=torch.float16),
                                    w_q, s, group, 64)
    assert seen == [(1, None), (4, None)]


def test_vllm_quark_takes_min_m_to_64_on_a_copy_and_leaves_the_rest(monkeypatch):
    """Quark W4A16 layers on RDNA3W4A16LinearKernel (gfx1100): hk's symmetric layout of Quark's own
    codes (repack_quark: AWQ's nibble order, signed codes), MIN_M to 64 tokens on hk's decode GEMM,
    the rest on vLLM's gptq_gemm_rdna3."""
    from hk.integration import vllm_quark
    from hk.ops.w4a16 import AWQ_ORDER

    _on_here("vllm_quark")
    seen = []
    monkeypatch.setattr(vllm_gptq, "_vllm_gptq", lambda x, *a: seen.append(x.shape[0]))
    n, k, group = 512, 1024, 128
    g = torch.Generator("cuda").manual_seed(8)
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)   # zero 8
    signed = (q ^ 8).to(torch.int64)
    quark = sum((signed[:, c::8] & 0xF) << (4 * AWQ_ORDER[c]) for c in range(8))
    quark = torch.where(quark >= 2 ** 31, quark - 2 ** 32, quark).to(torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    qw, sc, _ = w4a16.repack_quark(quark, s, group, dtype=torch.float16)
    w = ((q - 8).float() * s.float().repeat_interleave(group, 0)).t()
    for m in sorted({vllm_gptq.MIN_M, 2, 4, 64}):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_quark.PATCH.stats["hk"]
        y = vllm_quark.quark_w4a16(x, None, None, None, qw, sc, group)
        assert vllm_quark.PATCH.stats["hk"] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    vllm_quark.quark_w4a16(torch.randn(65, k, device="cuda", dtype=torch.float16),
                           None, None, None, qw, sc, group)
    assert seen == [65]


def test_vllm_quark_hybrid_takes_every_batch_vllm_sends_to_triton_in_place(monkeypatch):
    """Quark W4A16 layers on RDNAHybridW4A16LinearKernel (gfx1201), at Quark's group 128: hk reads
    vLLM's own skinny-layout tensors and takes the batches vLLM would run on Triton (decode forms to
    HYBRID_MAX_M tokens, the large-batch form above); vLLM's op the rest, without zero points."""
    from hk.integration import vllm_quark

    _on_here("vllm_quark")
    seen = []
    monkeypatch.setattr(vllm_gptq, "_SKINNY", {"max_m": 5, "limit": int(32768 * 1.2), "gfx115x": False})
    monkeypatch.setattr(torch.ops, "vllm", types.SimpleNamespace(
        rdna_hybrid_w4a16_apply=lambda x, w_q, w_s, w_zp, bias, cu, g: seen.append((x.shape[0], w_zp, g))),
        raising=False)
    n, k, group = 512, 7168, 128
    g = torch.Generator("cuda").manual_seed(7)
    q = torch.randint(0, 16, (n, k), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(n, k // group, device="cuda", generator=g) * 0.02 + 0.005).half()
    words = sum(q.to(torch.int64).reshape(n, k // 8, 8)[:, :, c] << (4 * i)
                for i, c in enumerate((0, 2, 4, 6, 1, 3, 5, 7)))     # ExLlama's nibble order
    w_q = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32).view(torch.int8)
    w = (q - 8).float() * s.float().repeat_interleave(group, 1)
    for m in (6, 13, 40, vllm_gptq.HYBRID_MAX_M, vllm_gptq.HYBRID_MAX_M + 1, 300):
        path = "hk" if m <= vllm_gptq.HYBRID_MAX_M else "hk_large"
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_quark.HYBRID.stats[path]
        y = vllm_quark.quark_w4a16_hybrid(x, w_q, s, group, 64)
        assert vllm_quark.HYBRID.stats[path] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    for m in (1, 4):
        vllm_quark.quark_w4a16_hybrid(torch.randn(m, k, device="cuda", dtype=torch.float16),
                                      w_q, s, group, 64)
    assert seen == [(1, None, group), (4, None, group)]


# ---------------------------------------------------------------- linear (SGLang, SGLang-Diffusion)


class _Layer:
    def __init__(self, weight):
        self.weight = weight


def test_sglang_linear_routes_its_apply_through_hk_and_reverts(monkeypatch):
    _on_here("sglang_linear")

    class Unquantized:
        def apply(self, layer, x, bias=None):
            raise AssertionError("SGLang's own apply ran for a bf16 call on the GPU")

    _stand_in(monkeypatch, hks._LINEAR_HOMES[0], UnquantizedLinearMethod=Unquantized)
    orig = Unquantized.apply
    try:
        assert "linear -> hk" in hks.apply_linear()
        w = torch.randn(4096, 4096, device="cuda", dtype=B) * 4096 ** -0.5
        for m in (1, 64):
            x = torch.randn(m, 4096, device="cuda", dtype=B)
            _close(Unquantized().apply(_Layer(w), x), torch.nn.functional.linear(x, w))
    finally:
        hks.revert()
        vllm_gemm.set_routes(None)
    assert Unquantized.apply is orig


def test_sglang_norm_takes_gemma_rmsnorm_through_hk_and_reverts(monkeypatch):
    """SGLang's GemmaRMSNorm (y = x_hat * (1 + w)) on hk's kernels: the fused add out of place
    (x and the residual untouched, as on SGLang's HIP paths) and within half a bf16 unit of float64 with the one exact, the plain norm the same; a call hk
    cannot take (not contiguous) goes back to SGLang's forward and is counted."""
    _on_here("sglang_norm")
    calls = []

    class GemmaRMSNorm:
        def __init__(self, weight):
            self.weight, self.variance_epsilon = weight, 1e-6

        def forward_hip(self, x, residual=None, post_residual_addition=None):
            calls.append(x.shape)
            return "sglang"

    _stand_in(monkeypatch, hks._NORM_HOME, GemmaRMSNorm=GemmaRMSNorm, _use_aiter=False,
              _has_vllm_rms_norm=False)
    orig = GemmaRMSNorm.forward_hip
    g = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(5120, device="cuda", generator=g) * 0.1).to(B)
    layer = GemmaRMSNorm(w)

    def ref(z):
        z = z.double()
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w.double())

    def ulps(got, want):
        e = torch.floor(torch.log2(want.to(B).double().abs().clamp_min(2.0 ** -14)))
        return ((got.double() - want).abs() / torch.pow(2.0, e - 7)).max().item()

    try:
        assert "norm -> hk" in hks.apply_norm()
        for m in (1, 8, 300):
            x = torch.randn(m, 5120, device="cuda", dtype=B, generator=g)
            r = (torch.randn(m, 5120, device="cuda", generator=g) * 4).to(B)
            z, want_r = x.double() + r.double(), (x.float() + r.float()).to(B)
            assert ulps(layer.forward_hip(x), ref(x)) <= 0.51
            x0, r0 = x.clone(), r.clone()
            y, h = layer.forward_hip(x, r)
            assert torch.equal(h, want_r) and torch.equal(x, x0) and torch.equal(r, r0)
            assert ulps(y, ref(z)) <= 0.51
        before = hks.NORM_STATS["hk"]
        assert layer.forward_hip(torch.randn(5120, 8, device="cuda", dtype=B).t()) == "sglang"
        assert hks.NORM_STATS["x not contiguous"] == 1 and hks.NORM_STATS["hk"] == before
        assert len(calls) == 1
    finally:
        assert "norm -> sglang" in hks.revert()
    assert GemmaRMSNorm.forward_hip is orig


def test_vllm_norm_takes_gemma_rmsnorm_through_hk_and_reverts(monkeypatch, tmp_path):
    """vLLM's GemmaRMSNorm (forward_cuda = its native PyTorch ops) on hk's kernels: y = x_hat *
    (1 + w) within half a bf16 unit of float64, the fused add out of place with its sum exact (x
    and the residual untouched, as vLLM's native path); a call hk cannot take goes to vLLM's
    forward and is counted, and one while torch.compile traces takes hk's op and counts nothing."""
    _on_here("vllm_norm")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))     # apply() names the compile cache
    calls = []

    class GemmaRMSNorm:
        def __init__(self, weight):
            self.weight, self.variance_epsilon = weight, 1e-6

        def forward_native(self, x, residual=None):
            calls.append(x.shape)
            return "vllm"

        def forward_cuda(self, x, residual=None):
            calls.append(x.shape)
            return "vllm"

    _stand_in(monkeypatch, vllm_norm._HOME, GemmaRMSNorm=GemmaRMSNorm)
    orig = GemmaRMSNorm.forward_cuda
    g = torch.Generator(device="cuda").manual_seed(6)
    w = (torch.randn(2048, device="cuda", generator=g) * 0.1).to(B)
    layer = GemmaRMSNorm(w)

    def ref(z):
        z = z.double()
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w.double())

    def ulps(got, want):
        e = torch.floor(torch.log2(want.to(B).double().abs().clamp_min(2.0 ** -14)))
        return ((got.double() - want).abs() / torch.pow(2.0, e - 7)).max().item()

    try:
        assert "norm -> hk" in vllm_norm.apply()
        for m in (1, 8, 300):
            x = torch.randn(m, 2048, device="cuda", dtype=B, generator=g)
            r = (torch.randn(m, 2048, device="cuda", generator=g) * 4).to(B)
            z, want_r = x.double() + r.double(), (x.float() + r.float()).to(B)
            assert ulps(layer.forward_cuda(x), ref(x)) <= 0.51
            x0, r0 = x.clone(), r.clone()
            y, h = layer.forward_cuda(x, r)
            assert torch.equal(h, want_r) and torch.equal(x, x0) and torch.equal(r, r0)
            assert ulps(y, ref(z)) <= 0.51
        before = vllm_norm.STATS["hk"]
        assert layer.forward_cuda(torch.randn(2048, 8, device="cuda", dtype=B).t()) == "vllm"
        assert vllm_norm.STATS["x not contiguous"] == 1 and vllm_norm.STATS["hk"] == before
        assert layer.forward_native(x) == "vllm"             # compiled calls: opt-in (HK_VLLM_NORM=compile)
        n = sum(vllm_norm.STATS.values())
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        x = torch.randn(8, 2048, device="cuda", dtype=B, generator=g)
        assert ulps(layer.forward_cuda(x), ref(x)) <= 0.51
        assert sum(vllm_norm.STATS.values()) == n          # nothing written while tracing
        assert len(calls) == 2
    finally:
        assert "norm -> vllm" in vllm_norm.revert()
    assert GemmaRMSNorm.forward_cuda is orig


def test_vllm_norm_compiles_into_one_op_and_replays_in_a_cuda_graph(monkeypatch, tmp_path):
    """Under torch.compile (vLLM traces forward_native, its custom ops off) the route is one opaque
    node per norm -- fullgraph, dynamic rows -- and the compiled function matches the eager route
    bit for bit. Captured in a CUDA graph, a replay on new inputs written into the captured
    buffers gives what an eager call on them gives: no host sync, nothing frozen at capture."""
    _on_here("vllm_norm")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("HK_VLLM_NORM", "compile")   # compiled calls are opt-in

    class GemmaRMSNorm:
        def __init__(self, weight):
            self.weight, self.variance_epsilon = weight, 1e-6

        def forward_native(self, x, residual=None):
            raise AssertionError("vLLM's native path ran")

        def forward_cuda(self, x, residual=None):
            return self.forward_native(x, residual)

    _stand_in(monkeypatch, vllm_norm._HOME, GemmaRMSNorm=GemmaRMSNorm)
    g = torch.Generator(device="cuda").manual_seed(7)
    w = (torch.randn(2048, device="cuda", generator=g) * 0.1).to(B)
    a, b = GemmaRMSNorm(w), GemmaRMSNorm(w.flip(0).contiguous())

    def block(x, r):                      # two norms, as a decoder layer: add+norm, then norm
        y, h = a.forward_native(x, r)
        return b.forward_native(y * 2.0), h

    try:
        assert "hk-norm" in vllm_norm.apply()
        torch._dynamo.reset()
        seen = []

        def backend(gm, example_inputs):
            seen.append(str(gm.graph))
            return gm.forward

        compiled = torch.compile(block, backend=backend, fullgraph=True, dynamic=True)
        for m in (5, 64, 300):
            x = torch.randn(m, 2048, device="cuda", dtype=B, generator=g)
            r = torch.randn(m, 2048, device="cuda", dtype=B, generator=g)
            want_y, want_h = block(x, r)
            got_y, got_h = compiled(x, r)
            assert torch.equal(got_y, want_y) and torch.equal(got_h, want_h)
        graph = "\n".join(seen)
        assert "hk_vllm.gemma_add_rms_norm" in graph and "hk_vllm.gemma_rms_norm" in graph
        assert len(seen) == 1                # dynamic rows: one graph for every m

        x = torch.randn(64, 2048, device="cuda", dtype=B, generator=g)
        r = torch.randn(64, 2048, device="cuda", dtype=B, generator=g)
        block(x, r)                          # warm: kernels built before capture
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        before = vllm_norm.ADD.captured["hk"]
        with torch.cuda.graph(cg):
            out_y, out_h = block(x, r)
        assert vllm_norm.ADD.captured["hk"] == before + 1
        for _ in range(3):
            x.copy_(torch.randn(64, 2048, device="cuda", dtype=B, generator=g))
            r.copy_(torch.randn(64, 2048, device="cuda", dtype=B, generator=g))
            cg.replay()
            want_y, want_h = block(x, r)
            assert torch.equal(out_y, want_y) and torch.equal(out_h, want_h)
    finally:
        assert "norm -> vllm" in vllm_norm.revert()


def test_sglang_norm_is_ordered_on_the_engines_stream(monkeypatch):
    """SGLang runs its model on a stream of its own. The route launches on torch's current
    stream, so a norm whose input a busy kernel on that stream is still writing waits for it:
    10 of 10 exact. (hk.ops' pybind entry points launch on the null stream on RDNA, which a
    non-blocking stream does not wait for: that was wrong 10 of 10 here.)"""
    _on_here("sglang_norm")

    class GemmaRMSNorm:
        def __init__(self, weight):
            self.weight, self.variance_epsilon = weight, 1e-6

        def forward_hip(self, x, residual=None, post_residual_addition=None):
            raise AssertionError("SGLang's own forward ran")

    _stand_in(monkeypatch, hks._NORM_HOME, GemmaRMSNorm=GemmaRMSNorm, _use_aiter=False,
              _has_vllm_rms_norm=False)
    layer = GemmaRMSNorm(torch.zeros(5120, device="cuda", dtype=B))
    side = torch.cuda.Stream()
    big = torch.randn(4096, 4096, device="cuda")
    try:
        assert "norm -> hk" in hks.apply_norm()
        exact = 0
        for it in range(10):
            with torch.cuda.stream(side):
                for _ in range(3):
                    big = big @ big * 1e-4                     # keep the stream busy
                x = torch.full((122, 5120), 1.0, device="cuda", dtype=B) + it
                r = torch.full((122, 5120), 2.0, device="cuda", dtype=B)
                y, h = layer.forward_hip(x, r)
                plain = layer.forward_hip(x)
            torch.cuda.synchronize()
            exact += int(torch.equal(h, x + r) and torch.equal(y, torch.ones_like(y))
                         and torch.equal(plain, torch.ones_like(plain)))
        assert exact == 10
    finally:
        hks.revert()


def test_sgld_linear_takes_a_dit_projection_through_hk(monkeypatch):
    _on_here("sgld_linear")

    class Unquantized:
        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    _stand_in(monkeypatch, sgld.LINEAR_HOME, UnquantizedLinearMethod=Unquantized)
    monkeypatch.setattr(sgld, "MARGIN", 0.0)         # take hk's GEMM whatever the timing says
    monkeypatch.setattr(sgld, "DECISIONS", {})
    orig = Unquantized.apply
    try:
        assert "linear -> hk" in sgld.apply_linear()
        w = torch.randn(3072, 3072, device="cuda", dtype=B) * 3072 ** -0.5
        x = torch.randn(1, sgld.MIN_M, 3072, device="cuda", dtype=B)
        before = dict(sgld.LINEAR_STATS)
        _close(Unquantized().apply(_Layer(w), x), torch.nn.functional.linear(x, w))
        assert sgld.LINEAR_STATS["hk"] == before["hk"] + 1
        small = torch.randn(16, 3072, device="cuda", dtype=B)   # below MIN_M: the engine's
        Unquantized().apply(_Layer(w), small)
        assert sgld.LINEAR_STATS["engine"] == before["engine"] + 1
        # a bias (Wan's projections): hk's GEMM with the bias in its store
        bias = torch.randn(3072, device="cuda", dtype=B)
        _close(Unquantized().apply(_Layer(w), x, bias), torch.nn.functional.linear(x, w, bias))
        assert sgld.LINEAR_STATS["hk"] == before["hk"] + 2
        monkeypatch.setattr(sgld, "LINEAR_BIAS", False)              # biased: the engine's
        Unquantized().apply(_Layer(w), x, bias)
        assert sgld.LINEAR_STATS["engine"] == before["engine"] + 2
        # rows of a wider buffer (a DiT fc2 reading half of its gate / up buffer): taken as they
        # are, the row stride passed to the GEMM; the other half must not leak into the result
        buf = torch.randn(1, sgld.MIN_M, 2 * 3072, device="cuda", dtype=B)
        for half in (buf[..., :3072], buf[..., 3072:]):
            assert not half.is_contiguous()
            _close(Unquantized().apply(_Layer(w), half),
                   torch.nn.functional.linear(half.contiguous(), w))
        assert sgld.LINEAR_STATS["hk"] == before["hk"] + 4
        odd = buf[..., 1:3073]                                       # not 16-byte aligned
        Unquantized().apply(_Layer(w), odd)
        assert sgld.LINEAR_STATS["engine"] == before["engine"] + 3
    finally:
        sgld.revert_linear()
    assert Unquantized.apply is orig


def test_sgld_vaeconv_runs_wans_causal_conv3d_in_frame_chunks_and_matches(monkeypatch):
    _on_here("sgld_vaeconv")

    class WanCausalConv3d(torch.nn.Conv3d):     # pads time itself, as Wan's VAE does
        def __init__(self, cin, cout, k):
            super().__init__(cin, cout, k, padding=(1, 1, 1))
            self.padding = (0, 0, 0)

        def forward(self, x):
            return super().forward(torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0)))

    def cl3d(weight):
        return True

    mod = _stand_in(monkeypatch, sgld.VAE_HOME, WanCausalConv3d=WanCausalConv3d,
                    _conv3d_weight_is_channels_last_3d=cl3d)
    orig = WanCausalConv3d._conv_forward
    conv = WanCausalConv3d(16, 8, 3).to("cuda", B)
    g = torch.Generator(device="cuda").manual_seed(0)
    try:
        assert "calls of 2 output frames" in sgld.apply_vaeconv()
        assert not mod._conv3d_weight_is_channels_last_3d(None)    # the VAE stays NCDHW
        for conv2d in (True, False):            # one conv2d over the stacked taps; frame chunks
            monkeypatch.setattr(sgld, "VAE_CONV2D", conv2d)
            before = dict(sgld.VAE_STATS)
            for frames in (1, 2, 5, 6):         # chunks: 1 and 2 one call, 5 and 6 chunked
                x = torch.randn(1, 16, frames, 12, 20, device="cuda", dtype=B, generator=g)
                want = torch.nn.functional.conv3d(
                    torch.nn.functional.pad(x, (1, 1, 1, 1, 2, 0)).float(), conv.weight.float(),
                    conv.bias.float())
                _close(conv(x), want)
            if conv2d:
                assert sgld.VAE_STATS["conv2d"] == before["conv2d"] + 4
            else:
                assert sgld.VAE_STATS["chunked"] == before["chunked"] + 2
                assert sgld.VAE_STATS["whole"] == before["whole"] + 2
    finally:
        sgld.revert_vaeconv()
    assert WanCausalConv3d._conv_forward is orig and mod._conv3d_weight_is_channels_last_3d is cl3d


def _complex_rope(x, rope):
    """The engine's expression (Qwen-Image 2.1's apply_rope fallback)."""
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def _rope_case(seq, g, heads=32, d=128):
    x = torch.randn(1, seq, heads, d, device="cuda", dtype=B, generator=g) * 3
    ang = torch.rand(seq, d // 2, device="cuda", generator=g) * 100
    return x, torch.polar(torch.ones_like(ang), ang)


@pytest.mark.parametrize("seq", [45, 4096, 4141])
def test_rotary_pairs_is_torchs_complex_rope_bit_for_bit(seq):
    from hk.ops import fused

    if not get_target(current_arch()).strips:
        pytest.skip("rotary_pairs runs on register strips")
    x, rope = _rope_case(seq, torch.Generator(device="cuda").manual_seed(seq))
    want = _complex_rope(x, rope)
    cos, sin = rope.real.contiguous(), rope.imag.contiguous()

    def same(got):
        # RDNA's torch multiplies complex numbers in the fused form rotary_pairs computes: bit
        # for bit. gfx950's torch contracts differently: a bf16 unit apart, on a few elements.
        if current_arch() in ("gfx1100", "gfx1201"):
            return torch.equal(got, want)
        _close(got, want, tol=1e-2)
        return (got != want).float().mean().item() < 1e-4

    assert same(fused.rotary_pairs(x, cos, sin))
    y = x.clone()
    assert fused.rotary_pairs(y, cos, sin, out=y) is y and same(y)                   # in place
    with pytest.raises(ValueError):
        fused.rotary_pairs(x.float(), cos, sin)


def test_sgld_rope_takes_qwen_image_21s_rope_through_hk_and_reverts(monkeypatch):
    _on_here("sgld_rope")

    def apply_rope(x, rope):
        return _complex_rope(x, rope)

    mod = _stand_in(monkeypatch, sgld.QWEN21_HOME, apply_rope=apply_rope)
    g = torch.Generator(device="cuda").manual_seed(1)
    x, rope = _rope_case(4141, g)
    try:
        assert "hk.ops.rotary_pairs" in sgld.apply_rope()
        before = dict(sgld.ROPE_STATS)
        assert torch.equal(mod.apply_rope(x, rope), _complex_rope(x, rope))
        assert torch.equal(mod.apply_rope(x, rope), _complex_rope(x, rope))     # cached tables
        assert sgld.ROPE_STATS["hk"] == before["hk"] + 2
        two = torch.cat([x, x])                                     # two sequences: the engine's
        assert torch.equal(mod.apply_rope(two, rope), _complex_rope(two, rope))
        assert sgld.ROPE_STATS["engine"] == before["engine"] + 1
        with torch.inference_mode():                                # as SGLang-Diffusion runs it
            xi, ri = _rope_case(4096, g)
            assert torch.equal(mod.apply_rope(xi, ri), _complex_rope(xi, ri))
    finally:
        sgld.revert_rope()
    assert mod.apply_rope is apply_rope


class _QwenQKNorm(torch.nn.Module):
    """SGLang's RMSNorm(cast_x_before_out_mul=True) as forward_native computes it."""
    def __init__(self, w, eps=1e-6):
        super().__init__()
        self.weight, self.variance_epsilon = w, eps
        self.cast_x_before_out_mul, self.variance_size_override = True, None
        self.override_orig_dtype = None

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return self.weight * xf.to(x.dtype)


def test_rmsnorm_rotary_pairs_matches_the_norm_then_rope_chain():
    from hk.ops import fused

    if not get_target(current_arch()).strips:
        pytest.skip("rmsnorm_rotary_pairs runs on register strips")
    g = torch.Generator(device="cuda").manual_seed(7)
    x, rope = _rope_case(4141, g)
    norm = _QwenQKNorm((torch.rand(128, device="cuda", generator=g) + 0.5).to(B))
    want = _complex_rope(norm(x), rope)
    got = fused.rmsnorm_rotary_pairs(x, norm.weight, 1e-6, rope.real.contiguous(),
                                     rope.imag.contiguous())
    # the mean's sum order is hk's: a bf16 unit apart on a few elements, never more
    assert (got != want).float().mean().item() < 1e-4
    _close(got, want, tol=1e-2)


def test_sgld_qknorm_takes_qwen_image_21s_qk_norm_and_rope_through_hk(monkeypatch):
    _on_here("sgld_qknorm")

    def apply_qk_norm_rope(x, norm, rope):
        return _complex_rope(norm(x), rope)

    mod = _stand_in(monkeypatch, sgld.QWEN21_HOME, apply_qk_norm_rope=apply_qk_norm_rope)
    g = torch.Generator(device="cuda").manual_seed(2)
    x, rope = _rope_case(4141, g)
    norm = _QwenQKNorm((torch.rand(128, device="cuda", generator=g) + 0.5).to(B))
    try:
        assert "rmsnorm_rotary_pairs" in sgld.apply_qknorm()
        before = dict(sgld.QKNORM_STATS)
        with torch.inference_mode():
            got = mod.apply_qk_norm_rope(x, norm, rope)
        assert (got != _complex_rope(norm(x), rope)).float().mean().item() < 1e-4
        assert sgld.QKNORM_STATS["hk"] == before["hk"] + 1
        norm.cast_x_before_out_mul = False                         # not what the kernel computes
        mod.apply_qk_norm_rope(x, norm, rope)
        assert sgld.QKNORM_STATS["engine"] == before["engine"] + 1
    finally:
        sgld.revert_qknorm()
    assert mod.apply_qk_norm_rope is apply_qk_norm_rope


def _lnss_ref(x, scale, shift, eps=1e-6, residual=None, gate=None, weight=None, bias=None):
    """torch's chain as SGLang-Diffusion runs it: r' = r + x * gate (the gate's dtype promotes),
    FP32LayerNorm(r') back in r''s dtype, then * (1 + scale) + shift."""
    r = x if residual is None else residual + (x if gate is None else x * gate)
    n = torch.nn.functional.layer_norm(r.float(), (r.shape[-1],), weight, bias, eps).to(r.dtype)
    y = (n.float() * (1 + scale.float()) + shift.float()).to(r.dtype)
    return y if residual is None else (y, r)


@pytest.mark.parametrize("fp32_gate", [False, True])
def test_layernorm_scale_shift_is_the_dit_norm_chain(fp32_gate):
    from hk.ops import norm

    if not get_target(current_arch()).strips:
        pytest.skip("layernorm_scale_shift runs on register strips")
    g = torch.Generator(device="cuda").manual_seed(3)
    d = 1536
    x = torch.randn(2, 640, d, device="cuda", dtype=B, generator=g)
    r = torch.randn(2, 640, d, device="cuda", dtype=B, generator=g) * 4
    sc = torch.randn(2, 1, d, device="cuda", generator=g) * 0.1
    sh = torch.randn(2, 1, d, device="cuda", generator=g) * 0.1
    gate = torch.randn(2, 1, d, device="cuda", generator=g)
    w = torch.rand(d, device="cuda", generator=g) + 0.5
    b = torch.randn(d, device="cuda", generator=g) * 0.1
    _close(norm.layernorm_scale_shift(x, sc, sh), _lnss_ref(x, sc, sh))
    gate = gate if fp32_gate else None
    rd = torch.float32 if fp32_gate else B
    y, ro = norm.layernorm_scale_shift(x, sc, sh, residual=r, gate=gate, weight=w, bias=b,
                                       round_residual=not fp32_gate, out_dtype=rd,
                                       residual_dtype=rd)
    want_y, want_r = _lnss_ref(x, sc, sh, residual=r, gate=gate, weight=w, bias=b)
    assert y.dtype == want_y.dtype and ro.dtype == want_r.dtype
    _close(ro, want_r, tol=1e-6 if fp32_gate else 1e-2)
    _close(y, want_y)


def test_sgld_ditnorm_takes_a_dit_blocks_norms_through_hk(monkeypatch):
    _on_here("sgld_ditnorm")

    class _Norm:
        def __init__(self, d, affine):
            self.eps = 1e-6
            self.weight = torch.rand(d, device="cuda") + 0.5 if affine else None
            self.bias = torch.randn(d, device="cuda") * 0.1 if affine else None

    class _ScaleResidualNormScaleShift:
        def __init__(self, d, affine=False, norm_type="layer"):
            self.norm_type, self.norm = norm_type, _Norm(d, affine)

        def forward_native(self, residual, x, gate, shift, scale):
            return _lnss_ref(x, scale, shift, self.norm.eps, residual=residual,
                             gate=None if isinstance(gate, int) else gate,
                             weight=self.norm.weight, bias=self.norm.bias)

    class _NormScaleShift:
        def __init__(self, d):
            self.norm_type, self.norm = "layer", _Norm(d, False)

        def forward_native(self, x, shift, scale):
            return _lnss_ref(x, scale, shift, self.norm.eps)

    _stand_in(monkeypatch, sgld.DITNORM_HOME, _ScaleResidualNormScaleShift=_ScaleResidualNormScaleShift,
              _NormScaleShift=_NormScaleShift)
    orig = _ScaleResidualNormScaleShift.forward_native
    g = torch.Generator(device="cuda").manual_seed(4)
    d = 1536
    h = torch.randn(1, 1024, d, device="cuda", dtype=B, generator=g) * 4
    a = torch.randn(1, 1024, d, device="cuda", dtype=B, generator=g)
    sc, sh, gate = (torch.randn(1, 1, d, device="cuda", generator=g) * 0.1 for _ in range(3))
    zero = torch.zeros((1,), device="cuda", dtype=B)
    try:
        assert "layernorm_scale_shift" in sgld.apply_ditnorm()
        before = dict(sgld.DITNORM_STATS)
        with torch.inference_mode():                       # Wan's three calls
            _close(_NormScaleShift(d).forward_native(h, sh, sc), _lnss_ref(h, sc, sh))
            aff = _ScaleResidualNormScaleShift(d, affine=True)
            y, r = aff.forward_native(h, a, gate, zero, zero)
            wy, wr = orig(aff, h, a, gate, zero, zero)
            assert y.dtype == wy.dtype == torch.float32 and r.dtype == wr.dtype
            _close(y, wy)
            y, r = _ScaleResidualNormScaleShift(d).forward_native(h, a, 1, sh, sc)
            assert torch.equal(r, h + a)
        assert sgld.DITNORM_STATS["hk"] == before["hk"] + 3
        _ScaleResidualNormScaleShift(d, norm_type="rms").forward_native(h, a, 1, sh, sc)
        assert sgld.DITNORM_STATS["engine"] == before["engine"] + 1     # not a LayerNorm
    finally:
        sgld.revert_ditnorm()
    assert _ScaleResidualNormScaleShift.forward_native is orig


# ---------------------------------------------------------------- MiniMax H3's image edit

class _H3Conv3d(torch.nn.Conv3d):
    """MiniMax H3's VAE BaseConv3d as SGLang writes it: spatial padding in pad_mode, then causal
    time padding (one frame: kt - 1 zero frames in front, or the frame repeated), then a conv3d."""

    def __init__(self, cin, cout, k, stride=1, padding=0, padding_mode="zeros", padding_mode_t=None,
                 causal=True):
        super().__init__(cin, cout, k, stride=stride, padding=padding, padding_mode=padding_mode)
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal

    def _apply_padding(self, x):
        F = torch.nn.functional
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
                  mode=self.pad_mode)
        if x.shape[2] > 1:
            return F.pad(x, (0, 0, 0, 0, self.padding[0] * 2, 0), mode=self.pad_mode_t)
        if self.pad_mode_t == "constant":
            return F.pad(x, (0, 0, 0, 0, self.kernel_size[0] - 1, 0), mode="constant")
        return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def forward(self, x, *, pre_padded=False):
        if sum(self.padding) == 0:
            return super().forward(x)
        if not pre_padded:
            x = self._apply_padding(x)
        return torch.nn.functional.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0,
                                          dilation=self.dilation)


def test_sgld_h3vae_runs_one_frame_as_a_conv2d_with_the_last_tap(monkeypatch):
    """Not a hk kernel (MIOpen's conv2d for its conv3d); RDNA only: on gfx950 the conv2d's mean error
    is 1.6x the conv3d's at these shapes, and the route leaves the engine's conv3d there."""
    _stand_in(monkeypatch, sgld.H3VAE_HOME, BaseConv3d=_H3Conv3d)
    if get_target(current_arch()).wave != 32:
        assert "not applied" in sgld.apply_h3vae() and not getattr(_H3Conv3d.forward, "_hk", False)
        return
    orig = _H3Conv3d.forward
    g = torch.Generator(device="cuda").manual_seed(5)
    convs = [  # conv_in / resnet (reflect), the stride-2 downsample, a 1x1x1 shortcut
        (_H3Conv3d(32, 64, 3, padding=1, padding_mode="reflect"), (1, 32, 1, 34, 18)),
        (_H3Conv3d(64, 64, 3, stride=(1, 2, 2), padding=(1, 0, 0)), (1, 64, 1, 33, 17)),
        (_H3Conv3d(64, 32, 1), (2, 64, 1, 16, 8)),
        (_H3Conv3d(32, 32, 3, padding=1), (1, 32, 1, 16, 8)),
    ]
    try:
        assert "last time tap" in sgld.apply_h3vae()
        before = dict(sgld.H3VAE_STATS)
        with torch.inference_mode():
            for conv, shape in convs:
                conv = conv.to("cuda", B)
                x = torch.randn(*shape, device="cuda", dtype=B, generator=g)
                got = conv(x)
                want = orig(conv, x)                    # the engine's conv3d
                ref = orig(conv.float(), x.float())     # and in fp32
                conv.to(B)
                assert got.shape == want.shape and got.dtype == want.dtype
                _close(got, ref, tol=1e-2)
                assert (got.float() - ref).abs().mean() <= 1.5 * (want.float() - ref).abs().mean()
            assert sgld.H3VAE_STATS["conv2d"] == before["conv2d"] + len(convs)
            # more than one frame, replicate time padding, pre-padded: the engine's
            conv = convs[3][0]
            x = torch.randn(1, 32, 3, 16, 8, device="cuda", dtype=B, generator=g)
            _close(conv(x), orig(conv, x), tol=0)
            rep = _H3Conv3d(32, 32, 3, padding=1, causal=False).to("cuda", B)
            x1 = x[:, :, :1].contiguous()
            _close(rep(x1), orig(rep, x1), tol=0)
            xp = torch.nn.functional.pad(x, (1, 1, 1, 1))
            _close(conv(xp, pre_padded=True), orig(conv, xp, pre_padded=True), tol=0)
            assert sgld.H3VAE_STATS["engine"] == before["engine"] + 3
    finally:
        sgld.revert_h3vae()
    assert _H3Conv3d.forward is orig


class _Conv3dLayer(torch.nn.Conv3d):
    enable_linear = False       # MiniMax H3's tower: disable_linear=True


class Qwen3VLVisionPatchEmbed(torch.nn.Module):
    """SGLang's, as MiniMax H3's vision tower builds it."""

    def __init__(self, c=3, t=2, p=16, e=1152):
        super().__init__()
        self.patch_size, self.temporal_patch_size, self.in_channels, self.embed_dim = p, t, c, e
        k = [t, p, p]
        self.proj = _Conv3dLayer(c, e, kernel_size=k, stride=k, bias=True)

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype)).view(-1, self.embed_dim)


def test_sgld_patchembed_is_the_conv3d_as_one_linear(monkeypatch):
    _stand_in(monkeypatch, sgld.PATCHEMBED_HOME, Qwen3VLVisionPatchEmbed=Qwen3VLVisionPatchEmbed)
    orig = Qwen3VLVisionPatchEmbed.forward
    g = torch.Generator(device="cuda").manual_seed(6)
    pe = Qwen3VLVisionPatchEmbed().to("cuda", B)
    x = torch.randn(1024, 3 * 2 * 16 * 16, device="cuda", dtype=torch.float32, generator=g)
    try:
        assert "one F.linear" in sgld.apply_patchembed()
        before = dict(sgld.PATCHEMBED_STATS)
        with torch.inference_mode():
            got, want = pe(x), orig(pe, x)
            ref = orig(pe.float(), x)
            pe.to(B)
            assert got.shape == want.shape == (1024, 1152) and got.dtype == want.dtype == B
            _close(got, ref, tol=1e-2)
            assert (got.float() - ref).abs().mean() <= 1.5 * (want.float() - ref).abs().mean()
            assert sgld.PATCHEMBED_STATS["linear"] == before["linear"] + 1
            pe.proj.enable_linear = True                # SGLang's own linear path: the engine's
            pe(x)
            assert sgld.PATCHEMBED_STATS["engine"] == before["engine"] + 1
    finally:
        sgld.revert_patchembed()
    assert Qwen3VLVisionPatchEmbed.forward is orig
