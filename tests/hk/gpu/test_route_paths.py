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
from hk.ops import moe, w4a16  # noqa: E402
from hk.target import current_arch  # noqa: E402

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
    vllm_int4.w4a16(torch.randn(top + 1, k, device="cuda", dtype=torch.float16),
                    qweight, scales, qzeros, qw, s, zs, group)
    assert seen == [top + 1]


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
    qw, sc, zs = w4a16.repack_gptq(qweight, qzeros, s, group, dtype=torch.float16)
    w = ((q - 8).float() * s.float().repeat_interleave(group, 0)).t()
    for m in (vllm_gptq.MIN_M, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        before = vllm_gptq.STATS["hk"]
        y = vllm_gptq.gptq_w4a16(x, None, None, None, qw, sc, zs, group)
        assert vllm_gptq.STATS["hk"] == before + 1 and y.shape == (m, n)
        _close(y, x.float() @ w.t())
    for m in (vllm_gptq.MIN_M - 1, 65):
        vllm_gptq.gptq_w4a16(torch.randn(m, k, device="cuda", dtype=torch.float16),
                             None, None, None, qw, sc, zs, group)
    assert seen == [vllm_gptq.MIN_M - 1, 65]


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
    finally:
        sgld.revert_linear()
    assert Unquantized.apply is orig
