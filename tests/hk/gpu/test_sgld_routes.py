"""The SGLang-Diffusion routes' kernels at Qwen-Image 2.1's own shapes (gfx1100), as the routes call
them: the DiT attention on attn_k3 (sgld_attn, through hk.ops.sdpa), the text encoder's causal GQA
attention (hk.ops.sdpa), and the projection GEMM (sgld_linear's torch op, `_gemm_op`).

The shapes are the ones a production server ran (1 card, h3gw's mix of t2i 1344x768 and 1024^2 at
30 and 40 steps, and i2i, whose reference image doubles the DiT's KV): q (B, H, N, D) views of
(B, N, H, D) memory, which is how SGLang-Diffusion passes them.

- **200 repeats over poisoned memory.** Before each call the memory the op is about to be given is
  NaN: an explicit output is filled with NaN; for the ops that allocate their own output and
  scratch, a NaN block larger than all of it is allocated and freed first, so the caching
  allocator hands its pieces out again. Every repeat must be bitwise the first, and the first
  must match the float64 reference: nothing reads an unwritten or stale element.
- **A producer on a side stream.** SGLang-Diffusion overlaps work on its own streams. The inputs
  are written last on a side stream behind busy work, and the op is called on that stream without
  synchronizing: it must launch on torch's current stream (attn_k3 and the attention through
  `codeobj`, the GEMM through its torch op), so the answer is the default stream's, bit for bit,
  and no null-stream warning is raised.
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx1100":
    pytest.skip("the SGLang-Diffusion routes are gfx1100's", allow_module_level=True)

pytestmark = pytest.mark.gpu

from hk.integration import sglang_diffusion as G  # noqa: E402
from hk.ops import attn_k3 as K3  # noqa: E402
from hk.ops import gemm  # noqa: E402
from hk.ops import sdpa as S  # noqa: E402

import _attn_ref  # noqa: E402

# DiT attention: (H, N_q, N_kv). 1344x768 (4032 image tokens + the prompt's), i2i (the reference
# image's tokens double N_kv), 1024^2.
ATTN = [(32, 4032, 4060), (32, 4032, 8092), (32, 4096, 4124)]
# The text encoder's causal attention on an i2i prompt (image tokens in it): 32 q heads, 8 kv heads.
CAUSAL = [(32, 8, 1050)]
# The DiT's projections: (M, N, K) at the three image token counts.
GEMMS = [(4032, 4096, 4096), (4060, 12288, 4096), (4096, 4096, 12288)]
REPEATS = 200


def _bnhd(h, n, seed, d=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((1, n, h, d), device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)


def _poison(nbytes):
    """Leave a NaN-filled free block of `nbytes` in torch's caching allocator."""
    junk = torch.full((nbytes // 2,), float("nan"), device="cuda", dtype=torch.bfloat16)
    del junk


def _dit(h, nq, nkv):
    q, k, v = _bnhd(h, nq, 1), _bnhd(h, nkv, 2), _bnhd(h, nkv, 3)
    assert K3.why(q, k, v) == "", K3.why(q, k, v)
    return q, k, v


def _sdpa_dit(q, k, v):
    before = S.STATS["k3"]
    o = S.scaled_dot_product_attention(q, k, v)
    assert S.STATS["k3"] == before + 1, "the route did not take attn_k3"
    return o


def _causal(h, hkv, n):
    return _bnhd(h, n, 4), _bnhd(hkv, n, 5), _bnhd(hkv, n, 6)


def _sdpa_causal(q, k, v):
    before = S.STATS["fallback"]
    o = S.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert S.STATS["fallback"] == before, "the route fell back to torch"
    return o


def _gemm(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn((n, k), device="cuda", dtype=torch.float32, generator=g) * k ** -0.5).bfloat16()
    s = gemm.schedule_for(m, n, k)
    assert not (k % s["k_step"] or n % s["block_n"]), s
    return x, w, G._gemm_op(s)


def _rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("h,nq,nkv", ATTN)
def test_dit_attention_poisoned_repeats_are_bitwise(h, nq, nkv):
    q, k, v = _dit(h, nq, nkv)
    out_bytes = q.numel() * 2
    first = _sdpa_dit(q, k, v).clone()
    torch.cuda.synchronize()
    _attn_ref.check(first.float(), _attn_ref.attention64(q, k, v))
    for _ in range(REPEATS):
        _poison(8 * out_bytes)
        o = _sdpa_dit(q, k, v)
        assert torch.equal(o.view(torch.int16), first.view(torch.int16))
    assert (K3.attention(q, k, torch.ones_like(v)) == 1).all()


@pytest.mark.parametrize("h,hkv,n", CAUSAL)
def test_causal_gqa_attention_poisoned_repeats_are_bitwise(h, hkv, n):
    q, k, v = _causal(h, hkv, n)
    first = _sdpa_causal(q, k, v).clone()
    torch.cuda.synchronize()
    _attn_ref.check(first.float(), _attn_ref.attention64(q, k, v, causal=True))
    for _ in range(REPEATS):
        _poison(8 * q.numel() * 2)
        o = _sdpa_causal(q, k, v)
        assert torch.equal(o.view(torch.int16), first.view(torch.int16))


@pytest.mark.parametrize("m,n,k", GEMMS)
def test_projection_gemm_poisoned_repeats_are_bitwise(m, n, k):
    x, w, op = _gemm(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    op(x, w, out)
    torch.cuda.synchronize()
    first = out.clone()
    assert _rel(first, x.float() @ w.float().T) < 2e-2
    for _ in range(REPEATS):
        out.fill_(float("nan"))
        op(x, w, out)
        assert torch.equal(out.view(torch.int16), first.view(torch.int16))


def _late_on(side, base):
    """`base`, written on `side` behind ~tens of ms of work: a launch that is not ordered after
    it reads the zeros."""
    with torch.cuda.stream(side):
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        for _ in range(16):
            a = a @ a * 2048 ** -0.5
        late = torch.zeros_like(base)
        # + 0, but only once `a` is done (`a` may overflow, so no arithmetic on its values:
        # inf * 0 is NaN; a boolean of it times 0 is 0)
        late.copy_(base + a[:1, :1].isfinite().to(base.dtype) * 0)
    return late


def _on_side_stream(make_inputs, call):
    """The op's answer with its first input written last on a side stream, called on that
    stream with no synchronization, against the default stream's."""
    inputs = make_inputs()
    want = call(*inputs).clone()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    late = _late_on(side, inputs[0])
    with torch.cuda.stream(side), warnings.catch_warnings():
        warnings.simplefilter("error")          # no null-stream warning: the launch is torch's
        got = call(late, *inputs[1:])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return got, want


def test_dit_attention_follows_a_side_stream_producer():
    got, want = _on_side_stream(lambda: _dit(*ATTN[1]), _sdpa_dit)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_causal_gqa_attention_follows_a_side_stream_producer():
    got, want = _on_side_stream(lambda: _causal(*CAUSAL[0]), _sdpa_causal)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_projection_gemm_follows_a_side_stream_producer():
    m, n, k = GEMMS[1]
    x, w, op = _gemm(m, n, k)

    def call(x_, w_):
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        op(x_, w_, out)
        return out
    got, want = _on_side_stream(lambda: (x, w), call)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_the_route_takes_these_shapes():
    """The linear route's own entry point sends these shapes to hk's GEMM (MIN_M, bf16, no bias,
    a schedule that tiles them) and returns the GEMM's answer: what the tests above call is what
    a server runs."""
    for m, n, k in GEMMS:
        x, w, op = _gemm(m, n, k)
        G.DECISIONS[(G._bucket(m), n, k)] = (True, "forced by the test")
        try:
            got = G._hk_linear(x.view(1, m, k), w, None)
        finally:
            G.DECISIONS.pop((G._bucket(m), n, k), None)
        assert got is not None and got.shape == (1, m, n)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        op(x, w, out)
        torch.cuda.synchronize()
        assert torch.equal(got.view(m, n), out)
