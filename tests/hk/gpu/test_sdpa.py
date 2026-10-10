"""The SDPA drop-in: does it compute torch's answer, and does it know when not
to try?

A drop-in has two failure modes and only one of them is loud. The loud one is
wrong numbers. The quiet one is a call the kernel should not have taken --
fp16, a mask, cross attention -- which has to reach torch unchanged rather than
raise or, worse, run on a kernel whose assumptions it breaks. Both are here.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402,N812

from hk.ops import sdpa  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a Radeon"
)

# bf16 attention accumulates in fp32 and rounds once at the end; torch's
# reference does the same, so what is left is the softmax's own ordering.
TOL = dict(rtol=2e-2, atol=2e-2)


def _qkv(b=1, h=8, n=1024, d=128, h_kv=None, dtype=torch.bfloat16):
    hk_ = h if h_kv is None else h_kv
    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hk_, n, d, device="cuda", dtype=dtype)
    v = torch.randn(b, hk_, n, d, device="cuda", dtype=dtype)
    return q, k, v


@pytest.mark.parametrize(
    "b,h,n,d,h_kv,causal",
    [
        (1, 8, 1024, 128, None, False),
        (1, 8, 1024, 128, None, True),
        (2, 4, 2048, 64, None, False),
        (1, 8, 1024, 128, 2, False),       # GQA
        (1, 8, 1000, 128, None, False),    # N does not divide the Q tile
        (1, 4, 1024, 64, None, True),
    ],
)
def test_it_matches_torch(b, h, n, d, h_kv, causal):
    q, k, v = _qkv(b, h, n, d, h_kv)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                          enable_gqa=h_kv is not None)
    got = sdpa.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                            enable_gqa=h_kv is not None)
    torch.testing.assert_close(got, want, **TOL)


def test_a_non_default_scale_matches_torch():
    q, k, v = _qkv()
    want = F.scaled_dot_product_attention(q, k, v, scale=0.05)
    got = sdpa.scaled_dot_product_attention(q, k, v, scale=0.05)
    torch.testing.assert_close(got, want, **TOL)


def test_a_transposed_input_is_copied_not_refused():
    # The layout a framework actually produces: (B, N, H, D) viewed and
    # transposed, so q is not contiguous.
    b, n, h, d = 1, 1024, 8, 128
    x = torch.randn(b, n, 3 * h * d, device="cuda", dtype=torch.bfloat16)
    q, k, v = (t.view(b, n, h, d).transpose(1, 2) for t in x.chunk(3, dim=-1))
    assert not q.is_contiguous()
    torch.testing.assert_close(
        sdpa.scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v), **TOL)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"dropout_p": 0.1}, "dropout"),
        ({"attn_mask": "mask"}, "attn_mask"),
    ],
)
def test_calls_the_kernel_cannot_take_go_to_torch(kwargs, reason):
    q, k, v = _qkv()
    if kwargs.get("attn_mask") == "mask":
        kwargs["attn_mask"] = torch.zeros(1024, 1024, device="cuda",
                                          dtype=torch.bool).tril()
    assert reason in sdpa.why_unsupported(q, k, v, **kwargs)
    # And it still answers: dropout_p is 0 at eval, so compare the masked case
    # only -- a dropout call is not deterministic to compare against.
    if "attn_mask" in kwargs:
        torch.testing.assert_close(
            sdpa.scaled_dot_product_attention(q, k, v, **kwargs),
            F.scaled_dot_product_attention(q, k, v, **kwargs), **TOL)


def test_fp16_goes_to_torch_and_is_still_correct():
    q, k, v = _qkv(dtype=torch.float16)
    assert "bf16 only" in sdpa.why_unsupported(q, k, v)
    torch.testing.assert_close(
        sdpa.scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v), rtol=2e-3, atol=2e-3)


def test_an_unsupported_head_dim_goes_to_torch():
    q, k, v = _qkv(d=96)
    assert "head_dim 96" in sdpa.why_unsupported(q, k, v)
    torch.testing.assert_close(
        sdpa.scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v), **TOL)


def test_the_strict_entry_point_raises_instead_of_falling_back():
    q, k, v = _qkv(d=96)
    with pytest.raises(ValueError, match="head_dim 96"):
        sdpa.attention(q, k, v)


def test_patch_and_unpatch_round_trip():
    original = F.scaled_dot_product_attention
    sdpa.patch()
    try:
        assert sdpa.patched()
        assert F.scaled_dot_product_attention is \
            sdpa.scaled_dot_product_attention
        sdpa.patch()  # idempotent: must not capture itself as the original
        q, k, v = _qkv()
        got = F.scaled_dot_product_attention(q, k, v)
        want = original(q, k, v)
        torch.testing.assert_close(got, want, **TOL)
    finally:
        sdpa.unpatch()
    assert F.scaled_dot_product_attention is original
    assert not sdpa.patched()


def test_the_op_behind_it_is_a_torch_custom_op():
    # Which is what makes it capturable and traceable; see test_torch_op.py.
    op = sdpa.op_for(128, False, None, 1024)
    assert "attn_fwd_d128" in str(op)
    assert sdpa.op_for(128, False, None, 1024) is op


# ------------------------------------------------- the strided-input contract
#
# vLLM's vision towers hand SDPA a `(b, s, h, d)` tensor permuted to
# `(b, h, s, d)` -- never contiguous -- and do it for every tower, including
# the ones whose head_dim this kernel does not have. Both halves of that are
# tested here because the first measurement of it found a 16% regression on
# the second half.


def _vit_qkv(heads=16, n=577, head_dim=64):
    """Exactly what `vllm...vit_attn_wrappers.apply_sdpa` passes on."""
    g = torch.Generator(device="cuda").manual_seed(0)
    bshd = [torch.randn(1, n, heads, head_dim, device="cuda",
                        dtype=torch.bfloat16, generator=g) for _ in range(3)]
    return tuple(x.permute(0, 2, 1, 3) for x in bshd)


def test_a_permuted_vit_input_is_not_a_reason_to_fall_back():
    q, k, v = _vit_qkv()
    assert not q.is_contiguous()
    assert sdpa.why_unsupported(q, k, v) == ""


def test_a_permuted_vit_input_computes_the_right_answer():
    q, k, v = _vit_qkv()
    sdpa.reset_stats()
    got = sdpa.scaled_dot_product_attention(q, k, v, scale=64 ** -0.5)
    want = F.scaled_dot_product_attention(q, k, v, scale=64 ** -0.5)
    assert sdpa.STATS == {"kernel": 1, "fallback": 0}
    torch.testing.assert_close(got, want, **TOL)


def test_a_call_that_falls_back_does_not_copy_anything_first(monkeypatch):
    # The regression this guards: `.contiguous()` ran before the support
    # check, so a SigLIP tower (head_dim 72) paid for three copies and then
    # handed torch the original strided tensors anyway -- 16% slower than not
    # patching at all. The fix is an ordering, so the test is on the ordering.
    q, k, v = _vit_qkv(head_dim=72)
    calls = []
    real = torch.Tensor.contiguous

    def counting(self, *a, **kw):
        calls.append(tuple(self.shape))
        return real(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "contiguous", counting)
    sdpa.reset_stats()
    got = sdpa.scaled_dot_product_attention(q, k, v, scale=72 ** -0.5)
    monkeypatch.undo()

    assert sdpa.STATS == {"kernel": 0, "fallback": 1}
    assert calls == [], f"copied {calls} on the way to falling back"
    torch.testing.assert_close(
        got, F.scaled_dot_product_attention(q, k, v, scale=72 ** -0.5))


def test_the_fallback_records_why():
    sdpa.reset_stats()
    q, k, v = _vit_qkv(head_dim=72)
    sdpa.scaled_dot_product_attention(q, k, v)
    assert len(sdpa.FALLBACKS) == 1
    assert "head_dim 72" in next(iter(sdpa.FALLBACKS.values()))


def test_the_fallback_record_is_bounded():
    # A serving process must not grow a dict because someone sent it a new
    # sequence length.
    sdpa.reset_stats()
    for n in range(sdpa._FALLBACK_CAP + 20):
        q, k, v = _vit_qkv(n=64 + n, head_dim=72)
        sdpa.scaled_dot_product_attention(q, k, v)
    assert len(sdpa.FALLBACKS) == sdpa._FALLBACK_CAP
    assert sdpa.STATS["fallback"] == sdpa._FALLBACK_CAP + 20
