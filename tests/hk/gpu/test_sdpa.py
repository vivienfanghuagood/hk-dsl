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
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from hk.ops import sdpa  # noqa: E402

import _attn_ref  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a Radeon"
)

# The kernel against exact (fp32) attention: P is rounded to bf16 for the
# second MMA and the output once at the end. Over 1000 seeds of every case of
# test_it_matches_torch on gfx950 the worst error is 0.015 (d128 causal) and
# the worst |error| / (atol + rtol |ref|) 0.48 (d64 causal): half of this.
TOL = dict(rtol=2e-2, atol=2e-2)


def _qkv(b=1, h=8, n=1024, d=128, h_kv=None, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hk_ = h if h_kv is None else h_kv
    q = torch.randn(b, h, n, d, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(b, hk_, n, d, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(b, hk_, n, d, device="cuda", dtype=dtype, generator=g)
    return q, k, v


def _ref(q, k, v, *, is_causal=False, enable_gqa=False, scale=None, attn_mask=None):
    """The answer, in float64 (_attn_ref): softmax(q k^T * scale) v with
    torch's masking and GQA conventions. Not torch's SDPA: in the vLLM image
    (torch 2.12, ROCm 10) AOTriton's bf16 flash forward returns whole 128-row
    tiles wrong under another process on the GPU, and an fp32 call moves by
    more than this kernel's own error from run to run."""
    b, h, n, d = q.shape
    nk = k.shape[2]
    if enable_gqa and k.shape[1] != h:
        k = k.repeat_interleave(h // k.shape[1], dim=1)
        v = v.repeat_interleave(h // v.shape[1], dim=1)
    scale = d ** -0.5 if scale is None else scale
    s = torch.einsum("bhnd,bhmd->bhnm", q.double(), k.double()) * scale
    if is_causal:
        keep = torch.ones(n, nk, device=q.device, dtype=torch.bool).tril()
        s = s.masked_fill(~keep, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            s = s.masked_fill(~attn_mask, float("-inf"))
        else:
            s = s + attn_mask.double()
    # a row every key of which is masked out answers 0, as torch's math
    # backend does, not 0/0
    p = torch.softmax(s, dim=-1).nan_to_num(nan=0.0)
    return torch.einsum("bhnm,bhmd->bhnd", p, v.double()).float()


def _close(got, want, **tol):
    _attn_ref.check(got, want)
    torch.testing.assert_close(got.float(), want.float(), **(tol or TOL))


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
    want = _ref(q, k, v, is_causal=causal, enable_gqa=h_kv is not None)
    got = sdpa.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                            enable_gqa=h_kv is not None)
    _close(got, want)


@pytest.mark.parametrize("causal", [False, True])
def test_a_short_call_after_a_long_one_gets_its_own_kernel(causal):
    # gfx1201 tiles a long Q 256 rows to a workgroup and keeps gfx1100's 192
    # below that; both lengths share a tuning key, so a memo on the key alone
    # ran the 200-row call on the 4096-row call's tiling.
    for n in (4096, 200):
        q, k, v = _qkv(h=4, n=n)
        _close(sdpa.scaled_dot_product_attention(q, k, v, is_causal=causal),
               _ref(q, k, v, is_causal=causal))


def test_a_non_default_scale_matches_torch():
    q, k, v = _qkv()
    want = _ref(q, k, v, scale=0.05)
    got = sdpa.scaled_dot_product_attention(q, k, v, scale=0.05)
    _close(got, want)


def test_a_transposed_input_is_copied_not_refused():
    # The layout a framework actually produces: (B, N, H, D) viewed and
    # transposed, so q is not contiguous.
    b, n, h, d = 1, 1024, 8, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(b, n, 3 * h * d, device="cuda", dtype=torch.bfloat16, generator=g)
    q, k, v = (t.view(b, n, h, d).transpose(1, 2) for t in x.chunk(3, dim=-1))
    assert not q.is_contiguous()
    _close(sdpa.scaled_dot_product_attention(q, k, v), _ref(q, k, v))


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
        with sdpa_kernel(SDPBackend.MATH):    # torch answers it: see _ref
            got = sdpa.scaled_dot_product_attention(q, k, v, **kwargs)
        _close(got, _ref(q, k, v, **kwargs))


def test_fp16_goes_to_torch_and_is_still_correct():
    q, k, v = _qkv(dtype=torch.float16)
    assert "bf16 only" in sdpa.why_unsupported(q, k, v)
    with sdpa_kernel(SDPBackend.MATH):        # torch answers it: see _ref
        got = sdpa.scaled_dot_product_attention(q, k, v)
    _close(got, _ref(q, k, v), rtol=2e-3, atol=2e-3)


def test_an_unsupported_head_dim_goes_to_torch():
    q, k, v = _qkv(d=96)
    assert "head_dim 96" in sdpa.why_unsupported(q, k, v)
    with sdpa_kernel(SDPBackend.MATH):        # torch answers it: see _ref
        got = sdpa.scaled_dot_product_attention(q, k, v)
    _close(got, _ref(q, k, v))


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
        _close(got, _ref(q, k, v))
    finally:
        sdpa.unpatch()
    assert F.scaled_dot_product_attention is original
    assert not sdpa.patched()


def test_a_patched_call_that_falls_back_reaches_torch_not_itself():
    """Once patched, F.scaled_dot_product_attention *is* the drop-in: a call
    it hands back (here a mask) must go to torch's function, not recurse
    through the module attribute -- SGLang-Diffusion's masked text-prefix
    attention hit exactly that."""
    q, k, v = _qkv(n=256)
    mask = torch.ones(256, 256, dtype=torch.bool, device="cuda").tril()
    ref = _ref(q, k, v, attn_mask=mask)
    sdpa.patch()
    try:
        with sdpa_kernel(SDPBackend.MATH):    # torch answers it: see _ref
            got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    finally:
        sdpa.unpatch()
    _close(got, ref)


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
    want = _ref(q, k, v, scale=64 ** -0.5)
    # On gfx1100 the permuted memory is read in place (the bnhd kernel).
    in_place = 0 if sdpa._attn._native_cdna() else 1
    assert sdpa.STATS == {"kernel": 1, "fallback": 0, "bnhd": in_place, "k3": 0}
    _close(got, want)


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
    with sdpa_kernel(SDPBackend.MATH):        # torch answers it: see _ref
        got = sdpa.scaled_dot_product_attention(q, k, v, scale=72 ** -0.5)
    monkeypatch.undo()

    assert sdpa.STATS == {"kernel": 0, "fallback": 1, "bnhd": 0, "k3": 0}
    assert calls == [], f"copied {calls} on the way to falling back"
    _close(got, _ref(q, k, v, scale=72 ** -0.5))


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


@pytest.mark.parametrize("h,h_kv,n,causal", [(32, 32, 4160, False), (8, 2, 1000, False),
                                              (8, 8, 1024, True)])
def test_a_bnhd_view_runs_in_place_and_matches_torch(h, h_kv, n, causal):
    """(B, H, N, D) views of (B, N, H, D) memory -- SGLang-Diffusion's DiT
    hands its projections over this way -- run on the bnhd kernel without a
    copy, and the output comes back as the same kind of view."""
    if sdpa._attn._native_cdna():
        pytest.skip("gfx1100's kernel")
    g = torch.Generator(device="cuda").manual_seed(n + h)
    q = torch.randn(1, n, h, 128, device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)
    k = torch.randn(1, n, h_kv, 128, device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)
    v = torch.randn(1, n, h_kv, 128, device="cuda", dtype=torch.bfloat16, generator=g).transpose(1, 2)
    before = sdpa.STATS["bnhd"]
    got = sdpa.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=h != h_kv)
    assert sdpa.STATS["bnhd"] == before + 1
    assert not got.is_contiguous() and got.transpose(1, 2).is_contiguous()
    _close(got, _ref(q, k, v, is_causal=causal, enable_gqa=h != h_kv))
