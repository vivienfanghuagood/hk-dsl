"""Numerics for the DSL attention kernel, against `F.scaled_dot_product_attention`.

Needs a Radeon and torch. The resource half of the Phase 4 gate -- ScratchSize
0 on all four variants, one occupancy wave better than the handwritten C++ --
is a build-time check and runs on the no-GPU tier. This file is the half that
catches what no resource gate can: a mask off by one row spills nothing and
occupies nothing extra, and is wrong.

The reference *is* SDPA, not a hand-rolled softmax, because that is what the
gate names and what a framework would otherwise have called. It is evaluated
one (batch, head, q-chunk) at a time in fp32: at N=49920 a single head's fp32
score matrix is 10 GB, so the chunking is not a nicety.

Two mask conventions differ and the difference is deliberate. torch aligns a
causal mask to the *bottom* right when n_kv != n_q; this kernel's diagonal is
top-left. `_why` rejects n_kv != n_q under causal rather than silently picking
one, so every causal shape here is square and the two agree.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402

#: bf16 operands rounded twice and then summed over thousands of terms land
#: around 1.2e-2 of relative error against fp32. 5e-2 is 4x that, and still
#: more than an order of magnitude below the O(1) error a wrong tile or a
#: shifted mask produces -- the failures this file exists to catch are not
#: subtle, they are total.
RTOL = 5e-2

D_HEAD = 128


def _qkv(b, h, n, d=D_HEAD, h_kv=None, n_kv=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hk_ = h if h_kv is None else h_kv
    nk = n if n_kv is None else n_kv
    mk = lambda hh, nn: torch.randn(b, hh, nn, d, device="cuda",  # noqa: E731
                                    dtype=torch.bfloat16, generator=g)
    return mk(h, n), mk(hk_, nk), mk(hk_, nk)


def _ref(q, k, v, causal=False, scale=None, q_chunk=512):
    """SDPA in fp32, one (batch, head, q-chunk) at a time.

    GQA is inferred from k's head count rather than passed, because that is
    all GQA is: a q head reads the kv head it maps to.
    """
    b, h, n, d = q.shape
    nk = k.shape[2]
    q_per_kv = h // k.shape[1]
    out = torch.empty(b, h, n, d, device=q.device, dtype=torch.float32)
    pos_k = torch.arange(nk, device=q.device)[None, :]
    for bi in range(b):
        for hi in range(h):
            kk = k[bi, hi // q_per_kv].float()[None, None]
            vv = v[bi, hi // q_per_kv].float()[None, None]
            for i in range(0, n, q_chunk):
                j = min(i + q_chunk, n)
                mask = None
                if causal:
                    # top-left aligned, matching the kernel; torch's own
                    # is_causal is bottom-right when nk != n.
                    pos_q = torch.arange(i, j, device=q.device)[:, None]
                    mask = (pos_k <= pos_q)[None, None]
                out[bi, hi, i:j] = F.scaled_dot_product_attention(
                    q[bi, hi, i:j].float()[None, None], kk, vv,
                    attn_mask=mask, scale=scale)[0, 0]
    return out


def _check(got, ref):
    got, ref = got.float(), ref.float()
    # An all-zero `got` against an all-zero `ref` passes every relative test.
    # That failure mode has happened here before, hence the magnitude guard.
    assert ref.abs().max().item() > 1e-6, "reference has no magnitude"
    assert torch.isfinite(got).all().item(), "output has non-finite elements"
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    assert rel < RTOL, f"relative error {rel:.2e} exceeds {RTOL}"
    return rel


def _run(b, h, n, **kw):
    causal = kw.pop("causal", False)
    scale = kw.pop("scale", None)
    q, k, v = _qkv(b, h, n, **kw)
    got = hk.attention(q, k, v, causal=causal, scale=scale)
    assert got.shape == q.shape and got.dtype == q.dtype
    rel = _check(got, _ref(q, k, v, causal=causal, scale=scale))
    del q, k, v, got
    torch.cuda.empty_cache()
    return rel


# h is small throughout: the head *count* is only a grid dimension, while the
# head *dimension* is what the whole tiling depends on. Paying for 56 heads
# would buy nothing and cost minutes in the fp32 reference.

@pytest.mark.parametrize("n", [4096, 16384])
def test_aligned(n):
    _run(1, 4, n)


def test_h3_480p():
    """The shape that drove every decision: 832x480, 125 frames -> 49920."""
    _run(1, 2, 49920)


def test_batch():
    _run(2, 4, 8192)


@pytest.mark.parametrize("n", [192, 384])
def test_few_q_tiles(n):
    """One and two Q tiles at the shipped 12 warps. With one, the loop's
    prologue and epilogue are the whole kernel and the body never runs."""
    _run(1, 4, n)


@pytest.mark.parametrize("n", [193, 4097, 5000, 12345])
def test_ragged_n(n):
    """Neither the Q tile (192) nor the KV block (32) divides these, so both
    back-ups run: the Q one recomputes an overlapping strip (safe only if a
    query row's output depends on that row alone) and the KV one masks the
    rows it does not own (safe only if the mask is exact -- a row counted
    twice shows up in the softmax denominator, not as a crash)."""
    _run(1, 4, n)


@pytest.mark.parametrize("n", [192, 193, 2048, 4096, 4097, 5000])
def test_causal(n):
    """The interesting lengths are where the diagonal misses a block boundary,
    and the first Q tile, where most waves skip every block but the first."""
    _run(1, 4, n, causal=True)


def test_causal_batch():
    _run(2, 4, 2048, causal=True)


@pytest.mark.parametrize("h, h_kv", [(8, 2), (8, 1), (4, 4)])
def test_gqa(h, h_kv):
    """Ratios 4 and 8, plus ratio 1 written the GQA way -- the case an
    off-by-one in the head divide would still pass."""
    _run(1, h, 4096, h_kv=h_kv)


def test_gqa_causal():
    _run(1, 8, 2048, h_kv=2, causal=True)


@pytest.mark.parametrize("n, kw", [
    (4096, {}), (4097, {}), (2048, dict(causal=True)),
    (2048, dict(h_kv=2)), (192, {}),
])
def test_head_dim_64(n, kw):
    """Half the registers for q and for the accumulator, and a different
    staging split -- a separate kernel, not a parameter of the d=128 one."""
    _run(1, 4 if "h_kv" not in kw else 8, n, d=64, **kw)


def test_custom_scale():
    """A non-default scale compiles a separate kernel, keyed on the float's
    bits so the on-disk cache hits on the second process rather than the
    second call."""
    _run(1, 4, 2048, scale=0.1)


def test_ragged_kv():
    """n_kv != n_q, non-causal. The Q and KV extents are independent: the
    kernel's KV loop is bounded by k's length, not q's."""
    _run(1, 4, 2048, n_kv=3000)


# ----------  what it refuses  ----------

def test_rejects_rather_than_falls_back():
    """A silent fallback inside an explicit `hk.attention` call is how a
    20 TFLOPs path gets mistaken for a 64 TFLOPs one."""
    q, k, v = _qkv(1, 4, 2048)
    with pytest.raises(ValueError, match="fp16|bf16|dtype"):
        hk.attention(q.half(), k.half(), v.half())
    with pytest.raises(ValueError):
        hk.attention(q[..., :96].contiguous(), k[..., :96].contiguous(),
                     v[..., :96].contiguous())
    with pytest.raises(ValueError):          # N below one Q tile
        hk.attention(q[:, :, :64], k[:, :, :64], v[:, :, :64])


def test_supported_agrees_with_attention():
    q, k, v = _qkv(1, 4, 2048)
    assert hk.ops.attn.supported(q, k, v)
    assert not hk.ops.attn.supported(q.half(), k.half(), v.half())
    # causal needs a square score matrix: this kernel's diagonal is top-left
    # and torch's is bottom-right, so n_kv != n_q would silently disagree.
    # .contiguous(): slicing N leaves a stride the kernel does not accept,
    # and that rejection would mask the one being tested here.
    kk = k[:, :, :1024].contiguous()
    vv = v[:, :, :1024].contiguous()
    assert hk.ops.attn.supported(q, kk, vv)
    assert not hk.ops.attn.supported(q, kk, vv, causal=True)


def test_out_parameter_is_written_in_place():
    q, k, v = _qkv(1, 4, 2048)
    o = torch.empty_like(q)
    got = hk.attention(q, k, v, out=o)
    assert got.data_ptr() == o.data_ptr()
    _check(o, _ref(q, k, v))


# ------------------------------------------------------- rounding, not error
#
# RTOL above is deliberately loose -- it exists to catch a wrong tile or a
# shifted mask, which are O(1) failures. It cannot see a *bias*, and a bias is
# what fp32 -> bf16 truncation was: half an ulp, always the same direction,
# applied to a tile of positive probabilities on every KV block. It passed
# every test in this file while making a VLM decode a different sentence.
#
# These two have exact answers, so there is no tolerance to hide in.


@pytest.mark.parametrize("n", [4096, 1025 + 4096 - 1025])
def test_uniform_values_come_back_exactly(n):
    """V all ones: the softmax weights sum to one, so every output element is
    exactly 1.0 -- whatever Q and K were, and whatever the running max did.

    Any deviation is the kernel disagreeing with itself: the numerator
    accumulates the rounded probability tile while the denominator sums the
    fp32 one, and with V = 1 nothing else can contribute. Truncation put this
    at 1.0 +- 2^-8 on every element.
    """
    q, k, _ = _qkv(1, 4, n)
    v = torch.ones_like(q)
    got = hk.attention(q, k, v)
    want = torch.ones_like(got)
    assert torch.equal(got, want), (
        f"max deviation from 1.0: "
        f"{(got.float() - 1.0).abs().max().item():.3e} "
        f"(one bf16 ulp here is {2 ** -8:.3e})"
    )


def test_equal_scores_give_the_mean_of_v():
    """Q all zeros: every score is equal, so the output is the mean of V along
    the sequence. Keeps the PV matmul and the V operand in play while removing
    the exponential, so a failure here is the accumulation and not the
    softmax."""
    n = 4096
    _, k, v = _qkv(1, 4, n)
    q = torch.zeros_like(k)
    got = hk.attention(q, k, v).float()
    want = v.double().mean(dim=2, keepdim=True).float().expand_as(got)
    rel = ((got - want).abs() / want.abs().clamp_min(1e-6)).max().item()
    # One bf16 ulp of the output, which is the floor for anything writing bf16.
    assert rel < 2 ** -7, f"relative error {rel:.3e}"
