"""Phase 2 numerics: norms, fused activations, RoPE, int8 quantization.

Needs a Radeon and torch. The gate for this phase is elementwise agreement
with torch *including shapes that do not divide the tile* -- that is the part
no other tier can check, because a tail handled wrongly still compiles, still
reports no spills, and still returns a plausible-looking tensor.

How the tail works, since every assertion here depends on it: include/rdna3
bounds-checks nothing, so a partial tile would read past the end of the
allocation. Instead of masking lanes, the kernels back the *address* up until
the tile is wholly in bounds and then mask the overlap with the reduction's
identity (0 for a sum, -inf for a max). Along rows nothing needs masking at
all: each row normalizes independently, so a backed-up row block recomputes
the same bytes. The shapes below are chosen to exercise both directions.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402

DTYPES = [torch.bfloat16, torch.float16, torch.float32]

TOL = {
    torch.bfloat16: (3e-2, 1e-2),
    torch.float16: (1e-3, 1e-3),
    torch.float32: (1e-5, 1e-6),
}

#: (rows, cols). The first divides both extents; the rest do not divide one or
#: the other, which is the Phase 2 gate. Every shape is at least 16x64 -- the
#: back-up trick needs one whole tile to exist, and the wrappers say so.
SHAPES = [
    (64, 256),  # divides
    (64, 200),  # ragged columns: the reduction's tail is masked
    (100, 256),  # ragged rows: no masking needed, blocks overlap
    (100, 200),  # both
    (17, 65),  # both, and only just larger than one tile
]


def _rand(shape, dtype):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _close(got, want, dtype, msg=""):
    rtol, atol = TOL[dtype]
    torch.testing.assert_close(got.float(), want.float(), rtol=rtol, atol=atol, msg=msg)


# -- norms --------------------------------------------------------------------


def _ref_rmsnorm(x, eps):
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm(shape, dtype):
    x = _rand(shape, dtype)
    _close(hk.ops.rmsnorm(x), _ref_rmsnorm(x, 1e-6), dtype, msg=f"{shape}")


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_layernorm(shape, dtype):
    x = _rand(shape, dtype)
    want = F.layer_norm(x.float(), (shape[-1],), eps=1e-5).to(dtype)
    _close(hk.ops.layernorm(x), want, dtype, msg=f"{shape}")


def test_layernorm_on_a_row_with_a_large_offset():
    """var is computed as E[x^2] - E[x]^2, which cancels. At an offset of 100
    the two terms agree to four digits and fp32 keeps seven, so ~1e-3 of
    relative error in the variance is expected and accepted -- see the note in
    norm.py. This pins how much: if a change makes it worse, it shows here."""
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float32) + 100.0
    want = F.layer_norm(x, (4096,), eps=1e-5)
    torch.testing.assert_close(hk.ops.layernorm(x), want, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(shape, dtype):
    x = _rand(shape, dtype)
    _close(hk.ops.softmax(x), torch.softmax(x.float(), -1).to(dtype), dtype, f"{shape}")


def test_softmax_is_max_shifted():
    """Without the shift, exp() of a logit in the hundreds is inf and the row
    comes back NaN. Large logits are what a real model produces, so this is
    the case that matters, not the randn one."""
    x = _rand((32, 256), torch.float32) + 400.0
    got = hk.ops.softmax(x)
    assert torch.isfinite(got).all()
    _close(got, torch.softmax(x, -1), torch.float32)


def test_the_ragged_tail_is_masked_not_double_counted():
    """The sharpest version of the gate. The backed-up last block re-reads
    columns the previous block already saw; if the overlap were not masked
    with the reduction's identity, those elements would count twice and every
    row's normalizer would be wrong by a shape-dependent amount -- small
    enough to pass a loose tolerance, which is why the reference here is
    exact-ish fp32."""
    x = _rand((16, 65), torch.float32)  # one column past a tile: 63 overlap
    _close(hk.ops.rmsnorm(x), _ref_rmsnorm(x, 1e-6), torch.float32)
    _close(hk.ops.softmax(x), torch.softmax(x, -1), torch.float32)


#: (cols, expected plan) for the three ways norm.plan can lower a wide row.
#: The plan is asserted, not just the numbers, because all three produce the
#: right answer on a dividing shape and the point of the shapes below is that
#: they do not divide -- if a change to plan() quietly moved one of these onto
#: another path, the test would still pass while covering nothing.
#:
#:   8192 -- FOLD: 128 blocks is 8 per warp, too many to hold, but the row
#:           reshapes to 16x512 and 512 is 8 blocks split 8 ways.
#:   8208 -- FOLD again (8208 = 16*513), and the folded row is 513 wide: a
#:           ragged tail *inside* the fold, nine warps for nine blocks of
#:           which the last is clamped.
#:   8199 -- not a multiple of 16, so the fold is unavailable and it streams.
#:
#: 8208 is also the only shape in this file with a non-power-of-two workgroup,
#: which is a thing `_split` only started producing when it stopped rounding
#: the warp count down; nine warps is 288 threads and a nine-entry cross-warp
#: LDS reduction.
WIDE = [(8192, (8, 1, 16)), (8208, (9, 1, 16)), (8199, (16, 0, 1))]


@pytest.mark.parametrize("cols,want_plan", WIDE, ids=[f"{c}" for c, _ in WIDE])
def test_wide_rows_take_the_path_they_are_meant_to_and_get_it_right(cols, want_plan):
    from hk.ops import norm

    assert norm.plan(cols) == want_plan, f"{cols}: plan moved to {norm.plan(cols)}"
    x = _rand((32, cols), torch.float32)
    _close(hk.ops.rmsnorm(x), _ref_rmsnorm(x, 1e-6), torch.float32, f"rms {cols}")
    _close(hk.ops.layernorm(x), F.layer_norm(x, (cols,), eps=1e-5),
           torch.float32, f"layer {cols}")
    _close(hk.ops.softmax(x), torch.softmax(x, -1), torch.float32, f"softmax {cols}")


def test_a_held_tile_is_not_masked_in_place():
    """The persistent variant keeps its tiles in registers across all three
    passes, so a mask applied to one is still there when the write pass stores
    it -- the overlap columns come back as zeros, and whichever warp also owns
    those columns races with them. A dividing shape never shows it, because
    there the mask is empty; this shape is the smallest that does.

    Kept separate from the parametrized cases so that the failure names the
    cause rather than a shape.
    """
    cols = 200  # 4 blocks over 4 warps, one tile each, 56 columns of overlap
    from hk.ops import norm

    assert norm.plan(cols)[1] > 0, "shape no longer takes the persistent path"
    x = _rand((64, cols), torch.float32)
    _close(hk.ops.rmsnorm(x), _ref_rmsnorm(x, 1e-6), torch.float32)
    _close(hk.ops.softmax(x), torch.softmax(x, -1), torch.float32)
    # Quantize holds its tiles too, and its mask is on the absmax. A zeroed
    # overlap does not raise an absmax, so the scale survives; what does not
    # is the quantized row, which comes back zero across the overlap.
    q, sc = hk.ops.quantize(x)
    wq, wsc = _ref_quant(x)
    torch.testing.assert_close(sc, wsc, rtol=1e-5, atol=1e-8)
    assert (q.int() - wq.int()).abs().max() <= 1


def test_rows_below_one_tile_are_refused_with_the_reason():
    x = _rand((8, 256), torch.float32)
    with pytest.raises(ValueError, match="at least 16x64"):
        hk.ops.rmsnorm(x)


def test_a_leading_batch_axis_normalizes_over_the_last_one():
    x = _rand((4, 32, 128), torch.float32)
    _close(hk.ops.rmsnorm(x), _ref_rmsnorm(x, 1e-6), torch.float32)
    assert hk.ops.rmsnorm(x).shape == x.shape


def test_eps_reaches_the_kernel():
    """eps is a constexpr, so a wrong one is baked into a *separately cached*
    build -- passing it and ignoring it would look right on the default and
    wrong nowhere else."""
    x = _rand((16, 64), torch.float32) * 1e-3
    _close(hk.ops.rmsnorm(x, eps=1.0), _ref_rmsnorm(x, 1.0), torch.float32)
    assert not torch.allclose(hk.ops.rmsnorm(x, eps=1.0), hk.ops.rmsnorm(x))


# -- fused --------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(128, 256), (100, 200), (17, 65)],
                         ids=["128x256", "100x200", "17x65"])
def test_silu_mul(dtype, shape):
    """The ragged shapes matter here for a different reason than in the norms.
    There is nothing to mask -- a pointwise result does not depend on the rest
    of the tile -- but a workgroup covers WARPS*TILES blocks and the last one
    is mostly clamped, so these are the shapes where several warps write the
    same bytes at once. They have to write the same *values*."""
    a, b = _rand(shape, dtype), _rand(shape, dtype)
    _close(hk.ops.silu_mul(a, b), F.silu(a.float()).to(dtype) * b, dtype, f"{shape}")


def test_silu_mul_needs_one_whole_tile():
    a = _rand((8, 200), torch.float32)
    with pytest.raises(ValueError, match="at least 16x64"):
        hk.ops.silu_mul(a, a)


def _ref_rope(x, cos, sin):
    f = x.float()
    half = x.shape[-1] // 2
    x1, x2 = f[..., :half], f[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [128, 256])
def test_rope(dtype, d):
    bh, seq = 6, 32
    x = _rand((bh, seq, d), dtype)
    ang = torch.rand(seq, d // 2, device="cuda", dtype=torch.float32) * 6.28
    cos, sin = ang.cos().contiguous(), ang.sin().contiguous()
    _close(hk.ops.rope(x, cos, sin), _ref_rope(x, cos, sin), dtype, msg=f"D={d}")


def test_rope_rotates_by_zero_when_the_table_is_identity():
    """cos=1, sin=0 must be a copy. A halves-swapped kernel passes the random
    test only by coincidence of tolerance; it fails this one outright."""
    x = _rand((2, 16, 128), torch.float32)
    seq, half = 16, 64
    cos = torch.ones(seq, half, device="cuda")
    sin = torch.zeros(seq, half, device="cuda")
    _close(hk.ops.rope(x, cos, sin), x, torch.float32)


def test_rope_refuses_a_head_dim_whose_half_is_not_a_tile():
    x = _rand((2, 16, 64), torch.float32)
    t = torch.zeros(16, 32, device="cuda")
    with pytest.raises(ValueError, match="not a multiple"):
        hk.ops.rope(x, t, t)


# -- quantization -------------------------------------------------------------


def _ref_quant(x):
    f = x.float()
    amax = f.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    q = torch.round(f / (amax / 127.0)).clamp(-128, 127).to(torch.int8)
    return q, (amax / 127.0).squeeze(-1)


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize(shape, dtype):
    x = _rand(shape, dtype)
    q, s = hk.ops.quantize(x)
    wq, ws = _ref_quant(x)
    assert q.dtype == torch.int8 and q.shape == x.shape
    torch.testing.assert_close(s, ws, rtol=1e-5, atol=1e-8)
    # The kernel multiplies by 127/amax where the reference divides by
    # amax/127; the two differ in the last fp32 bit, so an element sitting on
    # a .5 boundary may round the other way. One step is the whole budget.
    off = (q.int() - wq.int()).abs()
    assert off.max() <= 1, off.max().item()
    assert (off > 0).float().mean() < 0.01


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{r}x{c}" for r, c in SHAPES])
def test_dequantize_inverts_quantize(shape):
    x = _rand(shape, torch.float32)
    q, s = hk.ops.quantize(x)
    got = hk.ops.dequantize(q, s, dtype=torch.float32)
    # Round trip error is one quantization step, by construction.
    step = s.unsqueeze(-1).expand_as(x)
    assert (got - x).abs().max() <= (step.max() * 0.51 + 1e-6)
    torch.testing.assert_close(got, q.float() * step, rtol=1e-5, atol=1e-6)


#: (cols, expected quant.plan) -- the widths at which quantize folds the row
#: onto the tile's sixteen rows. The scale store is the thing being tested: a
#: folded workgroup owns one row and has one slot for its answer, and the
#: whole-vector store that the unfolded variant uses would write this row's
#: scale over the next fifteen rows' slots. Nothing about the *quantized*
#: tensor would look wrong if it did -- every element would still be scaled by
#: something plausible -- so the assertion that matters here is on `s`.
Q_WIDE = [
    (8192, (8, 1, 16)),     # first width too wide to hold unfolded: folds
    (16384, (16, 1, 16)),   # folded, and wide enough to need all sixteen warps
    (32768, (16, 2, 16)),   # folded and still two tiles a warp
    (4096, (16, 4, 1)),     # fits unfolded, so it is not folded
    (5120, (5, 1, 16)),     # folds to five blocks -- a five-warp workgroup,
                            # which is the shape the warp-count rule exists for
    (1001, (16, 1, 1)),     # 1001 % 16 != 0, so no folded view exists either
]


@pytest.mark.parametrize("cols,want_plan", Q_WIDE, ids=[f"{c}" for c, _ in Q_WIDE])
def test_a_folded_quantize_writes_one_scale_per_row(cols, want_plan):
    from hk.ops import quant

    assert quant.plan(cols) == want_plan, f"{cols}: plan moved to {quant.plan(cols)}"
    # Rows that differ by orders of magnitude, so that a scale landing on the
    # wrong row is a visible factor and not a rounding difference.
    x = _rand((33, cols), torch.float32)
    x *= torch.logspace(-3, 3, 33, device="cuda").unsqueeze(-1)
    q, s = hk.ops.quantize(x)
    wq, ws = _ref_quant(x)
    torch.testing.assert_close(s, ws, rtol=1e-5, atol=1e-30)
    off = (q.int() - wq.int()).abs()
    assert off.max() <= 1, off.max().item()


def test_the_scale_is_per_row_not_per_tensor():
    """A per-tensor scale is decided by the worst outlier anywhere; on a row
    that is 1e6 times smaller it quantizes the whole row to zero. That is the
    reason this kernel exists, so assert the difference is real."""
    x = torch.ones(32, 256, device="cuda", dtype=torch.float32)
    x[0] *= 1e4
    q, s = hk.ops.quantize(x)
    assert s[0] > s[1] * 1e3
    assert (q[1] == 127).all(), "a small row must still use the full range"


def test_an_all_zero_row_comes_back_as_zeros():
    """absmax is 0 and 127/0 is an infinity the saturating convertor turns
    into 127 -- a row of zeros returning as a row of 127s. The _TINY clamp is
    what prevents it, and nothing else would catch its removal."""
    x = _rand((32, 256), torch.float32)
    x[3] = 0.0
    q, s = hk.ops.quantize(x)
    assert (q[3] == 0).all()
    assert torch.isfinite(s).all()


def test_quantization_saturates_rather_than_wrapping():
    """A C cast of 200.f to int8 is implementation-defined and in practice
    wraps to -56: an outlier comes back with the opposite sign. convertor<
    int8, float> clamps instead."""
    x = _rand((16, 64), torch.float32)
    x[0, 0] = 1e30  # dominates its row's scale; every other element -> 0
    q, _ = hk.ops.quantize(x)
    assert q[0, 0] == 127
    assert q.min() >= -128


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_takes_every_float_dtype(dtype):
    """The round trip is accurate to a quantization step, not to the dtype:
    int8 keeps ~7 bits of a row whose scale is set by its largest element, so
    the error is half a step regardless of whether the input was fp32 or bf16.
    Comparing at the dtype's tolerance would be asserting that quantization is
    lossless."""
    x = _rand((32, 128), dtype)
    q, s = hk.ops.quantize(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    back = hk.ops.dequantize(q, s, dtype=dtype)
    step = s.unsqueeze(-1).expand_as(x).float()
    err = (back.float() - x.float()).abs()
    # Half a step of rounding, plus the at-most-one-step tie difference that
    # test_quantize bounds (the kernel multiplies by 127/amax where the
    # reference divides by amax/127), plus the output dtype's own rounding.
    budget = 1.51 * step + 4e-3 * x.float().abs() + 1e-6
    assert (err <= budget).all(), (err / step).max().item()


def test_dequantize_refuses_a_scale_vector_of_the_wrong_length():
    q = torch.zeros(32, 128, device="cuda", dtype=torch.int8)
    with pytest.raises(ValueError, match="scales for"):
        hk.ops.dequantize(q, torch.ones(16, device="cuda"))
