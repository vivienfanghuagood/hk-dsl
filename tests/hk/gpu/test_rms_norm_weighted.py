"""The RMSNorm a model runs -- weighted, and vLLM's fused residual add --
against torch.

Decode normalises one row per request, so the row counts below include fewer
than a tile's sixteen and counts that are not a multiple of it. The fused op
is in place, like vLLM's: the residual becomes x + residual (rounded to the
dtype, as stored) and x its norm, computed from the unrounded sum.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

SHAPES = [(1, 4096), (5, 4096), (16, 4096), (37, 2048), (300, 4096),
          (8192, 4096), (4100, 2560)]

#: Widths the strip kernels take with a bounded tail: a run of 4 (4100), of 2
#: (2050), of 1 (129), a ragged last chunk (3000), and one narrower than a
#: chunk (64).
RAGGED = [(3, 4100), (2, 2050), (7, 129), (5, 3000), (9, 64)]


@pytest.fixture(params=["strip", "tile"])
def layout(request, monkeypatch):
    from hk.ops import norm

    monkeypatch.setattr(norm, "LAYOUT", request.param)
    return request.param


def _ref(z, w, eps=1e-6):
    z = z.float()
    return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rel(got, want):
    return ((got.float() - want).abs().max() / want.abs().max()).item()


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_rms_norm_weighted(rows, cols, layout):
    g = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.rand(cols, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    assert _rel(hk.ops.rms_norm(x, w), _ref(x, w)) < 2e-2


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_fused_add_rms_norm_in_place(rows, cols, layout):
    g = torch.Generator(device="cuda").manual_seed(rows * 7 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    r = torch.randn_like(x)
    w = torch.rand(cols, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    z = x.float() + r.float()
    want_y, want_r = _ref(z, w), z.to(torch.bfloat16)
    y, r2 = hk.ops.fused_add_rms_norm(x, r, w)
    assert y.data_ptr() == x.data_ptr() and r2.data_ptr() == r.data_ptr()
    assert torch.equal(r, want_r)
    assert _rel(x, want_y) < 2e-2


@pytest.mark.parametrize("rows,cols", RAGGED)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_strip_tails(rows, cols, dtype, layout):
    """Bounded strips: the runs past the row read zero, are not written, and
    the residual stays bit-exact. On the tile kernels the same widths back
    their last block up instead."""
    g = torch.Generator(device="cuda").manual_seed(rows * 13 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, generator=g)
    r = torch.randn_like(x)
    w = (torch.rand(cols, device="cuda", generator=g) + 0.5).to(dtype)
    assert _rel(hk.ops.rms_norm(x, w), _ref(x, w)) < 2e-2
    z = x.float() + r.float()
    want_y, want_r = _ref(z, w), z.to(dtype)
    hk.ops.fused_add_rms_norm(x, r, w)
    assert torch.equal(r, want_r)
    assert _rel(x, want_y) < 2e-2


#: (rows, cols, FOLD) the tile kernels stream the fused add with a backed-up
#: last block -- more blocks than a warp may hold, the width not a whole
#: number of them: 3x4100 (65 blocks), 4100x3100 (49, past the residual
#: variant's cap of three a warp, and unfolded by its row count), and two
#: rows wider than a strip takes, so that RDNA3's default layout comes here
#: too: 40004 (not a multiple of 16, unfolded) and 65600 (folded to 4100).
STREAMED = [(3, 4100, 1), (4100, 3100, 1), (2, 40004, 1), (2, 65600, 16)]


@pytest.mark.parametrize("rows,cols,fold", STREAMED, ids=[f"{r}x{c}" for r, c, _ in STREAMED])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a_streamed_block_two_warps_cover_is_added_once(rows, cols, fold, dtype, monkeypatch):
    """The streamed plan re-reads each block to write it, and its backed-up
    last block overlaps the one before it, as do warps clamped onto it. In
    place, a re-read of those columns after a neighbour had written them
    added the residual twice: 3x4100 came back with its last 64 columns
    several times too large, on every target, wherever this plan ran. The
    plan is asserted so that the shapes cannot drift off it."""
    from hk.ops import norm

    monkeypatch.setattr(norm, "LAYOUT", "tile")
    _, tpw, got_fold, exact = norm._resolve_weighted("addrmsw", dtype, cols, rows)
    assert (tpw, got_fold, exact) == (0, fold, 0), "the plan moved"
    g = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, generator=g)
    r = torch.randn_like(x)
    w = (torch.rand(cols, device="cuda", generator=g) + 0.5).to(dtype)
    assert _rel(hk.ops.rms_norm(x, w), _ref(x, w)) < 2e-2
    x2 = x.clone()
    assert _rel(hk.ops.rms_norm(x2, w, out=x2), _ref(x, w)) < 2e-2
    z = x.float() + r.float()
    want_y, want_r = _ref(z, w), z.to(dtype)
    hk.ops.fused_add_rms_norm(x, r, w)
    assert torch.equal(r, want_r)
    assert _rel(x, want_y) < 2e-2


def _ulps(got, want, dtype):
    """max |got - want| in units of `dtype`'s spacing at `want` (float64)."""
    mant = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    e = torch.floor(torch.log2(want.to(dtype).double().abs().clamp_min(2.0 ** -14)))
    return ((got.double() - want).abs() / torch.pow(2.0, e - mant)).max().item()


#: A model that stores its weight as an offset from one, at its hidden widths
#: and decode / prefill row counts, and the strip tails.
OFFSET_SHAPES = [(1, 5120), (8, 5120), (1024, 5120), (2, 4096), (3, 4100), (5, 3000)]


@pytest.mark.parametrize("rows,cols", OFFSET_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_weight_offset_is_added_exactly(rows, cols, dtype, layout):
    """weight_offset=1 with the stored w is y = x_hat * (1 + w), the one added
    in fp32: within half a unit of the output dtype of a float64 reference
    (plus fp32's own rounding), where folding the one into a 16-bit weight
    first is not. The fused add's residual stays bit-exact, and the norm is of
    the unrounded sum, as without the offset."""
    g = torch.Generator(device="cuda").manual_seed(rows * 31 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype, generator=g)
    r = (torch.randn(rows, cols, device="cuda", generator=g) * 4).to(dtype)
    w = (torch.randn(cols, device="cuda", generator=g) * 0.1).to(dtype)

    def ref(z):
        z = z.double()
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w.double())

    assert _ulps(hk.ops.rms_norm(x, w, weight_offset=1.0), ref(x), dtype) <= 0.51
    folded = hk.ops.rms_norm(x, (w.float() + 1.0).to(dtype))
    assert _ulps(folded, ref(x), dtype) > 0.51, "the folded weight was exact: a weak probe"
    z = x.double() + r.double()
    xi, ri = x.clone(), r.clone()
    hk.ops.fused_add_rms_norm(xi, ri, w, weight_offset=1.0)
    assert torch.equal(ri, (x.float() + r.float()).to(dtype))
    assert _ulps(xi, ref(z), dtype) <= 0.51


@pytest.mark.parametrize("rows,cols", [(1, 5120), (300, 4096)])
def test_weight_offset_zero_is_the_plain_call(rows, cols, layout):
    g = torch.Generator(device="cuda").manual_seed(rows + 3 * cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.rand(cols, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
    assert torch.equal(hk.ops.rms_norm(x, w, weight_offset=0.0), hk.ops.rms_norm(x, w))
