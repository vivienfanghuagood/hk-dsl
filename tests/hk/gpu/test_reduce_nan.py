"""gfx950's fp32 max and min reductions are IEEE 754-2019 maximum / minimum (docs/language.md,
"fp32 max and min reductions"): a NaN anywhere in what is reduced gives NaN, and -0 orders below
+0 -- the max of -0 and +0 is +0, the min -0. Register tiles along both axes, a strip's rows and
lane groups within a wave, and a row split across warps (cross_warp's combine) and across blocks
(a partial folded in by max_). Elsewhere every value comes out as torch's amax / amin, bit for
bit. Elementwise hk.max / hk.min keep fmaxf / fminf: the last test holds that too.
"""

import math
import struct

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.dsl.collective import cross_warp  # noqa: E402


def _arch():
    from hk.target import current_arch

    return current_arch()


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(_arch() != "gfx950",
                                 reason="llvm.maximum / minimum reductions are gfx950's (gfx942: fmaxf)")]

T = hk.RT(hk.fp32, 16, 64)


def _ref(vals, kind):
    """The documented max / min of a list of floats."""
    if any(math.isnan(v) for v in vals):
        return math.nan
    m = max(vals) if kind == "max" else min(vals)
    if m == 0:
        neg = [math.copysign(1.0, v) < 0 for v in vals if v == 0]
        return -0.0 if (all(neg) if kind == "max" else any(neg)) else 0.0
    return m


def _bits(t):
    return t.contiguous().view(torch.int32)


def _assert_bits(got, want):
    """Bit-identical, NaN for NaN (any NaN: payloads are not part of the contract)."""
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan), (got[torch.isnan(got) != nan], want[torch.isnan(got) != nan])
    assert torch.equal(_bits(got)[~nan], _bits(want)[~nan]), (got[~nan], want[~nan])


def _special():
    """16 x 64 fp32: random, with columns of mixed zeros (40), of -0 (41), of NaN (42) and with
    one NaN (43); then rows with a NaN (row 0 col 5), of mixed zeros (1), of -0 (2), of -inf but
    one 3.0 (3), written over those columns."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn(16, 64, generator=g)
    x[:, 40] = torch.tensor([(-0.0 if i % 2 else 0.0) for i in range(16)])
    x[:, 41] = -0.0
    x[:, 42] = math.nan
    x[9, 43] = math.nan
    x[0, 5] = math.nan
    x[1] = torch.tensor([(-0.0 if i % 3 else 0.0) for i in range(64)])
    x[2] = -0.0
    x[3] = -math.inf
    x[3, 17] = 3.0
    return x


def test_tile_row_and_column_max_and_min():
    @hk.kernel(warps=1, grid=lambda p: (1, 1, 1))
    def reduce_nan_tile(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32], c: hk.Tensor[hk.fp32]):
        t = x.at[0, 0].load(T)
        one = hk.full(T, 1.0)
        # one * v: a broadcast that keeps -0 and NaN (0 + v would turn -0 into +0)
        o.at[0, 0] = one * t.max(axis=1)[:, None]
        o.at[16, 0] = one * t.min(axis=1)[:, None]
        c.at[0, 0] = one * t.max(axis=0)[None, :]
        c.at[16, 0] = one * t.min(axis=0)[None, :]

    x = _special()
    o = torch.full((32, 64), 7.0, device="cuda")
    c = torch.full((32, 64), 7.0, device="cuda")
    reduce_nan_tile(x.cuda(), o, c)
    rows = [x[r].tolist() for r in range(16)]
    cols = [x[:, j].tolist() for j in range(64)]
    want_o = torch.tensor([[_ref(rows[r], k)] * 64 for k in ("max", "min") for r in range(16)])
    want_c = torch.tensor([[_ref(cols[j], k) for j in range(64)] for k in ("max", "min") for _ in range(16)])
    _assert_bits(o.cpu(), want_o)
    _assert_bits(c.cpu(), want_c)
    # the special values are where they should be
    assert math.isnan(o[0, 0]) and math.isnan(o[16, 0])
    assert struct.pack("<f", float(o[1, 0])) == struct.pack("<f", 0.0)      # max(-0, +0) = +0
    assert struct.pack("<f", float(o[17, 0])) == struct.pack("<f", -0.0)    # min(-0, +0) = -0
    assert struct.pack("<f", float(o[2, 0])) == struct.pack("<f", -0.0)     # max of -0s
    assert float(o[3, 0]) == 3.0
    assert math.isnan(c[0, 42]) and math.isnan(c[0, 43]) and math.isnan(c[16, 43])
    assert struct.pack("<f", float(c[0, 41])) == struct.pack("<f", -0.0)    # max of -0s (and -inf)
    assert float(c[16, 40]) == -math.inf                                       # min: row 3's -inf


def test_strip_row_and_group_max_and_min():
    S = hk.RS(hk.fp32, 2, 512, 8)

    @hk.kernel(warps=1, grid=lambda p: (1, 1, 1))
    def reduce_nan_strip(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32], g: hk.Tensor[hk.fp32]):
        v = x.at[0, 0].load(S)
        one = hk.full(S, 1.0)
        o.at[0, 0].store(one * v.max(axis=1))
        o.at[2, 0].store(one * v.min(axis=1))
        g.at[0, 0].store(v.group_max(32))
        g.at[2, 0].store(v.group_min(32))

    gen = torch.Generator().manual_seed(7)
    x = torch.randn(2, 512, generator=gen)
    x[0, 100] = math.nan                      # row 0 and its group (96 .. 127)
    x[1, :32] = torch.tensor([(-0.0 if i % 2 else 0.0) for i in range(32)])   # a group of mixed zeros
    x[1, 32:64] = -0.0                        # a group of -0
    x[1, 64:96] = math.nan                    # a group of NaN
    o = torch.full((4, 512), 7.0, device="cuda")
    gr = torch.full((4, 512), 7.0, device="cuda")
    reduce_nan_strip(x.cuda(), o, gr)
    want_o = torch.tensor([[_ref(x[r].tolist(), k)] * 512 for k in ("max", "min") for r in range(2)])
    want_g = torch.tensor([[_ref(x[r, (j // 32) * 32:(j // 32 + 1) * 32].tolist(), k) for j in range(512)]
                           for k in ("max", "min") for r in range(2)])
    _assert_bits(o.cpu(), want_o)
    _assert_bits(gr.cpu(), want_g)


def _rows_with_specials(n, seed):
    """4 rows of n fp32 columns, a quarter per warp: row 0 with one NaN in warp 2's quarter; row 1
    all zeros, +0 only in warp 1's quarter; row 2 all -0; row 3 random."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, n, generator=g)
    q = n // 4
    x[0, 2 * q + 3] = math.nan
    x[1] = -0.0
    x[1, q:2 * q] = 0.0
    x[2] = -0.0
    return x


def test_a_row_across_warps_and_blocks():
    """The row's quarters in four warps, combined by cross_warp; each warp's quarter in two
    column blocks, the second folded in by max_ / min_ of a vector."""
    WARPS, COLS = 4, 512
    S = hk.RS(hk.fp32, 4, COLS // 2, 4)
    half = COLS // 2

    @hk.kernel(warps=WARPS, grid=lambda p: (1, 1, 1))
    def reduce_nan_warps(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        c0 = hk.arch.warp_id() * COLS
        a = x.at[0, c0].load(S)
        b = x.at[0, c0 + half].load(S)
        mx = a.max(axis=1).value()
        mx.max_(b.max(axis=1).value())        # a partial folded in: a vector
        mn = a.min(axis=1).value()
        mn.min_(b.min(axis=1).value())
        mx = cross_warp(mx, "max", WARPS)
        mn = cross_warp(mn, "min", WARPS)
        one = hk.full(S, 1.0)
        o.at[0, c0].store(one * mx)
        o.at[4, c0].store(one * mn)

    x = _rows_with_specials(WARPS * COLS, 11)
    o = torch.full((8, WARPS * COLS), 7.0, device="cuda")
    reduce_nan_warps(x.cuda(), o)
    got = o.cpu()
    want = torch.tensor([[_ref(x[r].tolist(), k)] * (WARPS * COLS) for k in ("max", "min") for r in range(4)])
    # each warp stored its own first half-block; compare those columns
    cols = [w * COLS + c for w in range(WARPS) for c in range(half)]
    _assert_bits(got[:, cols], want[:, cols])
    assert math.isnan(got[0, 0]) and math.isnan(got[4, 0])        # the NaN of warp 2, in warp 0
    assert struct.pack("<f", float(got[1, 0])) == struct.pack("<f", 0.0)     # warp 1's +0
    assert struct.pack("<f", float(got[5, 0])) == struct.pack("<f", -0.0)


def test_elementwise_max_and_min_keep_maxnum():
    """maximum / minimum (hk.max / hk.min) are elementwise, fmaxf / fminf on every target: a NaN operand gives the
    other operand."""
    T16 = hk.RT(hk.fp32, 16, 64)

    @hk.kernel(warps=1, grid=lambda p: (1, 1, 1))
    def elementwise_nan(a: hk.Tensor[hk.fp32], b: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        ta = a.at[0, 0].load(T16)
        tb = b.at[0, 0].load(T16)
        o.at[0, 0] = ta.maximum(tb)
        o.at[16, 0] = ta.minimum(tb)

    g = torch.Generator().manual_seed(3)
    a = torch.randn(16, 64, generator=g)
    b = torch.randn(16, 64, generator=g)
    a[0, :8] = math.nan
    b[1, :8] = math.nan
    o = torch.full((32, 64), 7.0, device="cuda")
    elementwise_nan(a.cuda(), b.cuda(), o)
    got = o.cpu()
    torch.testing.assert_close(got[:16], torch.fmax(a, b), rtol=0, atol=0)
    torch.testing.assert_close(got[16:], torch.fmin(a, b), rtol=0, atol=0)
