"""Register strips on the GPU: loads and stores with the element type
converting, bounded at a ragged edge, row and column reductions across the
warps, both broadcasts, the column fills, and a strip of several rows.

hk.ops.norm's strip kernels are the shipped users (test_phase2.py,
test_rms_norm_weighted.py); this is the layout on its own, so that a failure
names the primitive.
"""

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.dsl.collective import cross_warp  # noqa: E402

def _target():
    from hk.target import current_arch, get_target

    return get_target(current_arch())


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not _target().strips,
                                 reason="strips are include/rdna3's and gfx950's")]

#: Columns scale with the wave: a chunk is `wave` lanes of one run, so a
#: wave32 case's widths double on wave64 and every edge stays where it was
#: relative to the chunks.
W = _target().wave // 32

_DT = {torch.bfloat16: hk.bf16, torch.float16: hk.fp16, torch.float32: hk.fp32}


def _stats_kernel(dtype, rows, cols, run, warps, bounded):
    """Per strip of `rows` rows: o = x * w + rowsum(x) - rowmax(x), and cs =
    the strip's column sums, one row of cs per strip."""
    S = hk.RS(hk.fp32, rows, cols, run)
    dt = _DT[dtype]

    # No `from __future__ import annotations` in this file: the parameter
    # annotations name `dt`, a local, and are evaluated when `body` is made.
    def body(x: hk.Tensor[dt], w: hk.Tensor[dt], o: hk.Tensor[hk.fp32],
             cs: hk.Tensor[hk.fp32]):
        blk = hk.arch.block_idx.x
        r0, c0 = blk * rows, hk.arch.warp_id() * cols
        g = w.at[0, c0].load(hk.row_vec(S), bounded=bounded)
        v = x.at[r0, c0].load(S, bounded=bounded)
        s = cross_warp(v.sum(axis=1), "add", warps)
        vm = hk.zeros(S)
        v.right_fill_(x.cols - c0, float("-inf"), into=vm)
        m = cross_warp(vm.max(axis=1), "max", warps)
        col = hk.zeros(hk.row_vec(S))
        col += v.sum(axis=0)
        cs.at[blk, c0].store(col, bounded=bounded)
        v *= g
        v += s
        v -= m
        o.at[r0, c0].store(v, bounded=bounded)

    name = f"strip_stats_{dt.name}_{rows}x{cols}_r{run}_w{warps}_b{int(bounded)}"
    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=lambda p: (hk.cdiv(p.x.rows, rows), 1, 1),
                     name=name)


CASES = [
    # dtype, strip rows, strip cols (per warp), run, warps, tensor rows x cols, bounded
    (torch.bfloat16, 1, 256, 8, 4, 5, 1024, False),
    (torch.bfloat16, 3, 512, 8, 2, 7, 1000, True),     # ragged rows and columns
    (torch.float16, 2, 256, 8, 2, 4, 512, False),
    (torch.float32, 1, 128, 4, 4, 3, 500, True),
    (torch.float32, 4, 64, 2, 1, 9, 60, True),
    (torch.bfloat16, 1, 32, 1, 3, 2, 95, True),        # a run of 1, three warps
]


@pytest.mark.parametrize("dtype,rows,cols,run,warps,m,n,bounded", CASES)
def test_strip_loads_reduces_broadcasts_and_stores(dtype, rows, cols, run, warps, m, n,
                                                   bounded):
    cols, n = cols * W, n * W
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n)
    x = torch.randn(m, n, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(1, n, device="cuda", dtype=dtype, generator=g)
    blocks = -(-m // rows)
    o = torch.full((m, n), 7.0, device="cuda")
    cs = torch.full((blocks, n), 7.0, device="cuda")
    _stats_kernel(dtype, rows, cols, run, warps, bounded)(x, w, o, cs)
    xf, wf = x.float(), w.float()
    want = xf * wf + xf.sum(-1, keepdim=True) - xf.max(-1, keepdim=True).values
    torch.testing.assert_close(o, want, rtol=1e-5, atol=1e-4)
    pad = torch.zeros(blocks * rows, n, device="cuda")
    pad[:m] = xf
    torch.testing.assert_close(cs, pad.view(blocks, rows, n).sum(1), rtol=1e-5, atol=1e-5)


def test_a_16bit_strip_copies_bit_for_bit():
    S = hk.RS(hk.bf16, 2, 512)

    def strip_copy16(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        r0 = hk.arch.block_idx.x * 2
        o.at[r0, 0].store(x.at[r0, 0].load(S, bounded=True), bounded=True)

    k = hk.kernel(strip_copy16, warps=1, name="strip_copy16",
                  grid=lambda p: (hk.cdiv(p.x.rows, 2), 1, 1))
    x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    o = torch.zeros_like(x)
    k(x, o)
    assert torch.equal(o, x)


def test_a_bounded_store_leaves_the_rest_of_the_tensor_alone():
    """Runs past the last column and rows past the last row are not
    written: the guard band around a narrower view stays as it was."""
    S = hk.RS(hk.fp32, 2, 256 * W)

    def strip_fill(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        r0 = hk.arch.block_idx.x * 2
        v = x.at[r0, 0].load(S, bounded=True)
        v += 1.0
        o.at[r0, 0].store(v, bounded=True)

    k = hk.kernel(strip_fill, warps=1, name="strip_fill",
                  grid=lambda p: (hk.cdiv(p.x.rows, 2), 1, 1))
    big = torch.full((8, 320 * W), -5.0, device="cuda")
    view = big[:3, :200 * W]
    x = torch.randn(3, 200 * W, device="cuda")
    # A strided output: the kernel's row stride is the big tensor's.
    k(x, view)
    torch.testing.assert_close(big[:3, :200 * W], x + 1.0)
    assert (big[:3, 200 * W:] == -5.0).all() and (big[3:] == -5.0).all()


# ---------------------------------------------------------------- a stated reduction order (gfx950)

def _emulate(x, lanes, xor):
    """The ordered row sum on the host, in fp32 step by step: lane l holds columns 8 l .. 8 l + 7 of
    the row (one 512-column chunk, run 8); its values left to right ("seq") or as hk's balanced
    tree, then op with lane l ^ o for each o of `xor` in order."""
    rows = x.shape[0]
    a = x.view(rows, 64, 8).clone()
    if lanes == "seq":
        p = a[:, :, 0].clone()
        for i in range(1, 8):
            p = p + a[:, :, i]
    else:
        n = 8
        while n > 1:
            h = (n + 1) // 2
            for i in range(n - h):
                a[:, :, i] = a[:, :, i] + a[:, :, i + h]
            n = h
        p = a[:, :, 0].clone()
    lane = torch.arange(64)
    for o in xor:
        p = p + p[:, lane ^ o]
    return p                                           # [rows, 64]: every lane's final value


@pytest.mark.skipif(_target().wave != 64, reason="a stated order is gfx950's")
@pytest.mark.parametrize("lanes, xor", [("seq", (32, 16, 8, 4, 2, 1)), ("tree", (32, 16, 8, 4, 2, 1)),
                                         ("seq", (1, 2, 4, 8, 16, 32)), ("seq", (8, 1, 32, 2, 16, 4))])
def test_a_stated_order_is_bit_for_bit_that_order(lanes, xor):
    S = hk.RS(hk.fp32, 1, 512, 8)

    def ordered(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        v = x.at[row, 0].load(S)
        w = hk.zeros(S)
        w += v.sum(axis=1, lanes=lanes, xor=xor)
        o.at[row, 0].store(w)

    k = hk.kernel(ordered, warps=1, name=f"ordered_{lanes}_{'_'.join(map(str, xor))}",
                  grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(7)
    # a wide dynamic range, so that the grouping shows in the low bits
    x = (torch.randn(256, 512, generator=g) * torch.exp(torch.randn(256, 512, generator=g) * 3)).float()
    o = torch.zeros(256, 512, device="cuda")
    k(x.cuda(), o)
    torch.cuda.synchronize()
    want = _emulate(x, lanes, xor)                     # every lane ends with its own final value
    got = o.cpu().view(256, 64, 8)[:, :, 0]
    assert torch.equal(got.view(torch.int32), want.view(torch.int32)), \
        f"{int((got != want).sum())} of {got.numel()} lanes differ"
    if (lanes, xor) == ("seq", (32, 16, 8, 4, 2, 1)):
        # hk's own order groups differently: the test can tell orders apart
        assert not torch.equal(_emulate(x, "tree", (1, 2, 4, 8, 16, 32)), want)


def _fma(a, b, c):
    """fp32 fma: the exact product of two floats fits a double, the sum rounds once there and
    once to float (double rounding cannot change the result's float rounding here: the double
    keeps 29 extra bits)."""
    return (a.double() * b.double() + c.double()).float()


def _emulate_dot(x, y, run, fused, xor=(32, 16, 8, 4, 2, 1)):
    rows = x.shape[0]
    a, b = x.view(rows, 64, run), y.view(rows, 64, run)
    if not fused:                                   # rounded products, left to right
        p = a * b
        s = p[:, :, 0].clone()
        for i in range(1, run):
            s = s + p[:, :, i]
    elif run == 1:
        s = a[:, :, 0] * b[:, :, 0]
    else:                                           # LLVM's contraction of a compiled loop
        s = _fma(a[:, :, 0], b[:, :, 0], a[:, :, 1] * b[:, :, 1])
        for i in range(2, run):
            s = _fma(a[:, :, i], b[:, :, i], s)
    lane = torch.arange(64)
    for o in xor:
        s = s + s[:, lane ^ o]
    return s


@pytest.mark.skipif(_target().wave != 64, reason="a stated order is gfx950's")
@pytest.mark.parametrize("run", [1, 2, 8])
def test_a_sum_of_products_in_a_stated_order(run):
    # (x * y).sum(lanes="seq") adds rounded products; sum(times=y) fuses them the way a compiled
    # loop does (dsv41_mono's sums of squares); each is bit for bit its emulation and not the other
    S = hk.RS(hk.fp32, 1, 64 * run, run)
    XOR = (32, 16, 8, 4, 2, 1)

    def dots(x: hk.Tensor[hk.fp32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32],
             f: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        b = y.at[row, 0].load(S)
        w = hk.zeros(S)
        w += (a * b).sum(axis=1, lanes="seq", xor=XOR)
        o.at[row, 0].store(w)
        v = hk.zeros(S)
        v += a.sum(axis=1, lanes="seq", xor=XOR, times=b)
        f.at[row, 0].store(v)

    k = hk.kernel(dots, warps=1, name=f"dots_{run}", grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(run)
    x = (torch.randn(512, 64 * run, generator=g) * torch.exp(torch.randn(512, 64 * run, generator=g) * 2)).float()
    y = x.clone() if run != 2 else torch.randn(512, 64 * run, generator=g)     # squares, and a dot
    o = torch.zeros(512, 64 * run, device="cuda")
    f = torch.zeros(512, 64 * run, device="cuda")
    k(x.cuda(), y.cuda(), o, f)
    torch.cuda.synchronize()
    plain, fused = _emulate_dot(x, y, run, False), _emulate_dot(x, y, run, True)
    got_o = o.cpu().view(512, 64, run)[:, :, 0]
    got_f = f.cpu().view(512, 64, run)[:, :, 0]
    assert torch.equal(got_o.view(torch.int32), plain.view(torch.int32)), \
        f"rounded products: {int((got_o != plain).sum())} of {got_o.numel()} lanes differ"
    assert torch.equal(got_f.view(torch.int32), fused.view(torch.int32)), \
        f"fused: {int((got_f != fused).sum())} of {got_f.numel()} lanes differ"
    if run > 1:
        assert not torch.equal(plain, fused)        # the inputs tell the two apart


# ---------------------------------------------------------------- hk.map_cpp (gfx950)

@pytest.mark.skipif(_target().wave != 64, reason="hk.map_cpp is gfx950's")
def test_map_cpp_computes_bits_fmas_and_takes_scalars():
    S = hk.RS(hk.fp32, 1, 512, 8)

    def mapped(x: hk.Tensor[hk.fp32], y: hk.Tensor[hk.fp32], n: hk.Tensor[hk.i32],
               e: hk.Tensor[hk.i32], f: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        b = y.at[row, 0].load(S)
        k = n.at[0].load_scalar()
        # the biased exponent of the smallest power of two >= |a|, plus a device scalar
        ce = hk.map_cpp("((__builtin_bit_cast(int, __builtin_fabsf(x0)) >> 23) & 0xFF)"
                        " + ((__builtin_bit_cast(int, x0) & 0x7FFFFF) != 0) + x1", a, k, dtype=hk.i32)
        e.at[row, 0].store(ce)
        f.at[row, 0].store(hk.map_cpp("__builtin_fmaf(x0, x1, x0)", a, b))

    k = hk.kernel(mapped, warps=1, name="map_cpp_bits", grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(64, 512, generator=g) * torch.exp(torch.randn(64, 512, generator=g) * 4)
    y = torch.randn(64, 512, generator=g)
    n = torch.tensor([5], dtype=torch.int32)
    e = torch.zeros(64, 512, dtype=torch.int32, device="cuda")
    f = torch.zeros(64, 512, device="cuda")
    k(x.cuda(), y.cuda(), n.cuda(), e, f)
    torch.cuda.synchronize()
    bits = x.abs().view(torch.int32)
    want_e = ((bits >> 23) & 0xFF) + ((x.view(torch.int32) & 0x7FFFFF) != 0).int() + 5
    assert torch.equal(e.cpu(), want_e)
    want_f = (x.double() * y.double() + x.double()).float()   # one rounding of the exact a*b + a
    assert torch.equal(f.cpu().view(torch.int32), want_f.view(torch.int32))


@pytest.mark.skipif(_target().wave != 64, reason="hk.map_cpp is gfx950's")
def test_map_cpp_does_not_fuse_a_product_and_a_sum():
    # HIP compiles with contraction on: x0 * x1 + x0 would be one fma (one rounding) unless the
    # lambda turns it off; as written it is two roundings, and an fma is spelled __builtin_fmaf
    S = hk.RS(hk.fp32, 1, 512, 8)

    def unfused(x: hk.Tensor[hk.fp32], y: hk.Tensor[hk.fp32], f: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        b = y.at[row, 0].load(S)
        f.at[row, 0].store(hk.map_cpp("x0 * x1 + x0", a, b))

    k = hk.kernel(unfused, warps=1, name="map_cpp_unfused", grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(4)
    x = torch.randn(64, 512, generator=g)
    y = torch.randn(64, 512, generator=g)
    f = torch.zeros(64, 512, device="cuda")
    k(x.cuda(), y.cuda(), f)
    torch.cuda.synchronize()
    want = (x * y) + x                                          # two roundings
    assert torch.equal(f.cpu().view(torch.int32), want.view(torch.int32))
    fused = (x.double() * y.double() + x.double()).float()
    assert not torch.equal(fused, want)                         # the inputs tell the two apart


# ---------------------------------------------------------------- strip index / lane ops (gfx950)

_gfx950 = pytest.mark.skipif(_target().wave != 64, reason="gfx950's")


def _fold(a, tree):
    """A run folded as cdna.cuh does: left to right, or its balanced tree (a[i] op= a[i + h])."""
    a = list(a.unbind(-1))
    if not tree:
        v = a[0]
        for x in a[1:]:
            v = v + x
        return v
    n = len(a)
    while n > 1:
        h = (n + 1) // 2
        for i in range(n - h):
            a[i] = a[i] + a[i + h]
        n = h
    return a[0]


@_gfx950
@pytest.mark.parametrize("lanes,xor", [("seq", (1, 2)), ("seq", (2, 1)), ("tree", (1, 2))])
def test_group_sum_is_in_the_stated_order_and_group_max_min_are_the_groups(lanes, xor):
    S = hk.RS(hk.fp32, 1, 512, 8)

    def grouped(x: hk.Tensor[hk.fp32], s: hk.Tensor[hk.fp32], mx: hk.Tensor[hk.fp32],
                mn: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        s.at[row, 0].store(a.group_sum(32, lanes=lanes, xor=xor))
        mx.at[row, 0].store(a.group_max(32))
        mn.at[row, 0].store(a.group_min(32))

    k = hk.kernel(grouped, warps=1, name=f"group_{lanes}_{''.join(map(str, xor))}",
                  grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(5)
    x = torch.randn(16, 512, generator=g) * torch.exp(torch.randn(16, 512, generator=g) * 3)
    s, mx, mn = (torch.zeros(16, 512, device="cuda") for _ in range(3))
    k(x.cuda(), s, mx, mn)
    torch.cuda.synchronize()
    run = _fold(x.view(16, 16, 4, 8), lanes == "tree")         # [row, group, lane]
    for o in xor:                                               # lane l takes l ^ o, in order
        run = run + run[..., torch.arange(4) ^ o]
    want = run[..., None].expand(16, 16, 4, 8).reshape(16, 512)
    assert torch.equal(s.cpu().view(torch.int32), want.contiguous().view(torch.int32))
    grp = x.view(16, 16, 32)
    assert torch.equal(mx.cpu(), grp.amax(-1, keepdim=True).expand(16, 16, 32).reshape(16, 512))
    assert torch.equal(mn.cpu(), grp.amin(-1, keepdim=True).expand(16, 16, 32).reshape(16, 512))


@_gfx950
def test_iota_indexes_load_elems_and_store_elems_and_out_of_range_is_masked():
    S = hk.RS(hk.fp32, 1, 512, 8)

    def gathered(x: hk.Tensor[hk.fp32], t: hk.Tensor[hk.i32], o: hk.Tensor[hk.fp32],
                 ot: hk.Tensor[hk.i32], sc: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        c = hk.iota(S)
        # reversed within the row; every 7th column past the tensor (reads 0), every 11th at -1
        idx = hk.map_cpp("(x1 * 512 + 511 - x0) + (x0 % 7 == 3 ? (1 << 28) : 0)", c, row)
        idx = hk.map_cpp("x0 % 11 == 5 ? -1 : x1", c, idx)
        o.at[row, 0].store(x.load_elems(idx))
        ot.at[row, 0].store(t.load_elems(idx))
        # scattered back to the reversed column; the masked elements write nothing
        sc.store_elems(x.at[row, 0].load(S), idx)

    k = hk.kernel(gathered, warps=1, name="elems", grid=lambda p: (p.x.rows, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(6)
    x = torch.randn(8, 512, generator=g)
    t = torch.randint(-2**31, 2**31 - 1, (8, 512), generator=g, dtype=torch.int32)
    o = torch.zeros(8, 512, device="cuda")
    ot = torch.zeros(8, 512, dtype=torch.int32, device="cuda")
    sc = torch.full((8, 512), -7.0, device="cuda")
    k(x.cuda(), t.cuda(), o, ot, sc)
    torch.cuda.synchronize()
    col = torch.arange(512)
    live = (col % 7 != 3) & (col % 11 != 5)
    want = torch.where(live, x.flip(1), 0.0)
    want_t = torch.where(live, t.flip(1), 0)
    assert torch.equal(o.cpu(), want) and torch.equal(ot.cpu(), want_t)
    want_sc = torch.full((8, 512), -7.0)
    want_sc[:, (511 - col)[live]] = x[:, live]
    assert torch.equal(sc.cpu(), want_sc)


@_gfx950
def test_lane_gather_takes_the_run_element_of_another_lane():
    S = hk.RS(hk.fp32, 1, 512, 8)

    def perm(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        lane = hk.map_cpp("(x0 / 8) ^ (5 + x1)", hk.iota(S), row)
        o.at[row, 0].store(x.at[row, 0].load(S).lane_gather(lane))

    k = hk.kernel(perm, warps=1, name="lane_gather", grid=lambda p: (p.x.rows, 1, 1))
    x = torch.randn(4, 512, generator=torch.Generator(device="cpu").manual_seed(7))
    o = torch.zeros(4, 512, device="cuda")
    k(x.cuda(), o)
    torch.cuda.synchronize()
    for r in range(4):
        src = (torch.arange(64) ^ (5 + r))
        assert torch.equal(o[r].cpu().view(64, 8), x[r].view(64, 8)[src])


@_gfx950
def test_mfma_strip_is_one_f32_16x16x4_in_its_lane_layout():
    A1, C4 = hk.RS(hk.fp32, 1, 64, 1), hk.RS(hk.fp32, 1, 256, 4)

    def mm(a: hk.Tensor[hk.fp32], b: hk.Tensor[hk.fp32], c: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        o.at[0, 0].store(hk.mfma_strip("f32_16x16x4", a.at[0, 0].load(A1), b.at[0, 0].load(A1),
                                       c.at[0, 0].load(C4)))

    k = hk.kernel(mm, warps=1, name="mfma_strip", grid=lambda p: (1, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(8)
    A = torch.randint(-8, 8, (16, 4), generator=g).float()       # small integers: the sum is exact
    B = torch.randint(-8, 8, (16, 4), generator=g).float()
    C = torch.randint(-64, 64, (16, 16), generator=g).float()
    lane = torch.arange(64)
    a = A[lane % 16, lane // 16][None]                           # lane l holds A[l % 16][l // 16]
    b = B[lane % 16, lane // 16][None]
    rows = 4 * (lane // 16)[:, None] + torch.arange(4)           # lane l's 4 values: C rows 4 (l // 16) + i
    cols = (lane % 16)[:, None].expand(64, 4)
    c = C[rows, cols].reshape(1, 256)
    o = torch.zeros(1, 256, device="cuda")
    k(a.cuda(), b.cuda(), c.cuda(), o)
    torch.cuda.synchronize()
    D = C + A @ B.T
    assert torch.equal(o.cpu(), D[rows, cols].reshape(1, 256))
