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
