"""The ops DeepSeek-V4.1's index top-k needed in hk (gfx950): LDS atomic adds (a radix select's
histogram), MXFP4 MMA operands and their E8M0 scales from per-lane gathers (logits straight off a
preshuffled paged K cache), and a lane's value as a scalar for a workgroup-uniform early exit."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import fp32, i8, i32  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="gfx950's")]

S = hk.RS(fp32, 1, 512, 8)


def test_lds_atomic_adds_count_a_histogram():
    """Four warps add 1 per element to bin idx of a 512-bin LDS tile; out-of-range indices (-1, 512)
    add nothing. Every add to one bin lands: the counts are torch.bincount's."""
    st = hk.ST(fp32, 32, 16)                     # 512 bins, 16 columns: flat indices are linear

    def body(ix: hk.Tensor[i32], o: hk.Tensor[i32]):
        h = hk.lds.alloc(st, name="h")
        w = hk.arch.warp_id()
        c = hk.iota(S)
        h.store_elems(hk.map_cpp("0", c, dtype=i32), hk.map_cpp("x1 * 512 + x0 < 512 ? x1 * 512 + x0 : -1", c, w))
        hk.arch.barrier(drain=True)
        h.atomic_add_elems(hk.map_cpp("1", c, dtype=i32), ix.at[w, 0].load(hk.RS(i32, 1, 512, 8)))
        hk.arch.barrier(drain=True)
        if w == 0:
            o.store_elems(h.load_elems(c, i32), c)

    k = hk.kernel(body, arch="gfx950", warps=4, name="lds_hist", grid=lambda p: (1, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(20)
    ix = torch.randint(0, 64, (4, 512), generator=g, dtype=torch.int32) * 7 % 512   # heavy collisions
    ix[0, ::17] = -1
    ix[1, ::13] = 512
    o = torch.full((512,), -5, dtype=torch.int32, device="cuda")
    k(ix.cuda(), o)
    torch.cuda.synchronize()
    v = ix.view(-1)
    want = torch.bincount(v[(v >= 0) & (v < 512)].long(), minlength=512).int()
    assert torch.equal(o.cpu(), want)


def _gathered_fp4_kernel():
    """A (16 x 128) and B (16 x 128) FP4, packed [16, 64] bytes, scales [16, 4] E8M0: each lane gathers
    its own 16 bytes (row l % 16, bytes 16 (l / 16) ..) and its scale byte (row l % 16, block l / 16)."""
    W = hk.RS(i32, 1, 256, 4)
    L = hk.RS(i32, 1, 64, 1)

    def body(a: hk.Tensor[i32], sa: hk.Tensor[i8], b: hk.Tensor[i32], sb: hk.Tensor[i8], c: hk.Tensor[fp32]):
        lw, ln = hk.iota(W), hk.iota(L)
        widx = hk.map_cpp("((x0 >> 2) % 16) * 16 + 4 * ((x0 >> 2) / 16) + (x0 & 3)", lw)
        sidx = hk.map_cpp("(x0 % 16) * 4 + x0 / 16", ln)
        ta = hk.fp4_operand(a.load_elems(widx, runs=True))
        tb = hk.fp4_operand(b.load_elems(widx, runs=True))
        s_a = hk.mx_scales_loose(hk.map_cpp("x0 & 0xFF", sa.load_elems(sidx, i32), dtype=i32))
        s_b = hk.mx_scales_loose(hk.map_cpp("x0 & 0xFF", sb.load_elems(sidx, i32), dtype=i32))
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma_ABt_mx(ta, tb, hk.zeros(hk.RT(fp32, 16, 16, "col")), s_a, s_b,
                                                     a_fmt="fp4", b_fmt="fp4")

    return hk.kernel(body, arch="gfx950", warps=1, name="fp4_gathered", grid=lambda p: (1, 1, 1))


def test_fp4_operands_and_loose_scales_from_gathers_give_the_scaled_product():
    g = torch.Generator(device="cpu").manual_seed(21)
    a = torch.randint(0, 256, (16, 64), generator=g, dtype=torch.uint8)
    b = torch.randint(0, 256, (16, 64), generator=g, dtype=torch.uint8)
    sa = torch.randint(122, 133, (16, 4), generator=g, dtype=torch.uint8)
    sb = torch.randint(122, 133, (16, 4), generator=g, dtype=torch.uint8)
    c = torch.zeros(16, 16, device="cuda")
    _gathered_fp4_kernel()(a.cuda().view(torch.int32), sa.cuda().view(torch.int8),
                           b.cuda().view(torch.int32), sb.cuda().view(torch.int8), c)
    torch.cuda.synchronize()
    deq = lambda w, s: hk.mx.fp4_values(w).double() * torch.pow(2.0, s.double() - 127).repeat_interleave(32, -1)
    ref = deq(a, sa) @ deq(b, sb).T
    # fp4 x fp4 products are exact in fp32 and so are their 128-term sums at these scales: only the order differs
    assert ((c.cpu().double() - ref).abs().max() / ref.abs().max()).item() < 1e-6


def test_first_lane_decides_a_uniform_branch_from_one_lds_word():
    """Warp 0 writes a word to LDS; after the barrier every warp reads it, takes it as a scalar with
    first_lane, and the hk.uniform branch it drives is taken by all of them or none."""
    st = hk.ST(fp32, 16, 16)
    L = hk.RS(i32, 1, 64, 1)

    def body(v: hk.Tensor[i32], o: hk.Tensor[i32]):
        s = hk.lds.alloc(st, name="s")
        w = hk.arch.warp_id()
        ln = hk.iota(L)
        if w == 0:
            s.store_elems(v.load_elems(hk.map_cpp("x0 == 0 ? 0 : -1", ln)), hk.map_cpp("x0 == 0 ? 0 : -1", ln))
        hk.arch.barrier(drain=True)
        x = s.load_elems(hk.map_cpp("0", ln, dtype=i32), i32).first_lane()
        if hk.uniform(x > 5):
            o.store_elems(hk.map_cpp("x1 + 1", ln, x, dtype=i32), hk.map_cpp("x0 == 0 ? x1 : -1", ln, w))
        if hk.uniform(x <= 5):
            o.store_elems(hk.map_cpp("-x1", ln, x, dtype=i32), hk.map_cpp("x0 == 0 ? x1 : -1", ln, w))

    k = hk.kernel(body, arch="gfx950", warps=4, name="first_lane_exit", grid=lambda p: (1, 1, 1))
    for val, want in ((9, 10), (3, -3)):
        o = torch.zeros(4, dtype=torch.int32, device="cuda")
        k(torch.tensor([val], dtype=torch.int32, device="cuda"), o)
        torch.cuda.synchronize()
        assert o.cpu().tolist() == [want] * 4, val
