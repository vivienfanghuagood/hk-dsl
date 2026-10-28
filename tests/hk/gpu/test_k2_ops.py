"""What DeepSeek-V4.1's K2 MoE in one hk launch needed (gfx950): a work queue's loop with barriers inside
(`while hk.uniform(...)` over hk.comm.claim), an MXFP8 MMA operand from words kept in LDS
(hk.fp8_operand), and non-temporal loads for a weight stream (hk.Tensor[..., "stream"])."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import fp32, fp8e4m3, i8, i32  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="gfx950's")]

LANE = hk.RS(i32, 1, 64, 1)


def test_a_work_queue_loop_holds_barriers():
    """64 workgroups claim 1000 tasks; each task goes through LDS between the warps (a barrier each side),
    inside the claimed loop. Every task is done once, with every warp's part."""
    N = 1000

    def body(q: hk.Tensor[i32, "device"], o: hk.Tensor[i32]):
        s = hk.lds.alloc(hk.ST(fp32, 16, 16), name="s")
        w = hk.arch.warp_id()
        ln = hk.iota(LANE)
        t = hk.comm.claim(q.at[0])
        while hk.uniform(t < N):
            # warp w writes w + 1 to word w; after the barrier warp 0 sums the 4 words into o[t]
            s.store_elems(hk.map_cpp("x1 + 1 + 0 * x0", ln, w, dtype=i32), hk.map_cpp("x0 == 0 ? x1 : -1", ln, w))
            hk.arch.barrier(drain=True)
            if w == 0:
                v = s.load_elems(hk.map_cpp("x0 < 4 ? x0 : -1", ln, dtype=i32), i32)     # 1, 2, 3, 4, then 0
                tot = hk.zeros(LANE)
                tot += v.sum(axis=1, lanes="seq", xor=(32, 16, 8, 4, 2, 1)).value()
                o.store_elems(hk.map_cpp("x0 + x1 * 100", tot, t, dtype=i32), hk.map_cpp("x0 == 0 ? x1 : -1", ln, t))
            hk.arch.barrier(drain=True)
            t = hk.comm.claim(q.at[0])

    k = hk.kernel(body, arch="gfx950", warps=4, name="work_queue", grid=lambda p: (64, 1, 1))
    q = torch.zeros(1, dtype=torch.int32, device="cuda")
    o = torch.full((N,), -1, dtype=torch.int32, device="cuda")
    k(q, o)
    torch.cuda.synchronize()
    # every warp's word: 1 + 2 + 3 + 4, + t * 100
    assert torch.equal(o.cpu(), torch.arange(N, dtype=torch.int32) * 100 + 10)
    assert int(q.item()) == N + 64          # every workgroup's last claim ran past the end


def test_fp8_operand_from_lds_words_is_the_tile_loads_operand():
    """B (16 x 128 fp8) kept in LDS as words, each lane's two 16-byte halves read back and made an
    operand: the scaled MMA gives the bits a tile-loaded B gives."""
    T_W = hk.RT(fp8e4m3, 16, 128, "row")
    T_C = hk.RT(fp32, 16, 16, "col")
    X = hk.RS(i32, 1, 512, 4)

    def body(a: hk.Tensor[fp8e4m3], b: hk.Tensor[fp8e4m3], b32: hk.Tensor[i32], sa: hk.Tensor[i8], sb: hk.Tensor[i8],
             c0: hk.Tensor[fp32], c1: hk.Tensor[fp32]):
        lw = hk.lds.alloc(hk.ST(fp32, 32, 16), name="lw")          # 16 rows x 32 words
        q = hk.iota(hk.RS(i32, 1, 512, 8))
        lw.store_elems(b32.load_elems(q, runs=True), q)
        hk.arch.barrier(drain=True)
        x = hk.iota(X)                                              # lane l, run r, word e
        bw = hk.fp8_operand(lw.load_elems(hk.map_cpp(
            "((x0 % 256) / 4 % 16) * 32 + (x0 / 256) * 16 + ((x0 % 256) / 64) * 4 + x0 % 4", x, dtype=i32), i32, runs=True))
        ta = a.at[0, 0].load(T_W)
        s_a = hk.load_mx_scales(sa, 16, 4, 0, 0, loose=True)
        s_b = hk.load_mx_scales(sb, 16, 4, 0, 0, loose=True)
        c0[hk.elem_coord(0, 0, 0, 0)] = hk.mma("mma_ABt", ta.scaled(s_a),
                                               b.at[0, 0].load(T_W).scaled(s_b), hk.zeros(T_C))
        c1[hk.elem_coord(0, 0, 0, 0)] = hk.mma("mma_ABt", ta.scaled(s_a), bw.scaled(s_b),
                                               hk.zeros(T_C))

    k = hk.kernel(body, arch="gfx950", warps=1, name="fp8_operand_lds", grid=lambda p: (1, 1, 1))
    g = torch.Generator(device="cuda").manual_seed(30)
    a = (torch.randn(16, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(16, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (16, 4), dtype=torch.uint8, device="cuda", generator=g).view(torch.int8)
    sb = torch.randint(124, 131, (16, 4), dtype=torch.uint8, device="cuda", generator=g).view(torch.int8)
    c0, c1 = torch.zeros(16, 16, device="cuda"), torch.zeros(16, 16, device="cuda")
    k(a, b, b.view(torch.uint8).view(torch.int32), sa, sb, c0, c1)
    torch.cuda.synchronize()
    assert torch.equal(c0, c1)
    ref = hk.mx.dequant(a, sa.view(torch.uint8)) @ hk.mx.dequant(b, sb.view(torch.uint8)).T
    assert ((c0 - ref).abs().max() / ref.abs().max()).item() < 1e-4


def test_stream_loads_read_what_plain_loads_read():
    S = hk.RS(i32, 1, 512, 8)

    def body(w: hk.Tensor[i32, "stream"], x: hk.Tensor[i32], o0: hk.Tensor[i32], o1: hk.Tensor[i32]):
        c = hk.iota(S)
        o0.store_elems(w.load_elems(c, runs=True), c, runs=True)
        o1.store_elems(x.load_elems(c, runs=True), c, runs=True)

    k = hk.kernel(body, arch="gfx950", warps=1, name="stream_load", grid=lambda p: (1, 1, 1))
    v = torch.randint(-2**31, 2**31 - 1, (512,), dtype=torch.int32, device="cuda")
    o0, o1 = torch.zeros_like(v), torch.zeros_like(v)
    k(v, v, o0, o1)
    torch.cuda.synchronize()
    assert torch.equal(o0, v) and torch.equal(o1, v)
