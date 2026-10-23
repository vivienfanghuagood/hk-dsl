"""gfx950's GEMM and the three things it is written with: the asynchronous
group load (gfx950's DMA, buffer_load ... lds), the vmcnt hk-lds-pipeline
derives for it, and hk.arch.pingpong -- plus the cross-warp checks that make
a copy left in flight something the IR can vouch for.

The schedule is HipKittens' CDNA4 GEMM. Translating it found a read its
hand-counted waits do not cover; the last test here holds the refusal.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.errors import VerifyError  # noqa: E402
from hk.ops import gemm_cdna  # noqa: E402

ST = hk.ST(hk.bf16, 128, 64)
RT = hk.RT(hk.bf16, 64, 64, "row")


def _gemm(name, **sched):
    return gemm_cdna.gemm_kernel(name, **{**gemm_cdna.DEFAULT, **sched}, arch="gfx950")


def _trace(fn, arch="gfx950", warps=8):
    return hk.kernel(fn, arch=arch, warps=warps, grid=lambda p: (1, 1, 1)).trace()


def _src(fn, arch="gfx950"):
    return scaffold.bare(_trace(fn, arch))


def _waits(src):
    return [int(n) for n in re.findall(r"kittens::vm_wait<(\d+)>\(\);", src)]


# -- the language -------------------------------------------------------------


def test_a_copy_left_in_flight_is_retired_by_a_derived_vmcnt():
    """Two copies of a 128x64 bf16 tile over 512 threads: two 16-byte copies a
    thread each. Retiring the first leaves the second's two outstanding."""
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        As = hk.lds.alloc(ST, count=2, name="As")
        g.load(As[0], a.at[0, 0], wait=False)
        g.load(As[1], a.at[0, 64], wait=False)
        g.wait_for(As[0])
        hk.arch.barrier()
        x = As[0].subtile(64, 64, 0, 0).load(RT)
        g.wait_for(As[1])
        a.at[0, 0] = x

    src = _src(k)
    assert src.count("kittens::group_load_async<8>(") == 2
    assert _waits(src) == [2, 0]


def test_a_copy_left_in_flight_is_gfx950_only():
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0], wait=False)

    for arch in ("gfx1100", "gfx942"):
        with pytest.raises(TypeError, match="buffer_load ... lds"):
            _trace(k, arch, warps=4 if arch == "gfx1100" else 8)


def test_a_copy_left_in_flight_lays_out_a_whole_tile():
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(hk.ST(hk.bf16, 256, 64), name="s")
        g.load(s.subtile(128, 64, 0, 0), a.at[0, 0], wait=False)

    with pytest.raises(TypeError, match="into a subtile"):
        _trace(k)


def test_reading_before_every_share_has_landed_is_refused():
    """g.wait_for retires this warp's share; the others' land a barrier later."""
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0], wait=False)
        g.wait_for(s)
        a.at[0, 0] = s.subtile(64, 64, 0, 0).load(RT)

    with pytest.raises(VerifyError, match="0 barriers after this warp's share"):
        _trace(k)


def test_inside_a_pingpong_it_takes_two_barriers():
    def k(a: hk.Tensor[hk.bf16], *, BARRIERS: hk.Constexpr = 1):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0], wait=False)
        with hk.arch.pingpong(hk.arch.warp_id() // 4 == 1):
            g.wait_for(s)
            for _ in range(BARRIERS):
                hk.arch.barrier()
            a.at[0, 0] = s.subtile(64, 64, 0, 0).load(RT)

    kern = hk.kernel(k, arch="gfx950", warps=8, grid=lambda p: (1, 1, 1))
    with pytest.raises(VerifyError, match="half the workgroup is a barrier behind"):
        kern.trace(BARRIERS=1)
    src = scaffold.bare(kern.trace(BARRIERS=2))
    i = src.index("if (eq) __builtin_amdgcn_s_barrier();")
    assert src.index("if (!(eq)) __builtin_amdgcn_s_barrier();") > i


def test_a_copy_over_a_buffer_still_being_read_is_refused():
    """The read may cross a barrier; the copy into the buffer it reads may not
    be issued until the read has landed, a barrier (two in a pingpong) ago."""
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0])
        hk.arch.barrier()
        x = s.subtile(64, 64, 0, 0).load(RT, wait=False)
        hk.arch.barrier()
        g.load(s, a.at[0, 64], wait=False)
        hk.lds.wait_for(x)

    with pytest.raises(VerifyError, match="while this warp's reads of it into lds"):
        _trace(k)


def test_pingpong_takes_a_condition_from_warp_id():
    def by_lane(a: hk.Tensor[hk.bf16]):
        with hk.arch.pingpong(hk.arch.lane_id() < 32):
            hk.arch.barrier()

    with pytest.raises(VerifyError, match="depends on lane_id"):
        _trace(by_lane)

    def not_a_comparison(a: hk.Tensor[hk.bf16]):
        with hk.arch.pingpong(hk.arch.warp_id()):
            hk.arch.barrier()

    with pytest.raises(TypeError, match="a device comparison"):
        _trace(not_a_comparison)


def test_rdna_output_has_none_of_it():
    """A sync group load on gfx1100 is what it was: no DMA, no vmcnt."""
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(4)
        s = hk.lds.alloc(hk.ST(hk.bf16, 64, 32), name="s")
        g.load(s, a.at[0, 0])
        hk.arch.barrier()
        a.at[0, 0] = s.load(hk.RT(hk.bf16, 64, 32, "row"))

    src = _src(k, "gfx1100")
    assert "group_load_async" not in src and "vm_wait" not in src


# -- the kernel ---------------------------------------------------------------


def test_the_gemm_is_hipkittens_schedule_with_derived_waits():
    src = scaffold.bare(_gemm("g_t1").trace())
    # Four copies before the pingpong, three in its prologue, eight a loop
    # iteration, one in the epilogue.
    assert src.count("kittens::group_load_async<8>(") == 4 + 3 + 8 + 1
    # The prologue's two waits, the loop's two, the epilogue's two: all but
    # three copies in flight where HipKittens leaves three.
    assert _waits(src) == [4, 6, 6, 6, 2, 0]
    # Eight math phases a loop iteration, six in the epilogue.
    assert src.count("__builtin_amdgcn_s_setprio(1);") == 8 + 6
    assert "if (eq) __builtin_amdgcn_s_barrier();" in src
    assert "__launch_bounds__(512, 1)" in src
    assert "LDS    : 131072 B" in src


def test_hipkittens_own_waits_are_refused():
    """HipKittens waits for all but three copies at the end of each iteration
    (vmcnt(6)). The copy that leaves outstanding is the one into Bs[1][0] --
    hk's Bs[2] -- and the next iteration's fourth memory phase reads it before
    its own wait: right by timing, not by any wait."""
    with pytest.raises(VerifyError, match=r"reads Bs\[2\] while this warp's copy into it is in flight"):
        _gemm("g_t2", hipkittens_waits=True, late_b=False).trace()


def test_reading_that_tile_one_phase_later_makes_hipkittens_waits_right():
    """hk's schedule reads Bs[2] in the fifth memory phase with A's half,
    after the fourth phase's wait: HipKittens' own counts then cover every
    read, and the pass accepts them unchanged."""
    src = scaffold.bare(_gemm("g_t4", hipkittens_waits=True, late_b=True).trace())
    assert _waits(src) == [4, 6, 6, 6, 2, 0]


def test_split_k_writes_fp32_shares_and_grids_the_splits():
    """K cut across workgroups: the grid's y is the split, each share is the
    fp32 accumulators stored to the scratch's depth, and the waits inside are
    the unsplit kernel's."""
    src = scaffold.bare(_gemm("g_t5", splits=2).trace())
    assert "* (g.b.rows() / 256)), 2, 1);" in src
    assert src.count("kittens::store_bounded(g.ws,") == 4
    assert "g.c" not in src
    assert _waits(src) == [4, 6, 6, 6, 2, 0]


def test_shapes_pick_their_blocks():
    assert gemm_cdna.schedule_for(4096, 4096, 4096)["block_n"] == 256
    assert gemm_cdna.schedule_for(4096, 4224, 4096)["block_n"] == 128
    assert gemm_cdna.schedule_for(100, 4096, 4096)["block_m"] == 128
    assert gemm_cdna.schedule_for(4096, 4096, 4096)["block_m"] == 256


def test_an_odd_count_of_k_tiles_is_padded_with_zeros():
    """The loop takes K-tiles in pairs; the pad tile's copies start at the
    tensors' last row, where the tile descriptor ends."""
    src = scaffold.bare(_gemm("g_t3").trace())
    assert re.search(r"cdiv\w* = \(\w+ \+ c\w* - 1\) / c\w*;", src)


def test_a_row_limit_narrows_the_copy_and_needs_it_left_in_flight():
    def k(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0], wait=False, rows=a.rows - 5)
        g.wait_for(s)

    assert re.search(r"group_load_async<8>\(s, g\.a, idx, \w+\);", _src(k))

    def waited(a: hk.Tensor[hk.bf16]):
        g = hk.group(8)
        s = hk.lds.alloc(ST, name="s")
        g.load(s, a.at[0, 0], rows=4)

    with pytest.raises(TypeError, match="pass wait=False"):
        _trace(waited)


def test_shapes_route_to_hk_kernels():
    """Every point of the table is one of hk's candidates, and a shape the
    table did not measure falls to the rule."""
    for m in (1, 16, 64, 256, 1024, 4096, 8192):
        name = gemm_cdna.route(m, 4096, 4096)
        assert name in gemm_cdna.CANDIDATES
        assert gemm_cdna.fits(name, m, 4096, 4096)
    assert gemm_cdna.route(8192, 3968 * 2, 6016 + 128) == "pp"
    assert not gemm_cdna.fits("pp", 64, 4096, 96)
    # A measured point that is also a square is in the table: the squares
    # route to the pingpong, not to whatever an M either side of them took.
    assert gemm_cdna.route(4096, 4096, 4096) == "pp"
    assert gemm_cdna.route(8192, 8192, 8192) == "pp"


# -- the streaming and balanced blocks ----------------------------------------


def test_a_window_narrower_than_a_named_layout_is_refused():
    """include/cdna4's st_subtile addresses whole subtiles of the layout: a
    32-wide window of st_16x64 read its subtile's first 32 columns whatever
    its index (the four-warp GEMM gave wrong answers with it)."""
    def k(x: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 64, 64), name="s", shape="16x64")
        hk.group(1).load(s, x.at[0, 0])
        hk.arch.barrier()
        r = s.subtile(64, 32, 0, 1).load(hk.RT(hk.bf16, 64, 32, "row"))
        x.at[0, 0] = r

    with pytest.raises(Exception, match="is not whole 16x64 subtiles"):
        scaffold.bare(hk.kernel(k, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace())

    def whole(x: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 64, 128), name="s", shape="16x64")
        hk.group(1).load(s, x.at[0, 0])
        hk.arch.barrier()
        x.at[0, 0] = s.subtile(64, 64, 0, 1).load(hk.RT(hk.bf16, 64, 64, "row"))

    assert "st_shape::st_16x64>" in scaffold.bare(
        hk.kernel(whole, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace())


def test_the_streaming_blocks_read_whole_lines_past_the_caches():
    """dp32x96nt: st_16x64 allocations read 64 deep, and B -- the weights --
    copied non-temporal; A, which every block of N reads again, is not."""
    from hk.ops import gemm_cdna_dp

    kind, sched = gemm_cdna.CANDIDATES["dp32x96nt"]
    src = scaffold.bare(gemm_cdna_dp.gemm_kernel(
        "g_nt", **{**gemm_cdna_dp.DEFAULT, **sched}, arch="gfx950").trace())
    assert "st_shape::st_16x64>" in src and "st_16x32" not in src
    assert re.search(r"group_load_async<4, true>\(\w+, g\.b,", src)
    assert re.search(r"group_load_async<4>\(\w+, g\.a,", src)
    assert "subtile_inplace<16, 64>" in src and "subtile_inplace<48, 64>" in src


def test_blocks_take_the_n_they_divide():
    assert gemm_cdna.fits("dp32x96nt", 32, 24576, 4096)
    assert not gemm_cdna.fits("dp32x96nt", 32, 151936, 4096)    # 96 does not divide
    assert not gemm_cdna.fits("dp32x96nt", 32, 24576, 4096 + 128)  # 256-deep K-tiles
    assert gemm_cdna.fits("dp128x224", 256, 28672, 4096)
    assert not gemm_cdna.fits("dp128x224", 256, 24576, 4096)


def test_the_wave_split_cuts_at_whole_waves():
    """2048 x 12288: eight block-rows, 32 blocks a row to a wave of 256 CUs,
    so 8192 columns on the pingpong and 4096 apart; a shape whose N is less
    than a wave has no split."""
    assert gemm_cdna.wave_cut(2048, 12288) == 8192
    assert gemm_cdna.fits("ppw", 2048, 12288, 4096)
    assert gemm_cdna.wave_cut(256, 151936) == 131072
    assert not gemm_cdna.fits("ppw", 2048, 4096, 4096)


def test_the_pingpong_tail_is_split_along_k():
    """N = 151936's last 128 columns: a split four-warp kernel by M, the
    pingpong's own 128-wide block only where no split takes K."""
    assert gemm_cdna.tail_for(256, 4096) == "dp64x64k8"
    assert gemm_cdna.tail_for(2048, 4096) == "dp64x64k4"
    assert gemm_cdna.tail_for(4096, 4096) == "dp64k4"
    assert gemm_cdna.tail_for(8192, 4096) == "dp64x64"
    assert gemm_cdna.tail_for(300, 1152) == "dp64x64"


def test_the_split_reduce_is_on_strips():
    """Each share a 16-byte fp32 run a lane, rounded once into bf16; a grid
    row a row of the output."""
    from hk.ops import gemm_cdna_dp

    src = scaffold.bare(gemm_cdna_dp.strip_reduce_kernel("sr_t", 8, arch="gfx950").trace())
    assert re.search(r"rs_fp32_1x512_r4 \w+;", src)
    assert len(re.findall(r"kittens::load\(\w+, g\.ws, \w+\);", src)) == 8
    assert len(re.findall(r"kittens::store\(g\.c, \w+, \w+\);", src)) == 1
    assert "bid_y" in src


# -- stream-K -----------------------------------------------------------------


def _plan_segments(m, n, k, bm, bn, mode, grid=256):
    flat, slots = gemm_cdna.sk_plan(m, n, k, bm, bn, 64, grid, mode, 8)
    segs = []
    for g in range(grid):
        for r in range(flat[g], flat[g + 1], gemm_cdna.SK_REC):
            segs.append((g, r, flat[r:r + gemm_cdna.SK_REC]))
    return flat, slots, segs


@pytest.mark.parametrize("m,n,k", [(4096, 1536, 4096), (1024, 5120, 7168), (512, 7680, 5120),
                                   (128, 17920, 5120), (300, 3072, 2048), (64, 1024, 4096),
                                   (4096, 5120, 17920)])
@pytest.mark.parametrize("bm,bn", [(256, 256), (192, 256), (128, 256)])
@pytest.mark.parametrize("mode", [1, 2])
def test_a_stream_k_plan_covers_every_k_pair_once(m, n, k, bm, bn, mode):
    """Every (block, K-pair) unit in exactly one segment; a cut block's
    workgroups on one XCD (their shares meet in its L2), in consecutive
    slots, each naming the same count; a whole block has no slot; and each
    workgroup's first and last cut segments are the ones its fix-up fields
    name."""
    flat, slots, segs = _plan_segments(m, n, k, bm, bn, mode)
    per = k // 128
    seen, sharers = set(), {}
    for g, r, (m0, n0, k0, kt, slot, t, nsh, first, bit) in segs:
        for u in range(k0 // 128, k0 // 128 + kt // 2):
            assert (m0, n0, u) not in seen
            seen.add((m0, n0, u))
        if slot < 0:
            assert (k0, kt, nsh, bit) == (0, 2 * per, 1, 0)
        else:
            sharers.setdefault(t, []).append((g, slot, nsh, first))
            assert flat[256 + 1 + (bit - 1) * 256 + g] == r
    assert len(seen) == -(-m // bm) * (n // bn) * per
    assert sum(len(s) for s in sharers.values()) == slots
    for sh in sharers.values():
        assert len({g % 8 for g, *_ in sh}) == 1
        assert sorted(s for _, s, _, _ in sh) == list(range(sh[0][3], sh[0][3] + len(sh)))
        assert all(c == len(sh) for _, _, c, _ in sh)


def test_whole_waves_are_not_cut():
    """Mode 2 deals the grid's whole waves of blocks whole and cuts only the
    rest; mode 1 cuts every block. 16 x 20 blocks of 256 x 256 on 256
    workgroups: one whole wave, 64 blocks left."""
    _, slots1, segs1 = _plan_segments(4096, 5120, 4096, 256, 256, 1)
    _, slots2, segs2 = _plan_segments(4096, 5120, 4096, 256, 256, 2)
    assert sum(1 for *_, f in segs2 if f[4] < 0) == 256
    assert all(f[4] >= 0 for *_, f in segs1 if f[3] < 2 * 32) and slots2 < slots1


def test_a_stream_k_kernel_sums_shares_on_its_xcd():
    """The persistent grid is sk_grid workgroups; the shares meet through
    wg_arrive's XCD form (no device-wide fence) and are read back transposed."""
    src = scaffold.bare(_gemm("g_sk1", stream_k=1).trace())
    assert "wg_arrive_xcd(" in src and "__threadfence" not in src
    assert "return dim3(256, 1, 1);" in src
