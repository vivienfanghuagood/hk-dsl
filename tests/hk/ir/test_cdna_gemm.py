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
    # The prologue's two waits, the loop's two, the epilogue's two.
    assert _waits(src) == [4, 4, 6, 4, 2, 0]
    # Eight math phases a loop iteration, six in the epilogue.
    assert src.count("__builtin_amdgcn_s_setprio(1);") == 8 + 6
    assert "if (eq) __builtin_amdgcn_s_barrier();" in src
    assert "__launch_bounds__(512, 1)" in src
    assert "LDS    : 131072 B" in src


def test_hipkittens_own_waits_are_refused():
    """HipKittens waits for all but three copies at the end of each iteration
    (vmcnt(6)). The copy that leaves outstanding is the one into Bs[1][0] --
    hk's Bs[2] -- and the next iteration's fourth memory phase reads it before
    its own wait: right by timing, not by any wait. hk retires it one phase
    earlier, at vmcnt(4)."""
    with pytest.raises(VerifyError, match=r"reads Bs\[2\] while this warp's copy into it is in flight"):
        _gemm("g_t2", hipkittens_waits=True).trace()


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
