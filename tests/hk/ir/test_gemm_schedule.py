"""hk.ops.gemm's schedule options: s_setprio off by default and on in the thin
schedule, the sched_group_barrier interleave, and A streamed one fragment at
a time. No GPU, no hipcc."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.ops import gemm  # noqa: E402


def _src(**sched):
    return gemm.TUNER._kernel_for({**gemm.DEFAULT, **sched}).source()


def test_the_default_drops_setprio_and_the_thin_schedule_keeps_it():
    """s_setprio ends a scheduling region: without it the scheduler interleaves
    the WMMAs with the LDS reads and sinks a K-tile's last WMMAs under the
    next tile's writes, 3.5-4.3% at the scoreboard's shapes. At M <= 16 it
    measured within 1.2% either way and stays."""
    assert "s_setprio" not in _src()
    assert "__builtin_amdgcn_s_setprio(1);" in _src(setprio=True)
    assert "__builtin_amdgcn_s_setprio(1);" in gemm.TUNER._kernel_for(gemm.THIN).source()
    assert gemm.schedule_for(16, 4096, 4096)["setprio"] is True


def test_a_kernel_name_says_when_it_keeps_setprio():
    assert gemm._sched_name(gemm.DEFAULT).endswith("_l1")
    assert gemm._sched_name({**gemm.DEFAULT, "setprio": True}).endswith("_sp")
    assert gemm._sched_name({**gemm.DEFAULT, "sched": 2, "stream_a": True}).endswith("_a1_i2")


def test_the_interleave_asks_for_one_staging_load_then_n_wmmas():
    src = _src(sched=2)
    k = gemm.TUNER._kernel_for({**gemm.DEFAULT, "sched": 2})
    calls = k.trace().module.operation.get_asm().count("hk.sched_group_barrier 32, 1, 0")
    assert calls > 0
    assert src.count("__builtin_amdgcn_sched_group_barrier(32, 1, 0);") == calls
    assert src.count("__builtin_amdgcn_sched_group_barrier(8, 2, 0);") == calls


def test_a_streamed_a_holds_one_accumulator_a_fragment_row():
    """FlyDSL's tile in hk's schedule: four warps of 64x64, A two fragments
    live rather than four."""
    src = _src(k_step=32, warps=4, warp_rows=2, n_split=4, lds_buffers=2, stream_a=True)
    assert "rt_bf16_16x16_row" in src and "rt_bf16_64x16_row" not in src
    assert src.count("kittens::store_bounded(g.c,") == 16


def test_stream_a_needs_two_fragments():
    with pytest.raises(ValueError, match="at least two"):
        gemm.gemm_kernel("g", block_m=64, warps=4, warp_rows=4, stream_a=True)
