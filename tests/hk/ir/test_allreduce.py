"""hk.ops.allreduce without a GPU: the kernel's protocol and the plan's choices."""

from __future__ import annotations

from hk.ops import allreduce as ar


def test_the_kernel_moves_link_bytes_as_fp32_and_meets_twice():
    src = ar._kernel(32, 8).source()
    # phase 1 and the y copy: fp32 tiles through system-scoped views; the sum stored as bf16
    assert "kittens::store_scoped<17>(g.rs_peer_f, " in src
    assert "kittens::store_scoped<17>(g.y_peer_f, " in src and "kittens::store_scoped<17>(g.y_mine_b, " in src
    assert src.count("kittens::signal<17>(") == 2 and src.count("kittens::wait<17, 0>(") == 2
    # the epoch moves on device: one arrival on a device-scope counter
    assert "kittens::arrive<16>(g.ctl, " in src


def test_the_kernel_is_resident():
    assert ar._kernel(32, 8).resident


class _Fake:
    workgroups = 32
    _plan = ar.AllReduce._plan


def test_the_plan_takes_the_most_workgroups_that_divide():
    f = _Fake()
    assert f._plan(20480) == (32, 8)      # M = 8192 x 5120
    assert f._plan(640) == (4, 8)         # M = 128
    assert f._plan(16) == (1, 4)          # 4 warps x 4 tiles
    assert f._plan(10) is None            # not a multiple of 4 warps x 2 tiles
