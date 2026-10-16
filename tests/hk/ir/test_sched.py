"""hk.arch.sched_group_barrier and sched_barrier's mask: what they lower to
and what they refuse. No GPU, no hipcc."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402


def _src(body):
    body.__name__ = body.__name__.lstrip("_")
    k = hk.kernel(body, arch="gfx1100", warps=1, name=body.__name__,
                  grid=lambda p: (1, 1, 1))
    return k.trace().module.operation.get_asm(), k.source()


def test_an_interleave_lowers_to_the_builtin():
    def _interleave(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        for _ in range(2):
            hk.arch.sched_group_barrier("vmem_read", 1)
            hk.arch.sched_group_barrier(("mma",), 2, 1)
        hk.arch.sched_group_barrier(0x200, 3)

    ir, src = _src(_interleave)
    assert ir.count("hk.sched_group_barrier 32, 1, 0") == 2
    assert ir.count("hk.sched_group_barrier 8, 2, 1") == 2
    assert "__builtin_amdgcn_sched_group_barrier(32, 1, 0);" in src
    assert "__builtin_amdgcn_sched_group_barrier(8, 2, 1);" in src
    assert "__builtin_amdgcn_sched_group_barrier(512, 3, 0);" in src


def test_a_sched_barrier_says_what_may_cross():
    def _fences(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        hk.arch.sched_barrier()
        hk.arch.sched_barrier(("valu", "salu"))

    ir, src = _src(_fences)
    assert "hk.sched_barrier\n" in ir and "hk.sched_barrier {mask = 6 : i64}" in ir
    assert "__builtin_amdgcn_sched_barrier(0);" in src
    assert "__builtin_amdgcn_sched_barrier(6);" in src


@pytest.mark.parametrize("args,match", [
    (("vgpr", 1), "unknown instruction class"),
    (("mma", 0), "at least one"),
    ((0, 1), "at least one class"),
    ((0x800, 1), "outside"),
    (("mma", 1, -1), "group id"),
])
def test_a_bad_group_is_refused(args, match):
    def _bad(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        hk.arch.sched_group_barrier(*args)

    with pytest.raises(TypeError, match=match):
        _src(_bad)
