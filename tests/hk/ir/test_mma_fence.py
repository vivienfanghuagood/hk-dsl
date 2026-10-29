"""hk-mma-fence (CDNA): a scheduling fence between a matrix-core chain and the
first other access to its accumulator, as a DSL rule rather than a fence a
kernel remembers. On gfx950 LLVM threaded the softmax's reads of AGPR scores
through the two pages' MFMA chains of the paged decode and computed wrong
scores."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402

TA = hk.RT(hk.bf16, 32, 128, "row")
TS = hk.RT(hk.fp32, 32, 32, "col")
TS2 = hk.RT(hk.fp32, 64, 32, "col")


def _trace(body, arch="gfx950"):
    return str(hk.kernel(body, arch=arch, warps=1, grid=lambda p: (1, 1, 1)).trace().module)


def _fences(ir: str) -> int:
    m = re.search(r"hk.accum_fences = (\d+)", ir)
    return int(m.group(1)) if m else 0


def _before_first_read(ir: str) -> str:
    """The op right before the first hk.mul."""
    lines = [ln.strip() for ln in ir.splitlines()]
    i = next(i for i, ln in enumerate(lines) if ln.startswith("hk.mul "))
    return lines[i - 1]


def test_a_read_after_a_chain_gets_one_fence_and_the_chain_none():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        s += a @ a.T
        s += a @ a.T
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    ir = _trace(k)
    assert _fences(ir) == 1
    assert _before_first_read(ir) == "hk.sched_barrier"
    between = ir[ir.index("hk.mma_ABt"):ir.rindex("hk.mma_ABt")]
    assert "hk.sched_barrier" not in between


def test_a_fence_already_there_is_enough():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        s += a @ a.T
        hk.arch.sched_barrier()
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    assert _fences(_trace(k)) == 0


def test_a_partial_fence_is_not_one():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        s += a @ a.T
        hk.arch.sched_barrier("valu")
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    assert _fences(_trace(k)) == 1


def test_an_mma_into_a_subtile_marks_the_whole_tile():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS2)
        for p in range(2):
            sp = s.subtile(32, p)
            sp += a @ a.T
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    ir = _trace(k)
    assert _fences(ir) == 1
    assert _before_first_read(ir) == "hk.sched_barrier"


def test_an_unrelated_tile_is_not_fenced():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        t = hk.zeros(TS)
        s += a @ a.T
        t.__imul__(2.0)
        s += a @ a.T
        hk.arch.sched_barrier()
        o[hk.tile_coord(0, 0, 0, 0)].store(s)
        o[hk.tile_coord(0, 0, 0, 0)].store(t)

    assert _fences(_trace(k)) == 0


def test_the_end_of_a_loop_trip_reaches_the_next():
    def k(x: hk.Tensor[hk.bf16], n: hk.Tensor[hk.i32], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        trips = n[hk.elem_coord(0, 0, 0, 0)].load_scalar()
        s = hk.zeros(TS)
        hk.arch.sched_barrier()
        for _ in range(trips):
            s.__imul__(0.5)
            s += a @ a.T
        hk.arch.sched_barrier()
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    ir = _trace(k)
    assert _fences(ir) == 1
    assert _before_first_read(ir) == "hk.sched_barrier"


def test_a_kernel_can_opt_out_for_an_ab():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        hk.arch.mma_fence(False)
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        s += a @ a.T
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    assert _fences(_trace(k)) == 0
    def bad(x: hk.Tensor[hk.bf16]):
        hk.arch.mma_fence(0)

    with pytest.raises(TypeError, match="True or False"):
        _trace(bad)


def test_rdna_kernels_are_left_alone():
    from hk.ops import gemm

    ir = str(gemm.KERNELS["gemm_bf16_128x128"].trace().module)
    assert "hk.accum_fences" not in ir


def test_the_two_page_decode_fences_its_scores_without_a_hand_written_fence():
    from hk.ops import paged_cdna

    ir = str(paged_cdna.decode_kernel("mf_decode", splits=2, ppstep=2, arch="gfx950").trace().module)
    lines = [ln.strip() for ln in ir.splitlines()]
    reads = [i for i, ln in enumerate(lines) if ln.startswith("hk.mul ") and "scalar" in ln]
    assert reads and all(lines[i - 1] == "hk.sched_barrier" for i in reads)
    off = str(paged_cdna.decode_kernel("mf_decode_off", splits=2, ppstep=2, arch="gfx950",
                                       fence_qk=False).trace().module)
    assert _fences(off) == 0
