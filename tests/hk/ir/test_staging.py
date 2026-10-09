"""The staging layer: multi-buffer LDS, subtiles, and the register pipeline.

gfx11 has no global->LDS DMA. A prefetch is therefore two ops with the bytes
parked in VGPRs in between -- `stage` fills a register buffer from global,
`commit` drains it into shared -- and that register buffer is a live range
somebody has to size. These tests are about the contracts that forces: how big
the buffer is, which thread count it was sized for, and the lgkmcnt accounting,
which once there are interleaved stores has to count ds_writes and not just
ds_reads.

The last point is the one worth the file. The handwritten GEMM computes that
term by hand, under the name TAIL_OPS, with a paragraph explaining that an
lgkmcnt immediate which is too *large* does not over-wait -- it fails to wait
at all, and the kernel is then wrong only sometimes.
"""

import pytest

import hk
from hk import bf16, fp32
from hk.ir.nodes import StageBufferType
from hk.ir.verify import VerifyError
from hk.lang import ops


def _trace(body, **kw):
    k = hk.kernel(body, arch="gfx1100", warps=8,
                  grid=lambda p: (1, 1, 1), name=body.__name__)
    return k.trace(**kw)


def _src(body, **kw):
    k = hk.kernel(body, arch="gfx1100", warps=8,
                  grid=lambda p: (1, 1, 1), name=body.__name__)
    return k.source(**kw)


def _waits(ir):
    out = []

    def walk(ops_):
        for op in ops_:
            if op.body is not None:
                walk(op.body)
            elif op.opcode == "lds_wait_for":
                out.append(op)

    walk(ir.body)
    return out


# -- how big is the buffer ---------------------------------------------------


def test_stage_calls_is_the_cpp_expression_transcribed():
    """ceil(elements / float4 / threads), and it has to agree exactly.

    256 threads staging a 128x32 bf16 tile: 4096 elements, 8 to a float4, 512
    float4s, 2 apiece. This is not an estimate that can be rounded up for
    safety in one direction only -- a buffer one float4 short is an
    out-of-bounds write to the stack.
    """
    assert StageBufferType(ops.st(bf16, 128, 32), 256).calls == 2
    assert StageBufferType(ops.st(bf16, 128, 32), 64).calls == 8
    # fp32 packs half as many elements per float4, so twice the calls.
    assert StageBufferType(ops.st(fp32, 128, 32), 256).calls == 4
    # And it rounds up: three warps cannot divide 512 float4s evenly.
    assert StageBufferType(ops.st(bf16, 128, 32), 96).calls == 6


def test_the_buffer_reaches_the_source_at_that_size():
    def staged(a: hk.GL[bf16]):
        g = hk.group(8)
        As = ops.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        g.stage_buffer(As, name="buf")

    assert "float4 buf[2];" in _src(staged)


def test_depth_is_a_second_dimension_not_a_bigger_one():
    def staged(a: hk.GL[bf16]):
        g = hk.group(8)
        As = ops.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        g.stage_buffer(As, depth=2, name="buf")

    assert "float4 buf[2][2];" in _src(staged)


# -- multi-buffer LDS --------------------------------------------------------


def test_a_stack_is_one_allocation_of_n_tiles():
    """count=2 doubles the bytes and keeps a single offset, which is what makes
    the XOR swap legal: the two buffers have to differ in one address bit."""
    def two(a: hk.GL[bf16]):
        ops.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")

    ir = _trace(two)
    assert ir.lds_bytes == 2 * 128 * 32 * 2


def test_indexing_a_lone_tile_is_an_error():
    def lone(a: hk.GL[bf16]):
        ops.shared_at(ops.alloc_shared(ops.st(bf16, 64, 64)), 0)

    with pytest.raises(TypeError):
        _trace(lone)


def test_constant_index_is_bounds_checked():
    def oob(a: hk.GL[bf16]):
        ops.shared_at(ops.alloc_shared(ops.st(bf16, 64, 64), count=2), 2)

    with pytest.raises(IndexError):
        _trace(oob)


def test_a_stack_cannot_be_used_where_a_tile_is_meant():
    """The failure this prevents is silent: an un-indexed stack decays to its
    first tile in C++, so half the double buffering would just not happen."""
    def forgot(a: hk.GL[bf16]):
        g = hk.group(8)
        As = ops.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = g.stage_buffer(As, name="buf")
        g.commit(As, buf)

    with pytest.raises(TypeError):
        _trace(forgot)


def test_a_buffer_sized_for_another_group_is_refused():
    """Same silent class: the count is baked into the buffer, and handing it to
    a group of a different size overruns or underfills it with no diagnostic."""
    def mixed(a: hk.GL[bf16]):
        As = ops.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = hk.group(8).stage_buffer(As, name="buf")
        hk.group(4).commit(ops.shared_at(As, 0), buf)

    with pytest.raises(TypeError, match="threads"):
        _trace(mixed)


def test_vm_wait_fits_in_the_vmcnt_field():
    def over(a: hk.GL[bf16]):
        ops.vm_wait(64)

    with pytest.raises(TypeError, match="6 bits"):
        _trace(over)

    def ok(a: hk.GL[bf16]):
        ops.vm_wait(63)

    assert "kittens::vm_wait<63>();" in _src(ok)


# -- subtiles ----------------------------------------------------------------


def test_a_subtile_must_tile_its_parent():
    """subtile_inplace indexes in units of the subtile, so a window that does
    not divide the parent has no index to be given."""
    def ragged(a: hk.GL[bf16]):
        ops.subtile(ops.alloc_shared(ops.st(bf16, 128, 32)), 48, 16)

    with pytest.raises(ValueError, match="does not divide"):
        _trace(ragged)


def test_a_subtile_narrows_the_type():
    def win(a: hk.GL[bf16]):
        s = ops.alloc_shared(ops.st(bf16, 128, 32))
        v = ops.subtile(s, 32, 16, 1, 0)
        assert (v.type.rows, v.type.cols) == (32, 16)

    _trace(win)


def test_a_subtile_binds_by_value():
    """`subtile_inplace` returns a prvalue st_subtile -- a pointer plus the
    parent's swizzle constants -- so `auto &` does not compile. The register
    overload of the same name does return a reference, which is exactly the
    trap."""
    def win(a: hk.GL[bf16]):
        s = ops.alloc_shared(ops.st(bf16, 128, 32))
        ops.subtile(s, 32, 16, 1, 0)

    line = [ln.strip() for ln in _src(win).splitlines() if "subtile_inplace" in ln]
    assert len(line) == 1, line
    assert line[0].startswith("auto sub"), line


# -- lgkmcnt with stores in the mix ------------------------------------------


def _pipelined(write_pos: int):
    """A K-loop with the register-buffer drain at one of two positions.

    This is the GEMM's shape in miniature: read operand slices out of the
    resident LDS buffer and multiply, and somewhere among them drain the
    staging registers into the *other* LDS buffer. `write_pos` is where that
    lands -- the C++ has the same knob, for the same reason, because moving it
    changes how much math the ds_writes get to hide behind.
    """
    def body(a: hk.GL[bf16]):
        g = hk.group(8)
        As = hk.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = g.stage_buffer(As, name="buf")
        acc = ops.zeros(ops.rt(fp32, 32, 16, "col"))
        rt_a = ops.rt(bf16, 32, 16, "row")
        rt_b = ops.rt(bf16, 16, 16, "row")
        for i in hk.range(ops.rows(a)):
            s = ops.shared_at(As, i % 2)
            other = ops.shared_at(As, 1 - i % 2)

            def drain(at):
                if at == write_pos:
                    ops.vm_wait(0)
                    g.commit(other, buf)
                    g.stage(buf, a, ops.elem_coord(0, 0, 0, 0))

            t = ops.load_shared(ops.subtile(s, 32, 16, 0, 0), rt_a, wait=False)
            b0 = ops.load_shared(ops.subtile(s, 16, 16, 0, 1), rt_b, wait=False)
            drain(0)
            ops.lds_wait_for(t, b0)
            ops.mma_ABt(t, b0, acc, out=acc)
            drain(1)
            ops.barrier(drain=True)

    body.__name__ = f"pipelined{write_pos}"
    return _trace(body)


def test_an_interleaved_write_counts_toward_the_wait():
    """The point of the whole exercise, and the thing the C++ calls TAIL_OPS.

    lgkmcnt is one counter for both directions, retiring in issue order. With
    the drain sitting between the operand reads and the wait that retires
    them, the two ds_writes are younger than every read being waited for -- so
    the immediate has to be 2, not 0. 0 would be correct-looking and merely
    slower; the dangerous direction is the other one, because an lgkmcnt that
    exceeds what is outstanding does not over-wait, it does not wait at all.

    This is also the position that pays: the writes now have a whole chunk of
    WMMAs to complete behind, so the lgkmcnt(0) at the barrier is nearly free.
    """
    assert [w.attrs["n"] for w in _waits(_pipelined(write_pos=0))] == [2]


def test_a_write_after_the_last_wait_costs_a_full_drain():
    """The other position. Nothing is left to wait for when the store issues,
    so the derived immediate is 0 and the writes have nothing to hide behind
    -- the barrier's lgkmcnt(0) eats the whole latency. Same answer, worse
    schedule, and the IR says which is which without a benchmark."""
    assert [w.attrs["n"] for w in _waits(_pipelined(write_pos=1))] == [0]


def test_the_loop_reaches_a_steady_state():
    """Both positions have to. The pass walks the body twice and insists the
    second walk derives the same immediates as the first, because a loop whose
    queue grows every iteration has no correct immediate at all -- and the
    first iteration, whose queue is short, is exactly the one that would
    produce a plausible wrong answer."""
    for pos in (0, 1):
        _pipelined(pos)


# -- s_barrier is not a memory fence -----------------------------------------


def test_a_barrier_with_writes_in_flight_is_refused():
    """s_barrier orders execution, not memory. Past it the other warps read
    the buffer these writes are still filling, and nothing waits for them.

    The handwritten GEMM spends an explicit `lds_wait<0>()` here and explains
    why in a comment; a comment is not a mechanism, and the two of its three
    WRITE_POS settings that happen not to need one are safe by accident of
    where the last wait falls.
    """
    def racy(a: hk.GL[bf16]):
        g = hk.group(8)
        As = hk.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = g.stage_buffer(As, name="buf")
        g.commit(ops.shared_at(As, 1), buf)
        ops.barrier()

    with pytest.raises(VerifyError, match="ds_writes still in flight"):
        _trace(racy)


def test_a_barrier_with_reads_in_flight_is_refused_too():
    """The mirror hazard, and the one that is easier to talk yourself out of:
    the reads have landed *somewhere*, just not yet in registers, and the
    buffer they came from is about to be overwritten by another warp."""
    def racy(a: hk.GL[bf16]):
        s = ops.alloc_shared(ops.st(bf16, 128, 32))
        ops.load_shared(ops.subtile(s, 32, 16, 0, 0),
                        ops.rt(bf16, 32, 16, "row"), wait=False)
        ops.barrier()

    with pytest.raises(VerifyError, match="ds_reads still in flight"):
        _trace(racy)


def test_draining_costs_an_lgkmcnt_zero_in_the_source():
    """And it is a separate instruction before the barrier, not a flag on it:
    s_barrier has no memory semantics to attach one to."""
    def drained(a: hk.GL[bf16]):
        g = hk.group(8)
        As = hk.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = g.stage_buffer(As, name="buf")
        g.commit(ops.shared_at(As, 1), buf)
        ops.barrier(drain=True)

    src = _src(drained)
    i = src.index("kittens::lds_wait<0>();")
    assert src.index("__builtin_amdgcn_s_barrier();", i) > i
    # And a plain barrier does not pay for it.
    def plain(a: hk.GL[bf16]):
        ops.barrier()

    assert "lds_wait<0>" not in _src(plain)


def test_the_barrier_is_the_builtin_not_syncthreads():
    """`__syncthreads()` orders global memory as well, and on gfx11 pays for
    that with a `buffer_gl0_inv` -- a per-WGP vector L0 flush. In the GEMM that
    is one flush per K-tile and it was the whole measured gap against the
    handwritten kernel. This op only ever promised execution ordering."""
    def plain(a: hk.GL[bf16]):
        ops.barrier()

    src = _src(plain)
    assert "__builtin_amdgcn_s_barrier();" in src
    assert "__syncthreads" not in src


def test_a_wait_the_math_needed_anyway_is_enough():
    """The cheap fix the error message points at: no drain, because the wait
    that feeds the next WMMA already retires the writes."""
    def clean(a: hk.GL[bf16]):
        g = hk.group(8)
        As = hk.alloc_shared(ops.st(bf16, 128, 32), count=2, name="As")
        buf = g.stage_buffer(As, name="buf")
        acc = ops.zeros(ops.rt(fp32, 32, 16, "col"))
        g.commit(ops.shared_at(As, 1), buf)
        t = ops.load_shared(ops.subtile(ops.shared_at(As, 0), 32, 16, 0, 0),
                            ops.rt(bf16, 32, 16, "row"), wait=False)
        b0 = ops.load_shared(ops.subtile(ops.shared_at(As, 0), 16, 16, 0, 1),
                             ops.rt(bf16, 16, 16, "row"), wait=False)
        ops.lds_wait_for(t, b0)
        ops.mma_ABt(t, b0, acc, out=acc)
        ops.barrier()

    assert "lds_wait<0>" not in _src(clean)


def test_rebinding_an_already_retired_tile_is_free():
    """The rotated schedule names both operands of every chunk, including one a
    previous chunk already retired. Rejecting that would force the author to
    track which half of an operand pair is still moving, which is precisely the
    bookkeeping the pass exists to take over.
    """
    def rebind(a: hk.GL[bf16]):
        s = ops.alloc_shared(ops.st(bf16, 128, 32))
        acc = ops.zeros(ops.rt(fp32, 32, 16, "col"))
        rt_a = ops.rt(bf16, 32, 16, "row")
        rt_b = ops.rt(bf16, 16, 16, "row")
        t = ops.load_shared(ops.subtile(s, 32, 16, 0, 0), rt_a, wait=False)
        b0 = ops.load_shared(ops.subtile(s, 16, 16, 0, 1), rt_b, wait=False)
        ops.lds_wait_for(t, b0)
        ops.mma_ABt(t, b0, acc, out=acc)
        ops.lds_wait_for(t, b0)       # pure re-bind: waits for nothing
        ops.mma_ABt(t, b0, acc, out=acc)

    assert [w.attrs["n"] for w in _waits(_trace(rebind))] == [0, 0]


def test_naming_a_tile_that_never_moved_is_still_an_error():
    """The relaxation above is narrow on purpose: it forgives a tile whose
    reads this kernel issued and retired, not one that was never loaded."""
    def never(a: hk.GL[bf16]):
        ops.lds_wait_for(ops.zeros(ops.rt(bf16, 32, 16, "row")))

    with pytest.raises(VerifyError):
        _trace(never)
