"""`while` in a kernel: a test on Python values is a loop at trace time, a
test on device values is an scf.while the kernel runs."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import VerifyError, bf16, fp32  # noqa: E402

T_IN = hk.RT(bf16, 16, 16, "row")
T_ACC = hk.RT(fp32, 16, 16, "row")


def block_sum(step: int = 16):
    """o = the sum of x's 16x16 column blocks, walked with a runtime while."""

    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        acc = hk.zeros(T_ACC)
        k = 0
        while k < x.cols:
            acc += x[hk.elem_coord(0, 0, 0, k)].load(T_IN).to(fp32)
            k += step
        o[hk.elem_coord(0, 0, 0, 0)] = acc

    return hk.kernel(body, arch="gfx1100", warps=1, name="while_block_sum",
                     grid=lambda p: (1, 1, 1))


def test_a_device_test_is_a_loop_the_kernel_runs():
    src = block_sum().source()
    loop = src[src.index("while (true) {"):]
    # The index is a local declared in front of the loop, the test is the
    # loop's first statement and the step is written back at the end.
    assert "int k = " in src[:src.index("while (true) {")]
    assert loop.index("break;") < loop.index("kittens::load(")
    assert "k = " in loop[loop.index("break;"):]


def test_a_python_test_unrolls():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        acc = hk.zeros(T_ACC)
        i = 0
        while i < 3:
            acc += x[hk.elem_coord(0, 0, 0, 16 * i)].load(T_IN).to(fp32)
            i += 1
        o[hk.elem_coord(0, 0, 0, 0)] = acc

    k = hk.kernel(body, arch="gfx1100", warps=1, name="while_unrolled",
                  grid=lambda p: (1, 1, 1))
    src = k.source()
    assert "while (" not in src
    assert k.trace().dump().count("hk.load ") == 3


def test_a_barrier_in_a_runtime_while_is_refused():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        k = 0
        while k < x.cols:
            hk.arch.barrier()
            k += 16

    k = hk.kernel(body, arch="gfx1100", warps=2, name="while_barrier",
                  grid=lambda p: (1, 1, 1))
    with pytest.raises(VerifyError, match="inside a runtime while loop"):
        k.trace()


def test_a_barrier_in_a_uniform_while_traces():
    """A work queue's loop: the claimed index is the same in every warp, so the loop declared
    hk.uniform may hold barriers -- and the loop is marked so the verifier knows."""
    def body(q: hk.Tensor[hk.i32, "device"], o: hk.Tensor[fp32]):
        t = hk.comm.claim(q.at[0])
        while hk.uniform(t < 64):
            hk.arch.barrier()
            t = hk.comm.claim(q.at[0])

    k = hk.kernel(body, arch="gfx950", warps=2, name="while_uniform_barrier",
                  grid=lambda p: (1, 1, 1))
    ir = k.trace().dump()
    assert "scf.while" in ir and "hk.uniform" in ir
    assert "while (" in k.source() or "for (" in k.source()


def test_rebinding_a_tile_in_a_runtime_while_is_refused():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        acc = hk.zeros(T_ACC)
        k = 0
        while k < x.cols:
            acc = acc + x[hk.elem_coord(0, 0, 0, k)].load(T_IN).to(fp32)
            k += 16
        o[hk.elem_coord(0, 0, 0, 0)] = acc

    k = hk.kernel(body, arch="gfx1100", warps=1, name="while_rebind",
                  grid=lambda p: (1, 1, 1))
    with pytest.raises(TypeError, match="rebound inside a runtime while loop"):
        k.trace()


def test_the_loop_value_is_usable_after_it():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        k = 0
        while k < x.cols:
            k += 16
        o[hk.elem_coord(0, 0, 0, k - 16)] = hk.zeros(T_ACC)

    src = hk.kernel(body, arch="gfx1100", warps=1, name="while_after",
                    grid=lambda p: (1, 1, 1)).source()
    after = src[src.rindex("break;"):]
    assert "k - " in after or "(k, " in after or "k)" in after


@pytest.mark.hipcc
def test_it_compiles_without_spilling():
    b = block_sum().build()
    assert b.kernels
    for k in b.kernels:
        assert k.scratch == 0 and not k.spills, k.line()
