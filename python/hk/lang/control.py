"""Control flow.

Two constructs. A runtime loop:

    for kb in hk.range(nblocks):
        ...

Python's own `for` still works and still unrolls at trace time -- that is the
right thing when the trip count is a constexpr and small. `hk.range` is for the
case where it is neither: a normalization over a 4096-wide hidden dimension is
64 column blocks, and unrolling 64 copies of the body is a kernel that takes
half a minute to compile and blows the instruction cache for nothing.

The body is traced **once**, so the Python inside it runs once. Anything you
accumulate with `+=` on a Python variable will be wrong; accumulate into a
traced value with `out=` instead:

    acc = hk.zeros_vec(vt)
    for cb in hk.range(n):
        hk.row_sum(hk.load(...), out=acc, accumulate=True)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

from ..ir import builder
from ..ir.nodes import ScalarType, Value, i32


def range(bound: Union[int, Value], *, unroll: int = 1) -> Iterator[Value]:
    """A loop from 0 to `bound`, yielding the induction variable once.

    `unroll` becomes a `#pragma unroll` on the generated loop. Leave it at 1
    unless you have measured: unrolling a loop whose body already fills the
    register file is how a kernel starts to spill, and a spill here is silent
    corruption rather than slowness.
    """
    from .ops import _as_value

    b = builder.current()
    n = _as_value(bound, i32)
    iv = Value(ScalarType(i32), name="i")
    b.emit_region("for", [n], [iv], unroll=unroll)
    try:
        yield iv
    finally:
        b.pop_region()


@contextmanager
def if_(cond: Value, *, opaque: bool = False) -> Iterator[None]:
    """A region that runs only where `cond` is nonzero.

        with hk.if_(hk.s_lt(c, CHUNKS)):
            ...

    `cond` must come from a comparison (`s_lt`, `s_ge`, ...), not be a bare
    index. `if_(c)` reads like "if c is in range" and means "if c is not zero",
    which on the c == 0 lane is the opposite; requiring the comparison makes
    that a trace-time error instead of a wrong answer on one warp.

    Divergence is allowed -- this lowers to a plain C++ `if`, and the compiler
    will use exec-mask predication where the region is small. What is *not*
    allowed is a barrier or a group staging call inside: those are workgroup
    operations and a warp that skips one deadlocks the rest. The verifier
    checks for it.

    `opaque=True` makes the condition opaque to the optimizer
    (`asm volatile("" : "+v"(cond))`) so the branch survives even when the
    condition folds to a constant. That is not a micro-optimization: a
    straight-line region containing both a kernel's staging and its math can
    hold two live sets at once and spill, and a never-taken branch between them
    costs one VGPR and an s_cbranch and drops scratch to zero. See
    `hk.sched_barrier` for the lighter tool that does not branch at all.
    """
    from .ops import PREDICATES

    # `s_any_ne` is a comparison too, just one whose operands are register
    # vectors and whose answer is wave-level. It exists to be branched on.
    conditions = PREDICATES + ("any_ne",)

    if not isinstance(cond, Value) or not isinstance(cond.type, ScalarType):
        raise TypeError(
            f"if_ condition is {cond!r}, expected a scalar Value from a "
            f"comparison such as hk.s_lt(a, b)"
        )
    producer = cond.producer
    name = producer.opcode[2:] if producer is not None else None
    if name not in conditions:
        raise TypeError(
            f"if_ condition is {cond.name}, which is not a comparison. Write "
            f"the test you mean -- hk.s_gt({cond.name}, 0) -- rather than "
            f"relying on nonzero: for an index, the two differ exactly at 0."
        )

    b = builder.current()
    b.emit_region("if", [cond], [], opaque=bool(opaque))
    try:
        yield
    finally:
        b.pop_region()


@contextmanager
def scope() -> Iterator[None]:
    """An always-taken region whose only job is to be a boundary.

        with hk.scope():
            ...the math...

    It costs one VGPR and a never-taken `s_cbranch`, and it buys a register
    allocator that does not have to hold two live sets at once. The attention
    kernel needs it: with its staging and its math in a single straight-line
    iteration the allocator gives up at 256 VGPRs and ~480 bytes/lane of
    scratch, and on a kernel that hand-manages `s_waitcnt` scratch is a wrong
    answer rather than a slow one. Inside the region the same code fits in 249
    with none.

    `hk.sched_barrier` is the cheaper tool and should be tried first: it bounds
    the scheduler without branching. This one additionally bounds the *live
    ranges*, which is what the allocator is actually tripping over.

    The same restriction as `if_`: nothing workgroup-wide inside. The region is
    always taken, so that is conservative -- but "always" is a property of the
    opaque condition the optimizer cannot see, and a rule that holds only
    because of what the optimizer does not know is not a rule.
    """
    from .ops import _as_value

    b = builder.current()
    b.emit_region("if", [_as_value(1, i32)], [], opaque=True)
    try:
        yield
    finally:
        b.pop_region()


__all__ = ["range", "if_", "scope"]
