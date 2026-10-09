"""Control flow.

Only one construct, and it is a runtime loop:

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


__all__ = ["range"]
