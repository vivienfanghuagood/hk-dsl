"""The tracing context.

Tracing here means: run the user's Python once, and record the tile ops it
performs. There is no AST rewriting (FlyDSL does rewrite, to capture Python
`if`/`for`). We do not, because in a tile kernel the loops that matter are
either fully unrolled at compile time or are the one pipelined loop, and both
read better when written explicitly -- `hk.range(..., prefetch=1)` says what it
means, where a rewritten `for` hides whether the trip count is static.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence

from .nodes import KernelIR, Op, Type, Value, capture_loc

_tls = threading.local()


class Builder:
    def __init__(self, kernel: KernelIR):
        self.kernel = kernel
        #: Ops are appended to the innermost open region, so a loop body
        #: collects its own ops instead of leaking into the top level.
        self._regions: List[List[Op]] = [kernel.body]
        #: Integer literals, memoised. A tile coordinate built from literals
        #: would otherwise emit one `const` op per component per use, and the
        #: generated C++ fills up with `const int c1 = 0; const int c2 = 0;`
        #: that says nothing. Only safe for immutable scalars, which is why it
        #: is keyed on the literal and not on anything traced.
        self._consts: Dict[Any, Value] = {}

    # -- region management -------------------------------------------------

    def push_region(self) -> List[Op]:
        region: List[Op] = []
        self._regions.append(region)
        return region

    def pop_region(self) -> List[Op]:
        if len(self._regions) == 1:
            raise RuntimeError("cannot pop the kernel's top-level region")
        return self._regions.pop()

    @property
    def region(self) -> List[Op]:
        return self._regions[-1]

    # -- emission ----------------------------------------------------------

    def emit(
        self,
        opcode: str,
        operands: Sequence[Value] = (),
        result_type: Optional[Type] = None,
        *,
        name: str = "",
        **attrs: Any,
    ) -> Optional[Value]:
        results = []
        if result_type is not None:
            results.append(Value(result_type, name=name))
        op = Op(opcode, list(operands), results, dict(attrs), loc=capture_loc())
        for r in results:
            r.producer = op
        self.region.append(op)
        return results[0] if results else None

    def emit_into(
        self,
        opcode: str,
        dst: Value,
        operands: Sequence[Value] = (),
        **attrs: Any,
    ) -> Value:
        """Emit an op that writes an existing value instead of making one.

        This is what `out=` compiles to, and what a loop-carried accumulator
        needs: `kittens::row_sum(acc, src, acc)` updates a register declared
        outside the loop. The destination is operands[0] and `inplace` says so,
        so a reader does not have to know each opcode's convention.
        """
        op = Op(opcode, [dst, *operands], [], {**attrs, "inplace": True},
                loc=capture_loc())
        self.region.append(op)
        return dst

    def emit_region(
        self,
        opcode: str,
        operands: Sequence[Value] = (),
        block_args: Sequence[Value] = (),
        **attrs: Any,
    ) -> Op:
        """Emit an op with a nested region and open it for writing. The caller
        is responsible for closing it -- see lang.control.range, which does so
        in a finally, because a traced body that raises must not leave the
        builder pointing into a region nobody will emit."""
        op = Op(opcode, list(operands), [], dict(attrs), loc=capture_loc(),
                body=[], block_args=list(block_args))
        self.region.append(op)
        self._regions.append(op.body)
        return op

    def emit_op(self, op: Op) -> Op:
        self.region.append(op)
        return op

    def constant(self, value: Any, result_type: Type) -> Value:
        """A scalar literal, emitted at most once per kernel.

        Memoised at the top-level region only. A constant hoisted out of a loop
        body is still correct -- it is a literal -- but the op has to live
        somewhere a later use can see it, and the top level always can.
        """
        return self.hoisted(
            ("const", type(value).__name__, value, repr(result_type)),
            "const",
            result_type,
            name="c",
            value=value,
        )

    def hoisted(
        self,
        key: Any,
        opcode: str,
        result_type: Type,
        *,
        name: str = "",
        **attrs: Any,
    ) -> Value:
        """An operand-free op whose value is the same everywhere in the kernel
        -- a literal, `blockIdx.y`, `warpid()` -- emitted once at the top.

        Not an optimisation so much as a readability rule: without it a loop
        body that mentions `hk.block_idx.y` three times generates three
        identical `const int` lines, and the generated C++ is supposed to be
        the artifact you take to the disassembler.
        """
        if key not in self._consts:
            op = Op(opcode, [], [Value(result_type, name=name)], dict(attrs),
                    loc=capture_loc())
            op.results[0].producer = op
            self.kernel.body.insert(0, op)
            self._consts[key] = op.results[0]
        return self._consts[key]


def current() -> Builder:
    b = getattr(_tls, "builder", None)
    if b is None:
        raise RuntimeError(
            "no kernel is being traced. hk tile ops are only valid inside a "
            "function decorated with @hk.kernel."
        )
    return b


def current_or_none() -> Optional[Builder]:
    return getattr(_tls, "builder", None)


class tracing:
    """`with tracing(builder):` -- makes `builder` the target of hk ops."""

    def __init__(self, builder: Builder):
        self.builder = builder
        self._prev = None

    def __enter__(self) -> Builder:
        self._prev = getattr(_tls, "builder", None)
        _tls.builder = self.builder
        return self.builder

    def __exit__(self, *exc):
        _tls.builder = self._prev
        return False
