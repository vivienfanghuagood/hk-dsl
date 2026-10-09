"""Derive every `s_waitcnt lgkmcnt` count, and prove every async tile is bound.

This pass is the reason the IR exists.

`load_shared(wait=False)` issues its ds_reads through an inline-asm *output*
operand, so LLVM believes the destination is ready the instant the asm ends.
`s_waitcnt` is a separate volatile asm sharing no operands with it, and
`v_wmma` is a pure builtin with no memory effect. Nothing in the IR stops the
scheduler from hoisting a consuming WMMA above the wait, and it does --
nondeterministically, as soon as register pressure shifts. It is not a spill:
ScratchSize stays 0 while the answer changes from run to run. The fix is to
bind the tile's registers after the wait, which `kittens::lds_wait_for` does;
the problem is that a human writing it by hand forgets, and nothing tells them.

So two things are checked here, and they are errors, not warnings:

  * a tile whose reads are outstanding may not be read; and
  * a tile whose reads have been retired by somebody else's wait may not be
    read either, because retiring is not binding.

And one thing is *computed*, so that it cannot be got wrong: the `N` in
`lgkmcnt(N)`. LDS returns in order, so retiring tile T means waiting until only
the reads issued after T are outstanding. The handwritten GEMM derives exactly
this number with a constexpr function and a paragraph explaining why
(`CHUNK_READS` in kernels/rdna3/gemm/bf16fp32/gemm.cpp); here it falls out of
walking the issue queue, and `lds_wait_for` has no `n` parameter to get wrong.

**Loops.** A prefetching loop issues reads for iteration k+1 and waits for
iteration k's at the top of the *next* iteration, so the queue is non-empty
across the back edge and the derived N depends on state from the previous
iteration. The pass therefore walks each loop body twice, the second time
starting from the state the first left, and requires the two to agree. A body
whose derived counts differ between the two passes has no steady state -- the
queue grows or shrinks each iteration -- and that is reported rather than
guessed at, because either answer would be wrong for some iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..nodes import KernelIR, Op, RegTileType, Value
from ..verify import VerifyError


def _at(op: Op) -> str:
    return f"\n    at {op.loc}" if op.loc else ""


@dataclass
class _State:
    """Where the LDS read queue stands at a point in the program.

    `queue` is outstanding reads in issue order -- LDS retires in that order,
    which is the only reason a count is derivable at all. `unbound` is tiles
    whose reads have been retired by a wait that did not name them: no longer
    in flight, still missing the register dependence, still unusable.
    """

    queue: List[Tuple[int, int, Value]] = field(default_factory=list)  # (id, reads, v)
    unbound: Dict[int, Op] = field(default_factory=dict)  # id -> the wait that retired it

    def copy(self) -> "_State":
        return _State(list(self.queue), dict(self.unbound))

    def index_of(self, v: Value) -> Optional[int]:
        for i, (vid, _, _) in enumerate(self.queue):
            if vid == id(v):
                return i
        return None


def _check_reads(op: Op, st: _State) -> None:
    """Nothing may read a tile that is in flight or retired-but-unbound."""
    for v in op.operands:
        if not isinstance(v.type, RegTileType):
            continue
        if st.index_of(v) is not None:
            raise VerifyError(
                f"{op.opcode} reads {v.name}, whose LDS reads are still in "
                f"flight. Name it in an lds_wait_for before using it. This is "
                f"an error rather than a race you might get away with: the "
                f"reads have not landed, and the consumer will not wait for "
                f"them on its own." + _at(op)
            )
        w = st.unbound.get(id(v))
        if w is not None:
            raise VerifyError(
                f"{op.opcode} reads {v.name}. Its LDS reads were retired by an "
                f"lds_wait_for that did not name it, so the data has landed but "
                f"the register dependence has not been put back -- this "
                f"consumer is free to schedule above the wait, and will, "
                f"intermittently. Add {v.name} to that lds_wait_for."
                + _at(w) + _at(op)
            )


def _walk(ops: List[Op], st: _State, second: bool) -> _State:
    for op in ops:
        if op.body is not None:
            # A loop (or any region). Run it twice and require agreement; see
            # the module docstring.
            after_first = _walk(op.body, st.copy(), second)
            st = _walk(op.body, after_first, True)
            continue

        if op.opcode == "lds_wait_for":
            positions = []
            for v in op.operands:
                i = st.index_of(v)
                if i is None:
                    if id(v) in st.unbound:
                        raise VerifyError(
                            f"lds_wait_for names {v.name}, whose reads a "
                            f"previous wait already retired without binding "
                            f"it. Move {v.name} into that wait instead; this "
                            f"one is too late, the window is already open."
                            + _at(op)
                        )
                    raise VerifyError(
                        f"lds_wait_for names {v.name}, which has no LDS reads "
                        f"outstanding. Either it was loaded with wait=True -- "
                        f"in which case it is already retired and bound and "
                        f"the wait is dead code -- or it was already waited "
                        f"for." + _at(op)
                    )
                positions.append(i)

            cut = max(positions)
            # Everything issued after the last-named tile stays in flight, and
            # its read count is exactly the lgkmcnt to wait down to.
            n = sum(reads for _, reads, _ in st.queue[cut + 1:])
            named = {id(v) for v in op.operands}
            for vid, _, v in st.queue[: cut + 1]:
                if vid not in named:
                    # Retired by this wait but not bound by it. Legal to write,
                    # illegal to read -- _check_reads says so at the use.
                    st.unbound[vid] = op
            st.queue = st.queue[cut + 1:]
            for v in op.operands:
                st.unbound.pop(id(v), None)

            if "n" in op.attrs and op.attrs["n"] != n:
                raise VerifyError(
                    f"this lds_wait_for needs lgkmcnt({op.attrs['n']}) on one "
                    f"iteration and lgkmcnt({n}) on the next: the loop has no "
                    f"steady state, so the queue depth depends on how many "
                    f"iterations have run. Peel the prologue out of the loop, "
                    f"or issue the same reads every iteration." + _at(op)
                )
            op.attrs["n"] = n
            continue

        if op.opcode == "load_shared":
            # Before _check_reads, and that order is load-bearing. `out=`
            # compiles to emit_into, which puts the destination at operands[0]
            # -- so a load into an in-flight tile looks to _check_reads like a
            # *read* of one, and gets diagnosed as a missing wait before use
            # when the actual fault is that the previous reads have not landed
            # yet and will overwrite the new ones. Same verdict, wrong
            # instruction to the reader. load_shared is the only op here whose
            # first operand is written without being read; an in-place copy or
            # mul really does read it, and must go through _check_reads.
            dst = op.dst
            if st.index_of(dst) is not None:
                raise VerifyError(
                    f"load_shared writes {dst.name} while its previous LDS "
                    f"reads are still in flight. The earlier reads would land "
                    f"on top of the newer ones -- retire them with an "
                    f"lds_wait_for first." + _at(op)
                )
            if op.attrs.get("wait", True):
                # kittens::load<true> ends in lgkmcnt(0), which retires every
                # outstanding read, not only this tile's -- and binds none of
                # them. Anything else in the queue becomes unbound.
                for vid, _, _ in st.queue:
                    st.unbound[vid] = op
                st.queue = []
                st.unbound.pop(id(dst), None)
            else:
                st.unbound.pop(id(dst), None)
                st.queue.append((id(dst), op.attrs["lds_loads"], dst))
            continue

        _check_reads(op, st)
    return st


def lds_pipeline(ir: KernelIR) -> None:
    final = _walk(ir.body, _State(), False)
    if final.queue:
        names = ", ".join(v.name for _, _, v in final.queue)
        raise VerifyError(
            f"the kernel ends with LDS reads still in flight for {names}. "
            f"Either they are dead -- drop the load -- or a wait is missing."
        )
