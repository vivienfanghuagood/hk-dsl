"""`hk.group(W)` -- the ops a whole workgroup does together.

Everything in `lang.ops` is per-warp unless it says otherwise, and that is the
right default: a warp is the unit the register file and the WMMA are defined
over. But a global->LDS copy is not a warp's work. One K-tile of a 128x32 bf16
A-block is 8 KB, and splitting it across one wave means 32 lanes each walking a
256-byte stride; splitting it across eight means a contiguous float4 per lane.
The C++ library spells that `kittens::group<W>::load`, and this is that.

A group handle carries exactly one thing, its thread count, and it is the only
place that number appears -- which matters, because `stage_calls` is computed
from it and a staging buffer sized for the wrong thread count is an
out-of-bounds write to the stack rather than a mismatch anyone would see.

    g = hk.group(8)
    As = hk.alloc_shared(hk.st(hk.bf16, 128, 32), count=2, name="As")
    buf = g.stage_buffer(As, name="buf_a")

    g.load(As[0], a, coord)            # synchronous: issues and waits
    g.stage(buf, a, coord)             # issues into VGPRs, does not wait
    hk.vm_wait(0)                      # ... other work goes here
    g.commit(As[1], buf)               # VGPRs -> LDS

The split between `stage` and `commit` is the whole reason this file exists.
See `hk.lang.ops.stage_buffer`.
"""

from __future__ import annotations

from typing import Optional

from ..ir.nodes import CoordType, GlobalType, SharedTileType, Value
from . import ops


class Group:
    """A workgroup of `warps` waves. Make one with `hk.group(W)`."""

    __slots__ = ("warps",)

    def __init__(self, warps: int):
        if not isinstance(warps, int) or warps <= 0:
            raise TypeError(f"hk.group takes a positive warp count, got {warps!r}")
        if warps > 32:
            raise ValueError(
                f"hk.group({warps}): a gfx1100 workgroup is at most 1024 threads, "
                f"so at most 32 waves of 32."
            )
        self.warps = warps

    @property
    def threads(self) -> int:
        return self.warps * 32

    def __repr__(self) -> str:
        return f"group<{self.warps}>"

    # ------------------------------------------------------------ synchronous

    def load(self, dst: Value, src: Value, idx: Value) -> None:
        """Global -> LDS, issued and waited by the whole group.

        The simple form, for a prologue. In a steady-state loop you want
        `stage`/`commit` instead, so the latency lands under some math.
        """
        self._check_shared(dst, "group load destination")
        ops._expect(src, GlobalType, "group load source")
        ops._expect(idx, CoordType, "group load index")
        ops._b().emit("group_load", [dst, src, idx], threads=self.threads)

    def store(self, dst: Value, src: Value, idx: Value) -> None:
        """LDS -> global, by the whole group."""
        ops._expect(dst, GlobalType, "group store destination")
        self._check_shared(src, "group store source")
        ops._expect(idx, CoordType, "group store index")
        ops._b().emit("group_store", [dst, src, idx], threads=self.threads)

    # ----------------------------------------------------------- asynchronous

    def stage_buffer(self, dst: Value, depth: int = 1,
                     name: str = "buf") -> Value:
        """The register buffer for a group copy into `dst`. Declare at top level.

        `dst` is the allocation, not one buffer of it: a double-buffered copy
        stages once and commits to whichever half is free, so the buffer
        belongs to the pair.
        """
        return ops.stage_buffer(dst, self.threads, depth, name)

    def stage(self, buf: Value, src: Value, idx: Value, slot: int = 0) -> None:
        """Issue the global loads for one tile into `buf`. Does not wait."""
        self._check_buf(buf, "stage")
        ops.stage_load(buf, src, idx, slot)

    def commit(self, dst: Value, buf: Value, slot: int = 0,
               wait: bool = False) -> None:
        """Write a staged buffer to LDS. `vm_wait` must come first."""
        self._check_buf(buf, "commit")
        ops.stage_commit(dst, buf, slot, wait)

    # ------------------------------------------------------------------ check

    def _check_buf(self, buf: Value, what: str) -> None:
        ops._expect(buf, ops.StageBufferType, f"group {what} buffer")
        if buf.type.threads != self.threads:
            raise TypeError(
                f"{self}.{what}() on {buf.type}, whose buffer is sized for "
                f"{buf.type.threads} threads. The size is stage_calls, which is "
                f"a division by the thread count: using it from a differently "
                f"sized group writes past the end of the array."
            )

    @staticmethod
    def _check_shared(v: Value, what: str) -> None:
        ops._expect(v, SharedTileType, what)
        if v.type.count != 1:
            raise TypeError(
                f"{what} is {v.type}, a stack of {v.type.count}. Index it: "
                f"`As[tic]`."
            )


def group(warps: int) -> Group:
    """A handle for the collective ops of a `warps`-wave workgroup."""
    return Group(warps)


__all__ = ["Group", "group"]
