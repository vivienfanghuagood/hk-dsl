"""`hk.group(W)`: what a whole workgroup does together -- the global<->LDS
copies, synchronous and staged through registers.

    g = hk.group(8)
    As = hk.lds.alloc(hk.ST(hk.bf16, 128, 64), count=2, name="As")
    buf = g.stage_buffer(As, name="buf_a")
    g.load(As[0], a[idx])           # issue and wait
    g.stage(buf, a[idx])            # issue into VGPRs
    hk.arch.vm_wait(0)
    g.commit(As[1], buf)            # VGPRs -> LDS

    g.load(As[1], a[idx], wait=False)   # gfx950: the DMA, left in flight
    g.wait_for(As[1])                   # vmcnt, derived
    hk.arch.barrier()                   # every warp's share has landed
"""

from __future__ import annotations

from .._mlir.dialects import hk as hkd
from .module import mlir_type
from ..target import get_target
from ..types import StageBufferType
from . import builder as B
from .values import SharedTile, Stage, View


class Group:
    __slots__ = ("warps",)

    def __init__(self, warps: int):
        if not isinstance(warps, int) or warps <= 0 or warps > 32:
            raise TypeError(f"hk.group takes 1..32 warps, got {warps!r}")
        self.warps = warps

    @property
    def threads(self) -> int:
        # The wave of the kernel being traced: a group of W warps is 64 W
        # threads on CDNA, and a buffer sized for 32 W stages half the tile.
        t = B.current_or_none()
        return self.warps * (get_target(t.arch).wave if t else 32)

    def __repr__(self) -> str:
        return f"group<{self.warps}>"

    @staticmethod
    def _one(dst: SharedTile, what: str) -> None:
        if not isinstance(dst, SharedTile):
            raise TypeError(f"{what} is {dst!r}, expected an LDS tile")
        if dst.ty.count != 1:
            raise TypeError(f"{what} is {dst.ty}, a stack; index it: As[tic]")

    def load(self, dst: SharedTile, src: View, *, wait: bool = True) -> None:
        """Global -> LDS by the whole group, issued and waited.

        `wait=False` (gfx950) issues the DMA, buffer_load ... lds, and returns
        with it in flight: name the tile in `g.wait_for` and put a barrier
        after that before any warp reads it. `dst` is a whole allocation or
        one buffer of a stack picked by a Python int, so hk-lds-pipeline can
        check every read of it and every later copy into it."""
        self._one(dst, "group load destination")
        t = B.current()
        if not wait:
            tgt = get_target(t.arch)
            if not tgt.has_global_to_lds_dma or tgt.arch != "gfx950":
                raise TypeError(
                    f"g.load(wait=False) on {t.arch}: an asynchronous copy is "
                    f"gfx950's 16-byte buffer_load ... lds, which this target "
                    f"does not have. Use wait=True.")
            if dst.v.owner.name == "hk.subtile":
                raise TypeError(
                    "g.load(wait=False) into a subtile: the copy lays out a "
                    "whole LDS tile. Copy into the allocation, or one buffer "
                    "of a stack.")
        hkd.GroupLoadOp(dst.v, src.tensor.v, src.idx.v, self.threads,
                        async_=not wait, loc=t.loc())

    def wait_for(self, *tiles: SharedTile) -> None:
        """`s_waitcnt vmcnt(n)`: this warp's share of the asynchronous copies
        into `tiles` has landed, `n` derived by hk-lds-pipeline. The other
        warps' shares are visible only after a barrier (two inside an
        hk.arch.pingpong), and the pass refuses a read before that."""
        if not tiles:
            raise TypeError("g.wait_for() needs the LDS tiles it retires")
        for x in tiles:
            if not isinstance(x, SharedTile):
                raise TypeError(f"g.wait_for operand is {x!r}, expected an LDS tile")
        hkd.GroupWaitForOp([x.v for x in tiles], loc=B.current().loc())

    def store(self, dst: View, src: SharedTile) -> None:
        """LDS -> global, by the whole group."""
        self._one(src, "group store source")
        hkd.GroupStoreOp(dst.tensor.v, src.v, dst.idx.v, self.threads,
                         loc=B.current().loc())

    def stage_buffer(self, dst: SharedTile, depth: int = 1,
                     name: str = "buf") -> Stage:
        """The register buffer for copies into `dst`, sized from it so the two
        cannot disagree. Declare it outside any loop."""
        ty = StageBufferType(dst.ty.element(), self.threads, depth)
        t = B.current()
        return Stage(hkd.StageBufferOp(mlir_type(ty), dst.v,
                                       loc=t.loc(name)).result, ty)

    def _check(self, buf: Stage, what: str) -> None:
        if not isinstance(buf, Stage):
            raise TypeError(f"group {what} buffer is {buf!r}")
        if buf.ty.threads != self.threads:
            raise TypeError(f"{self}.{what}() on a buffer sized for "
                            f"{buf.ty.threads} threads")

    def stage(self, buf: Stage, src: View, slot: int = 0) -> None:
        """Issue the global loads for one tile into `buf`. Does not wait."""
        self._check(buf, "stage")
        if not isinstance(slot, int) or not 0 <= slot < buf.ty.depth:
            raise TypeError(f"stage slot={slot!r}: an int in [0, {buf.ty.depth})")
        hkd.StageLoadOp(buf.v, src.tensor.v, src.idx.v, slot,
                        loc=B.current().loc())

    def commit(self, dst: SharedTile, buf: Stage, slot: int = 0,
               wait: bool = False) -> None:
        """Write a staged buffer to LDS. hk.arch.vm_wait must come first."""
        self._check(buf, "commit")
        self._one(dst, "commit destination")
        hkd.StageCommitOp(dst.v, buf.v, slot, wait=bool(wait),
                          loc=B.current().loc())


def group(warps: int) -> Group:
    return Group(warps)


__all__ = ["Group", "group"]
