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

from typing import Optional

from .._mlir.dialects import hk as hkd
from .module import mlir_type
from ..target import get_target
from ..types import StageBufferType
from . import builder as B
from .values import RowIndex, SharedTile, Stage, View, _deprecated, as_int


def _max(a, b):
    """max of two row counts, Python ints or device Ints."""
    from .values import Value, scalar_op  # noqa: PLC0415

    if isinstance(a, Value) or isinstance(b, Value):
        return scalar_op("max", a, b)
    return a if a >= b else b


class Group:
    """What `hk.group(warps)` returns: copies between global memory and LDS
    done by `warps` warps together."""

    __slots__ = ("warps",)

    def __init__(self, warps: int):
        if not isinstance(warps, int) or warps <= 0 or warps > 32:
            raise TypeError(f"hk.group takes 1..32 warps, got {warps!r}")
        self.warps = warps

    @property
    def threads(self) -> int:
        """warps x the wave of the kernel being traced (32 on RDNA, 64 on CDNA)."""
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

    def load(self, dst: SharedTile, src: View, *, wait: bool = True,
             rows=None, nontemporal: bool = False, gather=None,
             staged: Optional[int] = None) -> None:
        """Global -> LDS by the whole group, issued and waited. Rows of the
        tile past the tensor's last read zero and touch no memory, on every
        target (gfx950's copy used to span the tile's whole rows: a last,
        partial tile read past the tensor).

        `wait=False` (gfx950) issues the DMA, buffer_load ... lds, and returns
        with it in flight: name the tile in `g.wait_for` and put a barrier
        after that before any warp reads it. `dst` is a whole allocation or
        one buffer of a stack picked by a Python int, so hk-lds-pipeline can
        check every read of it and every later copy into it. `rows` (with
        wait=False) ends the copy that many rows from the tile's first, as
        the tensor's last row does: rows past it read zero. `nontemporal`
        (with wait=False) reads past the caches' keeping, sc0 nt, for data
        read once -- in runs of a whole line it streams faster; in the
        half-line runs of a 16x32 tile it fetched lines twice (4.8 against
        6.15 TB/s in the paged decode). `gather=ix` (with wait=False), an
        `hk.RowIndex`: the tile's rows come by index, row r being the source's
        row `ids[at + r] & 0xFFFFFF` -- AITER's MoE sort ids (token | choice <<
        24; its padding is a row past the tensor, which reads zero); `src`'s
        coordinate gives the column. With `rows` (or, left out, the index's
        `count`: rows = count - at), the tile's rows from `rows` on read zero and
        their ids are not read -- an index list that ends inside the tile (a
        CSR's last row) is not read past. An index staged by
        `g.stage_rows(ix, N)` is read from LDS, so a lane's row costs no vmcnt
        wait (which, in order, would retire the whole prefetch).
        `gather=(ids, at)` and `staged=N` are deprecated spellings of the
        same."""
        self._one(dst, "group load destination")
        t = B.current()
        if rows is not None and wait:
            raise TypeError("g.load(rows=...) narrows the asynchronous copy's "
                            "descriptor; pass wait=False")
        if nontemporal and wait:
            raise TypeError("g.load(nontemporal=True) is the asynchronous "
                            "copy's cache policy; pass wait=False")
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
        if gather is not None and wait:
            raise TypeError("g.load(gather=...) is the asynchronous copy's (wait=False)")
        if isinstance(gather, RowIndex):
            if staged is not None:
                raise TypeError("g.load: staged is the RowIndex's (g.stage_rows), not a keyword")
            if gather.stage_n and gather.keep_choice:
                raise TypeError("g.load(gather=): this index was staged for scatters (the raw "
                                "ids); a gathered copy reads the token rows -- "
                                "g.stage_rows(ix, rows)")
            if gather.div != 1:
                raise TypeError("g.load(gather=): the copy decodes AITER sort ids or plain rows, "
                                "not div=")
            if gather.stage_n:
                gather._check_staged("g.load(gather=)")
            if rows is None and gather.count is not None:
                # the index list ends at count: rows from count - at on are not read
                rows = _max(gather.count - gather.at, 0)
            gids, gat, staged = gather.ids, gather.at, gather.stage_n
        else:
            if gather is not None:
                _deprecated("g.load(..., gather=(ids, at))",
                            "g.load(..., gather=hk.RowIndex(ids, at))")
            if staged is not None:
                _deprecated("g.load(..., staged=N)", "g.load(..., gather=g.stage_rows(ix, N))")
            gids, gat = (None, None) if gather is None else gather
            staged = staged or 0
        if gids is not None and (not hasattr(gids, "dtype") or gids.dtype.name != "i32"):
            raise TypeError(f"g.load(gather=(ids, at)): ids is an int32 tensor, got {gids!r}")
        hkd.GroupLoadOp(dst.v, src.tensor.v, src.idx.v, self.threads,
                        rows=None if rows is None else as_int(rows).v,
                        gather_ids=None if gids is None else gids.v,
                        gather_at=None if gat is None else as_int(gat).v,
                        gather_staged=int(staged),
                        async_=not wait, nontemporal=bool(nontemporal),
                        loc=t.loc())

    def stage_rows(self, ix: RowIndex, rows: Optional[int] = None, *,
                   keep_choice: Optional[bool] = None) -> RowIndex:
        """`ix`'s positions [at, at + rows) copied into LDS, then a barrier (every warp must
        call it); returns the staged index, which `g.load(..., gather=)` and
        `Global.scatter` read from LDS -- no vmcnt wait for an index, which in order would
        retire every load in flight. For gathers it holds the token rows (`ids & 0xFFFFFF`);
        with `keep_choice=True` the ids themselves, which a scatter of sort ids needs (`at` a
        multiple of `rows`). A consumer refuses the other form, and refuses a staged index whose
        copy no `stage_rows` has made before it.

        `ix` may already be described as staged (`ix.staged(rows, keep_choice=...)`, made where
        the copies that read it are written, e.g. outside the loop whose body stages it): then
        this makes its copy, here."""
        if not isinstance(ix, RowIndex):
            raise TypeError(f"g.stage_rows(ix, rows): ix is an hk.RowIndex, got {ix!r}")
        if ix.stage_n:
            if (rows is not None and int(rows) != ix.stage_n) or (
                    keep_choice is not None and bool(keep_choice) != ix.keep_choice):
                raise TypeError(f"g.stage_rows: {ix!r} is described as staged already; "
                                f"rows / keep_choice are its own")
            out = ix
        else:
            if rows is None:
                raise TypeError("g.stage_rows(ix, rows): how many positions to stage")
            out = ix.staged(rows, keep_choice=bool(keep_choice))
        t = B.current()
        hkd.GatherPrepOp(out.ids.v, as_int(out.at).v, out.stage_n, raw=out.keep_choice,
                         loc=t.loc())
        staged = getattr(t, "staged_rows", None)
        if staged is None:
            staged = t.staged_rows = set()
        staged.add((out.stage_n, out.keep_choice))
        return out

    def gather_prep(self, ids, at, rows: int, *, raw: bool = False) -> None:
        """Deprecated: `g.stage_rows(hk.RowIndex(ids, at), rows, keep_choice=raw)`."""
        _deprecated("g.gather_prep(ids, at, rows, raw=...)",
                    "g.stage_rows(hk.RowIndex(ids, at), rows, keep_choice=...)")
        if not hasattr(ids, "dtype") or ids.dtype.name != "i32":
            raise TypeError(f"gather_prep: ids is an int32 tensor, got {ids!r}")
        hkd.GatherPrepOp(ids.v, as_int(at).v, int(rows), raw=bool(raw), loc=B.current().loc())

    def wait_for(self, *tiles: SharedTile, exact: bool = False) -> None:
        """`s_waitcnt vmcnt(n)`: this warp's share of the asynchronous copies
        into `tiles` has landed, `n` derived by hk-lds-pipeline. The other
        warps' shares are visible only after a barrier (two inside an
        hk.arch.pingpong), and the pass refuses a read before that."""
        if not tiles:
            raise TypeError("g.wait_for() needs the LDS tiles it retires")
        for x in tiles:
            if not isinstance(x, SharedTile):
                raise TypeError(f"g.wait_for operand is {x!r}, expected an LDS tile")
        op = hkd.GroupWaitForOp([x.v for x in tiles], loc=B.current().loc())
        if exact:
            # keep the tracked register loads issued after the copies in flight (in-order vmcnt)
            from .._mlir import ir  # noqa: PLC0415

            op.attributes["exact"] = ir.UnitAttr.get()

    def store(self, dst: View, src: SharedTile) -> None:
        """LDS -> global, by the whole group. Rows of the tile past the
        tensor's last are not written."""
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
    """The workgroup's `warps` warps acting together: `g = hk.group(8)`, then
    `g.load`, `g.store`, `g.stage`/`g.commit`, `g.wait_for`. `warps` is the
    kernel's `warps=` (or the part of it that copies)."""
    return Group(warps)


__all__ = ["Group", "group"]
