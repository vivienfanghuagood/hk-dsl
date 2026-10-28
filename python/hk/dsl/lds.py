"""hk.lds: shared memory -- allocation, and the wait that retires reads."""

from __future__ import annotations

from typing import Optional

from .._mlir.dialects import hk as hkd
from .module import mlir_type
from ..types import RegVecType, SharedTileType, SharedVecType, fp32
from . import builder as B
from .values import SharedTile, SharedVec, Tile, lds_loads

__all__ = ["alloc", "alloc_vec", "shared_vec", "reserve", "wait_for", "lds_loads", "phase"]


#: Layouts an allocation may name (gfx950): include/cdna4's st_shapes that
#: also serve a read of the fragment that would otherwise pick another.
SHAPES = ("16x64", "32x32", "16x128")


def _divergent_ifs(t):
    """The enclosing `scf.if`s, innermost first, that not every warp may take (no `hk.uniform` declaration, not an
    always-taken `hk.scope`), and whether the point is inside the rewriter's scratch probe of a branch."""
    ips = getattr(t, "_ips", None)
    out, probe = [], False
    if not ips:
        return out, probe
    op = ips[-1].block.owner
    while op is not None:
        o = getattr(op, "operation", op)
        if o.name == "scf.if":
            if "hk.probe" in o.attributes:
                probe = True
            if "hk.uniform" not in o.attributes and "hk.opaque" not in o.attributes:
                out.append(o)
        if o.name in ("func.func", "builtin.module"):
            break
        op = o.parent
    return out, probe


def _check_uniform_place(t, name: str) -> None:
    # HipKittens' shared_allocator hands out offsets in execution order (a pointer bumped at each allocation), so a
    # tile allocated where only some warps go moves every LATER tile for those warps alone: past such a branch, two
    # warps see one "tile" at two addresses (the index chain's top-k read another warp's histogram). An allocation
    # inside a divergent branch is fine as long as every later one stays inside that branch.
    divs, probe = _divergent_ifs(t)
    seen = getattr(t, "_lds_divergent", None)
    if seen is None:
        seen = t._lds_divergent = []
    for d, first in seen:
        if not any(d == x for x in divs):
            raise TypeError(f"lds alloc {name!r} after {first!r}, which was allocated inside a branch not every warp "
                            f"takes: LDS offsets are handed out in execution order, so the warps that skipped that "
                            f"branch would see {name!r} at another address. Allocate {first!r} before its branch (or "
                            f"declare the branch workgroup-uniform, hk.uniform).")
    if divs and not probe and not any(divs[0] == d for d, _ in seen):
        seen.append((divs[0], name))


def alloc(tile: SharedTileType, name: str = "smem",
          count: Optional[int] = None, *, shape: Optional[str] = None) -> SharedTile:
    """Reserve an LDS tile, or `count` of them in one size-aligned block (the
    two buffers of a pair must differ in one address bit for the XOR swap).
    Offsets and the 64 KB check come from the hk-lds-alloc pass.

    `shape` (gfx950) lays the allocation out as include/cdna4's st_<shape>
    rather than the layout its reads' fragment picks: "16x64" for tiles read
    as 16x32 row operands, "32x32" for 32x16 stride-4 col ones. A DMA copy
    into it reads each row in 128- or 64-byte runs instead of 64 or 32 (the
    paged decode's copies: 6.15 to 6.9 TB/s with `nontemporal`), and its
    reads may conflict on banks more; a read of another fragment is
    refused. "16x128" (byte tiles): what an fp8 16x128 operand read would pick,
    named for an allocation only `load_fp4_frags_lds` reads."""
    if count is not None:
        tile = SharedTileType(tile.dtype, tile.rows, tile.cols, count)
    t = B.current()
    _check_uniform_place(t, name)
    if shape is not None:
        from ..target import get_target  # noqa: PLC0415

        if shape not in SHAPES:
            raise TypeError(f"lds.alloc(shape={shape!r}): one of {SHAPES}")
        if get_target(t.arch).arch != "gfx950":
            raise TypeError(f"lds.alloc(shape=...) names an include/cdna4 st_shape; "
                            f"{t.arch} has none")
    op = hkd.AllocSharedOp(mlir_type(tile), loc=t.loc(name))
    if shape is not None:
        from .._mlir import ir  # noqa: PLC0415

        op.attributes["shape"] = ir.StringAttr.get(f"st_{shape}")
    return SharedTile(op.result, tile)


def shared_vec(vt: RegVecType, count: int = 1) -> SharedVecType:
    """The LDS type a register vector round-trips through."""
    if not isinstance(vt, RegVecType):
        raise TypeError(f"shared_vec takes a register vector type, got {vt!r}")
    if vt.tile.layout == "strip" and vt.kind == "col":
        # One value a row, and a strip may have fewer rows than the sixteen
        # an sv holds; the vector uses its first `length` entries.
        return SharedVecType(vt.dtype, -(-vt.length // 16) * 16, count)
    return SharedVecType(vt.dtype, vt.length, count)


def alloc_vec(ty: SharedVecType, name: str = "svec") -> SharedVec:
    """Reserve an LDS vector stack of type `ty` (`hk.lds.shared_vec(...)`
    or `hk.SV`): what `hk.collective.cross_warp` combines through."""
    if not isinstance(ty, SharedVecType):
        raise TypeError(f"alloc_vec takes a SharedVecType, got {ty!r}")
    t = B.current()
    _check_uniform_place(t, name)
    return SharedVec(hkd.AllocSharedVecOp(mlir_type(ty), loc=t.loc(name)).result,
                     ty)


def reserve(nbytes: int, name: str = "reserve") -> None:
    """Take `nbytes` of LDS and use none of it: an occupancy cap.

    A WGP has 128 KB of LDS, so a workgroup holding 30 KB leaves room for
    four of them, 60 KB for two. A kernel that streams HBM can run faster
    with fewer rows in flight than its registers would allow (chapter 27);
    this is the knob, at the granularity LDS gives it."""
    if not isinstance(nbytes, int) or nbytes <= 0:
        raise TypeError(f"reserve({nbytes!r}): a positive number of bytes")
    alloc_vec(SharedVecType(fp32, 16, -(-nbytes // 64)), name=name)


def phase() -> None:
    """Start an LDS phase: the allocations from here on reuse the LDS of the phases before (a union).

    For a persistent launch whose workgroups run stages in order (K2's attention back, then its MoE): the
    allocations before the first `phase()` are every phase's; each phase starts at their end, and the launch
    takes the largest phase. At the kernel's top level only (every workgroup passes it: the allocator is
    rewound there at run time). A tile from an earlier phase must not be read in a later one. A barrier
    between phases is the caller's: a slow warp may still read the last phase's tiles."""
    t = B.current()
    hkd.LdsPhaseOp(loc=t.loc("lds_phase"))


def wait_for(*tiles: Tile) -> None:
    """Retire the outstanding LDS reads of `tiles` and re-bind their
    registers. The lgkmcnt is derived by hk-lds-pipeline; there is no bare
    wait, because a wait that names nothing puts back no dependence."""
    if not tiles:
        raise TypeError("wait_for() needs the tiles it retires")
    for x in tiles:
        if not isinstance(x, Tile):
            raise TypeError(f"wait_for operand is {x!r}, expected a register tile")
    hkd.LdsWaitForOp([x.v for x in tiles], loc=B.current().loc())
