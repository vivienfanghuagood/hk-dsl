"""hk.arch: the machine -- where this workgroup and warp are, and the
instructions that order and schedule them."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .._mlir import ir
from .._mlir.dialects import hk as hkd
from ..types import CoordType, GlobalType, ScalarType, i32
from . import builder as B
from .values import Bool, Coord, Global, Int, Scalar, View, as_int, i1


def _i32():
    return ir.IntegerType.get_signless(32)


class _BlockIdx:
    """`hk.arch.block_idx.x` / `.y` / `.z`: hoisted, read once per kernel."""

    def __getattr__(self, axis: str) -> Int:
        if axis not in ("x", "y", "z"):
            raise AttributeError(f"block_idx has no axis {axis!r}")
        t = B.current()

        def build():
            return hkd.BlockIdxOp(_i32(), getattr(hkd.Axis, axis),
                                  loc=t.loc(f"bid_{axis}")).result

        return Int(t.hoisted(("block_idx", axis), build), ScalarType(i32))

    def __call__(self):
        return self


block_idx = _BlockIdx()


def target():
    """The hk.target.Target this kernel is being traced for: wave size,
    register file, matrix-core fragments. For a kernel body whose tiling is
    a property of the target, chosen while tracing."""
    from ..target import get_target  # noqa: PLC0415

    return get_target(B.current().arch)


def warp_id() -> Int:
    """This warp's index in the workgroup (wave-uniform)."""
    t = B.current()
    return Int(t.hoisted("warp_id", lambda: hkd.WarpIdOp(
        _i32(), loc=t.loc("warp_id")).result), ScalarType(i32))


def xcd_id() -> Int:
    """The XCD this workgroup runs on, 0..7 on a gfx950 (read from the hardware: s_getreg
    HW_REG_XCC_ID), wave-uniform. Workgroups are dealt to the XCDs round robin, so it is usually
    block id mod 8 -- but a kernel that relies on sharing an XCD's L2 should ask, not assume."""
    t = B.current()
    if not t.arch.startswith("gfx9"):
        raise TypeError(f"hk.arch.xcd_id() is gfx94x/gfx950's (XCDs); {t.arch} has none")
    return Int(t.hoisted("xcd_id", lambda: hkd.XcdIdOp(_i32(), loc=t.loc("xcd")).result),
               ScalarType(i32))


def xcd_count() -> int:
    """XCDs (L2 domains) of the GPU this kernel is traced for, a constant: 8 on an MI355X,
    4 on an MI350P or a DPX partition, 1 on a part with one L2 (KFD's num_xcc, else the arch's
    default; HK_XCDS overrides). What hk.dsl.placement deals workgroups over."""
    from .placement import xcd_count as _count  # noqa: PLC0415

    return _count(B.current().arch)


def lane_id() -> Int:
    """This lane's index in its wave: 0..31 on gfx11, 0..63 on gfx950. Not
    wave-uniform, so not a branch condition for a workgroup operation."""
    t = B.current()
    return Int(t.hoisted("lane_id", lambda: hkd.LaneIdOp(
        _i32(), loc=t.loc("lane_id")).result), ScalarType(i32))


def barrier(drain: bool = False) -> None:
    """`s_barrier`: orders execution across the workgroup, not memory.
    `drain=True` spends an lgkmcnt(0) in front of it."""
    hkd.BarrierOp(drain=bool(drain), loc=B.current().loc())


#: The instruction classes of __builtin_amdgcn_sched_barrier and
#: sched_group_barrier, by name. An inline-asm instruction is in none of them:
#: include/rdna3's LDS fragment reads and writes are asm, its staging loads
#: and WMMAs are not.
SCHED_CLASSES = {
    "alu": 0x1, "valu": 0x2, "salu": 0x4, "mma": 0x8, "wmma": 0x8,
    "vmem": 0x10, "vmem_read": 0x20, "vmem_write": 0x40, "ds": 0x80,
    "ds_read": 0x100, "ds_write": 0x200, "trans": 0x400,
}


def _sched_mask(classes) -> int:
    if isinstance(classes, int) and not isinstance(classes, bool):
        mask = classes
    else:
        names = (classes,) if isinstance(classes, str) else tuple(classes)
        bad = [n for n in names if n not in SCHED_CLASSES]
        if bad:
            raise TypeError(f"unknown instruction class {bad[0]!r}; the classes "
                            f"are {', '.join(SCHED_CLASSES)}")
        mask = 0
        for n in names:
            mask |= SCHED_CLASSES[n]
    if not 0 <= mask <= 0x7FF:
        raise TypeError(f"instruction class mask {mask:#x} is outside 0x1..0x400")
    return mask


def sched_barrier(allow=()) -> None:
    """Nothing but the classes in `allow` crosses this line in the scheduler;
    it costs no instruction. `allow` is a class name, several, or a mask."""
    hkd.SchedBarrierOp(mask=_sched_mask(allow), loc=B.current().loc())


def sched_group_barrier(classes, size: int, sync_id: int = 0) -> None:
    """Ask the scheduler for `size` instructions of `classes` here, after
    the groups before this one with the same `sync_id`: a run of these is an
    interleave. `classes` is a name from SCHED_CLASSES, several, or a mask.

        for _ in range(4):                     # one staging load, two WMMAs
            hk.arch.sched_group_barrier("vmem_read", 1)
            hk.arch.sched_group_barrier("mma", 2)
    """
    mask = _sched_mask(classes)
    if not mask:
        raise TypeError("sched_group_barrier needs at least one class")
    if not isinstance(size, int) or size < 1:
        raise TypeError(f"sched_group_barrier size={size!r}: at least one")
    if not isinstance(sync_id, int) or sync_id < 0:
        raise TypeError(f"sched_group_barrier sync_id={sync_id!r}: a group id >= 0")
    hkd.SchedGroupBarrierOp(mask, size, sync_id, loc=B.current().loc())


@contextmanager
def pingpong(behind) -> Iterator[None]:
    """Run the warps where `behind` holds one barrier behind the rest.

    On entry those warps wait at one extra s_barrier; on exit the others do,
    and the halves meet again. In between, every barrier pairs a program
    point in one half with the next one in the other, so a schedule that
    alternates memory and math between barriers runs the two halves out of
    phase: with eight warps, the two waves on a SIMD take turns at the
    matrix core. HipKittens' CDNA4 GEMM is built on it.

        with hk.arch.pingpong(warp_id // 4 == 1):
            ...                     # memory, barrier, math, barrier, ...

    `behind` is wave-uniform (from warp_id, not lane_id). hk-lds-pipeline
    knows the skew: inside, a warp is only sure another has passed a point
    two barriers later, not one."""
    if not isinstance(behind, Bool):
        raise TypeError(f"pingpong({behind!r}): a device comparison, e.g. "
                        f"warp_id() // 4 == 1")
    t = B.current()
    hkd.PingPongBeginOp(i1(behind), loc=t.loc())
    try:
        yield
    finally:
        hkd.PingPongEndOp(i1(behind), loc=t.loc())


def pin(*values) -> None:
    """Keep these register tiles' and vectors' values here: what computes
    them stays above this point and what reads them below. No instruction.
    `sched_barrier` fences only the scheduler; LLVM still sinks a value's
    computation to its first use, across barriers and into another phase of
    a hand-written schedule. A value with an LDS read in flight is refused,
    as any read of it is, until it is waited for."""
    from .values import Tile, Vec  # noqa: PLC0415

    if not values:
        raise TypeError("pin() needs the tiles or vectors to keep here")
    for x in values:
        if not isinstance(x, (Tile, Vec)):
            raise TypeError(f"pin takes register tiles and vectors, got {x!r}")
    hkd.PinOp([x.v for x in values], loc=B.current().loc())


def agpr(tile) -> None:
    """Hold `tile`, an fp32 MMA accumulator just made by hk.zeros, in AGPRs
    (CDNA): every mma into it runs on the AGPRs (inline asm, C/D "+a"), every
    other access reads it into a VGPR copy after the MFMAs' wait states and
    writes it back (include/hk/cdna_agpr.cuh). Outputs are bitwise the VGPR
    form's. LLVM picks the VGPR form unless register pressure forces AGPRs,
    and an asm MFMA is one its scheduler cannot see: measured slower on
    attn_cdna (docs: KernelWiki pitfalls/gfx950-agpr-mma-chain-interleave.md),
    so the kernels that offer it keep it off by default."""
    from .values import Tile  # noqa: PLC0415

    t = B.current()
    if not t.arch.startswith("gfx9"):
        raise TypeError(f"hk.arch.agpr: AGPRs are CDNA's, not {t.arch}'s")
    if not isinstance(tile, Tile):
        raise TypeError(f"hk.arch.agpr takes a register tile, got {tile!r}")
    owner = tile.v.owner
    name = getattr(getattr(owner, "operation", owner), "name", "")
    if name not in ("hk.zero", "hk.full"):
        raise TypeError("hk.arch.agpr takes a tile hk.zeros just made "
                        "(`acc = hk.zeros(T); hk.arch.agpr(acc)`)")
    owner.attributes["hk.agpr"] = ir.UnitAttr.get()


def mma_fence(on: bool = True) -> None:
    """Whether hk-mma-fence guards this kernel (CDNA; on by default). The
    pass puts a scheduling fence between a matrix-core chain and the first
    other access to its accumulator: on gfx950 LLVM threaded accumulator
    reads through AGPR MFMA chains and computed wrong scores (docs: KernelWiki
    pitfalls/gfx950-agpr-mma-chain-interleave.md). Turning it off is for an
    A/B, not a fix."""
    if not isinstance(on, bool):
        raise TypeError(f"mma_fence({on!r}): True or False")
    t = B.current()
    t.func.attributes["hk.mma_fence"] = ir.BoolAttr.get(on)


def setprio(level: int) -> None:
    """`s_setprio`: this wave's issue priority, 0..3."""
    if not isinstance(level, int) or not 0 <= level <= 3:
        raise TypeError(f"setprio({level!r}): the priority field is 2 bits, 0..3")
    hkd.SetPrioOp(level, loc=B.current().loc())


def vm_wait(keep: int = 0) -> None:
    """`s_waitcnt vmcnt(keep)`: retire vector-memory loads, `keep` in flight."""
    if not isinstance(keep, int) or not 0 <= keep <= 63:
        raise TypeError(f"vm_wait(keep={keep!r}): vmcnt is 6 bits, 0..63")
    hkd.VmWaitOp(keep, loc=B.current().loc())


def wg_arrive(counter, expected: Scalar, *, scope: str = "device") -> Int:
    """Count this workgroup's arrival at `counter` (an element view of an
    int32 global) and return how many arrived before it. The last to arrive
    resets the counter. Contains barriers: every warp must reach it.

    `scope="xcd"`: every workgroup arriving at this counter runs on one XCD
    of a gfx950 (block ids equal modulo 8: workgroups are dealt to the XCDs
    round robin), so the stores before it need only reach the shared L2 --
    no device-wide fence, which writes back and invalidates the L2 under
    every other workgroup of the XCD. The caller guarantees the placement."""
    if scope not in ("device", "xcd"):
        raise TypeError(f"wg_arrive(scope={scope!r}): 'device' or 'xcd'")
    if not isinstance(counter, View):
        raise TypeError("wg_arrive takes a view of the counter: cnt.at[i]")
    if counter.idx.unit != "element":
        raise TypeError("wg_arrive takes an element coordinate: cnt.at[i]")
    if counter.tensor.dtype is not i32:
        raise TypeError(f"wg_arrive counts in an int32 tensor, not "
                        f"{counter.tensor.dtype.name}")
    t = B.current()
    return Int(hkd.WgArriveOp(_i32(), counter.tensor.v, counter.idx.v,
                              as_int(expected).v, xcd=True if scope == "xcd" else None,
                              loc=t.loc("arrived")).result, ScalarType(i32))


__all__ = ["block_idx", "warp_id", "lane_id", "xcd_id", "xcd_count", "barrier", "sched_barrier",
           "sched_group_barrier", "SCHED_CLASSES", "pingpong", "pin", "setprio",
           "vm_wait", "wg_arrive", "mma_fence", "agpr"]
