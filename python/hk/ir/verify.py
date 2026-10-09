"""Rules the IR has to satisfy, checked before anything is compiled.

Two severities, and the distinction is the point:

* **Error** -- the kernel would be *wrong*, or would not build. Raises.
* **Warning** -- the kernel would be *slow* in a way we can name. Reported, not
  raised, because a deliberate slow path is a legitimate thing to write and a
  verifier that cannot be overruled gets disabled.

The register budget is an **estimate**. The authoritative number comes from the
compiler, after codegen, via hk.runtime.compile -> hip_resources.py. The
estimate exists to fail fast and to guide the autotuner, and it says so in every
message it produces; it is never presented as measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..target import get_target
from .nodes import (
    CoordType,
    GlobalType,
    KernelIR,
    RegTileType,
    SharedTileType,
    Value,
)


class VerifyError(Exception):
    """The kernel is wrong or unbuildable. Never downgrade one of these."""


@dataclass
class Warning_:
    code: str
    message: str
    loc: str = ""

    def __str__(self) -> str:
        at = f"\n    at {self.loc}" if self.loc else ""
        return f"[{self.code}] {self.message}{at}"


def _fail(op, msg: str) -> "VerifyError":
    at = f"\n    at {op.loc}" if getattr(op, "loc", None) else ""
    return VerifyError(f"{msg}{at}")


# ---------------------------------------------------------------- structure


def check_structure(ir: KernelIR) -> None:
    defined = {p.name for p in ir.params if isinstance(p.type, GlobalType)}
    seen: set = set()

    for op in ir.body:
        for v in op.operands:
            if isinstance(v.type, GlobalType):
                if v.name not in defined:
                    raise _fail(op, f"{op.opcode} reads unknown tensor {v.name!r}")
            elif v.producer is None:
                raise _fail(op, f"{op.opcode} operand {v!r} has no definition")
            elif id(v) not in seen:
                raise _fail(
                    op,
                    f"{op.opcode} uses {v!r} before it is defined. This normally "
                    f"means a value escaped a loop body -- carry it with "
                    f"hk.range(..., init=[...]) instead.",
                )
        for r in op.results:
            seen.add(id(r))

    if ir.warps < 1:
        raise VerifyError(f"warps={ir.warps}: a kernel needs at least one warp")

    tgt = get_target(ir.arch)
    max_warps = tgt.max_waves_per_simd * tgt.simds_per_lds_pool
    if ir.warps > max_warps:
        raise VerifyError(
            f"warps={ir.warps} exceeds {max_warps}, the most {tgt.arch} can put "
            f"in one workgroup ({tgt.max_waves_per_simd} waves/SIMD x "
            f"{tgt.simds_per_lds_pool} SIMDs)"
        )


# ---------------------------------------------------------------- layout


def _flat(ir: KernelIR) -> List:
    """Every op in the kernel, nested ones included, in program order.

    Regions are flattened rather than skipped. For the register estimate that
    is an approximation -- a loop body's live set is scanned once, as if it ran
    once -- but the alternative in force until the GEMM arrived was to not look
    inside loops at all, which on a kernel whose entire body is one K loop
    meant estimating zero.
    """
    return [inner for op in ir.body for inner in op.walk()]


def check_layouts(ir: KernelIR) -> List[Warning_]:
    """The one that pays for itself: a col-layout 16-bit operand read out of LDS
    costs 8x the instructions of a row-layout one, and nothing about the result
    says so."""
    tgt = get_target(ir.arch)
    warns: List[Warning_] = []

    for op in _flat(ir):
        if op.opcode != "load_shared":
            continue
        d = op.dst
        if d is None:
            continue
        dst = d.type
        if not isinstance(dst, RegTileType):
            continue
        if dst.dtype.bits == 16 and dst.layout not in tgt.fast_smem_layouts:
            warns.append(
                Warning_(
                    "slow-smem-layout",
                    f"loading {dst} from LDS: a {dst.dtype} operand in "
                    f"'{dst.layout}' layout does not vectorise. Expect 16 scalar "
                    f"ds_read_u16 per base tile instead of 2 ds_read_b128 -- 8x "
                    f"the instructions. Fast layouts on {tgt.arch}: "
                    f"{', '.join(tgt.fast_smem_layouts)}. If you need the "
                    f"transpose, stage it transposed rather than loading and "
                    f"swapping: swap_layout moves data on this architecture, it "
                    f"is not a relabel.",
                    op.loc or "",
                )
            )
    return warns


# ---------------------------------------------------------------- budgets


def estimate_vgprs(ir: KernelIR) -> int:
    """Peak live register-tile VGPRs, by linear scan.

    An estimate and nothing more: it counts tile storage only, so it misses
    addresses, loop induction variables, prefetch buffers and anything the
    compiler decides to keep. Treat it as a lower bound -- useful to reject a
    tiling that cannot possibly fit, useless to certify one that can.
    """
    tgt = get_target(ir.arch)
    ops = _flat(ir)
    last_use = {}
    for i, op in enumerate(ops):
        for v in op.operands:
            last_use[id(v)] = i

    live = {}
    peak = 0
    for i, op in enumerate(ops):
        for r in op.results:
            if isinstance(r.type, RegTileType):
                live[id(r)] = tgt.tile_vgprs(r.type.dtype.bits, r.type.rows, r.type.cols)
        peak = max(peak, sum(live.values()))
        for v in op.operands:
            if last_use.get(id(v)) == i:
                live.pop(id(v), None)
    return peak


def check_budgets(ir: KernelIR) -> List[Warning_]:
    tgt = get_target(ir.arch)
    warns: List[Warning_] = []

    if ir.lds_bytes > tgt.lds_per_workgroup:
        raise VerifyError(
            f"LDS request is {ir.lds_bytes} B but one workgroup on {tgt.arch} "
            f"gets at most {tgt.lds_per_workgroup} B. Shrink a staged tile, drop "
            f"double buffering, or move a tile back into registers."
        )

    est = estimate_vgprs(ir)
    budget = tgt.vgprs_per_simd // tgt.max_waves_per_simd * tgt.max_waves_per_simd
    if est > tgt.vgprs_per_simd:
        msg = (
            f"register tiles alone need an estimated {est} VGPRs, more than the "
            f"{tgt.vgprs_per_simd} a SIMD has. This cannot be scheduled."
        )
        if tgt.verified:
            raise VerifyError(msg)
        warns.append(Warning_("vgpr-budget", msg + " (target unverified)"))
    elif est > 256:
        warns.append(
            Warning_(
                "vgpr-budget",
                f"estimated {est} VGPRs of live register tiles "
                f"(occupancy {tgt.occupancy(est)} waves/SIMD, headroom "
                f"{tgt.vgpr_headroom(est)}). This is an estimate that counts tile "
                f"storage only; the real number comes from the compiler after "
                f"codegen. If it spills, the kernel is wrong, not slow.",
            )
        )

    occ_lds = tgt.occupancy_by_lds(ir.lds_bytes, ir.warps) if ir.lds_bytes else None
    occ_reg = tgt.occupancy(est) if est else None
    if occ_lds is not None and occ_reg is not None and occ_lds < occ_reg:
        warns.append(
            Warning_(
                "lds-bound",
                f"LDS binds occupancy below registers: {occ_lds} waves/SIMD from "
                f"{ir.lds_bytes} B of LDS vs {occ_reg} from registers. The "
                f"compiler's occupancy remark reports the register number only "
                f"and will look better than what you get.",
            )
        )
    return warns


# ---------------------------------------------------------------- entry


def verify(ir: KernelIR) -> List[Warning_]:
    """Run every check. Raises on the first error; returns the warnings."""
    check_structure(ir)
    return check_layouts(ir) + check_budgets(ir)
