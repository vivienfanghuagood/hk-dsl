"""IR-to-IR passes that run before layouts are chosen."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

from . import ir
from . import types as T
from .ir import ScalarType, Value

__all__ = ["pipeline_loads"]

_PURE_SCALAR = {"sop", "scast", "const"}


def _walk(block: ir.Block):
    for op in block.ops:
        yield op
        for b in op.blocks:
            yield from _walk(b)


def _defined_in(block: ir.Block) -> Set[int]:
    out = {p.id for p in block.params}
    for op in _walk(block):
        out |= {r.id for r in op.results}
        for b in op.blocks:
            out |= {p.id for p in b.params}
    return out


def _has(block: ir.Block, kind: str) -> bool:
    return any(op.kind == kind for op in _walk(block))


def _replace_uses(block: ir.Block, old: Value, new: Value):
    for op in _walk(block):
        op.args = [new if a is old else a for a in op.args]


class _Cloner:
    """Clone the scalar slice computing a value, with the loop index mapped to another value."""

    def __init__(self, body: ir.Block, local: Set[int], defs: Dict[int, ir.Op], iv: Value, iv_to: Value):
        self.body, self.local, self.defs = body, local, defs
        self.map: Dict[int, Value] = {iv.id: iv_to}
        self.new_ops: List[ir.Op] = []

    def ok(self, v: Value) -> bool:
        if v.id in self.map or v.id not in self.local:
            return True
        op = self.defs.get(v.id)
        return op is not None and op.kind in _PURE_SCALAR and all(self.ok(a) for a in op.args)

    def get(self, v: Value) -> Value:
        if v.id in self.map:
            return self.map[v.id]
        if v.id not in self.local:
            return v
        op = self.defs[v.id]
        nop = ir.Op(op.kind, [self.get(a) for a in op.args], [Value(r.type, r.loc) for r in op.results],
                    dict(op.attrs), [], op.loc)
        self.new_ops.append(nop)
        for r, nr in zip(op.results, nop.results):
            self.map[r.id] = nr
        return self.map[v.id]


def _dma_ok(ld: ir.Op, uses, dma) -> bool:
    """A load can go global -> LDS directly (no registers) when its only consumers read it as a dot operand
    and its rows are whole 16-byte chunks. ``dma``: (bytes per lane per instruction, warps), or None."""
    if not dma:
        return False
    t = ld.result.type
    if len(t.shape) != 2 or t.dtype.bits != 16 or ld.attrs.get("vec") != 8 or t.shape[1] < 8:
        return False
    per_inst, nwarps = dma[0], dma[1]
    total = t.shape[0] * t.shape[1] * 2
    if total % (per_inst * 64 * nwarps):
        return False
    if not uses:
        return False
    for u, idx in uses:
        if u.kind == "dot" and idx == 0:
            continue  # A operand straight from its (M, K) storage
        return False
    return True


def pipeline_loads(block: ir.Block, dma=None) -> int:
    """Software-pipeline the loads of every loop that feeds the matrix cores: the loads for iteration i+1 are
    issued during iteration i, so their latency overlaps this iteration's work. With ``dma`` (gfx950), loads
    whose only use is a dot operand go straight to LDS (``dma`` attribute) instead of through registers.
    Returns the number of loads moved."""
    moved = 0
    for idx in range(len(block.ops)):
        op = block.ops[idx]
        for b in op.blocks:
            moved += pipeline_loads(b, dma)
    i = 0
    while i < len(block.ops):
        op = block.ops[i]
        if op.kind == "for" and _has(op.blocks[0], "dot"):
            n, inserted = _pipeline_for(block, i, dma)
            moved += n
            i += 1 + inserted
            continue
        i += 1
    return moved


def _uses(body: ir.Block, v: Value):
    """(op, operand index) of every use of v at the top level of body; None if used deeper."""
    out = []
    for op in body.ops:
        for i, a in enumerate(op.args):
            if a is v:
                out.append((op, i))
        for b in op.blocks:
            if any(a is v for o in _walk(b) for a in o.args):
                return None
    return out


def _const_op(v: int) -> ir.Op:
    return ir.Op("const", [], [Value(ScalarType(T.i32))], {"value": v})


def _pipeline_for(parent: ir.Block, at: int, dma=None):
    fop = parent.ops[at]
    body = fop.blocks[0]
    iv = body.params[0]
    lb, step = fop.args[0], fop.args[2]
    local = _defined_in(body)
    defs = {r.id: o for o in body.ops for r in o.results}
    check = _Cloner(body, local, defs, iv, iv)
    cands = [o for o in body.ops if o.kind == "load" and all(check.ok(a) for a in o.args)]
    if not cands:
        return 0, 0
    # which loads go global -> LDS directly (B operands are read through .T of an (N, K) tile)
    is_dma = {}
    for ld in cands:
        uses = _uses(body, ld.result)
        direct = [] if uses is not None else None
        for u, idx in (uses or []):
            if u.kind == "trans":
                tu = _uses(body, u.result)
                if tu is None or not all(x.kind == "dot" and j == 1 for x, j in tu):
                    direct = None
                    break
                direct += [(x, 0) for x, j in tu]
            elif direct is not None:
                direct.append((u, idx))
        is_dma[id(ld)] = direct is not None and _dma_ok(ld, direct, dma)
    # DMA tiles sit in a ring of S stages: tiles i+1 .. i+S-1 are in flight while tile i is used
    stages = 2
    per_stage = sum(ld.result.type.shape[0] * ld.result.type.shape[1] * ld.result.type.dtype.bytes
                    for ld in cands if is_dma[id(ld)])
    if per_stage:
        budget = dma[2] * 3 // 4
        stages = max(2, min(dma[3] if len(dma) > 3 else 4, budget // per_stage))
        if os.environ.get("HK2_STAGES"):
            stages = int(os.environ["HK2_STAGES"])
        fop.attrs["stages"] = stages
        fop.attrs["dma_insts"] = per_stage // (dma[0] * 64 * dma[1])
    head: List[ir.Op] = []
    pro: List[ir.Op] = []

    def at_iv(dist, where):
        """Ops computing iv + dist*step (body) or lb + dist*step (prologue)."""
        base = iv if where == "body" else lb
        if dist == 0:
            return [], base
        c = _const_op(dist)
        m = ir.Op("sop", [step, c.result], [Value(ScalarType(T.i32))], {"fn": "mul"})
        a = ir.Op("sop", [base, m.result], [Value(ScalarType(T.i32))], {"fn": "add"})
        return [c, m, a], a.result

    nxt_ops, nxt = at_iv(1, "body")
    head += nxt_ops
    yields = body.ops[-1]
    for ld in cands:
        dma_tag = f"dma{id(fop)}_{len(body.params) + 1}" if is_dma[id(ld)] else None
        n_pro = stages - 1 if dma_tag else 1
        l0 = None
        for st in range(n_pro):
            ops, at0 = at_iv(st, "pro")
            c0 = _Cloner(body, local, defs, iv, at0)
            args0 = [c0.get(a) for a in ld.args]
            attrs = dict(ld.attrs)
            if dma_tag:
                attrs.update(dma=dma_tag, stage=st, stages=stages)
            li = ir.Op("load", args0, [Value(ld.result.type, ld.result.loc)], attrs, [], ld.loc)
            pro += ops + c0.new_ops + [li]
            l0 = l0 or li
        if dma_tag and stages > 2:
            ops, at1 = at_iv(stages - 1, "body")
            head += ops
        else:
            at1 = nxt
        c1 = _Cloner(body, local, defs, iv, at1)
        args1 = [c1.get(a) for a in ld.args]
        # "prefetch": the emitter issues it right after this iteration's tiles are written to (or read from)
        # LDS, so the registers or LDS slot it fills are free (no copies across the back edge)
        attrs = dict(ld.attrs, prefetch=True)
        if dma_tag:
            attrs.update(dma=dma_tag, dist=stages - 1, stages=stages)
        l1 = ir.Op("load", args1, [Value(ld.result.type, ld.result.loc)], attrs, [], ld.loc)
        head += c1.new_ops + [l1]
        p = Value(ld.result.type, ld.result.loc)
        body.params.append(p)
        fop.args.append(l0.result)
        fop.results.append(Value(ld.result.type, ld.result.loc))
        if dma_tag:
            fop.attrs.setdefault("dma_params", {})[len(body.params) - 2] = dma_tag  # index among carried
        body.ops.remove(ld)
        _replace_uses(body, ld.result, p)
        yields.args.append(l1.result)
    body.ops[0:0] = head
    parent.ops[at:at] = pro
    return len(cands), len(pro)
