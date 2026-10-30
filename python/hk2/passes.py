"""IR-to-IR passes that run before layouts are chosen."""
from __future__ import annotations

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


def pipeline_loads(block: ir.Block) -> int:
    """Software-pipeline the loads of every loop that feeds the matrix cores: the loads for iteration i+1 are
    issued at the top of iteration i and carried in registers, so their latency overlaps this iteration's
    work. Returns the number of loads moved."""
    moved = 0
    for idx in range(len(block.ops)):
        op = block.ops[idx]
        for b in op.blocks:
            moved += pipeline_loads(b)
    i = 0
    while i < len(block.ops):
        op = block.ops[i]
        if op.kind == "for" and _has(op.blocks[0], "dot"):
            n, inserted = _pipeline_for(block, i)
            moved += n
            i += 1 + inserted
            continue
        i += 1
    return moved


def _pipeline_for(parent: ir.Block, at: int):
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
    # i+1 in the body, computed first
    nxt = ir.Op("sop", [iv, step], [Value(ScalarType(T.i32))], {"fn": "add"})
    head: List[ir.Op] = [nxt]
    pro: List[ir.Op] = []
    yields = body.ops[-1]
    for ld in cands:
        c0 = _Cloner(body, local, defs, iv, lb)
        args0 = [c0.get(a) for a in ld.args]
        l0 = ir.Op("load", args0, [Value(ld.result.type, ld.result.loc)], dict(ld.attrs), [], ld.loc)
        pro += c0.new_ops + [l0]
        c1 = _Cloner(body, local, defs, iv, nxt.result)
        args1 = [c1.get(a) for a in ld.args]
        # "prefetch": the emitter issues it right after this iteration's tiles are written to LDS, so the
        # registers holding them are free to receive the next ones (no copies across the back edge)
        l1 = ir.Op("load", args1, [Value(ld.result.type, ld.result.loc)], dict(ld.attrs, prefetch=True), [],
                   ld.loc)
        head += c1.new_ops + [l1]
        p = Value(ld.result.type, ld.result.loc)
        body.params.append(p)
        fop.args.append(l0.result)
        fop.results.append(Value(ld.result.type, ld.result.loc))
        body.ops.remove(ld)
        _replace_uses(body, ld.result, p)
        yields.args.append(l1.result)
    body.ops[0:0] = head
    parent.ops[at:at] = pro
    return len(cands), len(pro)
