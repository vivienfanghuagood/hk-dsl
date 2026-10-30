"""Run a traced kernel on the CPU with torch, one program at a time: ``HK2_INTERPRET=1`` or
``kernel[grid](..., interpret=True)``.

The same tile IR the compiler sees, executed with element-wise torch semantics: a debugging aid
(``hk2.print`` shows values here) and a differential oracle for the compiler (tests compare both).
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Tuple

import torch

from . import ir
from . import types as T

_TORCH = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "int32": torch.int32,
          "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "int1": torch.bool}


def tdt(dt: T.DType):
    return _TORCH[dt.name]


class _Tensor:
    """A kernel argument: the CPU copy that loads read and stores write."""

    def __init__(self, t: torch.Tensor):
        self.dev = t
        self.cpu = t.detach().to("cpu", copy=True)


def _binary(fn, a, b, dt: T.DType):
    if fn == "add":
        r = a + b
    elif fn == "sub":
        r = a - b
    elif fn == "mul":
        r = a * b
    elif fn == "truediv":
        r = a / b
    elif fn == "floordiv":
        r = torch.div(a, b, rounding_mode="trunc") if dt.is_int else torch.floor(a / b)
    elif fn == "mod":
        r = torch.fmod(a, b)
    elif fn == "and":
        r = a & b
    elif fn == "or":
        r = a | b
    elif fn == "xor":
        r = a ^ b
    elif fn == "shl":
        r = a << b
    elif fn == "shr":
        r = a >> b
    elif fn == "max":
        r = torch.maximum(a, b) if not dt.is_float else torch.fmax(a, b)
    elif fn == "min":
        r = torch.minimum(a, b) if not dt.is_float else torch.fmin(a, b)
    elif fn in ("lt", "le", "gt", "ge", "eq", "ne"):
        r = getattr(torch, fn)(a, b)
    else:
        raise NotImplementedError(fn)
    return r


def _unary(fn, x):
    return {"exp": torch.exp, "exp2": torch.exp2, "log": torch.log, "log2": torch.log2, "sqrt": torch.sqrt,
            "rsqrt": torch.rsqrt, "abs": torch.abs}[fn](x)


class Interpreter:
    def __init__(self, top: ir.Block, binding: Dict[int, Any], printer=print):
        self.top, self.binding, self.printer = top, binding, printer

    def run(self, grid: Tuple[int, int, int]):
        for z, y, x in itertools.product(range(grid[2]), range(grid[1]), range(grid[0])):
            env = dict(self.binding)
            self.pid = (x, y, z)
            self.block(self.top, env)

    def block(self, blk: ir.Block, env: Dict[int, Any]):
        for op in blk.ops:
            if op.kind == "yield":
                return [env[a.id] for a in op.args]
            self.op(op, env)
        return []

    def _s(self, env, v):
        x = env[v.id]
        return x.item() if isinstance(x, torch.Tensor) and x.dim() == 0 else x

    def op(self, op: ir.Op, env):
        k = op.kind
        res = op.results[0] if op.results else None
        rdt = res.type.dtype if res is not None and hasattr(res.type, "dtype") else None
        a = [env.get(x.id) for x in op.args]
        if k == "const":
            env[res.id] = torch.tensor(op.attrs["value"], dtype=tdt(rdt))
        elif k == "program_id":
            env[res.id] = torch.tensor(self.pid[op.attrs["axis"]], dtype=torch.int32)
        elif k in ("sop", "ew") and len(a) == 2:
            dt = op.args[0].type.dtype
            r = _binary(op.attrs["fn"], a[0], a[1], dt)
            env[res.id] = r.to(tdt(rdt)) if r.dtype != tdt(rdt) else r
        elif k in ("sunary", "ew"):
            x = a[0]
            env[res.id] = _unary(op.attrs["fn"], x.float() if rdt.is_float else x).to(tdt(rdt))
        elif k in ("scast", "cast"):
            x = a[0]
            env[res.id] = (x != 0) if rdt == T.i1 else x.to(tdt(rdt))
        elif k == "full":
            env[res.id] = torch.full(res.type.shape, op.attrs["value"], dtype=tdt(rdt))
        elif k == "splat":
            env[res.id] = torch.full(res.type.shape, 0, dtype=tdt(rdt)) + a[0].to(tdt(rdt))
        elif k == "arange":
            env[res.id] = torch.arange(res.type.shape[0], dtype=torch.int32)
        elif k == "broadcast":
            env[res.id] = a[0].expand(res.type.shape).clone()
        elif k == "expand_dims":
            env[res.id] = a[0].unsqueeze(op.attrs["axis"])
        elif k == "trans":
            env[res.id] = a[0].t().contiguous()
        elif k == "where":
            env[res.id] = torch.where(a[0], a[1], a[2])
        elif k == "reduce":
            x, ax, fn = a[0], op.attrs["axis"], op.attrs["fn"]
            r = x.sum(ax, dtype=x.dtype) if fn == "add" else (x.amax(ax) if fn == "max" else x.amin(ax))
            env[res.id] = r
        elif k == "dot":
            env[res.id] = a[0].float() @ a[1].float() + a[2].float()
        elif k == "load":
            env[res.id] = self.load(op, a, res)
        elif k == "store":
            self.store(op, a)
        elif k == "print":
            vals = [env[x.id] for x in op.args]
            self.printer(f"[hk2 pid {self.pid}] {op.attrs['msg']}", *vals)
        elif k == "for":
            lb, ub, step = (int(self._s(env, v)) for v in op.args[:3])
            carried = [env[v.id] for v in op.args[3:]]
            blk = op.blocks[0]
            for i in range(lb, ub, step) if step > 0 else range(lb, ub, step):
                env[blk.params[0].id] = torch.tensor(i, dtype=torch.int32)
                for p, c in zip(blk.params[1:], carried):
                    env[p.id] = c
                carried = self.block(blk, env)
            for r, c in zip(op.results, carried):
                env[r.id] = c
        elif k == "if":
            blk = op.blocks[0] if bool(self._s(env, op.args[0])) else op.blocks[1]
            for r, c in zip(op.results, self.block(blk, env)):
                env[r.id] = c
        else:
            raise NotImplementedError(f"hk2 interpreter: {k}")

    def _view(self, op, a):
        nd = op.attrs["ndim"]
        tensor = a[0]
        offs = [int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in a[1 + 2 * nd:1 + 3 * nd]]
        return tensor, offs

    def _index(self, tensor: _Tensor, offs, shape):
        t = tensor.cpu
        nd = t.dim()
        lead = nd - len(shape)
        valid_lead = all(0 <= o < s for o, s in zip(offs[:lead], t.shape[:lead]))
        sub = t[tuple(offs[:lead])] if lead and valid_lead else (t if not lead else None)
        grids = torch.meshgrid(*[torch.arange(n) + o for n, o in zip(shape, offs[lead:])], indexing="ij")
        if sub is None:
            return None, None, grids, torch.zeros(shape, dtype=torch.bool)
        mask = torch.ones(shape, dtype=torch.bool)
        for g, s in zip(grids, sub.shape):
            mask &= (g >= 0) & (g < s)
        clamped = [g.clamp(0, max(s - 1, 0)) for g, s in zip(grids, sub.shape)]
        return sub, clamped, grids, mask

    def load(self, op, a, res):
        tensor, offs = self._view(op, a)
        shape = res.type.shape
        sub, idx, _, mask = self._index(tensor, offs, shape)
        other = torch.full(shape, op.attrs["other"], dtype=tdt(res.type.dtype))
        if sub is None:
            return other
        vals = sub[tuple(idx)] if sub.numel() else other
        return torch.where(mask, vals.to(other.dtype), other)

    def store(self, op, a):
        tensor, offs = self._view(op, a)
        val = a[-1]
        sub, idx, _, mask = self._index(tensor, offs, tuple(val.shape))
        if sub is None:
            return
        sel = tuple(i[mask] for i in idx)
        sub[sel] = val[mask].to(sub.dtype)


def bind(params: List[Tuple[ir.Value, str, str]], packing, args: Dict[str, Any]):
    """Kernel parameter values -> interpreter bindings; returns (binding, tensors to copy back)."""
    binding, tensors = {}, {}
    for (val, _, _), (kind, src) in zip(params, packing):
        x = args[src[0]]
        if kind == "p":
            tensors.setdefault(src[0], _Tensor(x))
            binding[val.id] = tensors[src[0]]
        elif len(src) == 3:
            binding[val.id] = torch.tensor(x.shape[src[2]] if src[1] == "shape" else x.stride(src[2]),
                                           dtype=torch.int32)
        elif kind == "i":
            binding[val.id] = torch.tensor(int(x), dtype=torch.int32)
        else:
            binding[val.id] = torch.tensor(float(x), dtype=torch.float32)
    return binding, tensors


def run(top, params, packing, args, grid):
    binding, tensors = bind(params, packing, args)
    Interpreter(top, binding).run(grid)
    for t in tensors.values():
        t.dev.copy_(t.cpu.to(t.dev.device))
