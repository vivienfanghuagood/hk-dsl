"""Tile IR -> LLVM IR for one architecture.

Two passes:

1. **Layouts.** Each tile value gets a layout (``hk2.layout``). Loads are coalesced, ``dot`` results are
   matrix-core accumulators, reductions keep their input's layout minus the reduced axis. Constants,
   ``arange`` and broadcasts are *flexible*: they are built at each use, directly in the layout that use
   needs, so ``x - m[:, None]`` never moves data.
2. **Emission.** Every tile becomes, per thread, a list of SSA registers. A value needed in another
   layout is converted through LDS; barriers are placed by tracking which LDS bytes were written or read
   since the last one, so a kernel never states a wait or a barrier.
"""
from __future__ import annotations

import itertools
import math
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import arch as A
from . import ir
from . import types as T
from .layout import Layout, blocked, log2

FLEX = "flex"
INF = 1 << 40


def _fhex(v: float, dt: T.DType) -> str:
    """An exact LLVM constant for ``v`` in ``dt``."""
    import numpy as np
    if dt == T.bf16:
        bits = int(np.array([v], dtype=np.float32).view(np.uint32)[0])
        # round to nearest even, as fptrunc does
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 if not math.isnan(v) else 0x7FC0
        return f"0xR{bits & 0xFFFF:04X}"
    if dt == T.f16:
        return f"0xH{int(np.array([v], dtype=np.float16).view(np.uint16)[0]):04X}"
    d = float(np.float32(v))
    return f"0x{struct.unpack('<Q', struct.pack('<d', d))[0]:016X}"


def _const(v, dt: T.DType) -> str:
    if dt.is_float:
        return _fhex(float(v), dt)
    if dt == T.i1:
        return "true" if v else "false"
    return str(int(v))


@dataclass
class TV:
    """A tile as emitted: its layout and one SSA name per register (copies share a name)."""
    layout: Layout
    regs: List[str]
    dtype: T.DType


class Compiler:
    def __init__(self, arch: A.Arch, nwarps: int, name: str, llvm_major: int = 22):
        self.arch, self.nwarps, self.name = arch, nwarps, name
        self.llvm_major = llvm_major
        self.wave = arch.wave
        self.lay: Dict[int, object] = {}
        self.defs: Dict[int, ir.Op] = {}
        self.vals: Dict[int, object] = {}
        self.memo: List[Dict] = [{}]
        self.entry: List[str] = []
        self.entry_memo: Dict = {}
        self.blocks: List[List] = []
        self.decls: set = set()
        self._n = itertools.count()
        self.lds_size = 0
        self.pend_r: List[Tuple] = []
        self.pend_w: List[Tuple] = []
        self.loops: List[Dict] = []
        self.div: Dict[int, int] = {}
        self.ivs: Dict[int, ir.Op] = {}
        self.param_div: Dict[int, int] = {}
        self.n_barriers = 0
        self.deferred: List[Tuple[int, ir.Op]] = []
        self.rowmode = False
        self.depth = 0

    # ================================================================ pass 1: layouts
    def blocked_for(self, shape, dt: T.DType, vec: int = 0) -> Layout:
        v = vec or max(1, 16 // dt.bytes)
        return blocked(shape, self.wave, self.nwarps, min(v, shape[-1]))

    def infer(self, block: ir.Block):
        for op in block.ops:
            for r in op.results:
                self.defs[r.id] = op
            self.infer_op(op)

    @staticmethod
    def _score(l) -> int:
        if l is FLEX:
            return -1
        return 3 if "mma_c" in l.name else 2 if "mma" in l.name else 1

    def infer_op(self, op: ir.Op):
        k, L = op.kind, self.lay
        if not op.results or not op.results[0].is_tile:
            if k == "for":
                self._infer_for(op)
            elif k == "if":
                self._infer_if(op)
            return
        r = op.result
        sh, dt = r.type.shape, r.type.dtype
        if k in ("full", "splat", "arange", "broadcast"):
            L[r.id] = FLEX
        elif k == "expand_dims":
            x = L[op.args[0].id]
            L[r.id] = FLEX if x is FLEX else x.expand(op.attrs["axis"])
        elif k == "trans":
            x = L[op.args[0].id]
            L[r.id] = FLEX if x is FLEX else x.transpose()
        elif k == "load":
            L[r.id] = self.blocked_for(sh, dt)
        elif k in ("ew", "where", "cast"):
            cands = [L[a.id] for a in op.args if a.is_tile and a.type.shape == sh]
            best = max(cands, key=self._score) if cands else FLEX
            L[r.id] = best
        elif k == "dot":
            atom, grid = self.dot_plan(op.args[0].type.dtype, sh[0], sh[1])
            L[r.id] = A.acc_layout(atom, sh[0], sh[1], self.nwarps, grid)
        elif k == "reduce":
            x = op.args[0]
            xl = L[x.id]
            if xl is FLEX:
                xl = self.blocked_for(x.type.shape, x.type.dtype)
            L[r.id] = xl.slice(op.attrs["axis"])
        elif k in ("for", "if"):
            (self._infer_for if k == "for" else self._infer_if)(op)
        else:
            raise NotImplementedError(k)

    def _default(self, v: ir.Value, l):
        if l is FLEX:
            return self.blocked_for(v.type.shape, v.type.dtype)
        return l

    def _infer_for(self, op: ir.Op):
        blk = op.blocks[0]
        self.ivs[blk.params[0].id] = op
        inits = op.args[3:]
        params = blk.params[1:]
        # A carried tile starting as a constant (a running max, a zero accumulator) takes the layout the body
        # gives it: iterate with such parameters left flexible until the yields stop changing.
        guess = [self.lay.get(i.id) if i.is_tile else None for i in inits]
        for _ in range(8):
            for p, g in zip(params, guess):
                if p.is_tile:
                    self.lay[p.id] = g
            self.infer(blk)
            new = []
            for p, g, y in zip(params, guess, blk.ops[-1].args):
                yl = self.lay.get(y.id) if p.is_tile else None
                new.append(None if not p.is_tile else (yl if yl is not FLEX else g))
            if new == guess:
                break
            guess = new
        for p, g in zip(params, guess):
            if p.is_tile:
                self.lay[p.id] = self._default(p, g)
        self.infer(blk)
        for p, res in zip(params, op.results):
            if p.is_tile:
                self.lay[res.id] = self.lay[p.id]

    def _infer_if(self, op: ir.Op):
        for blk in op.blocks:
            self.infer(blk)
        for j, res in enumerate(op.results):
            if not res.is_tile:
                continue
            a = self.lay[op.blocks[0].ops[-1].args[j].id]
            b = self.lay[op.blocks[1].ops[-1].args[j].id]
            pick = a if a is not FLEX else b
            self.lay[res.id] = self._default(res, pick)

    # ================================================================ matrix-core plans
    def dot_plan(self, dtype: T.DType, m: int, n: int):
        """The atom and warp grid of a dot. When a kernel reduces dot results along rows (softmax,
        attention), every dot gives each warp whole rows and uses one atom shape, so row statistics stay in
        a warp and meet the other dots' rows without moving."""
        if self.rowmode:
            atom = self.arch.mma(dtype)
            return atom, A.rows_grid(m, n, self.nwarps, atom)
        atom = self.arch.mma(dtype, m, n, self.nwarps)
        return atom, A.warp_grid(m, n, self.nwarps, atom)

    def _detect_rowmode(self, block: ir.Block) -> bool:
        producer = {}
        reduces = []

        def walk(b):
            for op in b.ops:
                for r in op.results:
                    producer[r.id] = op
                if op.kind == "reduce" and op.attrs["axis"] == len(op.args[0].type.shape) - 1:
                    reduces.append(op)
                for bb in op.blocks:
                    walk(bb)
        walk(block)

        def from_dot(v, depth=0):
            op = producer.get(v.id)
            if op is None or depth > 16:
                return False
            if op.kind == "dot":
                return True
            if op.kind in ("ew", "cast", "where"):
                return any(a.is_tile and from_dot(a, depth + 1) for a in op.args)
            return False
        return any(from_dot(r.args[0]) for r in reduces)

    # ================================================================ divisibility of scalars
    def divisibility(self, v: ir.Value) -> int:
        """The largest power of two (up to 1024) known to divide an i32 scalar."""
        if v.id in self.div:
            return self.div[v.id]
        op = self.defs.get(v.id)
        d = 1
        if op is not None:
            if op.kind == "const" and isinstance(op.attrs["value"], int):
                c = abs(op.attrs["value"])
                d = 1024 if c == 0 else min(1024, c & -c)
            elif op.kind == "sop":
                a, b = (self.divisibility(x) for x in op.args)
                fn = op.attrs["fn"]
                if fn == "mul":
                    d = min(1024, a * b)
                elif fn in ("add", "sub"):
                    d = math.gcd(a, b)
                elif fn == "shl" and self.defs.get(op.args[1].id) and self.defs[op.args[1].id].kind == "const":
                    d = min(1024, a << self.defs[op.args[1].id].attrs["value"])
            elif op.kind == "scast":
                d = self.divisibility(op.args[0]) if op.args[0].type.dtype.is_int else 1
        elif v.id in self.ivs:
            f = self.ivs[v.id]
            d = math.gcd(self.divisibility(f.args[0]), self.divisibility(f.args[2]))
        else:
            d = self.param_div.get(v.id, 1)
        self.div[v.id] = d
        return d

    # ================================================================ emission helpers
    def t(self) -> str:
        return f"%v{next(self._n)}"

    def emit(self, line: str):
        self.blocks[-1][1].append(line)

    def label(self, prefix="b") -> str:
        return f"{prefix}{next(self._n)}"

    def start(self, label: str):
        self.blocks.append([label, []])

    @property
    def cur(self) -> str:
        return self.blocks[-1][0]

    def decl(self, s: str):
        self.decls.add(s)

    def e(self, rhs: str) -> str:
        n = self.t()
        self.emit(f"  {n} = {rhs}")
        return n

    def en(self, key, rhs_fn) -> str:
        """An entry-block value, computed once."""
        if key not in self.entry_memo:
            n = self.t()
            self.entry.append(f"  {n} = {rhs_fn()}")
            self.entry_memo[key] = n
        return self.entry_memo[key]

    def lane(self) -> str:
        return self.en("lane", lambda: f"and i32 {self.tid()}, {self.wave - 1}")

    def tid(self) -> str:
        self.decl("declare i32 @llvm.amdgcn.workitem.id.x()")
        return self.en("tid", lambda: "call i32 @llvm.amdgcn.workitem.id.x()")

    def warp(self) -> str:
        self.decl("declare i32 @llvm.amdgcn.readfirstlane.i32(i32)")
        w0 = self.en("warp0", lambda: f"lshr i32 {self.tid()}, {log2(self.wave)}")
        return self.en("warp", lambda: f"call i32 @llvm.amdgcn.readfirstlane.i32(i32 {w0})")

    def _bits(self, src: str, bases, dim: int, tag) -> str:
        """sum over input bits b of ((src >> b) & 1) * bases[b][dim], in the entry block."""
        terms = []
        for b, basis in enumerate(bases):
            if basis[dim]:
                s = log2(basis[dim])
                x = self.en((tag, src, b, "bit"), lambda b=b: f"lshr i32 {src}, {b}")
                y = self.en((tag, src, b, "and"), lambda x=x: f"and i32 {x}, 1")
                terms.append(self.en((tag, src, b, s, "shl"), lambda y=y, s=s: f"shl i32 {y}, {s}"))
        if not terms:
            return "0"
        acc = terms[0]
        for i, x in enumerate(terms[1:]):
            acc = self.en((tag, src, dim, tuple(map(str, bases)), i), lambda a=acc, x=x: f"or i32 {a}, {x}")
        return acc

    def lw(self, L: Layout, dim: int) -> str:
        """The (lane, warp) part of this thread's coordinate along ``dim`` in layout L."""
        def calc():
            a = self._bits(self.lane(), L.lane, dim, "l")
            w = self._bits(self.warp(), L.warp, dim, "w")
            if a == "0":
                return f"add i32 {w}, 0"
            if w == "0":
                return f"add i32 {a}, 0"
            return f"or i32 {a}, {w}"
        return self.en(("lw", L, dim), calc)

    def owner(self, L: Layout) -> Optional[str]:
        """i1: this thread holds the first copy of its elements (None when nothing is copied)."""
        conds = [("lane", b) for b in L.dup_lane_bits()] + [("warp", b) for b in L.dup_warp_bits()]
        if not conds:
            return None
        mask_l = sum(1 << b for k, b in conds if k == "lane")
        mask_w = sum(1 << b for k, b in conds if k == "warp")

        def calc():
            parts = []
            if mask_l:
                x = self.en(("ownl", mask_l), lambda: f"and i32 {self.lane()}, {mask_l}")
                parts.append(x)
            if mask_w:
                x = self.en(("ownw", mask_w), lambda: f"and i32 {self.warp()}, {mask_w}")
                parts.append(x)
            s = parts[0] if len(parts) == 1 else self.en(("ownor", mask_l, mask_w),
                                                         lambda: f"or i32 {parts[0]}, {parts[1]}")
            return f"icmp eq i32 {s}, 0"
        return self.en(("owner", mask_l, mask_w), calc)

    # ---------------------------------------------------------------- scalar ops
    def ty(self, dt: T.DType) -> str:
        return dt.llvm

    def binop(self, fn: str, a: str, b: str, dt: T.DType) -> str:
        key = ("bin", fn, a, b, dt)
        m = self.memo[-1].get(key)
        if m:
            return m
        t = self.ty(dt)
        if dt.is_float:
            ops = {"add": "fadd", "sub": "fsub", "mul": "fmul", "truediv": "fdiv", "mod": "frem"}
            if fn in ops:
                r = self.e(f"{ops[fn]} {t} {a}, {b}")
            elif fn in ("max", "min"):
                name = f"llvm.{'maxnum' if fn == 'max' else 'minnum'}.{self._fsuf(dt)}"
                self.decl(f"declare {t} @{name}({t}, {t})")
                r = self.e(f"call {t} @{name}({t} {a}, {t} {b})")
            elif fn in ("lt", "le", "gt", "ge", "eq", "ne"):
                r = self.e(f"fcmp {dict(lt='olt', le='ole', gt='ogt', ge='oge', eq='oeq', ne='une')[fn]} {t} {a}, {b}")
            elif fn == "floordiv":
                q = self.e(f"fdiv {t} {a}, {b}")
                name = f"llvm.floor.{self._fsuf(dt)}"
                self.decl(f"declare {t} @{name}({t})")
                r = self.e(f"call {t} @{name}({t} {q})")
            else:
                raise NotImplementedError(f"{fn} on floats")
        else:
            u = dt.kind == "u"
            ops = {"add": "add", "sub": "sub", "mul": "mul", "floordiv": "udiv" if u else "sdiv",
                   "mod": "urem" if u else "srem", "and": "and", "or": "or", "xor": "xor", "shl": "shl",
                   "shr": "lshr" if u else "ashr"}
            if fn in ops:
                r = self.e(f"{ops[fn]} {t} {a}, {b}")
            elif fn in ("max", "min"):
                name = f"llvm.{('u' if u else 's') + fn}.{t}"
                self.decl(f"declare {t} @{name}({t}, {t})")
                r = self.e(f"call {t} @{name}({t} {a}, {t} {b})")
            elif fn in ("lt", "le", "gt", "ge", "eq", "ne"):
                p = {"lt": "lt", "le": "le", "gt": "gt", "ge": "ge"}.get(fn)
                pred = fn if fn in ("eq", "ne") else ("u" if u else "s") + p
                r = self.e(f"icmp {pred} {t} {a}, {b}")
            else:
                raise NotImplementedError(f"{fn} on integers")
        self.memo[-1][key] = r
        return r

    @staticmethod
    def _fsuf(dt: T.DType) -> str:
        return {"float": "f32", "half": "f16", "bfloat": "bf16"}[dt.llvm]

    def cast(self, x: str, src: T.DType, dst: T.DType) -> str:
        if src == dst:
            return x
        key = ("cast", x, src, dst)
        m = self.memo[-1].get(key)
        if m:
            return m
        s, d = self.ty(src), self.ty(dst)
        if src.is_float and dst.is_float:
            r = self.e(f"{'fpext' if dst.bits > src.bits else 'fptrunc'} {s} {x} to {d}") if src.bits != dst.bits \
                else self.cast(self.cast(x, src, T.f32), T.f32, dst)
        elif src.is_float:
            r = self.e(f"{'fptoui' if dst.kind == 'u' else 'fptosi'} {s} {x} to {d}") if dst != T.i1 else \
                self.e(f"fcmp une {s} {x}, {_const(0.0, src)}")
        elif dst.is_float:
            r = self.e(f"{'uitofp' if src.kind in 'ub' else 'sitofp'} {s} {x} to {d}")
        elif dst == T.i1:
            r = self.e(f"icmp ne {s} {x}, 0")
        elif dst.bits > src.bits:
            r = self.e(f"{'zext' if src.kind in 'ub' else 'sext'} {s} {x} to {d}")
        elif dst.bits < src.bits:
            r = self.e(f"trunc {s} {x} to {d}")
        else:
            r = x
        self.memo[-1][key] = r
        return r

    def unary(self, fn: str, x: str, dt: T.DType) -> str:
        key = ("un", fn, x, dt)
        m = self.memo[-1].get(key)
        if m:
            return m
        t, suf = self.ty(dt), (self._fsuf(dt) if dt.is_float else dt.llvm)
        if fn == "exp":
            y = self.e(f"fmul {t} {x}, {_fhex(1.4426950408889634, dt)}")
            r = self._call1("exp2", y, dt)
        elif fn == "log":
            r = self.e(f"fmul {t} {self._call1('log2', x, dt)}, {_fhex(0.6931471805599453, dt)}")
        elif fn in ("exp2", "log2", "sqrt"):
            r = self._call1(fn, x, dt)
        elif fn == "rsqrt":
            r = self.e(f"fdiv {t} {_fhex(1.0, dt)}, {self._call1('sqrt', x, dt)}")
        elif fn == "abs":
            if dt.is_float:
                r = self._call1("fabs", x, dt)
            else:
                self.decl(f"declare {t} @llvm.abs.{suf}({t}, i1)")
                r = self.e(f"call {t} @llvm.abs.{suf}({t} {x}, i1 false)")
        else:
            raise NotImplementedError(fn)
        self.memo[-1][key] = r
        return r

    def _call1(self, name: str, x: str, dt: T.DType) -> str:
        t = self.ty(dt)
        if name in ("exp2", "log2") and dt == T.f32 and not os.environ.get("HK2_PRECISE_EXP"):
            # the hardware instruction itself (v_exp_f32 / v_log_f32): results below 2^-126 flush to zero,
            # which is what softmax-style code wants; LLVM's generic form adds range reduction around it
            full = f"llvm.amdgcn.{name}.f32"
            self.decl(f"declare float @{full}(float)")
            return self.e(f"call float @{full}(float {x})")
        full = f"llvm.{name}.{self._fsuf(dt)}"
        self.decl(f"declare {t} @{full}({t})")
        return self.e(f"call afn {t} @{full}({t} {x})") if name in ("exp2", "log2") else \
            self.e(f"call {t} @{full}({t} {x})")

    def select(self, c: str, a: str, b: str, dt: T.DType) -> str:
        key = ("sel", c, a, b)
        m = self.memo[-1].get(key)
        if m:
            return m
        r = self.e(f"select i1 {c}, {self.ty(dt)} {a}, {self.ty(dt)} {b}")
        self.memo[-1][key] = r
        return r

    # ---------------------------------------------------------------- vectors
    def pack(self, regs: Sequence[str], dt: T.DType, as_ty: Optional[str] = None) -> str:
        n = len(regs)
        vt = f"<{n} x {self.ty(dt)}>"
        v = "poison"
        for i, r in enumerate(regs):
            v = self.e(f"insertelement {vt} {v}, {self.ty(dt)} {r}, i32 {i}")
        if as_ty and as_ty != vt:
            v = self.e(f"bitcast {vt} {v} to {as_ty}")
        return v

    def unpack(self, v: str, n: int, dt: T.DType, from_ty: Optional[str] = None) -> List[str]:
        vt = f"<{n} x {self.ty(dt)}>"
        if from_ty and from_ty != vt:
            v = self.e(f"bitcast {from_ty} {v} to {vt}")
        if n == 1 and from_ty == self.ty(dt):
            return [v]
        return [self.e(f"extractelement {vt} {v}, i32 {i}") for i in range(n)]

    @staticmethod
    def int_ty(bits: int) -> str:
        return {128: "<4 x i32>", 64: "<2 x i32>", 32: "i32", 16: "i16", 8: "i8"}[bits]

    @staticmethod
    def int_suf(bits: int) -> str:
        return {128: "v4i32", 64: "v2i32", 32: "i32", 16: "i16", 8: "i8"}[bits]

    # ================================================================ values in layouts
    def lookup(self, key):
        for m in reversed(self.memo):
            if key in m:
                return m[key]
        return None

    def get(self, v: ir.Value, L: Layout) -> TV:
        lv = self.lay.get(v.id)
        if lv is FLEX:
            key = ("remat", v.id, L)
            got = self.lookup(key)
            if got is None:
                got = self.remat(self.defs[v.id], L)
                self.memo[-1][key] = got
            return got
        tv = self.vals[v.id]
        if tv.layout == L:
            return tv
        moved = self.remap(tv, L)
        if moved is not None:
            return moved
        return self.convert([(v, L)])[0]

    @staticmethod
    def remap(tv: TV, L: Layout) -> Optional[TV]:
        """The same elements, numbered differently in registers: no data moves. Applies when lanes and warps
        hold the same elements in both layouts (copies included), e.g. a row statistic of one dot meeting
        another dot with the same rows."""
        S = tv.layout
        if S.shape != L.shape or S.lane != L.lane or S.warp != L.warp:
            return None
        where = {}
        for r in range(S.nregs):
            where.setdefault(S.reg_offset(r), r)
        regs = []
        for r in range(L.nregs):
            src = where.get(L.reg_offset(r))
            if src is None:
                return None
            regs.append(tv.regs[src])
        return TV(L, regs, tv.dtype)

    def concrete(self, v: ir.Value) -> TV:
        lv = self.lay[v.id]
        if lv is FLEX:
            return self.get(v, self.blocked_for(v.type.shape, v.type.dtype))
        return self.vals[v.id]

    def remat(self, op: ir.Op, L: Layout) -> TV:
        k, dt = op.kind, op.result.type.dtype
        n = L.nregs
        if k == "full":
            return TV(L, [_const(op.attrs["value"], dt)] * n, dt)
        if k == "splat":
            return TV(L, [self.vals[op.args[0].id]] * n, dt)
        if k == "arange":
            base = self.lw(L, 0)
            regs = []
            for r in range(n):
                c = L.reg_offset(r)[0]
                regs.append(self.en(("ar", base, c), lambda c=c: f"add i32 {base}, {c}") if c else base)
            return TV(L, regs, dt)
        if k == "broadcast":
            x = op.args[0]
            return TV(L, list(self.get(x, L.project(x.type.shape)).regs), dt)
        if k == "expand_dims":
            x = op.args[0]
            return TV(L, list(self.get(x, L.slice(op.attrs["axis"])).regs), dt)
        if k == "trans":
            return TV(L, list(self.get(op.args[0], L.transpose()).regs), dt)
        if k in ("ew", "where", "cast"):
            return self.emit_ew(op, L)
        raise NotImplementedError(f"rematerialize {k}")

    # ================================================================ LDS
    # A pending access is (slot, lo, hi, region_lo, region_hi). slot "" is an absolute byte range; "cur" and
    # "oth" are byte ranges inside one slot of a double-buffered region [region_lo, region_hi) of the
    # innermost loop: this iteration's slot and the other one. They swap at the back edge.
    @staticmethod
    def _conflict(a, b) -> bool:
        sa, la, ha, ra0, ra1, pa = a
        sb, lb, hb, rb0, rb1, pb = b
        if pa is not None and pa == pb:
            return False  # the same warp-local partition: each warp touches only its own bytes
        if sa and sb and (ra0, ra1) == (rb0, rb1):
            return sa == sb and la < hb and lb < ha
        x0, x1 = (ra0, ra1) if sa else (la, ha)
        y0, y1 = (rb0, rb1) if sb else (lb, hb)
        return x0 < y1 and y0 < x1

    def barrier(self):
        self.decl("declare void @llvm.amdgcn.s.barrier()")
        self.emit('  fence syncscope("workgroup") release')
        self.emit("  call void @llvm.amdgcn.s.barrier()")
        self.emit('  fence syncscope("workgroup") acquire')
        self.pend_r, self.pend_w = [], []
        self.n_barriers += 1

    def lds_write(self, acc):
        if any(self._conflict(acc, p) for p in self.pend_r + self.pend_w):
            self.barrier()
        self.pend_w.append(acc)

    def lds_read(self, acc):
        if any(self._conflict(acc, p) for p in self.pend_w):
            self.barrier()
        self.pend_r.append(acc)

    @staticmethod
    def _absolute(pend):
        return sorted({("", r0, r1, 0, 0, p) if s else ("", lo, hi, 0, 0, p) for s, lo, hi, r0, r1, p in pend},
                      key=repr)

    @staticmethod
    def _swap(pend):
        sw = {"cur": "oth", "oth": "cur", "": ""}
        return sorted({(sw[s], lo, hi, r0, r1, p) for s, lo, hi, r0, r1, p in pend}, key=repr)

    def parity(self) -> str:
        """0 or 1: which slot of the innermost loop's double buffers this iteration uses."""
        lp = self.loops[-1]
        if lp["par"] is None:
            lp["par"] = self.t()
        return lp["par"]

    def lds_ptr(self, off: str) -> str:
        return self.e(f"getelementptr inbounds i8, ptr addrspace(3) @hk2.lds, i32 {off}")

    def _geometry(self, shape, dt, D: Layout, S: Layout):
        """LDS storage for a conversion: the destination's contiguous dimension is the fast one."""
        if len(shape) == 1:
            return 0, 0, shape[0]
        fast = 0 if D.contig(0) > D.contig(1) else 1
        if D.contig(0) == D.contig(1) == 1:
            fast = 0 if S.contig(0) > S.contig(1) else 1
        n = shape[fast]
        pad = (16 // dt.bytes) if n * dt.bytes >= 64 else 0
        return fast, 1 - fast, n + pad

    def _lds_addr(self, L: Layout, geom, base: int, r: int, dt: T.DType) -> Tuple[str, int]:
        fast, slow, pitch = geom
        if L.rank == 1:
            lin = self.lw(L, 0)
            c = L.reg_offset(r)[0]
            const = c
        else:
            lf, ls = self.lw(L, fast), self.lw(L, slow)
            lin = self.en(("ldslin", L, geom), lambda: f"mul i32 {ls}, {pitch}")
            lin = self.en(("ldslin2", L, geom), lambda: f"add i32 {lin}, {lf}")
            off = L.reg_offset(r)
            const = off[slow] * pitch + off[fast]
        lb = self.en(("ldsbytes", lin, dt.bytes), lambda: f"mul i32 {lin}, {dt.bytes}")
        return lb, base + const * dt.bytes

    def convert(self, items: Sequence[Tuple[ir.Value, Layout]], double: bool = False) -> List[TV]:
        """Move tiles into new layouts through LDS: all writes, one barrier, all reads. ``double``: inside a
        loop, alternate between two copies of the buffers so the next iteration's writes need no barrier
        against this iteration's reads."""
        out: List[Optional[TV]] = [None] * len(items)
        todo = []
        for i, (v, L) in enumerate(items):
            key = ("cvt", v.id, L)
            got = self.lookup(key)
            if got is None and self.lay.get(v.id) is FLEX:
                got = self.get(v, L)
            if got is None and self.vals[v.id].layout == L:
                got = self.vals[v.id]
            if got is None:
                got = self.remap(self.vals[v.id], L)
            if got is not None:
                out[i] = got
            else:
                todo.append(i)
        if not todo:
            return out
        base, plans = 0, []
        for i in todo:
            v, D = items[i]
            src = self.vals[v.id]
            geom = self._geometry(v.type.shape, src.dtype, D, src.layout)
            rows = v.type.shape[geom[1]] if len(v.type.shape) == 2 else 1
            size = rows * geom[2] * src.dtype.bytes
            plans.append((i, src, D, geom, base))
            base += (size + 15) // 16 * 16
        if os.environ.get("HK2_TRACE_CVT"):
            for i, src, D, geom, b0 in plans:
                v = items[i][0]
                op = self.defs.get(v.id)
                print(f"[hk2 cvt] depth {self.depth} {op.kind if op else 'param'} {v.type}\n  from {src.layout}\n"
                      f"  to   {D}\n  warp-local {self.warp_local(src.layout, D)}")
        dyn = None
        local = all(self.warp_local(src.layout, D) for _, src, D, _, _ in plans)
        part = tuple((src.layout.warp, D.warp, geom, b0, src.dtype.bytes) for _, src, D, geom, b0 in plans) \
            if local and not os.environ.get("HK2_NO_WARP_LOCAL") else None
        if part is None and double and self.loops and 2 * base <= self.arch.lds_bytes and \
                not os.environ.get("HK2_NO_DOUBLE"):
            par = self.parity()
            dyn = self.e(f"mul i32 {par}, {base}")
            acc = ("cur", 0, base, 0, 2 * base, None)
            self.lds_size = max(self.lds_size, 2 * base)
        else:
            acc = ("", 0, base, 0, 0, part)
            self.lds_size = max(self.lds_size, base)
        self.lds_write(acc)
        for i, src, D, geom, b0 in plans:
            self._lds_store_tile(src, geom, b0, dyn)
        self.flush_deferred()
        if part is not None:
            # every warp reads back only what it wrote: order this wave's own LDS accesses, no s_barrier
            self.emit('  fence syncscope("wavefront") release')
            self.emit('  fence syncscope("wavefront") acquire')
        self.lds_read(acc)
        for i, src, D, geom, b0 in plans:
            tv = self._lds_load_tile(D, src.dtype, geom, b0, dyn)
            out[i] = tv
            self.memo[-1][("cvt", items[i][0].id, D)] = tv
        return out

    @staticmethod
    def warp_local(S: Layout, D: Layout) -> bool:
        """Each warp holds the same elements in both layouts, so a change between them never crosses warps."""
        if S.warp != D.warp or S.shape != D.shape:
            return False
        span = lambda L: {b for b in L.reg + L.lane if any(b)}
        return span(S) == span(D)

    def _dyn_addr(self, lb: str, dyn: Optional[str]) -> str:
        if dyn is None:
            return lb
        key = ("dynaddr", lb, dyn)
        got = self.memo[-1].get(key)
        if got is None:
            got = self.memo[-1][key] = self.e(f"add i32 {lb}, {dyn}")
        return got

    def _vec(self, L: Layout, geom, dt: T.DType) -> int:
        if L.rank == 1:
            return min(L.contig(0), max(1, 16 // dt.bytes))
        return min(L.contig(geom[0]), max(1, 16 // dt.bytes))

    def _lds_store_tile(self, src: TV, geom, b0: int, dyn: Optional[str] = None):
        L, dt = src.layout, src.dtype
        v = self._vec(L, geom, dt)
        for r in range(0, L.nregs, v):
            if L.canonical_reg(r) != r:
                continue
            lb, const = self._lds_addr(L, geom, b0, r, dt)
            addr = self.e(f"add i32 {self._dyn_addr(lb, dyn)}, {const}")
            p = self.lds_ptr(addr)
            regs = src.regs[r:r + v]
            if v == 1:
                self.emit(f"  store {self.ty(dt)} {regs[0]}, ptr addrspace(3) {p}, align {dt.bytes}")
            else:
                vec = self.pack(regs, dt)
                self.emit(f"  store <{v} x {self.ty(dt)}> {vec}, ptr addrspace(3) {p}, align {v * dt.bytes}")

    def _lds_load_tile(self, D: Layout, dt: T.DType, geom, b0: int, dyn: Optional[str] = None) -> TV:
        v = self._vec(D, geom, dt)
        regs: List[Optional[str]] = [None] * D.nregs
        for r in range(0, D.nregs, v):
            if D.canonical_reg(r) != r:
                continue
            lb, const = self._lds_addr(D, geom, b0, r, dt)
            addr = self.e(f"add i32 {self._dyn_addr(lb, dyn)}, {const}")
            p = self.lds_ptr(addr)
            if v == 1:
                regs[r] = self.e(f"load {self.ty(dt)}, ptr addrspace(3) {p}, align {dt.bytes}")
            else:
                vt = f"<{v} x {self.ty(dt)}>"
                x = self.e(f"load {vt}, ptr addrspace(3) {p}, align {v * dt.bytes}")
                regs[r:r + v] = self.unpack(x, v, dt)
        for r in range(D.nregs):
            if regs[r] is None:
                regs[r] = regs[D.canonical_reg(r)]
        return TV(D, regs, dt)

    # ================================================================ ops
    def emit_block(self, block: ir.Block):
        for op in block.ops:
            if op.kind == "yield":
                continue
            if op.attrs.get("prefetch"):
                self.deferred.append((self.depth, op))
                continue
            self.emit_op(op)

    def flush_deferred(self):
        """Issue the pipelined loads waiting at this nesting level."""
        keep = []
        for d, op in self.deferred:
            if d == self.depth:
                self.emit_op(op)
            else:
                keep.append((d, op))
        self.deferred = keep

    def emit_op(self, op: ir.Op):
        k = op.kind
        fn = getattr(self, "op_" + k, None)
        if fn is None:
            raise NotImplementedError(f"hk2: op {k}")
        fn(op)

    # scalars ----------------------------------------------------------
    def op_const(self, op):
        self.vals[op.result.id] = _const(op.attrs["value"], op.result.type.dtype)

    def op_program_id(self, op):
        ax = "xyz"[op.attrs["axis"]]
        self.decl(f"declare i32 @llvm.amdgcn.workgroup.id.{ax}()")
        self.vals[op.result.id] = self.en(("pid", ax), lambda: f"call i32 @llvm.amdgcn.workgroup.id.{ax}()")

    def op_sop(self, op):
        a, b = (self.vals[x.id] for x in op.args)
        self.vals[op.result.id] = self.binop(op.attrs["fn"], a, b, op.args[0].type.dtype)

    def op_scast(self, op):
        self.vals[op.result.id] = self.cast(self.vals[op.args[0].id], op.args[0].type.dtype, op.result.type.dtype)

    def op_sunary(self, op):
        self.vals[op.result.id] = self.unary(op.attrs["fn"], self.vals[op.args[0].id], op.result.type.dtype)

    # flexible tiles: built at their uses ------------------------------
    def _flex_or(self, op, fn):
        if self.lay[op.result.id] is FLEX:
            return
        self.vals[op.result.id] = fn()

    def op_full(self, op):
        pass

    op_splat = op_arange = op_broadcast = op_full

    def op_expand_dims(self, op):
        L = self.lay[op.result.id]
        if L is FLEX:
            return
        x = self.vals[op.args[0].id]
        self.vals[op.result.id] = TV(L, list(x.regs), x.dtype)

    def op_trans(self, op):
        L = self.lay[op.result.id]
        if L is FLEX:
            return
        x = self.vals[op.args[0].id]
        self.vals[op.result.id] = TV(L, list(x.regs), x.dtype)

    def op_ew(self, op):
        L = self.lay[op.result.id]
        if L is not FLEX:
            self.vals[op.result.id] = self.emit_ew(op, L)

    op_where = op_cast = op_ew

    def emit_ew(self, op, L: Layout) -> TV:
        k, rt = op.kind, op.result.type.dtype
        ins = [self.get(a, L.project(a.type.shape) if a.type.shape != op.result.type.shape else L)
               for a in op.args]
        regs = []
        for r in range(L.nregs):
            c = L.canonical_reg(r)
            if c != r:
                regs.append(regs[c])
                continue
            if k == "cast":
                regs.append(self.cast(ins[0].regs[r], ins[0].dtype, rt))
            elif k == "where":
                regs.append(self.select(ins[0].regs[r], ins[1].regs[r], ins[2].regs[r], rt))
            elif len(ins) == 1:
                regs.append(self.unary(op.attrs["fn"], ins[0].regs[r], rt))
            else:
                regs.append(self.binop(op.attrs["fn"], ins[0].regs[r], ins[1].regs[r], ins[0].dtype))
        return TV(L, regs, rt)

    # memory ------------------------------------------------------------
    def _tensor(self, op):
        nd = op.attrs["ndim"]
        a = op.args
        ptr = self.vals[a[0].id]
        dims = [self.vals[x.id] for x in a[1:1 + nd]]
        strides = [self.vals[x.id] for x in a[1 + nd:1 + 2 * nd]]
        offs = a[1 + 2 * nd:1 + 3 * nd]
        return ptr, dims, strides, offs

    def _rsrc(self, ptr: str, strides, offs_vals, dt: T.DType) -> str:
        """A buffer resource whose base is the tile's origin (uniform math, in scalar registers)."""
        lin = None
        for s, o in zip(strides, offs_vals):
            o64 = self.e(f"sext i32 {o} to i64")
            if s == "1":
                term = o64
            else:
                s64 = self.e(f"sext i32 {s} to i64")
                term = self.e(f"mul i64 {o64}, {s64}")
            lin = term if lin is None else self.e(f"add i64 {lin}, {term}")
        base = self.e(f"getelementptr inbounds {self.ty(dt)}, ptr addrspace(1) {ptr}, i64 {lin}")
        nr = "i64" if self.llvm_major >= 21 else "i32"  # LLVM 21 widened num_records
        self.decl(f"declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, {nr}, i32)")
        return self.e(f"call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) {base}, i16 0, "
                      f"{nr} 2147483647, i32 {self.arch.buffer_flags})")

    def _mem_plan(self, op, L: Layout, dt: T.DType):
        ptr, dims, strides, offs = self._tensor(op)
        nd = op.attrs["ndim"]
        offv = [self.vals[o.id] for o in offs]
        rsrc = self._rsrc(ptr, strides, offv, dt)
        lead = nd - L.rank  # leading dimensions are indexed, the trailing ones tiled
        rem = [self.e(f"sub i32 {d}, {o}") for d, o in zip(dims[lead:], offv[lead:])]
        lead_ok = None
        for d, o in zip(dims[:lead], offv[:lead]):
            c = self.e(f"icmp ult i32 {o}, {d}")
            lead_ok = c if lead_ok is None else self.e(f"and i1 {lead_ok}, {c}")
        v = 1
        if strides[nd - 1] == "1":
            v = min(L.contig(L.rank - 1), op.attrs["vec"] or 1, max(1, 16 // dt.bytes))
            v = min(v, max(1, self.divisibility(offs[nd - 1])))
        return rsrc, rem, strides[lead:], v, lead_ok

    def _and(self, a: Optional[str], b: Optional[str]) -> Optional[str]:
        if a is None or b is None:
            return a or b
        return self.e(f"and i1 {a}, {b}")

    def _elem_offset(self, L: Layout, r: int, rem, strides, v: int, dt: T.DType, extra_ok: Optional[str]):
        off = L.reg_offset(r)
        coords, oks = [], []
        for d in range(L.rank):
            lw = self.lw(L, d)
            c = self.e(f"add i32 {lw}, {off[d]}") if off[d] else lw
            coords.append(c)
            oks.append(self.e(f"icmp slt i32 {c}, {rem[d]}"))
        lin = None
        for d, c in enumerate(coords):
            term = c if strides[d] == "1" else self.e(f"mul i32 {c}, {strides[d]}")
            lin = term if lin is None else self.e(f"add i32 {lin}, {term}")
        ok = oks[0]
        for o in oks[1:]:
            ok = self.e(f"and i1 {ok}, {o}")
        if extra_ok:
            ok = self.e(f"and i1 {ok}, {extra_ok}")
        b = self.e(f"mul i32 {lin}, {dt.bytes}")
        return self.e(f"select i1 {ok}, i32 {b}, i32 -2147483648"), ok

    def op_load(self, op):
        L = self.lay[op.result.id]
        dt = op.result.type.dtype
        rsrc, rem, strides, v, lead_ok = self._mem_plan(op, L, dt)
        bits = v * dt.bits
        ity, suf = self.int_ty(bits), self.int_suf(bits)
        self.decl(f"declare {ity} @llvm.amdgcn.raw.ptr.buffer.load.{suf}(ptr addrspace(8), i32, i32, i32)")
        other = op.attrs["other"]
        regs: List[Optional[str]] = [None] * L.nregs
        for r in range(0, L.nregs, v):
            if L.canonical_reg(r) != r:
                continue
            voff, ok = self._elem_offset(L, r, rem, strides, v, dt, lead_ok)
            x = self.e(f"call {ity} @llvm.amdgcn.raw.ptr.buffer.load.{suf}(ptr addrspace(8) {rsrc}, i32 {voff}, "
                       f"i32 0, i32 0)")
            vals = self.unpack(x, v, dt, ity) if v > 1 else [self.e(f"bitcast {ity} {x} to {self.ty(dt)}")
                                                             if ity != self.ty(dt) else x]
            if other not in (0, 0.0, False):
                oc = _const(other, dt)
                vals = [self.e(f"select i1 {ok}, {self.ty(dt)} {y}, {self.ty(dt)} {oc}") for y in vals]
            regs[r:r + v] = vals
        for r in range(L.nregs):
            if regs[r] is None:
                regs[r] = regs[L.canonical_reg(r)]
        self.vals[op.result.id] = TV(L, regs, dt)

    def op_store(self, op):
        val = op.args[-1]
        dt = val.type.dtype
        L = self.lay.get(val.id)
        Lb = self.blocked_for(val.type.shape, dt, op.attrs["vec"] or 1)
        if L is FLEX or L is None:
            tv = self.get(val, Lb)
        else:
            tv = self.vals[val.id]
            last = len(val.type.shape) - 1
            if tv.layout.contig(last) < min(Lb.contig(last), max(1, 16 // dt.bytes)):
                # Scattered stores cost more than one trip through LDS, unless that trip would raise the
                # kernel's LDS high-water mark (and so cost occupancy for the whole kernel): then store as is.
                geom = self._geometry(val.type.shape, dt, Lb, tv.layout)
                rows = val.type.shape[geom[1]] if len(val.type.shape) == 2 else 1
                need = rows * geom[2] * dt.bytes
                if need <= max(self.lds_size, 32 << 10):
                    tv = self.get(val, Lb)
        L = tv.layout
        rsrc, rem, strides, v, lead_ok = self._mem_plan(op, L, dt)
        bits = v * dt.bits
        ity, suf = self.int_ty(bits), self.int_suf(bits)
        self.decl(f"declare void @llvm.amdgcn.raw.ptr.buffer.store.{suf}({ity}, ptr addrspace(8), i32, i32, i32)")
        own = self._and(self.owner(L), lead_ok)
        for r in range(0, L.nregs, v):
            if L.canonical_reg(r) != r:
                continue
            voff, _ = self._elem_offset(L, r, rem, strides, v, dt, own)
            if v > 1:
                data = self.pack(tv.regs[r:r + v], dt, ity)
            else:
                data = tv.regs[r] if ity == self.ty(dt) else self.e(f"bitcast {self.ty(dt)} {tv.regs[r]} to {ity}")
            self.emit(f"  call void @llvm.amdgcn.raw.ptr.buffer.store.{suf}({ity} {data}, ptr addrspace(8) {rsrc}, "
                      f"i32 {voff}, i32 0, i32 0)")

    # matrix cores --------------------------------------------------------
    def op_dot(self, op):
        a, b, c = op.args
        m, k = a.type.shape
        n = b.type.shape[1]
        atom, grid = self.dot_plan(a.type.dtype, m, n)
        if m % atom.m or n % atom.n or k % atom.k:
            raise ValueError(f"hk2.dot {m}x{k} @ {k}x{n}: on {self.arch.name} the tile sizes must be multiples of "
                             f"{atom.m}x{atom.k} @ {atom.k}x{atom.n}")
        La = A.a_layout(atom, m, k, n, self.nwarps, grid)
        Lb = A.b_layout(atom, m, k, n, self.nwarps, grid)
        Lc = self.lay[op.result.id]
        ta, tb = self.convert([(a, La), (b, Lb)], double=self.depth > 0)
        tc = self.get(c, Lc)
        wm, wn = grid
        rm, rn, rk = (m // wm) // atom.m, (n // wn) // atom.n, k // atom.k
        rm, rn = max(rm, 1), max(rn, 1)
        dt = a.type.dtype
        self.decl(f"declare {atom.c_ty} @{atom.intrinsic}({atom.op_ty}, {atom.op_ty}, {atom.c_ty}"
                  f"{', i32 immarg, i32 immarg, i32 immarg' if atom.tail else ''})")
        accs = {}
        for i in range(rm):
            for j in range(rn):
                base = atom.c_regs * (j + rn * i)
                accs[i, j] = self.pack(tc.regs[base:base + atom.c_regs], T.f32)
        op_n = int(atom.op_ty.split()[0][1:])
        for kk in range(rk):
            avecs = {}
            for i in range(rm):
                ab = atom.a_regs * (kk + rk * i)
                avecs[i] = self._operand(ta.regs[ab:ab + atom.a_regs], dt, atom, op_n)
            for j in range(rn):
                bb = atom.b_regs * (kk + rk * j)
                bvec = self._operand(tb.regs[bb:bb + atom.b_regs], dt, atom, op_n)
                for i in range(rm):
                    accs[i, j] = self.e(f"call {atom.c_ty} @{atom.intrinsic}({atom.op_ty} {avecs[i]}, "
                                        f"{atom.op_ty} {bvec}, {atom.c_ty} {accs[i, j]}{atom.tail})")
        regs = [None] * Lc.nregs
        for i in range(rm):
            for j in range(rn):
                base = atom.c_regs * (j + rn * i)
                regs[base:base + atom.c_regs] = self.unpack(accs[i, j], atom.c_regs, T.f32)
        for r in range(Lc.nregs):
            if regs[r] is None:
                regs[r] = regs[Lc.canonical_reg(r)]
        self.vals[op.result.id] = TV(Lc, regs, T.f32)

    def _operand(self, regs, dt, atom, op_n):
        assert len(regs) == op_n, (len(regs), op_n)
        return self.pack(regs, dt, atom.op_ty)

    # reductions ------------------------------------------------------------
    def dpp_pair(self, x: str, b: int, lower_reduced: bool, dt: T.DType) -> Optional[str]:
        """The value of the partner lane for a reduction step over lane bit ``b`` (b < 4), by DPP: a VALU
        move, no LDS traffic. RDNA has an exact xor (row_xmask); CDNA pairs lanes with quad permutes and,
        when the lower bits are already reduced, with the half-row and row mirrors."""
        if dt.bits != 32 or b > 3 or os.environ.get("HK2_NO_DPP"):
            return None
        if self.arch.family == "rdna":
            ctrl = 0x160 | (1 << b)
        elif b == 0:
            ctrl = 0xB1  # quad_perm [1,0,3,2]
        elif b == 1:
            ctrl = 0x4E  # quad_perm [2,3,0,1]
        elif lower_reduced:
            ctrl = 0x141 if b == 2 else 0x140  # row_half_mirror, row_mirror
        else:
            return None
        self.decl("declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)")
        xi = x if dt.is_int else self.e(f"bitcast {self.ty(dt)} {x} to i32")
        y = self.e(f"call i32 @llvm.amdgcn.update.dpp.i32(i32 0, i32 {xi}, i32 {ctrl}, i32 15, i32 15, i1 true)")
        return y if dt.is_int else self.e(f"bitcast i32 {y} to {self.ty(dt)}")

    def shfl_xor(self, x: str, mask: int, dt: T.DType) -> str:
        self.decl("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)")
        addr = self.en(("shfladdr", mask), lambda: f"xor i32 {self.lane()}, {mask}")
        addr4 = self.en(("shfladdr4", mask), lambda: f"shl i32 {addr}, 2")
        if dt.bits == 32:
            xi = x if dt.is_int else self.e(f"bitcast {self.ty(dt)} {x} to i32")
        else:
            xi = self.e(f"zext {self.int_ty(dt.bits)} {self.e(f'bitcast {self.ty(dt)} {x} to {self.int_ty(dt.bits)}') if dt.is_float else x} to i32")
        y = self.e(f"call i32 @llvm.amdgcn.ds.bpermute(i32 {addr4}, i32 {xi})")
        if dt.bits == 32:
            return y if dt.is_int else self.e(f"bitcast i32 {y} to {self.ty(dt)}")
        t = self.e(f"trunc i32 {y} to {self.int_ty(dt.bits)}")
        return self.e(f"bitcast {self.int_ty(dt.bits)} {t} to {self.ty(dt)}") if dt.is_float else t

    def op_reduce(self, op):
        x = op.args[0]
        tv = self.concrete(x)
        L, dt = tv.layout, tv.dtype
        ax, fn = op.attrs["axis"], op.attrs["fn"]
        regs = list(tv.regs)
        n = L.nregs
        for b, basis in enumerate(L.reg):
            if not basis[ax]:
                continue
            for r in range(n):
                if not r >> b & 1:
                    regs[r] = self.binop(fn, regs[r], regs[r | 1 << b], dt)
            for r in range(n):
                if r >> b & 1:
                    regs[r] = regs[r ^ (1 << b)]
        R = L.slice(ax)
        for b, basis in enumerate(L.lane):
            if not basis[ax]:
                continue
            lower = all(L.lane[i][ax] for i in range(b))
            done = {}
            for r in range(n):
                c = regs[r]
                if c not in done:
                    other = self.dpp_pair(c, b, lower, dt) or self.shfl_xor(c, 1 << b, dt)
                    done[c] = self.binop(fn, c, other, dt)
                regs[r] = done[c]
        wbits = [b for b, basis in enumerate(L.warp) if basis[ax]]
        if wbits:
            regs = self._reduce_warps(regs, R, L, wbits, fn, dt)
        self.vals[op.result.id] = TV(R, regs, dt)

    def _reduce_warps(self, regs, R: Layout, L: Layout, wbits, fn, dt):
        K = 1 << len(wbits)
        P = R.shape[0]
        kidx = "0"
        for i, b in enumerate(wbits):
            bit = self.en(("rwb", b), lambda b=b: f"lshr i32 {self.warp()}, {b}")
            bit1 = self.en(("rwb1", b), lambda bit=bit: f"and i32 {bit}, 1")
            sh = self.en(("rwbs", b, i), lambda bit1=bit1, i=i: f"shl i32 {bit1}, {i}")
            kidx = sh if kidx == "0" else self.en(("rwk", kidx, sh), lambda k=kidx, s=sh: f"or i32 {k}, {s}")
        size = K * P * dt.bytes
        self.lds_size = max(self.lds_size, size)
        pos0 = self.lw(R, 0)
        self.lds_write(("", 0, size, 0, 0, None))
        seen = {}
        for r in range(R.nregs):
            c = R.canonical_reg(r)
            if c != r or regs[r] in seen:
                continue
            seen[regs[r]] = r
            p = self.e(f"add i32 {pos0}, {R.reg_offset(r)[0]}")
            kp = self.e(f"mul i32 {kidx}, {P}")
            idx = self.e(f"add i32 {kp}, {p}")
            byt = self.e(f"mul i32 {idx}, {dt.bytes}")
            self.emit(f"  store {self.ty(dt)} {regs[r]}, ptr addrspace(3) {self.lds_ptr(byt)}, align {dt.bytes}")
        self.lds_read(("", 0, size, 0, 0, None))
        out, done = list(regs), {}
        for r in range(R.nregs):
            if regs[r] in done:
                out[r] = done[regs[r]]
                continue
            p = self.e(f"add i32 {pos0}, {R.reg_offset(r)[0]}")
            acc = None
            for kk in range(K):
                idx = self.e(f"add i32 {p}, {kk * P}")
                byt = self.e(f"mul i32 {idx}, {dt.bytes}")
                y = self.e(f"load {self.ty(dt)}, ptr addrspace(3) {self.lds_ptr(byt)}, align {dt.bytes}")
                acc = y if acc is None else self.binop(fn, acc, y, dt)
            done[regs[r]] = acc
            out[r] = acc
        return out

    # control flow ----------------------------------------------------------
    def _phi_vals(self, v: ir.Value, x):
        """Canonical registers of a tile (or the scalar) for phi nodes."""
        if not v.is_tile:
            return [x]
        L = x.layout
        return [x.regs[r] for r in range(L.nregs) if L.canonical_reg(r) == r]

    def _from_phis(self, v: ir.Value, names, L):
        if not v.is_tile:
            return names[0]
        regs, it = [None] * L.nregs, iter(names)
        for r in range(L.nregs):
            if L.canonical_reg(r) == r:
                regs[r] = next(it)
        for r in range(L.nregs):
            if regs[r] is None:
                regs[r] = regs[L.canonical_reg(r)]
        return TV(L, regs, v.type.dtype)

    def _llty(self, v: ir.Value) -> str:
        return self.ty(v.type.dtype)

    def _in_layout(self, v: ir.Value, target: ir.Value):
        """The emitted value of v, as the phi for ``target`` needs it."""
        if not v.is_tile:
            return self.vals[v.id]
        return self.get(v, self.lay[target.id])

    def op_for(self, op):
        blk = op.blocks[0]
        lb, ub, step = (self.vals[a.id] for a in op.args[:3])
        params = blk.params[1:]
        inits = [self._in_layout(i, p) for i, p in zip(op.args[3:], params)]
        self._hoist_operands(blk)
        if self.pend_r or self.pend_w:
            self.barrier()  # once here, rather than at the top of every iteration
        pre = self.cur
        hdr, body, exit_ = self.label("for"), self.label("body"), self.label("done")
        self.emit(f"  br label %{hdr}")
        self.start(hdr)
        hdr_block = self.blocks[-1]
        iv = self.t()
        pnames = [[self.t() for _ in self._phi_vals(p, i)] for p, i in zip(params, inits)]
        cond = self.t()
        self.vals[blk.params[0].id] = iv
        for p, names in zip(params, pnames):
            self.vals[p.id] = self._from_phis(p, names, self.lay[p.id] if p.is_tile else None)
        self.start(body)
        # The LDS state at the loop head is the fixpoint of (state before the loop) and (state at the end of
        # the body, slots swapped): emit the body under an assumed head state, and emit it again from a wider
        # one if the end state is not covered.
        n_blocks, deferred0, nbar0 = len(self.blocks), list(self.deferred), self.n_barriers
        head_r, head_w = self._absolute(self.pend_r), self._absolute(self.pend_w)
        for attempt in range(6):
            self.memo.append({})
            self.loops.append({"par": None})
            self.pend_r, self.pend_w = list(head_r), list(head_w)
            self.depth += 1
            self.emit_block(blk)
            self.flush_deferred()
            self.depth -= 1
            end_r, end_w = self._swap(self.pend_r), self._swap(self.pend_w)
            if set(end_r) <= set(head_r) and set(end_w) <= set(head_w):
                break
            head_r = sorted(set(head_r) | set(end_r), key=repr)
            head_w = sorted(set(head_w) | set(end_w), key=repr)
            self.memo.pop()
            self.loops.pop()
            del self.blocks[n_blocks:]
            self.blocks[-1][1].clear()
            self.deferred, self.n_barriers = list(deferred0), nbar0
        else:
            raise RuntimeError("hk2: LDS barrier analysis did not converge")
        ys = [self._in_layout(y, p) for y, p in zip(blk.ops[-1].args, params)]
        ivn = self.e(f"add i32 {iv}, {step}")
        lp = self.loops.pop()
        parn = self.e(f"xor i32 {lp['par']}, 1") if lp["par"] else None
        latch = self.cur
        self.emit(f"  br label %{hdr}")
        self.memo.pop()
        lines = [f"  {iv} = phi i32 [ {lb}, %{pre} ], [ {ivn}, %{latch} ]"]
        if parn:
            lines.append(f"  {lp['par']} = phi i32 [ 0, %{pre} ], [ {parn}, %{latch} ]")
        for p, names, i0, y in zip(params, pnames, inits, ys):
            for nm, a, b in zip(names, self._phi_vals(p, i0), self._phi_vals(p, y)):
                lines.append(f"  {nm} = phi {self._llty(p)} [ {a}, %{pre} ], [ {b}, %{latch} ]")
        pred = "sgt" if op.attrs.get("neg_step") else "slt"
        lines.append(f"  {cond} = icmp {pred} i32 {iv}, {ub}")
        lines.append(f"  br i1 {cond}, label %{body}, label %{exit_}")
        hdr_block[1].extend(lines)
        self.start(exit_)
        self.pend_r, self.pend_w = self._absolute(head_r), self._absolute(head_w)
        for p, res in zip(params, op.results):
            self.vals[res.id] = self.vals[p.id]

    def _hoist_operands(self, blk: ir.Block):
        """Matrix-core operands that do not change in the loop (attention's Q) take their operand layout once,
        before it, instead of every iteration."""
        inside, dots = set(), []

        def walk(b):
            for p in b.params:
                inside.add(p.id)
            for o in b.ops:
                inside.update(r.id for r in o.results)
                if o.kind == "dot":
                    dots.append(o)
                for bb in o.blocks:
                    walk(bb)
        walk(blk)
        items = []
        for d in dots:
            a, b = d.args[:2]
            m, k = a.type.shape
            n = b.type.shape[1]
            atom, grid = self.dot_plan(a.type.dtype, m, n)
            for v, L in ((a, A.a_layout(atom, m, k, n, self.nwarps, grid)),
                         (b, A.b_layout(atom, m, k, n, self.nwarps, grid))):
                if v.id not in inside and self.lay.get(v.id) is not FLEX and v.id in self.vals and \
                        self.vals[v.id].layout != L:
                    items.append((v, L))
        if items:
            self.convert(items)

    def op_if(self, op):
        cond = self.vals[op.args[0].id]
        tl, el, ml = self.label("then"), self.label("else"), self.label("endif")
        self.emit(f"  br i1 {cond}, label %{tl}, label %{el}")
        ends, outs = [], []
        r0, w0 = list(self.pend_r), list(self.pend_w)
        merged_r, merged_w = set(), set()
        for lbl, blk in zip((tl, el), op.blocks):
            self.start(lbl)
            self.memo.append({})
            self.pend_r, self.pend_w = list(r0), list(w0)
            self.depth += 1
            self.emit_block(blk)
            self.flush_deferred()
            self.depth -= 1
            outs.append([self._in_layout(y, res) for y, res in zip(blk.ops[-1].args, op.results)])
            merged_r |= set(self.pend_r)
            merged_w |= set(self.pend_w)
            ends.append(self.cur)
            self.emit(f"  br label %{ml}")
            self.memo.pop()
        self.start(ml)
        self.pend_r, self.pend_w = sorted(merged_r, key=repr), sorted(merged_w, key=repr)
        for j, res in enumerate(op.results):
            a, b = (self._phi_vals(res, outs[k][j]) for k in range(2))
            names = []
            for x, y in zip(a, b):
                n = self.t()
                self.emit(f"  {n} = phi {self._llty(res)} [ {x}, %{ends[0]} ], [ {y}, %{ends[1]} ]")
                names.append(n)
            self.vals[res.id] = self._from_phis(res, names, self.lay[res.id] if res.is_tile else None)

    # ================================================================ the kernel
    def compile(self, top: ir.Block, params: List[Tuple[ir.Value, str, str]], param_div: Dict[int, int]) -> str:
        """``params``: (value, LLVM type, name) for each kernel argument in order."""
        self.param_div = param_div
        self.rowmode = self._detect_rowmode(top)
        self.infer(top)
        self.start("entry")
        for v, _, name in params:
            self.vals[v.id] = f"%{name}"
        self.start("body")
        self.emit_block(top)
        self.emit("  ret void")
        if self.lds_size > self.arch.lds_bytes:
            raise ValueError(f"hk2: {self.name} needs {self.lds_size} bytes of LDS for its layout changes; "
                             f"{self.arch.name} has {self.arch.lds_bytes} per workgroup. Use smaller tiles "
                             f"(the dot operands are staged in LDS: about (BM + BN) * BK * 2 bytes for 16-bit types)")
        out = ['target triple = "amdgcn-amd-amdhsa"']
        if self.lds_size:
            out.append(f"@hk2.lds = internal addrspace(3) global [{self.lds_size} x i8] poison, align 16")
        out += sorted(self.decls)
        sig = ", ".join(f"{t} {'noalias ' if t.startswith('ptr') else ''}%{name}" for _, t, name in params)
        out.append(f"define amdgpu_kernel void @{self.name}({sig}) #0 {{")
        out.append("entry:")
        out += self.entry
        out.append("  br label %body")
        for label, lines in self.blocks[1:]:
            out.append(f"{label}:")
            out += lines
        out.append("}")
        threads = self.nwarps * self.wave
        out.append(f'attributes #0 = {{ "amdgpu-flat-work-group-size"="{threads},{threads}" '
                   f'"uniform-work-group-size"="true" }}')
        return "\n".join(out) + "\n"
