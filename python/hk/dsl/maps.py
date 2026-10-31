"""Integer, mask and bit expressions on strips, and `hk.map` (docs/design/dsl-surface.md §3 G1).

Each is an `hk.map` op: a region of scalar `arith` / `math` ops (and `hk.rcp`) applied to element
(r, c) of each strip operand. An operator on an i32 strip, a comparison, `hk.where` and the
`hk.math` functions build a one-op map; the `hk-map-fuse` pass merges a chain of them into one.
`hk.map(fn, ...)` traces `fn` once into one map's region. Floating-point contraction is off in a
map: `a * b + c` is two roundings, `hk.math.fma(a, b, c)` one.

The IR is arch-neutral; the lowering is `kittens::map_cpp`, in include/hk/cdna (gfx950) and
include/hk/rdna/map.cuh (gfx1100, gfx1201). On RDNA a mask lives only inside a map: the trees'
strips hold no bool, and hk-map-fuse refuses a mask that would leave its map.
"""

from __future__ import annotations

from typing import Callable

from .._mlir import ir
from .._mlir.dialects import arith, hk as hkd, math as mathd
from ..types import RegTileType, b1, bf16, fp32, i32

#: a map's element kinds: hk's dtype name -> the MLIR scalar type's maker
_KINDS = {"i32": lambda: ir.IntegerType.get_signless(32), "fp32": ir.F32Type.get,
          "b1": lambda: ir.IntegerType.get_signless(1), "bf16": ir.BF16Type.get}
_DTYPES = {"i32": i32, "fp32": fp32, "b1": b1, "bf16": bf16}


def _v():
    from . import values  # noqa: PLC0415

    return values


def _loc(name: str):
    return _v()._t().loc(name)


class Scalar:
    """An element inside an `hk.map`: what `fn` receives, and what its operators and the hk.math
    functions make. `kind` is "i32", "fp32", "b1" (a mask) or "bf16" (`.to(hk.bf16)`'s, to be
    converted back). The operators are C's on that type: `//` and `%` truncate, `>>` is
    arithmetic, `<<` / `+` / `*` wrap as C++ int arithmetic does on the GPU."""

    __slots__ = ("v", "kind")

    def __init__(self, v, kind: str):
        self.v = v
        self.kind = kind

    def __repr__(self) -> str:
        return f"<hk.map element {self.kind}>"

    @property
    def dtype(self):
        return _DTYPES[self.kind]

    def __add__(self, o): return _bin("+", self, o)
    def __radd__(self, o): return _bin("+", o, self)
    def __sub__(self, o): return _bin("-", self, o)
    def __rsub__(self, o): return _bin("-", o, self)
    def __mul__(self, o): return _bin("*", self, o)
    def __rmul__(self, o): return _bin("*", o, self)
    def __truediv__(self, o): return _bin("/", self, o)
    def __rtruediv__(self, o): return _bin("/", o, self)
    def __floordiv__(self, o): return _bin("//", self, o)
    def __rfloordiv__(self, o): return _bin("//", o, self)
    def __mod__(self, o): return _bin("%", self, o)
    def __rmod__(self, o): return _bin("%", o, self)
    def __lshift__(self, o): return _bin("<<", self, o)
    def __rlshift__(self, o): return _bin("<<", o, self)
    def __rshift__(self, o): return _bin(">>", self, o)
    def __rrshift__(self, o): return _bin(">>", o, self)
    def __and__(self, o): return _bin("&", self, o)
    def __rand__(self, o): return _bin("&", o, self)
    def __or__(self, o): return _bin("|", self, o)
    def __ror__(self, o): return _bin("|", o, self)
    def __xor__(self, o): return _bin("^", self, o)
    def __rxor__(self, o): return _bin("^", o, self)
    def __lt__(self, o): return _bin("<", self, o)
    def __le__(self, o): return _bin("<=", self, o)
    def __gt__(self, o): return _bin(">", self, o)
    def __ge__(self, o): return _bin(">=", self, o)
    def __eq__(self, o): return _bin("==", self, o)  # type: ignore[override]
    def __ne__(self, o): return _bin("!=", self, o)  # type: ignore[override]
    __hash__ = object.__hash__

    def __neg__(self):
        if self.kind == "fp32":
            return Scalar(arith.NegFOp(self.v, loc=_loc("neg")).result, "fp32")
        return _bin("-", 0, self)

    def __invert__(self):
        if self.kind not in ("i32", "b1"):
            raise TypeError(f"~ is on an i32 or a mask, not {self.kind}")
        return _bin("^", self, -1 if self.kind == "i32" else True)

    def __abs__(self):
        return abs_(self)

    def __bool__(self):
        raise TypeError("an hk.map element was used where Python needs a bool (an `if`, `and`, "
                        "`or`, `not`): a map is one expression -- choose with hk.where(m, a, b) "
                        "and combine masks with &, | and ~")

    def to(self, dtype) -> "Scalar":
        """The value converted: i32 <-> fp32 (C's conversions: fp32 -> i32 truncates), fp32 ->
        bf16 (round to nearest even) and back, a mask -> i32 (0 / 1)."""
        k, d = self.kind, dtype.name
        if k == d:
            return self
        t = _KINDS.get(d)
        if t is None:
            raise TypeError(f"an hk.map element converts to i32, fp32 or bf16, not {d}")
        cls = {("i32", "fp32"): arith.SIToFPOp, ("fp32", "i32"): arith.FPToSIOp,
               ("fp32", "bf16"): arith.TruncFOp, ("bf16", "fp32"): arith.ExtFOp,
               ("b1", "i32"): arith.ExtUIOp}.get((k, d))
        if cls is None:
            raise TypeError(f"no conversion of an hk.map element from {k} to {d} (i32 <-> fp32, "
                            f"fp32 <-> bf16, a mask to i32)")
        return Scalar(cls(t(), self.v, loc=_loc("cvt")).result, d)

    def bitcast(self, dtype) -> "Scalar":
        """The same 32 bits read as the other of i32 and fp32."""
        if {self.kind, dtype.name} != {"i32", "fp32"}:
            raise TypeError(f"an hk.map element bitcasts between i32 and fp32, not {self.kind} "
                            f"and {dtype.name}")
        return Scalar(arith.BitcastOp(_KINDS[dtype.name](), self.v, loc=_loc("bc")).result,
                      dtype.name)


# ---------------------------------------------------------------- scalar ops


def _const(x, kind: str) -> Scalar:
    t = _KINDS[kind]()
    if kind == "fp32":
        a = ir.FloatAttr.get(t, float(x))
    else:
        a = ir.IntegerAttr.get(t, int(x))
    return Scalar(arith.ConstantOp(t, a, loc=_loc("c")).result, kind)


def _num_kind(x) -> str:
    return "b1" if isinstance(x, bool) else "i32" if isinstance(x, int) else "fp32"


def _scalars(what: str, *xs) -> list:
    """`xs` as Scalars of one kind: a Python number takes its partner's kind (an int is an exact
    fp32 constant beside an fp32; a float beside an i32 is refused)."""
    kinds = {x.kind for x in xs if isinstance(x, Scalar)}
    for x in xs:
        if not isinstance(x, (Scalar, int, float)):
            raise TypeError(f"{what}: {x!r} inside an hk.map. A map's elements are its "
                            f"arguments' and Python numbers; pass a strip or an hk.Int as an "
                            f"argument of hk.map")
    if len(kinds) > 1:
        raise TypeError(f"{what} of {' and '.join(sorted(kinds))} elements: convert one with "
                        f".to() (no implicit conversion in a map)")
    if not kinds:
        raise TypeError(f"{what} of Python numbers only: compute it in Python")
    k = kinds.pop()
    out = []
    for x in xs:
        if isinstance(x, Scalar):
            out.append(x)
        elif k == "fp32" and not isinstance(x, bool):
            out.append(_const(x, "fp32"))
        elif _num_kind(x) != k:
            raise TypeError(f"{what}: the Python {type(x).__name__} {x!r} beside an {k} element")
        else:
            out.append(_const(x, k))
    return out


_INT = {"+": arith.AddIOp, "-": arith.SubIOp, "*": arith.MulIOp, "//": arith.DivSIOp,
        "%": arith.RemSIOp, "<<": arith.ShLIOp, ">>": arith.ShRSIOp, "&": arith.AndIOp,
        "|": arith.OrIOp, "^": arith.XOrIOp}
_FLOAT = {"+": arith.AddFOp, "-": arith.SubFOp, "*": arith.MulFOp, "/": arith.DivFOp}
_MASK = {"&": arith.AndIOp, "|": arith.OrIOp, "^": arith.XOrIOp}
_CMPI = {"<": "slt", "<=": "sle", ">": "sgt", ">=": "sge", "==": "eq", "!=": "ne"}
# ordered, but != unordered: C's operators (a NaN compares unequal to everything)
_CMPF = {"<": "olt", "<=": "ole", ">": "ogt", ">=": "oge", "==": "oeq", "!=": "une"}


def _bin(sym: str, a, b) -> Scalar:
    a, b = _scalars(f"`{sym}`", a, b)
    k = a.kind
    if sym in _CMPI:
        if k == "b1":
            raise TypeError(f"`{sym}` compares i32 or fp32 elements; a mask combines with & | ^ ~")
        if k == "i32":
            p = getattr(arith.CmpIPredicate, _CMPI[sym])
            return Scalar(arith.CmpIOp(p, a.v, b.v, loc=_loc("cmp")).result, "b1")
        p = getattr(arith.CmpFPredicate, _CMPF[sym].upper())
        return Scalar(arith.CmpFOp(p, a.v, b.v, loc=_loc("cmp")).result, "b1")
    table = {"i32": _INT, "fp32": _FLOAT, "b1": _MASK}.get(k, {})
    cls = table.get(sym)
    if cls is None:
        why = {("fp32", "//"): "a float divides with /", ("fp32", "%"): "there is no float %",
               ("i32", "/"): "an integer divides with // (C's truncating division)"}.get(
            (k, sym), f"{k} elements have + - * / and comparisons (fp32), the C integer "
                      f"operators (i32), & | ^ ~ (masks)")
        raise TypeError(f"`{sym}` on {k} elements: {why}")
    return Scalar(cls(a.v, b.v, loc=_loc(sym)).result, k)


def _call(what: str, kinds, cls, *xs, kind=None) -> Scalar:
    xs = _scalars(what, *xs)
    if xs[0].kind not in kinds:
        raise TypeError(f"{what} takes {' or '.join(kinds)} elements, not {xs[0].kind}")
    return Scalar(cls(*[x.v for x in xs], loc=_loc(what)).result, kind or xs[0].kind)


def where_(m, a, b) -> Scalar:
    if not isinstance(m, Scalar) or m.kind != "b1":
        raise TypeError(f"hk.where's first argument is a mask (a comparison), not {m!r}")
    if not isinstance(a, Scalar) and not isinstance(b, Scalar):
        # two numbers: i32 if both are ints, else fp32
        k = "i32" if all(_num_kind(x) == "i32" for x in (a, b)) else "fp32"
        for x in (a, b):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise TypeError(f"hk.where: {x!r} is not an element, a strip or a number")
        a = _const(a, k)
    a, b = _scalars("hk.where", a, b)
    return Scalar(arith.SelectOp(m.v, a.v, b.v, loc=_loc("where")).result, a.kind)


def maximum_(a, b) -> Scalar:
    a, b = _scalars("hk.max", a, b)
    cls = {"i32": arith.MaxSIOp, "fp32": arith.MaxNumFOp}.get(a.kind)
    if cls is None:
        raise TypeError(f"hk.max of {a.kind} elements")
    return Scalar(cls(a.v, b.v, loc=_loc("max")).result, a.kind)


def minimum_(a, b) -> Scalar:
    a, b = _scalars("hk.min", a, b)
    cls = {"i32": arith.MinSIOp, "fp32": arith.MinNumFOp}.get(a.kind)
    if cls is None:
        raise TypeError(f"hk.min of {a.kind} elements")
    return Scalar(cls(a.v, b.v, loc=_loc("min")).result, a.kind)


def abs_(x) -> Scalar:
    return _call("hk.math.abs", ("fp32",), mathd.AbsFOp, x)


#: The scalar functions of hk.math, by name: what they make inside a map (and, through a one-op
#: map, on strips for the ones that have no tile op).
SCALAR = {
    "fma": lambda a, b, c: _call("hk.math.fma", ("fp32",), mathd.FmaOp, a, b, c),
    "rcp": lambda x: _call("hk.math.rcp", ("fp32",),
                           lambda v, loc: hkd.RcpOp(ir.F32Type.get(), v, loc=loc), x),
    "popcount": lambda x: _call("hk.math.popcount", ("i32",), mathd.CtPopOp, x),
    "ctz": lambda x: _call("hk.math.ctz", ("i32",), mathd.CountTrailingZerosOp, x),
    "ceil": lambda x: _call("hk.math.ceil", ("fp32",), mathd.CeilOp, x),
    "floor": lambda x: _call("hk.math.floor", ("fp32",), mathd.FloorOp, x),
    "exp2": lambda x: _call("hk.math.exp2", ("fp32",), mathd.Exp2Op, x),
    "log2": lambda x: _call("hk.math.log2", ("fp32",), mathd.Log2Op, x),
    "sqrt": lambda x: _call("hk.math.sqrt", ("fp32",), mathd.SqrtOp, x),
    "abs": abs_,
}


# ---------------------------------------------------------------- hk.map


def _check_lowering(what: str) -> None:
    from .arch import target  # noqa: PLC0415

    if not target().strips:
        raise TypeError(f"{what}: hk.map maps strips, and {target().arch} has none")


def map_(fn: Callable, *args, name: str = "map"):
    """Element (r, c) of the result is `fn` of element (r, c) of each strip argument -- an
    `hk.Int` or a Python number argument is passed as itself. `fn` is traced once on
    `hk.map` elements (`Scalar`): Python's operators, `hk.where`, `hk.max` / `hk.min` and the
    hk.math functions, with C's semantics (module docstring). The result's dtype is what `fn`
    returns: i32, fp32 or a mask (on RDNA a mask only feeds another map)."""
    v = _v()
    if not callable(fn):
        raise TypeError(f"hk.map(fn, *args): fn is a function of the elements, not {fn!r}")
    strips = [a for a in args if isinstance(a, v.Tile)]
    if not strips:
        raise TypeError("hk.map needs at least one strip argument (its shape is the result's)")
    s0 = strips[0].ty
    if not v._is_strip_tile(s0):
        raise TypeError(f"hk.map maps strips (hk.RS), not {s0}")
    _check_lowering("hk.map")
    operands, params = [], []
    for i, a in enumerate(args):
        if isinstance(a, v.Tile):
            if (a.ty.rows, a.ty.cols, getattr(a.ty, "run", None)) != (s0.rows, s0.cols, s0.run):
                raise TypeError(f"hk.map: argument {i} is {a.ty}, every strip has {s0}'s shape "
                                f"and run")
            if a.ty.dtype.name not in ("fp32", "i32", "b1"):
                raise TypeError(f"hk.map: argument {i} is a {a.ty.dtype.name} strip; strips "
                                f"are fp32, i32 or masks (convert a 16-bit strip with .to)")
            operands.append((a.v, a.ty.dtype.name))
            params.append(None)
        elif isinstance(a, v.Int):
            operands.append((a.v, "i32"))
            params.append(None)
        elif isinstance(a, (int, float)):
            params.append(a)                   # a constant inside the region
        else:
            raise TypeError(f"hk.map: argument {i} is {a!r}: a strip, an hk.Int or a number")
    t = v._t()
    loc = t.loc(name)
    # one operand a value: an argument passed twice is one block argument
    uniq, slot = [], []
    for o, k in operands:
        j = next((j for j, (u, _) in enumerate(uniq) if u == o), None)
        if j is None:
            j = len(uniq)
            uniq.append((o, k))
        slot.append(j)
    # the region is traced into a scratch op's block, then moved to the op of the result's type
    scratch = hkd.MapOp(v.mlir_type(s0), [o for o, _ in uniq], loc=loc)
    block = ir.Block.create_at_start(scratch.regions[0], [_KINDS[k]() for _, k in uniq])
    args = [Scalar(a, k) for a, (_, k) in zip(block.arguments, uniq)]
    it = iter(slot)
    xs = [args[next(it)] if p is None else p for p in params]
    try:
        with ir.InsertionPoint(block):
            r = fn(*xs)
            if isinstance(r, (int, float)):
                raise TypeError(f"hk.map: fn returned the constant {r!r}; a constant strip is "
                                f"hk.full")
            if not isinstance(r, Scalar):
                raise TypeError(f"hk.map: fn returns an element of its arguments, not {r!r}")
            if r.kind == "bf16":
                raise TypeError("hk.map: fn returned a bf16 element; strips are fp32, i32 or "
                                "masks (convert back with .to(hk.fp32))")
            hkd.YieldOp(r.v, loc=loc)
    except BaseException:
        scratch.operation.erase()
        raise
    ty = RegTileType(_DTYPES[r.kind], s0.rows, s0.cols, s0.layout, s0.frag, run=s0.run)
    op = hkd.MapOp(v.mlir_type(ty), [o for o, _ in uniq], loc=loc)
    block.append_to(op.regions[0])
    scratch.operation.erase()
    return v.wrap(op.result, ty)


def is_int_strip(x) -> bool:
    """An i32 strip or a mask: what the operators build maps for."""
    v = _v()
    return (isinstance(x, v.Tile) and v._is_strip_tile(x.ty)
            and x.ty.dtype.name in ("i32", "b1"))


def lift(f: Callable, *args, name: str = "map"):
    """`f` of strips (and scalars): a one-op hk.map, `f` applied to its elements."""
    return map_(f, *args, name=name)
