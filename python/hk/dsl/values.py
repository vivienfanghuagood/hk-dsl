"""DSL values: what a kernel body computes with.

Each wraps an MLIR value and the type it has, and each operation on one
appends an op to the kernel being traced. Scalars are SSA (`Int`, `Bool`);
tiles and vectors are buffers, so they also have in-place forms -- `t += u`,
`t.exp_()`, `into=` -- that write the registers they already occupy, which is
what a loop-carried accumulator needs.

    v  = x.at[row, col].load(hk.RT(hk.fp32, 16, 64))   # global -> registers
    sq += (v * v).sum(axis=1)                           # row_sum, accumulated
    o.at[row, col] = (v * scale).to(hk.bf16)            # mul_row, cast, store
    acc += a @ b.T                                      # mma_ABt into acc
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .._mlir import ir
from .._mlir.dialects import arith, hk as hkd
from .module import mlir_type
from ..types import (
    FP8, CoordType, DType, GlobalType, GranuleSetType, MxScalesType, RegTileType,
    RegVecType, ScalarType, SharedTileType, SharedVecType, StageBufferType, fp32, i32,
    fp8e4m3, i8, bf16,
)
from ..target import get_target
from . import builder as B

Scalar = Union[int, float, "Int"]

#: Elementwise ops the headers specialise only for fp32 (include/rdna3, and
#: include/cdna4 with hk's include/hk/cdna.cuh). Asking for one it
#: does not have is a link error at the end of a long compile, naming a
#: template rather than the line that asked for it.
_FP32_ONLY = ("gelu", "dgelu", "sqrt", "rsqrt", "silu", "step")
_ALL_FLOAT = ("bf16", "fp16", "fp32")


def _t():
    return B.current()


def _i64(v: int):
    """An i64 attribute (an op's integer attribute: a scoped access's aux)."""
    from .._mlir import ir  # noqa: PLC0415

    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(v))


def _i32():
    return ir.IntegerType.get_signless(32)


def _check_map_dtype(op: str, ty) -> None:
    want = ("fp32",) if op in _FP32_ONLY else _ALL_FLOAT
    if ty.dtype.name not in want:
        raise TypeError(
            f"{op} has no {ty.dtype.name} implementation: on every arch hk's "
            f"headers specialise it for {', '.join(want)} only. Cast with "
            f"t.to(hk.fp32) first, or load into an fp32 tile -- the global "
            f"load converts for free. This is caught here because the "
            f"alternative is an undefined-symbol link error at the end of the "
            f"compile that names a template, not this line.")


def _is_strip_tile(ty) -> bool:
    """A strip tile (not a vector of one)."""
    return isinstance(ty, RegTileType) and _is_strip(ty)


def _is_strip(ty) -> bool:
    """A strip tile, or a vector of one."""
    if isinstance(ty, RegVecType):
        ty = ty.tile
    return isinstance(ty, RegTileType) and ty.layout == "strip"


def _no_strip(what: str, *tys) -> None:
    for ty in tys:
        if _is_strip(ty):
            raise TypeError(
                f"{what} takes WMMA-layout tiles, and {ty!r} is a strip. A "
                f"strip loads and stores, maps, reduces along rows and "
                f"columns, broadcasts its vectors and fills by column; it is "
                f"never an mma operand or an LDS tile.")


def _strip_access(what: str, ty, idx: "Coord", bounded) -> None:
    """A strip moves in elements; its col_vec loads unbounded and stores only
    through store_scalar; bounded=n (a row limit) is the tiles'."""
    if not _is_strip(ty):
        return
    arch = _t().arch
    tgt = get_target(arch)
    if not tgt.strips:
        raise TypeError(f"{what} of {ty!r}: strips (kittens::rs) are in "
                        f"include/rdna3, include/rdna4 and hk's wave64 port for gfx950, and "
                        f"this kernel is for {arch}")
    strip = ty.tile if isinstance(ty, RegVecType) else ty
    if strip.cols % (tgt.wave * strip.run):
        raise TypeError(f"{what} of {ty!r}: on {arch} a strip's chunk is "
                        f"{tgt.wave} lanes x {strip.run}, and {strip.cols} "
                        f"columns are not a whole number of them")
    if idx.unit != "element":
        raise TypeError(f"{what} of {ty!r}: a strip indexes in elements "
                        f"(x.at[r, c] or hk.elem_coord), not tiles")
    if bounded is not False and bounded is not True:
        raise TypeError(f"{what}(bounded=n) limits a tile's rows; a strip's "
                        f"bound is the tensor's, bounded=True")
    if isinstance(ty, RegVecType) and ty.kind == "col" and (
            bounded or what == "store"):
        raise TypeError(f"{what} of a strip's col_vec: it loads unbounded, one "
                        f"value a row, and stores with store_scalar")


# ---------------------------------------------------------------- base


def _deprecated(old: str, new: str) -> None:
    """A deprecation shim's warning (docs/design/dsl-surface.md §5): the old spelling still
    emits what the new one does."""
    import warnings  # noqa: PLC0415

    warnings.warn(f"{old} is deprecated: use {new} (docs/design/dsl-surface.md)",
                  DeprecationWarning, stacklevel=3)


class Value:
    """A device value of the kernel being traced."""

    __slots__ = ("v", "ty")

    def __init__(self, v: ir.Value, ty):
        self.v = v
        self.ty = ty

    __hash__ = object.__hash__

    @property
    def type(self):
        """The value's hk type (`RegTileType`, `RegVecType`, ...)."""
        return self.ty

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.ty!r}>"


def wrap(v: ir.Value, ty) -> "Value":
    """The DSL value for an MLIR value of hk type `ty`."""
    if isinstance(ty, RegTileType):
        return Tile(v, ty)
    if isinstance(ty, RegVecType):
        return Vec(v, ty)
    if isinstance(ty, SharedTileType):
        return SharedTile(v, ty)
    if isinstance(ty, SharedVecType):
        return SharedVec(v, ty)
    if isinstance(ty, GranuleSetType):
        return Granules(v, ty)
    if isinstance(ty, StageBufferType):
        return Stage(v, ty)
    if isinstance(ty, CoordType):
        return Coord(v, ty)
    if isinstance(ty, GlobalType):
        return Global(v, ty)
    if isinstance(ty, ScalarType):
        return Int(v, ty)
    if isinstance(ty, MxScalesType):
        return MxScales(v, ty)
    raise TypeError(f"no DSL value for {ty!r}")


# ---------------------------------------------------------------- scalars


def const(x: int) -> "Int":
    """An i32 literal, hoisted and emitted once per kernel."""
    t = _t()
    ty = ScalarType(i32)
    key = ("const", type(x).__name__, x, repr(ty))

    def build():
        return arith.ConstantOp(_i32(), int(x), loc=t.loc("c")).result

    return Int(t.hoisted(key, build), ty)


def as_int(x: Scalar) -> "Int":
    if isinstance(x, Int):
        return x
    if isinstance(x, Bool):
        return x.to_int()
    if isinstance(x, bool) or isinstance(x, int):
        return const(int(x))
    raise TypeError(f"expected an index (an int or a device Int), got {x!r}")


_ARITH = {"add": "AddIOp", "sub": "SubIOp", "mul": "MulIOp", "div": "DivSIOp",
          "mod": "RemSIOp", "min": "MinSIOp", "max": "MaxSIOp",
          "cdiv": "CeilDivSIOp"}
_PRED = {"lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge", "eq": "eq",
         "ne": "ne"}


def scalar_op(op: str, a: Scalar, b: Scalar) -> "Int":
    t = _t()
    va, vb = as_int(a), as_int(b)
    cls = getattr(arith, _ARITH[op])
    return Int(cls(va.v, vb.v, loc=t.loc(op)).result, ScalarType(i32))


def compare(op: str, a: Scalar, b: Scalar) -> "Bool":
    t = _t()
    va, vb = as_int(a), as_int(b)
    pred = getattr(arith.CmpIPredicate, _PRED[op])
    return Bool(arith.CmpIOp(pred, va.v, vb.v, loc=t.loc(op)).result,
                ScalarType(i32))


class Int(Value):
    """A wave-uniform 32-bit integer: an index, an extent, a loop counter.

    Python's operators build device arithmetic; `//` and `%` are C++'s
    truncating division and remainder, which agree with Python's on the
    non-negative values an index takes.
    """

    __slots__ = ()

    def __add__(self, o): return scalar_op("add", self, o)
    def __radd__(self, o): return scalar_op("add", o, self)
    def __sub__(self, o): return scalar_op("sub", self, o)
    def __rsub__(self, o): return scalar_op("sub", o, self)
    def __mul__(self, o): return scalar_op("mul", self, o)
    def __rmul__(self, o): return scalar_op("mul", o, self)
    def __floordiv__(self, o): return scalar_op("div", self, o)
    def __rfloordiv__(self, o): return scalar_op("div", o, self)
    def __mod__(self, o): return scalar_op("mod", self, o)
    def __rmod__(self, o): return scalar_op("mod", o, self)
    def __neg__(self): return scalar_op("sub", 0, self)
    def __lt__(self, o): return compare("lt", self, o)
    def __le__(self, o): return compare("le", self, o)
    def __gt__(self, o): return compare("gt", self, o)
    def __ge__(self, o): return compare("ge", self, o)
    def __eq__(self, o): return compare("eq", self, o)  # type: ignore[override]
    def __ne__(self, o): return compare("ne", self, o)  # type: ignore[override]
    __hash__ = Value.__hash__

    def __truediv__(self, o):
        raise TypeError("an index divides with //; there is no rounding "
                        "division on a device integer")

    def __bool__(self):
        from .rewrite import unrewritten_hint  # noqa: PLC0415

        raise TypeError(
            "a device Int was used where Python needs a bool. A runtime "
            "condition is a comparison (`i < n`), and it can only steer an "
            "`if` or a loop the kernel compiles, not Python logic at trace "
            "time." + unrewritten_hint())

    def __index__(self):
        from .rewrite import unrewritten_hint  # noqa: PLC0415

        raise TypeError(
            "a device Int was used where Python needs an int -- a tile "
            "extent, a Python range, a list index. Those are decided at "
            "trace time; make the value a constexpr parameter, or use it as "
            "a loop bound (`for i in range(n)` compiles a loop for a device "
            "n)." + unrewritten_hint())

    __int__ = __index__


class Bool(Value):
    """A comparison's answer: steers an `if`, or combines with & | ~."""

    __slots__ = ()

    def _logic(self, op: str, o) -> "Bool":
        t = _t()
        if isinstance(o, bool):
            o = Bool(arith.ConstantOp(ir.IntegerType.get_signless(1), int(o),
                                      loc=t.loc()).result, self.ty)
        if not isinstance(o, Bool):
            raise TypeError(f"{op} combines two conditions, not {o!r}")
        cls = {"and": arith.AndIOp, "or": arith.OrIOp, "xor": arith.XOrIOp}[op]
        return Bool(cls(self.v, o.v, loc=t.loc(op)).result, self.ty)

    def __and__(self, o): return self._logic("and", o)
    def __rand__(self, o): return self._logic("and", o)
    def __or__(self, o): return self._logic("or", o)
    def __ror__(self, o): return self._logic("or", o)
    def __xor__(self, o): return self._logic("xor", o)

    def __invert__(self):
        return self._logic("xor", True)

    def to_int(self) -> Int:
        """1 where true, else 0, as an Int."""
        t = _t()
        return Int(arith.ExtUIOp(_i32(), self.v, loc=t.loc()).result,
                   ScalarType(i32))

    def __bool__(self):
        from .rewrite import unrewritten_hint  # noqa: PLC0415

        raise TypeError(
            "a runtime condition was used where Python needs a bool -- an "
            "`and`, an `or`, a `not`, or an `if` with a `return` or `break` "
            "inside. Combine conditions with &, | and ~, and branch with a "
            "plain `if`, which the kernel compiles." + unrewritten_hint())


def i1(x) -> ir.Value:
    """`x` as a branch condition.

    A comparison, or a wave-level any_ne -- not a bare index. `if c:` reads
    like "if c is in range" and means "if c is not zero", which on the c == 0
    warp is the opposite, so that is refused rather than compiled."""
    t = _t()
    if isinstance(x, Bool):
        return x.v
    if isinstance(x, Int):
        owner = x.v.owner if isinstance(x.v, ir.OpResult) else None
        name = owner.name if owner is not None else ""
        if name not in ("hk.any_ne", "hk.any_above", "arith.constant"):
            raise TypeError(
                "a branch condition is an index, not a comparison. Write the "
                "test you mean -- `if i > 0:` -- rather than relying on "
                "nonzero: for an index, the two differ exactly at 0.")
        return arith.TruncIOp(ir.IntegerType.get_signless(1), x.v,
                              loc=ir.Location.unknown()).result
    raise TypeError(f"a branch condition is a comparison, got {x!r}")


# ---------------------------------------------------------------- coords


class Coord(Value):
    """A (b, d, r, c) index, in tiles or in elements."""

    __slots__ = ()

    @property
    def unit(self) -> str:
        return self.ty.unit


def coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0, *,
          unit: str = "tile") -> Coord:
    """A (batch, depth, row, col) index. `unit` is "tile" (C++ `coord<RT>`)
    or "element" (`coord<>`); picking the wrong one reads the wrong memory
    without any diagnostic, which is why it is in the type."""
    if unit not in ("tile", "element"):
        raise ValueError(f"coord unit must be 'tile' or 'element', got {unit!r}")
    t = _t()
    parts = [as_int(x).v for x in (b, d, r, c)]
    ty = CoordType(unit)
    return Coord(hkd.CoordOp(mlir_type(ty), *parts, loc=t.loc("idx")).result, ty)


def tile_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Coord:
    """A (batch, depth, row, col) index in tiles: row r is rows
    [r*R, (r+1)*R) for a load of an R-row tile. `x[hk.tile_coord(...)]`."""
    return coord(b, d, r, c, unit="tile")


def elem_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Coord:
    """A (batch, depth, row, col) index in elements: what `x.at[r, c]`
    builds. Strips index only this way."""
    return coord(b, d, r, c, unit="element")


def _pad4(idx) -> tuple:
    idx = idx if isinstance(idx, tuple) else (idx,)
    if len(idx) > 4:
        raise IndexError(f"a global is 4D (b, d, r, c); got {len(idx)} indices")
    return (0,) * (4 - len(idx)) + idx


# ---------------------------------------------------------------- globals


class _Indexer:
    __slots__ = ("tensor", "unit")

    def __init__(self, tensor: "Global", unit: str):
        self.tensor, self.unit = tensor, unit

    def __getitem__(self, idx) -> "View":
        return View(self.tensor, coord(*_pad4(idx), unit=self.unit))

    def __setitem__(self, idx, value) -> None:
        self[idx].store(value)


class Global(Value):
    """A global tensor parameter, 4D (b, d, r, c).

    `x[idx]` is a view at a coordinate; `x.at[r, c]` builds an element
    coordinate and `x.tile[r, c]` a tile one, left-padding with zeros.
    """

    __slots__ = ("name",)

    def __init__(self, v, ty, name: str = ""):
        super().__init__(v, ty)
        self.name = name

    def _extent(self, axis: str) -> Int:
        t = _t()
        return Int(hkd.ExtentOp(_i32(), self.v, getattr(hkd.Extent, axis),
                                loc=t.loc(axis)).result, ScalarType(i32))

    @property
    def batch(self) -> Int: return self._extent("batch")
    @property
    def depth(self) -> Int: return self._extent("depth")
    @property
    def rows(self) -> Int: return self._extent("rows")
    @property
    def cols(self) -> Int: return self._extent("cols")

    @property
    def dtype(self) -> DType:
        return self.ty.dtype

    @property
    def at(self) -> _Indexer:
        return _Indexer(self, "element")

    @property
    def tile(self) -> _Indexer:
        return _Indexer(self, "tile")

    def __getitem__(self, idx) -> "View":
        if not isinstance(idx, Coord):
            raise TypeError(f"index a global with a coordinate (hk.coord, "
                            f"x.at[...], x.tile[...]), got {idx!r}")
        return View(self, idx)

    def __setitem__(self, idx, value) -> None:
        self[idx].store(value)

    def _indexed(self, what: str, tile, ids, at, count, col):
        if not isinstance(tile, RegTileType) or tile.layout != "row":
            raise TypeError(f"{what} moves row-layout tiles, not {tile!r}: a "
                            f"lane's elements of a row tile are all in its "
                            f"row, which is what lets it address its own row")
        if not isinstance(ids, Global) or ids.dtype is not i32:
            raise TypeError(f"{what} takes its row indices from an int32 "
                            f"global, not {ids!r}")
        n = ids.cols if count is None else count
        return (ids.v, as_int(at).v, as_int(n).v, as_int(col).v)

    def load_elems(self, idx: "Tile", dtype: Optional[DType] = None, *, runs: bool = False,
                   row=None) -> "Tile":
        """Element by element at computed indices (gfx950): element (r, c) of the result is element
        `idx[r, c]` of this tensor, counted flat (rows over batch and depth, at the row stride),
        converted to `dtype` (fp32 or i32; default fp32, i32 for an i32 tensor). An index below
        zero or past the tensor reads 0 -- a table lookup with a fallback, a key table's slots.
        `idx` is an i32 strip (hk.map_cpp computes one).

        `runs=True`: the kernel says each lane's run comes from consecutive elements from its first
        element's index, aligned to the run's bytes; the run is one load (1, 2, 4, 8 or 16 k bytes),
        read as 0 if that first index is out of range -- a lane's 4 fp32 in one 16-byte load, not
        four gathered ones. `row`: as store_elems."""
        if not _is_strip_tile(idx.ty) or idx.ty.dtype.name != "i32":
            raise TypeError(f"load_elems takes an i32 strip of indices, not {idx.ty}")
        if runs:
            nb = idx.ty.run * self.dtype.bits // 8
            if nb not in (1, 2, 4, 8) and nb % 16:
                raise TypeError(f"load_elems(runs=True): a run of {nb} bytes; 1, 2, 4, 8 or 16 k")
        dtype = (i32 if self.dtype is i32 else fp32) if dtype is None else dtype
        ty = RegTileType(dtype, idx.ty.rows, idx.ty.cols, idx.ty.layout, idx.ty.frag, run=idx.ty.run)
        t = _t()
        aux = scope_aux(self, "load_elems") if getattr(self.ty, "scope", "") else 0
        return wrap(hkd.LoadElemsOp(mlir_type(ty), self.v, idx.v, aux=_i64(aux), runs=bool(runs),
                                    row=None if row is None else as_int(row).v, loc=t.loc("elems")).result, ty)

    def store_elems(self, val: "Tile", idx: "Tile", *, runs: bool = False, row=None) -> None:
        """Element by element at computed indices (gfx950): element (r, c) of `val` (an fp32 or
        i32 strip), converted to this tensor's type, to element `idx[r, c]` (flat, as
        load_elems). An index below zero or past the tensor writes nothing: -1 masks an element
        out. Two elements to one index race.

        `runs=True`: the kernel says each lane's run goes to consecutive elements from its first
        element's index, aligned to the run's bytes; the run is one store (1, 2, 4, 8 or 16 k
        bytes), masked by that first index -- 8 fp8 values in one 8-byte store, not eight.

        `row` (a device Int): the indices count from that row, whose base is formed in 64 bits and
        whose range is the one row -- a tensor past a 32-bit offset (a paged cache of many GB)
        addressed a row (block) at a time."""
        if not _is_strip_tile(idx.ty) or idx.ty.dtype.name != "i32":
            raise TypeError(f"store_elems takes an i32 strip of indices, not {idx.ty}")
        if runs:
            nb = idx.ty.run * self.dtype.bits // 8
            if nb not in (1, 2, 4, 8) and nb % 16:
                raise TypeError(f"store_elems(runs=True): a run of {nb} bytes; 1, 2, 4, 8 or 16 k")
        t = _t()
        aux = scope_aux(self, "store_elems") if getattr(self.ty, "scope", "") else 0
        hkd.StoreElemsOp(self.v, val.v, idx.v, aux=_i64(aux), runs=bool(runs),
                         row=None if row is None else as_int(row).v, loc=t.loc())

    def post_elems(self, val: "Tile", idx: "Tile", tag, *, runs: bool = False, bf16: bool = False) -> None:
        """Tagged mailbox pairs, an element at a time (gfx950): element (r, c) of `val` (fp32 or i32
        strip) and `tag` go to pair `idx[r, c]` of this scoped int32 tensor (int32 elements 2 i,
        2 i + 1) as one 8-byte store of its scope; an index below zero posts nothing. No drain --
        the value travels with its tag -- so a hand-off between CTAs is one store and, on the
        other side, one poll (dsv41_mono's Mailbox; hk.comm.post for one scalar).

        `runs=True`: each lane's run goes to consecutive pairs from its first element's index (the
        lane's other indices are not read), two pairs a 16-byte store; the run is a whole number of
        pairs of pairs. `bf16=True` (with runs): two values a pair, as bf16 -- the TOP HALVES of the
        fp32 values, so they must already be bf16 values (rounded by the caller); the pairs from a
        run's first index are then run / 2."""
        if bf16 and not runs:
            raise TypeError("post_elems(bf16=True) is a layout of runs: runs=True as well")
        if getattr(self.ty, "scope", "") not in HANDOFF_SCOPES:
            raise TypeError(f"post_elems on {self.name!r}, an unscoped tensor: a mailbox other CTAs "
                            f"read is hk.Tensor[hk.i32, \"device\"] (\"system\": other GPUs)")
        aux = scope_aux(self, "post_elems")
        t = _t()
        hkd.PostElemsOp(self.v, val.v, idx.v, as_int(tag).v, _i64(aux), runs=bool(runs), bf16=bool(bf16), loc=t.loc())

    def take_elems(self, idx: "Tile", tag, dtype: Optional[DType] = None, *,
                   timeout_s: float = 2.0, runs: bool = False, bf16: bool = False) -> "Tile":
        """The other side of post_elems: element (r, c) is the value of pair `idx[r, c]` once its tag
        is `tag`; every element's pair is loaded with the tensor's scope, and all are re-loaded
        (s_sleep between) while any is stale. An index below zero is ready at 0. `dtype`: fp32
        (default) or i32. After `timeout_s` the kernel traps (0: never) -- a lost post is an error,
        not a hang. `runs=True` / `bf16=True`: the layout post_elems wrote with them."""
        if bf16 and not runs:
            raise TypeError("take_elems(bf16=True) is a layout of runs: runs=True as well")
        if getattr(self.ty, "scope", "") not in HANDOFF_SCOPES:
            raise TypeError(f"take_elems on {self.name!r}, an unscoped tensor: a mailbox other CTAs "
                            f"read is hk.Tensor[hk.i32, \"device\"] (\"system\": other GPUs)")
        aux = scope_aux(self, "take_elems")
        dtype = fp32 if dtype is None else dtype
        ty = RegTileType(dtype, idx.ty.rows, idx.ty.cols, idx.ty.layout, idx.ty.frag, run=idx.ty.run)
        t = _t()
        from .comm import REALTIME_HZ  # noqa: PLC0415
        if runs and idx.ty.run % 2:
            raise TypeError(f"take_elems(runs=True): runs of pairs of pairs, not run {idx.ty.run}")
        return wrap(hkd.TakeElemsOp(mlir_type(ty), self.v, idx.v, as_int(tag).v,
                                    _i64(int(round(timeout_s * REALTIME_HZ))), _i64(aux),
                                    runs=bool(runs), bf16=bool(bf16), loc=t.loc("taken")).result, ty)

    def gather(self, tile, ids, at=None, *, count=None, col=0, div=None,
               into=None) -> "Tile":
        """Rows by index: tile row r is row `ids[at + r]` of this tensor (rows flattened over
        batch and depth), from column `col` -- `ids` an `hk.RowIndex`, which also says how an
        index names a row (`div`), or an int32 tensor and `at` (and `count`). An index outside the
        tensor, or a position at or past `count` (default: all of `ids`), reads zeros -- the
        padding a MoE sort fills its index list with costs no branch. `div=` here is deprecated:
        `hk.RowIndex(ids, at, div=...)`."""
        ids, at, count, div = _plain_rows("gather", ids, at, count, div=div)
        if div is None:
            div = 1
        idx, at_v, n_v, col_v = self._indexed("gather", tile, ids, at, count, col)
        return _make(hkd.GatherRowsOp, tile,
                     (self.v, idx, at_v, n_v, as_int(div).v, col_v),
                     into=into, name="gat")

    def scatter(self, value: "Tile", ids, at=None, *, count=None,
                col=0, sort_topk: Optional[int] = None, add: bool = False,
                staged: Optional[int] = None) -> None:
        """Rows by index: tile row r goes to row `ids[at + r]`, from column `col` -- `ids` an
        `hk.RowIndex`, or an int32 tensor and `at` (and `count`). An index outside the tensor, or
        a position at or past `count`, writes nothing. Two rows naming one row race. With an
        index of AITER's MoE sort ids (`RowIndex(..., sort_topk=k)`: token | choice << 24) this
        tensor is [k * M, ...] choice-major -- row r goes to row choice * M + token, and padding
        (token >= M) nowhere.

        `add=True` adds the rows atomically instead (an fp32 or bf16 global; bf16 adds element
        pairs, gfx950's global_atomic_pk_add_bf16). With sort ids the tensor is then [M, ...] and
        every choice of a token adds into its row: the top-k reduce, in no fixed order -- the
        result varies run to run by the rounding of the adds.

        An index staged in LDS by `g.stage_rows(ix, N, keep_choice=True)` is read there: a global
        index load is a vmcnt wait, which on gfx950 retires every load and store issued before
        it, in order -- in an epilogue between K loops, the next loop's whole prefetch and this
        epilogue's own earlier stores (K1 k1-prefill c36). `sort_topk=` and `staged=` here are
        deprecated: they are `hk.RowIndex`'s and `g.stage_rows`'."""
        if isinstance(ids, RowIndex):
            if sort_topk is not None or staged is not None:
                raise TypeError("scatter: sort_topk / staged are the RowIndex's, not keywords "
                                "next to one")
            if ids.stage_n and not ids.keep_choice:
                raise TypeError("scatter: this index was staged for gathers (the token rows); a "
                                "scatter reads the raw ids -- "
                                "g.stage_rows(ix, rows, keep_choice=True)")
            if ids.stage_n:
                ids._check_staged("scatter")
            sort_topk, staged = ids.sort_topk, ids.stage_n
            if ids.div != 1:
                raise TypeError("scatter: an index with div= names rows for a gather")
            ids, at, count = ids.ids, ids.at, ids.count if count is None else count
        else:
            if sort_topk is not None:
                _deprecated("scatter(..., sort_topk=k)",
                            "scatter(t, hk.RowIndex(ids, at, sort_topk=k))")
            if staged is not None:
                _deprecated("scatter(..., staged=N)",
                            "scatter(t, g.stage_rows(hk.RowIndex(ids, at), N, keep_choice=True))")
            if at is None:
                raise TypeError("scatter: rows by index take an hk.RowIndex, or ids and at")
            sort_topk, staged = sort_topk or 0, staged or 0
        value = _vec_of(value)
        if staged and get_target(_t().arch).arch != "gfx950":
            raise TypeError(f"scatter(staged=...): gfx950's LDS-staged ids, not {_t().arch}")
        if not isinstance(value, Tile):
            raise TypeError(f"scatter stores a tile, not {value!r}")
        ops = self._indexed("scatter", value.ty, ids, at, count, col)
        hkd.ScatterRowsOp(self.v, value.v, *ops, sort_topk=int(sort_topk), add=bool(add),
                          staged=int(staged), loc=_t().loc())


class RowIndex:
    """Which rows a gather, a scatter or a gathered LDS copy moves: row r of the tile is named by
    `ids[at + r]`, an int32 global, for positions below `count` (default: all of `ids`) -- and
    how such an index names a row:

    - plain: it is the row;
    - `div=k`: the row is `ids[at + r] // k` (a MoE's top-k, its sorted list counting
      (token, choice) pairs); for a gather;
    - `sort_topk=k`: AITER's MoE sort ids, token | choice << 24, padding token >= M. A gather or
      a gathered copy reads the token's row; a scatter writes row choice * M + token of a
      [k * M, ...] tensor, and with add=True adds into the token's row.

    Positions at or past `count`, padding and indices outside the tensor read zero and write
    nothing. `g.stage_rows(ix, rows)` copies [at, at + rows) into LDS and returns the staged
    index the copies then read there (no vmcnt wait for an index). Made in the kernel body; no
    instruction of its own."""

    __slots__ = ("ids", "at", "count", "div", "sort_topk", "stage_n", "keep_choice")

    def __init__(self, ids: "Global", at=0, *, count=None, div: int = 1, sort_topk: int = 0):
        if not isinstance(ids, Global) or ids.dtype is not i32:
            raise TypeError(f"hk.RowIndex: the indices are an int32 global, not {ids!r}")
        if int(div) < 1 or int(sort_topk) < 0 or (int(div) != 1 and int(sort_topk)):
            raise TypeError(f"hk.RowIndex: div >= 1 or sort_topk >= 1, not both "
                            f"(div={div}, sort_topk={sort_topk})")
        self.ids, self.at, self.count = ids, at, count
        self.div, self.sort_topk = int(div), int(sort_topk)
        self.stage_n, self.keep_choice = 0, False

    def offset(self, n) -> "RowIndex":
        """The same index from position at + n (a later tile of the same block). A staged
        index stays staged: [at, at + n + the tile's rows) must lie inside what was staged."""
        out = self._copy()
        out.at = self.at + n
        return out

    def staged(self, rows: int, *, keep_choice: bool = False) -> "RowIndex":
        """This index as `g.stage_rows(self, rows, keep_choice=...)` stages it, without the copy:
        for code written before the place that stages it (a loop body's copies, defined above
        the loop). A consumer still refuses it until a `stage_rows` has made that copy."""
        if self.stage_n:
            raise TypeError(f"{self!r} is staged already")
        if int(rows) <= 0:
            raise TypeError(f"RowIndex.staged: rows > 0, not {rows}")
        out = self._copy()
        out.stage_n, out.keep_choice = int(rows), bool(keep_choice)
        return out

    def _check_staged(self, what: str) -> None:
        """A staged index is read from LDS: some stage_rows of its size and form must come
        first."""
        done = getattr(_t(), "staged_rows", None) or set()
        if (self.stage_n, self.keep_choice) not in done:
            raise TypeError(f"{what}: {self!r} is read from LDS, but no g.stage_rows of "
                            f"{self.stage_n} {'raw ids' if self.keep_choice else 'token rows'} "
                            f"has made that copy before this point")

    def _copy(self) -> "RowIndex":
        out = RowIndex.__new__(RowIndex)
        for k in RowIndex.__slots__:
            setattr(out, k, getattr(self, k))
        return out

    def __repr__(self) -> str:
        bits = [f"div={self.div}"] if self.div != 1 else []
        bits += [f"sort_topk={self.sort_topk}"] if self.sort_topk else []
        bits += [f"staged={self.stage_n}" + (", keep_choice" if self.keep_choice else "")] \
            if self.stage_n else []
        return f"RowIndex({self.ids!r}, {self.at!r}{''.join(', ' + b for b in bits)})"


def _plain_rows(what: str, ids, at, count, *, div=None):
    """(ids, at, count, div) of a gather's rows: from a RowIndex, or the deprecated keywords."""
    if isinstance(ids, RowIndex):
        if at is not None or div is not None:
            raise TypeError(f"{what}: at / div are the RowIndex's, not arguments next to one")
        if ids.stage_n:
            raise TypeError(f"{what}: a staged index is read by g.load(gather=) and scatter; a "
                            f"register gather reads its ids from the global")
        if ids.sort_topk:
            raise TypeError(f"{what}: a register gather takes plain or div= indices, not sort ids")
        return ids.ids, ids.at, ids.count if count is None else count, ids.div
    if at is None:
        raise TypeError(f"{what}: rows by index take an hk.RowIndex, or ids and at")
    if div is not None:
        _deprecated(f"{what}(..., div=k)", f"{what}(ty, hk.RowIndex(ids, at, div=k))")
    return ids, at, count, div


#: A scoped tensor's cache-policy bits on gfx950: SC1 (device scope) or
#: SC0|SC1 (system scope). Measured (KernelWiki gfx950-cross-gpu-memory-model):
#: without them a store stays in this XCD's L2 and a load re-reads a stale line.
#: "stream" is not a scope but a policy: NT (non-temporal), for data a launch reads
#: once -- a weight stream that should not displace what the L2 holds for reuse.
SCOPE_AUX = {"": 0, "device": 16, "system": 17, "stream": 2}
#: the annotations a hand-off (mailbox pair, flag) needs: another CTA's or GPU's stores seen
HANDOFF_SCOPES = ("device", "system")


def scope_aux(tensor: "Global", what: str, tile=None) -> int:
    """The aux field of an access to `tensor` (0 for an unscoped one); checks
    that the access can carry it."""
    scope = getattr(tensor.ty, "scope", "")
    if not scope:
        return 0
    arch = _t().arch
    if get_target(arch).arch != "gfx950":
        raise TypeError(f"{what} through a {scope}-scoped tensor: scoped accesses are gfx950's "
                        f"(on {arch} a peer's writes stay stale in the L2 and no instruction "
                        f"invalidates it -- KernelWiki gfx1100-peer-write-l2-stale)")
    if tile is not None:
        ok = isinstance(tile, RegTileType) and not _is_strip(tile)
        ok = ok or (isinstance(tile, RegVecType) and tensor.dtype.bits == 32
                    and tile.kind == "col" and tile.tile.layout == "row")
        if not ok:
            raise TypeError(f"{what} through a {scope}-scoped tensor moves register tiles, or "
                            f"col_vecs of a row tile with a 32-bit tensor (a per-row scale), not "
                            f"{tile!r} with {tensor.dtype.name}")
    return SCOPE_AUX[scope]


class View:
    """A global at a coordinate: what a load reads and a store writes."""

    __slots__ = ("tensor", "idx")

    def __init__(self, tensor: "Global", idx: Coord):
        self.tensor, self.idx = tensor, idx

    def load(self, tile, *, into: Optional["Value"] = None,
             bounded=False, period: Optional[int] = None):
        """Global -> register tile or vector. The element type converts on the
        way in, for free. `bounded=True` predicates rows past the global's
        last (they read zero and touch no memory); `bounded=n`, an int or a
        device Int, also predicates rows at or past `n` counted from the
        tile's first -- for a tile whose rows end before the tensor's.

        `period` (a strip, unbounded): column c of the strip reads column
        c % period from the view's -- `period` values repeated along it, as
        one token's angles under each of its heads."""
        if not isinstance(tile, (RegTileType, RegVecType)):
            raise TypeError(f"load destination type is {tile!r}, expected a "
                            f"tile or a vector type")
        _strip_access("load", tile, self.idx, bounded)
        if (bounded is not False and not isinstance(tile, RegTileType)
                and not _is_strip(tile)):
            raise TypeError(f"load(bounded=...) predicates tile rows; {tile!r} "
                            f"is not a tile")
        if period is not None:
            if not _is_strip(tile) or bounded is not False:
                raise TypeError(f"load(period=...) repeats an unbounded strip's "
                                f"columns, not {tile!r}" +
                                (" with bounded" if bounded is not False else ""))
            if not isinstance(period, int) or period <= 0 or period % tile.run:
                raise TypeError(f"load(period={period!r}): a positive number of "
                                f"whole runs of {tile.run}")
            from .._mlir import ir  # noqa: PLC0415

            return _make(hkd.LoadOp, tile, (self.tensor.v, self.idx.v), into=into,
                         name="ld", bounded=False,
                         period=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), period))
        aux = scope_aux(self.tensor, "load", tile)
        if aux:
            if bounded is not True and bounded is not False:
                if not isinstance(tile, RegTileType) or _is_strip(tile):
                    raise TypeError("load(bounded=n) through a scoped tensor: register tiles only")
                return _make(hkd.LoadRowsOp, tile, (self.tensor.v, self.idx.v, as_int(bounded).v),
                             into=into, name="ld", aux=_i64(aux))
            return _make(hkd.LoadOp, tile, (self.tensor.v, self.idx.v), into=into, name="ld",
                         bounded=bounded, aux=_i64(aux))
        if bounded is True or bounded is False:
            return _make(hkd.LoadOp, tile, (self.tensor.v, self.idx.v),
                         into=into, name="ld", bounded=bounded)
        return _make(hkd.LoadRowsOp, tile,
                     (self.tensor.v, self.idx.v, as_int(bounded).v),
                     into=into, name="ld")

    def store(self, value, *, bounded=False) -> None:
        """Register tile or vector -> global. `bounded` as for load: True
        drops rows past the global's last, `n` also rows at or past `n`."""
        value = _vec_of(value)
        if not isinstance(value, (Tile, Vec)):
            raise TypeError(f"store value is {value!r}, expected a tile or a vector")
        _strip_access("store", value.ty, self.idx, bounded)
        if (bounded is not False and not isinstance(value, Tile)
                and not _is_strip(value.ty)):
            raise TypeError("store(bounded=...) predicates tile rows; the "
                            "value is not a tile")
        t = _t()
        aux = scope_aux(self.tensor, "store", value.ty)
        if aux:
            if bounded is not True and bounded is not False:
                raise TypeError("store(bounded=n) through a scoped tensor: not supported; "
                                "scoped stores are bounded by the tensor's rows")
            hkd.StoreOp(self.tensor.v, value.v, self.idx.v, bounded=bounded, aux=_i64(aux),
                        loc=t.loc())
        elif bounded is True or bounded is False:
            hkd.StoreOp(self.tensor.v, value.v, self.idx.v, bounded=bounded,
                        loc=t.loc())
        else:
            hkd.StoreRowsOp(self.tensor.v, value.v, self.idx.v,
                            as_int(bounded).v, loc=t.loc())

    def load_scalar(self, *, invariant: bool = False) -> Int:
        """One int32 element -> a wave-uniform scalar (a page index, a length).

        `invariant`: nothing writes the element while the kernel runs (a page
        table, the sequence lengths), so on gfx950 it is a scalar load
        (s_load_dword), counted on lgkmcnt -- not a vector load whose wait
        also waits out every copy issued before it."""
        if self.idx.unit != "element":
            raise TypeError(
                f"load_scalar takes an element coordinate, got coord<"
                f"{self.idx.unit}>. It reads one element, so a tile-unit index "
                f"would be off by the tile size -- index with x.at[...] or "
                f"hk.elem_coord.")
        if self.tensor.dtype is not i32:
            raise TypeError(
                f"load_scalar reads an integer index, so the global must be "
                f"i32, not {self.tensor.dtype.name}. A page table, a sequence "
                f"length and a slot mapping are all int32 tensors; a float out "
                f"of memory is a load into a vector.")
        t = _t()
        return Int(hkd.LoadScalarOp(_i32(), self.tensor.v, self.idx.v,
                                    invariant=bool(invariant),
                                    loc=t.loc("scalar")).result, ScalarType(i32))

    def store_scalar(self, vec: "Vec") -> None:
        """One entry of a *uniform* register vector -> one element."""
        if self.idx.unit != "element":
            raise TypeError(
                f"store_scalar takes an element coordinate, got coord<"
                f"{self.idx.unit}>. It writes one element, so a tile-unit index "
                f"would be off by the tile size -- index with x.at[...].")
        if not isinstance(vec, Vec) or not vec.ty.uniform:
            got = vec.ty if isinstance(vec, Value) else vec
            raise TypeError(
                f"store_scalar needs a uniform vector, got {got}. Every entry "
                f"of a uniform vector holds the same value, so writing one of "
                f"them is writing the answer; writing one entry of anything "
                f"else would pick an arbitrary row's value and be wrong on the "
                f"other fifteen -- silently, since the shapes all agree. "
                f"Uniformity comes from hk.collective.fold_rows, which is what "
                f"a FOLD'd kernel already calls to combine its row; carry that "
                f"vector here, or use x[idx].store(v) if the sixteen entries "
                f"really are sixteen rows.")
        t = _t()
        hkd.StoreScalarOp(self.tensor.v, vec.v, self.idx.v, loc=t.loc())

    def atomic_add(self, value: "Tile") -> None:
        """Add an fp32 tile into this float global, element by element,
        atomically: the reduction for partials in different workgroups."""
        value = _vec_of(value)
        _no_strip("atomic_add", getattr(value, "ty", None))
        if not isinstance(value, Tile) or value.ty.dtype is not fp32:
            raise TypeError(f"atomic_add adds an fp32 tile, not {value!r}: a "
                            f"16-bit operand tile is replicated across the "
                            f"wave halves and would be added twice")
        if self.tensor.dtype is not fp32:
            raise TypeError(f"atomic_add adds into an fp32 global, not "
                            f"{self.tensor.dtype.name}")
        hkd.AtomicAddOp(self.tensor.v, value.v, self.idx.v, loc=_t().loc())

    def _atomic_scalar(self, kind: str, value) -> Int:
        if self.idx.unit != "element":
            raise TypeError(f"atomic_{kind}_scalar takes an element coordinate "
                            f"(x.at[...]), not coord<{self.idx.unit}>")
        if self.tensor.dtype is not i32:
            raise TypeError(f"atomic_{kind}_scalar works on an int32 global, "
                            f"not {self.tensor.dtype.name}")
        aux = scope_aux(self.tensor, f"atomic_{kind}_scalar")
        op = hkd.AtomicScalarOp(_i32(), self.tensor.v, self.idx.v,
                                as_int(value).v, kind, aux=_i64(aux), loc=_t().loc("old"))
        return Int(op.result, ScalarType(i32))

    def atomic_add_scalar(self, value) -> Int:
        """One int32 atomic add per warp; the element's value before it, in
        every lane. W warps that call it add W times."""
        return self._atomic_scalar("add", value)

    def atomic_max_scalar(self, value) -> Int:
        """One int32 atomic max per warp; the value before it."""
        return self._atomic_scalar("max", value)

    def atomic_min_scalar(self, value) -> Int:
        """One int32 atomic min per warp; the value before it."""
        return self._atomic_scalar("min", value)

    def atomic_or_scalar(self, value) -> Int:
        """One int32 atomic or per warp; the value before it (a set of flags: bit w = workgroup w)."""
        return self._atomic_scalar("or", value)

    def atomic_and_scalar(self, value) -> Int:
        """One int32 atomic and per warp; the value before it."""
        return self._atomic_scalar("and", value)

    def atomic_xor_scalar(self, value) -> Int:
        """One int32 atomic xor per warp; the value before it."""
        return self._atomic_scalar("xor", value)

    def atomic_exch_scalar(self, value) -> Int:
        """One int32 atomic exchange per warp: stores `value`, returns the value before it (a
        lock's take, a slot's claim). W warps that call it exchange W times."""
        return self._atomic_scalar("exch", value)


# ---------------------------------------------------------------- tiles


def _make(cls, ty, operands, *, into=None, name: str, mark: Optional[str] = None, **attrs):
    """Make a fresh value of type `ty`, or write `into` an existing one. `mark`: a unit attribute
    set on the op (the C++ emitter reads it)."""
    t = _t()
    if into is None:
        op = cls(mlir_type(ty), *operands, loc=t.loc(name), **attrs)
        if mark:
            op.operation.attributes[mark] = _unit_attr()
        return wrap(op.result, ty)
    if not isinstance(into, Value):
        raise TypeError(f"into= takes a tile or vector, got {into!r}")
    plain = getattr(into.ty, "plain", into.ty)
    want = getattr(ty, "plain", ty)
    if plain != want or (getattr(into.ty, "uniform", False)
                         and not getattr(ty, "uniform", False)):
        raise TypeError(f"into is {into.ty} but the result is {ty}. An into= "
                        f"that does not match is a different tile, not a "
                        f"conversion.")
    op = cls(None, *operands, dst=into.v, loc=t.loc(name), **attrs)
    if mark:
        op.operation.attributes[mark] = _unit_attr()
    return into


def _unit_attr():
    from .._mlir import ir  # noqa: PLC0415

    return ir.UnitAttr.get()


def _meet(a, b):
    """The result type of an elementwise op: a's, minus a refinement b lacks."""
    return a if getattr(b, "uniform", True) else a.plain


_BINARY = {"add": hkd.AddOp, "sub": hkd.SubOp, "mul": hkd.MulOp,
           "div": hkd.DivOp, "max": hkd.MaxOp, "min": hkd.MinOp}
_UNARY = {"exp": hkd.ExpOp, "exp2": hkd.Exp2Op, "log": hkd.LogOp,
          "log2": hkd.Log2Op, "abs": hkd.AbsOp, "relu": hkd.ReluOp,
          "copy": hkd.CopyOp, "gelu": hkd.GeluOp, "dgelu": hkd.DgeluOp,
          "sqrt": hkd.SqrtOp, "rsqrt": hkd.RsqrtOp, "silu": hkd.SiluOp,
          "step": hkd.StepOp, "swap_halves": hkd.SwapHalvesOp}
_BROADCAST = {("add", "row"): hkd.AddRowOp, ("sub", "row"): hkd.SubRowOp,
              ("mul", "row"): hkd.MulRowOp, ("div", "row"): hkd.DivRowOp,
              ("add", "col"): hkd.AddColOp, ("sub", "col"): hkd.SubColOp,
              ("mul", "col"): hkd.MulColOp, ("div", "col"): hkd.DivColOp}
_REDUCE = {("max", "row"): hkd.RowMaxOp, ("min", "row"): hkd.RowMinOp,
           ("sum", "row"): hkd.RowSumOp, ("prod", "row"): hkd.RowProdOp,
           ("max", "col"): hkd.ColMaxOp, ("min", "col"): hkd.ColMinOp,
           ("sum", "col"): hkd.ColSumOp, ("prod", "col"): hkd.ColProdOp}


def binary(op: str, a: "Value", b, *, into=None, reduce: bool = False):
    """Elementwise `op` on a tile or vector and another of the same type, a
    device scalar, or a number. A tile and a vector broadcast: a column
    vector (one value per row) applies along rows (`mul_row`), a row vector
    along columns. `reduce=True` on a max or min of two vectors: the
    combine of two partial reductions, with a reduction's NaN and signed-zero
    rules (docs/language.md) rather than the elementwise op's."""
    if reduce:
        if op not in ("max", "min") or not isinstance(a, Vec) or not isinstance(b, Vec):
            raise TypeError(f"binary(reduce=True) combines two vectors' max or min, not {op} "
                            f"of {getattr(a, 'ty', a)!r} and {getattr(b, 'ty', b)!r}")
    if not isinstance(a, (Tile, Vec)):
        raise TypeError(f"{op}: left operand is {a!r}, expected a tile or a vector")
    if isinstance(a, Tile) and isinstance(b, Vec) and op in ("add", "sub", "mul", "div"):
        return broadcast(op, a, b, into=into)
    if (isinstance(a, Vec) and isinstance(b, Tile) and into is None
            and op in ("add", "sub", "mul", "div")):
        # The vector on the left, as in `scale * x` or `m - x`: the same
        # broadcast, the operands swapped -- one more map for `-` and `/`.
        if op in ("add", "mul"):
            return broadcast(op, b, a)
        if op == "sub":
            return binary("mul", broadcast("sub", b, a), -1.0)
        return broadcast("mul", binary("div", full(b.ty, 1.0), b), a)
    _check_map_dtype(op, a.ty)
    if isinstance(b, Int):
        return _make(_BINARY[op], a.ty, (a.v,), into=into, name=op, rhs=b.v)
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return _make(_BINARY[op], a.ty, (a.v,), into=into, name=op,
                     scalar=float(b))
    if not isinstance(b, (Tile, Vec)):
        raise TypeError(f"{op}: right operand is {b!r}, expected a tile, a "
                        f"vector or a number")
    if b.ty.plain != a.ty.plain:
        raise TypeError(f"{op}: operand types differ, {a.ty} vs {b.ty}. "
                        f"Elementwise ops do not broadcast between tiles.")
    return _make(_BINARY[op], _meet(a.ty, b.ty), (a.v,), into=into, name=op,
                 rhs=b.v, mark="hk.reduce" if reduce else None)


def unary(op: str, a: "Value", *, into=None):
    if not isinstance(a, (Tile, Vec)):
        raise TypeError(f"{op}: operand is {a!r}, expected a tile or a vector")
    _check_map_dtype(op, a.ty)
    return _make(_UNARY[op], a.ty, (a.v,), into=into, name=op)


def broadcast(op: str, t: "Tile", v: "Vec", *, into=None) -> "Tile":
    """`op` vec[i] into every element of row i (column vector) or column i
    (row vector) of the tile."""
    axis = "row" if v.ty.kind == "col" else "col"
    if (v.ty.tile.rows != t.ty.rows or v.ty.tile.cols != t.ty.cols
            or v.ty.tile.run != t.ty.run):
        raise TypeError(f"{op}_{axis}: the vector came off a {v.ty.tile} but is "
                        f"being applied to a {t.ty}. The lane replication "
                        f"would not line up.")
    return _make(_BROADCAST[(op, axis)], t.ty, (t.v, v.v), into=into,
                 name=f"{op}_{axis}")


class Reduction:
    """`t.sum(axis=1)` before it lands: a vector `+=` folds it in place."""

    __slots__ = ("kind", "axis", "src", "order", "times")

    def __init__(self, kind: str, axis: str, src: "Tile", order: Optional[str] = None):
        self.kind, self.axis, self.src, self.order = kind, axis, src, order
        self.times = None

    @property
    def op(self) -> str:
        return f"{self.axis}_{self.kind}"

    @property
    def vtype(self) -> RegVecType:
        return RegVecType(self.src.ty, "col" if self.axis == "row" else "row")

    def value(self, *, into=None, accumulate: bool = False) -> "Vec":
        if accumulate and into is None:
            raise TypeError(f"{self.op}(accumulate=True) needs into=")
        if into is not None and into.ty != self.vtype:
            raise TypeError(f"{self.op}: into is {into.ty}, but reducing "
                            f"{self.src.ty} gives {self.vtype}. A vector's "
                            f"layout is tied to the tile it came off.")
        extra = {} if self.order is None else {"order": self.order}
        if self.times is not None:
            extra["times"] = self.times.v
        return _make(_REDUCE[(self.kind, self.axis)], self.vtype, (self.src.v,),
                     into=into, name=self.op, accumulate=bool(accumulate), **extra)

    def __getattr__(self, name):
        # Anything else a vector does: materialize, then do it.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.value(), name)

    def __add__(self, o): return self.value() + o
    def __sub__(self, o): return self.value() - o
    def __mul__(self, o): return self.value() * o
    def __truediv__(self, o): return self.value() / o
    def __getitem__(self, key): return self.value()[key]


def _reduce_order(t: "Tile", kind: str, axis: int, lanes, xor) -> str:
    """The order attribute of a stated reduction order: "seq|tree:o1,o2,..."."""
    if axis != 1 or getattr(t.ty, "run", None) is None:
        raise TypeError(f"{kind}(lanes=, xor=): an order is for the rows of a strip "
                        f"(hk.RS), not {t.ty} along axis {axis}: a WMMA tile's reduction "
                        f"order is its layout's")
    from .arch import target  # noqa: PLC0415

    if target().wave != 64:
        raise TypeError(f"{kind}(lanes=, xor=): a stated order is gfx950's (a wave of 64 "
                        f"lanes), not {target().arch}'s")
    lanes = "tree" if lanes is None else lanes
    if lanes not in ("seq", "tree"):
        raise ValueError(f"{kind}(lanes={lanes!r}): \"seq\" (a lane's values left to right) "
                         f"or \"tree\"")
    xor = (1, 2, 4, 8, 16, 32) if xor is None else tuple(int(o) for o in xor)
    if sorted(xor) != [1, 2, 4, 8, 16, 32]:
        raise ValueError(f"{kind}(xor={xor}): every offset of a 64-lane wave, 1 2 4 8 16 32, "
                         f"once each")
    return f"{lanes}:" + ",".join(str(o) for o in xor)


def _vec_of(x) -> "Vec":
    return x.value() if isinstance(x, Reduction) else x


class _Elementwise(Value):
    __slots__ = ()

    def __add__(self, o): return binary("add", self, _vec_of(o))
    def __sub__(self, o): return binary("sub", self, _vec_of(o))
    def __mul__(self, o): return binary("mul", self, _vec_of(o))
    def __truediv__(self, o): return binary("div", self, _vec_of(o))

    def __radd__(self, o): return binary("add", self, o)
    def __rmul__(self, o): return binary("mul", self, o)
    # A number on the left of `-` or `/`: `1.0 - p`, `127.0 / amax`.
    def __rsub__(self, o): return binary("add", binary("mul", self, -1.0), o)
    def __rtruediv__(self, o): return binary("div", full(self.ty, float(o)), self)

    def __iadd__(self, o): return self._inplace("add", o)
    def __isub__(self, o): return self._inplace("sub", o)
    def __imul__(self, o): return self._inplace("mul", o)
    def __itruediv__(self, o): return self._inplace("div", o)

    # Index arithmetic only: `//` on a float tile would have to round, and
    # kittens::div does not.
    def __floordiv__(self, o): return self._index_only("//")
    def __rfloordiv__(self, o): return self._index_only("//")
    def __mod__(self, o): return self._index_only("%")
    def __rmod__(self, o): return self._index_only("%")

    def _index_only(self, sym):
        raise TypeError(f"{sym} is index arithmetic and {self.ty} is not an "
                        f"index. There is no rounding division on a tile.")

    def _inplace(self, op, o):
        binary(op, self, _vec_of(o), into=self)
        return self

    def __neg__(self):
        return binary("mul", self, -1.0)

    def __abs__(self):
        return unary("abs", self)

    def maximum(self, o, *, into=None): return binary("max", self, o, into=into)
    def minimum(self, o, *, into=None): return binary("min", self, o, into=into)

    # unary maps, functional and in place
    def exp(self, *, into=None): return unary("exp", self, into=into)
    def exp2(self, *, into=None): return unary("exp2", self, into=into)
    def log(self, *, into=None): return unary("log", self, into=into)
    def log2(self, *, into=None): return unary("log2", self, into=into)
    def abs(self, *, into=None): return unary("abs", self, into=into)
    def relu(self, *, into=None): return unary("relu", self, into=into)
    def gelu(self, *, into=None): return unary("gelu", self, into=into)
    def dgelu(self, *, into=None): return unary("dgelu", self, into=into)
    def sqrt(self, *, into=None): return unary("sqrt", self, into=into)
    def rsqrt(self, *, into=None): return unary("rsqrt", self, into=into)
    def silu(self, *, into=None): return unary("silu", self, into=into)
    def step(self, *, into=None):
        """1 where the element is positive, else 0: a comparison as a factor,
        `x * (r - p).step()` keeping where r > p. Register tiles and
        strips, not vectors."""
        if not isinstance(self, Tile):
            raise TypeError("step is defined on tiles, not on register vectors")
        return unary("step", self, into=into)

    def copy(self, *, into=None): return unary("copy", self, into=into)

    def swap_halves(self, span: Optional[int] = None, *, negate_first: bool = False, into=None):
        """A strip with the two halves of each `span`-column segment of its rows exchanged (span
        None: the whole row): in a segment's first half column c takes column c + span/2, in
        its second c - span/2. `negate_first` negates what lands in the first half -- NeoX's
        rotate_half, [x1 | x2] -> [-x2 | x1] (`rotate_half(span)` is this). Strips only, on the
        wave64 port (gfx950). The halves are whole chunks apart (a register swap) or 8, 16 or 32
        lanes apart (a DPP row_ror:8, a v_permlane16_swap, a v_permlane32_swap): a whole row of
        one chunk or of an even number of them, or a head of `span` columns -- a 128-column head
        at eight bf16 a lane is 16 lanes, its halves eight apart."""
        ty = self.ty
        what = "rotate_half" if negate_first else "swap_halves"
        if not (isinstance(self, Tile) and _is_strip(ty)):
            raise TypeError(f"{what} takes a strip, not {ty!r}")
        tgt = get_target(_t().arch)
        if not tgt.strips or tgt.wave != 64:
            raise TypeError(f"{what} is in hk's wave64 strip port; this "
                            f"kernel is for {_t().arch}")
        if not negate_first:
            _check_map_dtype("swap_halves", ty)
        if span is None:
            chunks = ty.cols // (tgt.wave * ty.run)
            if chunks != 1 and chunks % 2:
                raise TypeError(f"{what} of {ty!r}: {chunks} chunks a row, "
                                f"neither one nor an even number")
        elif not isinstance(span, int) or span <= 0 or ty.cols % span:
            raise TypeError(f"{what}(span={span!r}) of {ty!r}: a span "
                            f"divides the strip's columns")
        from .._mlir import ir  # noqa: PLC0415

        attrs = {}
        if span is not None:
            attrs["span"] = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), span)
        if negate_first:
            attrs["negate_first"] = ir.UnitAttr.get()
        return _make(hkd.SwapHalvesOp, ty, (self.v,), into=into, name=what, **attrs)

    def rotate_half(self, span: int, *, into=None):
        """NeoX's rotate_half on each `span`-column segment of a strip's rows:
        [x1 | x2] -> [-x2 | x1], one head of `span` columns a segment --
        `swap_halves(span, negate_first=True)`."""
        if not isinstance(span, int):
            raise TypeError(f"rotate_half(span={span!r}): a span is a Python int")
        return self.swap_halves(span, negate_first=True, into=into)

    def exp_(self): return unary("exp", self, into=self)
    def exp2_(self): return unary("exp2", self, into=self)
    def rsqrt_(self): return unary("rsqrt", self, into=self)

    # fills of the whole value
    def zero_(self): return zeros(self.ty, into=self)
    def fill_(self, v: float): return full(self.ty, v, into=self)
    def neg_infty_(self): return neg_infty(self.ty, into=self)


class Tile(_Elementwise):
    """A register tile: kittens::rt<T, rows, cols, layout>."""

    __slots__ = ()

    @property
    def dtype(self) -> DType:
        return self.ty.dtype

    # reductions -- axis=1 reduces along each row: one value per row
    def _reduce(self, kind: str, axis: int) -> Reduction:
        if axis not in (0, 1):
            raise ValueError(f"axis is 1 (along each row, one value per row) "
                             f"or 0 (along each column), got {axis}")
        return Reduction(kind, "row" if axis == 1 else "col", self)

    def _ordered(self, kind: str, axis: int, lanes, xor) -> Reduction:
        # _reduce keeps its (kind, axis) signature: the templates wrap it to refuse reductions
        r = self._reduce(kind, axis)
        if lanes is not None or xor is not None:
            r.order = _reduce_order(self, kind, axis, lanes, xor)
        return r

    def sum(self, axis: int = 1, *, lanes: Optional[str] = None,
            xor: Optional[Sequence[int]] = None, times: Optional["Tile"] = None) -> Reduction:
        """Sum along each row (axis=1) or column. On a strip's rows, `lanes` and `xor` state the
        order -- a lane's own values "seq" (left to right) or "tree" (balanced), then the lanes
        combined over the xor offsets `xor` in that order, each of 1 2 4 8 16 32 once -- for a
        result that must match another kernel bit for bit. Unstated, the order is hk's (a tree,
        then 1 .. 32). gfx950 only.

        A stated order sums the strip's values as they are: `(x * x).sum(...)` adds rounded
        products (the compiler does not fuse them into the sum). `times=u` with `lanes="seq"` is
        the fused form, the sum of self * u as LLVM contracts a compiled loop `s = a0 * b0 + a1 *
        b1; s += ai * bi` -- s = fma(a0, b0, a1 * b1), then s = fma(ai, bi, s) -- then the xor
        sums: dsv41_mono's and aiter's sums of squares in their norms are `x.sum(axis=1,
        lanes="seq", xor=(32, 16, 8, 4, 2, 1), times=x)`."""
        r = self._ordered("sum", axis, lanes, xor)
        if times is not None:
            if lanes != "seq":
                raise ValueError("sum(times=): the products are fused into the running sum left "
                                 "to right; say lanes=\"seq\"")
            if not isinstance(times, Tile) or times.ty != self.ty:
                raise TypeError(f"sum(times=): a strip of the same type as this one, {self.ty}, "
                                f"not {getattr(times, 'ty', times)!r}")
            r.times = times
        return r

    def max(self, axis: int = 1, *, lanes: Optional[str] = None,
            xor: Optional[Sequence[int]] = None) -> Reduction:
        return self._ordered("max", axis, lanes, xor)

    def _group(self, kind: str, cols: int, lanes: str, xor) -> "Tile":
        from .arch import target  # noqa: PLC0415

        run = getattr(self.ty, "run", None)
        if not _is_strip_tile(self.ty):
            raise TypeError(f"group_{kind} reduces a strip's lane groups (hk.RS), not {self.ty}")
        if target().wave != 64:
            raise TypeError(f"group_{kind} is gfx950's, not {target().arch}'s")
        if cols % run or (cols // run) & (cols // run - 1) or cols // run > 32:
            raise ValueError(f"group_{kind}(cols={cols}): a group is 1, 2, 4 .. 32 lanes' runs of "
                             f"{run} columns")
        k = (cols // run).bit_length() - 1
        offs = tuple(1 << i for i in range(k)) if xor is None else tuple(int(o) for o in xor)
        if (len(offs) != k or len(set(offs)) != k
                or any(o <= 0 or o > 32 or o & (o - 1) for o in offs)):
            raise ValueError(f"group_{kind}(cols={cols}, xor={offs}): a group of {1 << k} lanes "
                             f"takes {k} offsets, distinct powers of two up to 32")
        if lanes not in ("seq", "tree"):
            raise ValueError(f"group_{kind}(lanes={lanes!r}): \"seq\" or \"tree\"")
        t = _t()
        op = hkd.GroupReduceOp(mlir_type(self.ty), self.v, kind,
                               f"{lanes}:" + ",".join(str(o) for o in offs), loc=t.loc(f"group_{kind}"))
        return wrap(op.result, self.ty)

    def first_lane(self) -> "Int":
        """Lane 0's first element of this i32 strip, as a scalar (v_readfirstlane, gfx950). Uniform
        across the workgroup when every wave holds the same value -- one LDS word read after a
        barrier -- so it may drive hk.uniform branches (an early exit decided by one wave)."""
        if getattr(self.ty, "run", None) is None or self.ty.dtype is not i32:
            raise TypeError(f"first_lane reads an i32 strip, not {self.ty}")
        t = _t()
        return Int(hkd.FirstLaneOp(_i32(), self.v, loc=t.loc("fl")).result, ScalarType(i32))

    def lane_strip(self, run: int = 8) -> "Tile":
        """This tile's registers as a 1-row strip, lane by lane (gfx950): no data moves. Lane l's
        elements, in the tile's per-lane order (a matrix-core tile: its fragments row by row of
        fragments, each fragment's elements in order; a strip: its rows, chunks and runs), become
        the lane's runs of `run`, chunk after chunk -- so `hk.map_cpp`, the group reductions and
        `store_elems` work on an MMA result where it sits. `hk.lane_coords(tile, run)` says which
        row and column of the tile each element is. 32-bit elements."""
        from .arch import target  # noqa: PLC0415

        if target().wave != 64:
            raise TypeError(f"lane_strip is gfx950's, not {target().arch}'s")
        if self.ty.dtype.bits != 32:
            raise TypeError(f"lane_strip takes 32-bit elements, not {self.ty.dtype.name}")
        ty = _lane_strip_type(self.ty, run, self.ty.dtype)
        t = _t()
        return wrap(hkd.LaneStripOp(mlir_type(ty), self.v, loc=t.loc("ls")).result, ty)

    def run_xor(self, mask: int) -> "Tile":
        """Each element from another element of its lane's run (gfx950): element e of a run becomes
        element e ^ `mask` of the same run, a register move. A rotary pair's partner, when the pair
        sits in one run (an MMA result taken by `lane_strip`)."""
        run = getattr(self.ty, "run", None)
        if not _is_strip_tile(self.ty):
            raise TypeError(f"run_xor permutes a strip's runs (hk.RS), not {self.ty}")
        if not 0 < int(mask) < run:
            raise ValueError(f"run_xor({mask}): the mask is in 1 .. {run - 1}, below the run")
        t = _t()
        return wrap(hkd.RunXorOp(mlir_type(self.ty), self.v, int(mask), loc=t.loc("rx")).result, self.ty)

    def lane_gather(self, lanes: "Tile") -> "Tile":
        """Each element from another lane (gfx950): element (r, c) of lane l is the same run element
        of lane `lanes[r, c]` (an i32 strip of this strip's shape, 0..63) -- ds_bpermute, one an
        element. A lane group's small-matrix work (a 4x4 Sinkhorn's row and column sums) without
        LDS. 32-bit elements."""
        if not _is_strip_tile(self.ty) or not _is_strip_tile(lanes.ty):
            raise TypeError(f"lane_gather takes strips, not {self.ty} / {lanes.ty}")
        t = _t()
        return wrap(hkd.LaneGatherOp(mlir_type(self.ty), self.v, lanes.v, loc=t.loc("lg")).result, self.ty)

    def group_max(self, cols: int, *, lanes: str = "seq", xor: Optional[Sequence[int]] = None) -> "Tile":
        """A strip's rows reduced over groups of `cols` columns (a lane group's runs: an MX group is
        32 columns of a run-8 strip), the group's max in every element. `lanes` and `xor` state the
        order, as for `sum(axis=1, ...)`; the offsets are 1 .. cols / run / 2 (neighbouring lanes),
        or as many other distinct powers of two up to 32: the group is then a lane's run and those of
        lanes l ^ (a sum of the offsets). A 32x32 matrix-core result taken as a strip
        (`lane_strip(16)`) holds 32 rows of a column in lane l's run and lane l ^ 32's:
        `group_max(32, xor=(32,))`. gfx950."""
        return self._group("max", cols, lanes, xor)

    def group_min(self, cols: int, *, lanes: str = "seq", xor: Optional[Sequence[int]] = None) -> "Tile":
        """As `group_max`, the minimum."""
        return self._group("min", cols, lanes, xor)

    def group_sum(self, cols: int, *, lanes: str = "seq", xor: Optional[Sequence[int]] = None) -> "Tile":
        """As `group_max`, the sum, in the stated order."""
        return self._group("sum", cols, lanes, xor)

    def min(self, axis: int = 1, *, lanes: Optional[str] = None,
            xor: Optional[Sequence[int]] = None) -> Reduction:
        return self._ordered("min", axis, lanes, xor)

    def prod(self, axis: int = 1, *, lanes: Optional[str] = None,
             xor: Optional[Sequence[int]] = None) -> Reduction:
        return self._ordered("prod", axis, lanes, xor)

    def to(self, dtype: DType, *, into=None) -> "Tile":
        """Change the element type, keeping shape and layout (kittens::copy)."""
        ty = RegTileType(dtype, self.ty.rows, self.ty.cols, self.ty.layout, self.ty.frag,
                         run=self.ty.run)
        return _make(hkd.CastOp, ty, (self.v,), into=into, name="cvt")

    def bitcast(self, dtype: DType, *, into=None) -> "Tile":
        """The same registers read as another element type of the same width (fp32 <-> i32,
        bf16 <-> fp16 ...): no instruction, unlike `.to`. E.g. an i32 accumulator started at the
        bits of 1.5 * 2^23 read as fp32 is 1.5 * 2^23 + S exactly for |S| < 2^22 -- the int ->
        float convert of an iu8 product, folded away (experiment 1's c7 needed a C++ rewrite)."""
        if dtype.bits != self.ty.dtype.bits:
            raise TypeError(f"bitcast keeps the element width: {self.ty.dtype.name} is "
                            f"{self.ty.dtype.bits} bits, {dtype.name} {dtype.bits}; use .to()")
        ty = RegTileType(dtype, self.ty.rows, self.ty.cols, self.ty.layout, self.ty.frag,
                         run=self.ty.run)
        return _make(hkd.BitcastOp, ty, (self.v,), into=into, name="bc")

    @property
    def T(self) -> "Transposed":
        return Transposed(self)

    def __matmul__(self, other) -> "MatMul":
        return MatMul(self, other)

    # masks, in place: an empty mask is a no-op in the library, so a fresh
    # destination would keep garbage -- `into=` a new tile copies the source
    # in first
    def _fill(self, op: str, idx: Scalar, value: float, into=None):
        cls = {"left_fill": hkd.LeftFillOp, "right_fill": hkd.RightFillOp,
               "upper_fill": hkd.UpperFillOp, "lower_fill": hkd.LowerFillOp,
               "triu": hkd.TriuOp, "tril": hkd.TrilOp}[op]
        if op not in ("left_fill", "right_fill"):
            _no_strip(op, self.ty)
        out = self if into is None else into
        if out is not self:
            if out.ty != self.ty:
                raise TypeError(f"{op}: into is {out.ty} but the tile is {self.ty}")
            _make(hkd.CopyOp, self.ty, (self.v,), into=out, name="copy")
        t = _t()
        cls(None, self.v, as_int(idx).v, float(value), dst=out.v,
            loc=t.loc(op))
        return out

    def left_fill_(self, idx, value=0.0, *, into=None): return self._fill("left_fill", idx, value, into)
    def right_fill_(self, idx, value=0.0, *, into=None): return self._fill("right_fill", idx, value, into)
    def upper_fill_(self, idx, value=0.0, *, into=None): return self._fill("upper_fill", idx, value, into)
    def lower_fill_(self, idx, value=0.0, *, into=None): return self._fill("lower_fill", idx, value, into)
    def triu_(self, idx, value=0.0, *, into=None): return self._fill("triu", idx, value, into)
    def tril_(self, idx, value=0.0, *, into=None): return self._fill("tril", idx, value, into)

    def scaled(self, scales: "MxScales", *, fmt: str = "fp8", at=(0, 0)) -> "Scaled":
        """This operand with its E8M0 block scales, for gfx950's block-scaled MFMA:
        `hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb, fmt="fp4"), acc)`. `fmt` is "fp8"
        (e4m3) or "fp4" (the FP4 container `hk.load_fp4_frags` fills). `at` = (tile, step):
        the scales may cover more than the operand -- its 16-row base tile i and 128-wide step s
        take the scales' tile i + tile and step s + step (a 32-row block's two-step AITER
        dwords, loaded once, feeding two token tiles over two steps)."""
        return Scaled(self, scales, fmt, at)

    def as_operand(self, dtype: Optional[DType] = None, *, into=None) -> "Tile":
        """This accumulator's registers read as an MFMA operand: no data
        moves across lanes (gfx950).

        `self` is a col-layout tile in an accumulator fragment -- an MFMA
        result, or its cast to bf16. Each lane holds a column and four rows
        at a time; read as a stride-4 operand fragment (32x32 -> 16x32/4,
        16x16 -> 32x16/4), its rows are the reduction index in another order,
        and an MMA whose other operand has the same stride (hk-fragments
        lays it out so) computes the same product. So with the scores kept
        transposed, `o_t += v.T @ p_t.as_operand(bf16)` is P·V straight off
        the softmax, with no reshuffle across lanes. `dtype` converts an
        fp32 accumulator on the way (to nearest even); a 16-bit one is read
        as it is."""
        t = self.ty
        dtype = dtype or t.dtype
        if t.layout != "col" or (t.dtype is not fp32 and t.dtype.bits != 16):
            raise TypeError(f"as_operand reads a col-layout accumulator tile, "
                            f"fp32 or 16-bit, not {t}")
        if dtype.bits != 16 or (t.dtype is not fp32 and dtype != t.dtype):
            raise TypeError(f"as_operand makes a bf16 or fp16 operand of {t}, "
                            f"not {dtype}")
        if t.frag == (32, 32):
            f = (16, 32, 4)
        elif t.frag in (None, (16, 16)):
            f = (32, 16, 4)
            if t.rows % 32:
                raise TypeError(f"as_operand: {t} in 16x16 fragments is read "
                                f"as 32x16 stride-4 ones, two to a 32-row band; "
                                f"{t.rows} rows is not a whole number of them")
        else:
            raise TypeError(f"as_operand: {t} is not in an accumulator "
                            f"fragment (16x16, or 32x32)")
        ty = RegTileType(dtype, t.rows, t.cols, "col", f)
        return _make(hkd.AsOperandOp, ty, (self.v,), into=into, name="opnd")

    def subtile(self, rows: int, index: int) -> "Tile":
        """Rows [index*rows, (index+1)*rows) of this tile, as a tile of
        `rows` rows that is the same registers: an op writing into it writes
        this tile (include/cdna4's subtile_inplace). `index` is a Python int
        -- a runtime one would index the register file. Not an LDS load's
        destination: load into the whole tile."""
        t = self.ty
        _no_strip("subtile", t)
        if not isinstance(index, int) or not isinstance(rows, int):
            raise TypeError("subtile(rows, index) takes Python ints: a register "
                            "tile has no runtime-indexed window")
        fr = t.frag[0] if t.frag else 16
        if rows <= 0 or t.rows % rows or rows % fr:
            raise ValueError(f"subtile rows={rows} of {t}: a whole number of "
                             f"{fr}-row fragments that tiles its {t.rows} rows")
        if not 0 <= index < t.rows // rows:
            raise IndexError(f"subtile {index} of {t.rows // rows}")
        ty = RegTileType(t.dtype, rows, t.cols, t.layout, t.frag)
        tt = _t()
        return Tile(hkd.RegSubtileOp(mlir_type(ty), self.v, index,
                                     loc=tt.loc("sub")).result, ty)

    def make_causal(self, value: float = 0.0, *, into=None, transposed: bool = False):
        """Deprecated: `tril_(0, value)` (`triu_(0, value)` with `transposed=True`), which takes
        any diagonal. A copy of this tile with the elements above the diagonal set to `value`
        (below it with `transposed=True`), or into `into`; this tile is unchanged."""
        import warnings  # noqa: PLC0415

        warnings.warn("Tile.make_causal is deprecated: use tril_(0, value), or triu_(0, value) "
                      "for transposed=True (docs/design/dsl-surface.md, D5)",
                      DeprecationWarning, stacklevel=2)
        _no_strip("make_causal", self.ty)
        if self.ty.rows != self.ty.cols:
            raise TypeError(f"make_causal: the tile is {self.ty.rows}x"
                            f"{self.ty.cols}, not square; mask the off-diagonal "
                            f"blocks with left_fill_/right_fill_.")
        fill = "triu" if transposed else "tril"
        if into is not None:
            return self._fill(fill, 0, value, into)
        out = self.copy()
        return out._fill(fill, 0, value)


class Transposed:
    """`t.T`: an mma operand read the other way, or a relabelling transpose."""

    __slots__ = ("tile",)

    def __init__(self, tile: Tile):
        self.tile = tile

    def value(self, *, into=None) -> Tile:
        return transpose(self.tile, into=into)

    def __matmul__(self, other) -> "MatMul":
        return MatMul(self, other)


class MatMul:
    """`a @ b` before it lands: `acc += a @ b.T` is one mma into acc."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def variant(self):
        at = isinstance(self.a, Transposed)
        bt = isinstance(self.b, Transposed)
        a = self.a.tile if at else self.a
        b = self.b.tile if bt else self.b
        return "mma_" + ("At" if at else "A") + ("Bt" if bt else "B"), a, b


_MMA_CLS = {"mma_AB": hkd.MmaABOp, "mma_ABt": hkd.MmaABtOp,
            "mma_AtB": hkd.MmaAtBOp, "mma_AtBt": hkd.MmaAtBtOp}
_MMA_SPEC = {  # (A layout, B layout, D.rows from A, D.cols from B, A red, B red)
    "mma_AB": ("row", "col", "rows", "cols", "cols", "rows"),
    "mma_ABt": ("row", "row", "rows", "rows", "cols", "cols"),
    "mma_AtB": ("col", "col", "cols", "cols", "rows", "rows"),
    "mma_AtBt": ("col", "row", "cols", "rows", "rows", "cols"),
}


def mma(variant: str, a, b, acc: Tile, *, into=None) -> Tile:
    """`acc + A x B` on the WMMA unit. The variant is decided by the layouts
    you have; the wrong one is how a kernel buys eight ds_read_u16 per base
    tile instead of two ds_read_b128.

    Scaled operands (`Tile.scaled`) make it gfx950's block-scaled MFMA
    (v_mfma_scale_f32_16x16x128_f8f6f4): `hk.mma("mma_ABt", a.scaled(sa),
    b.scaled(sb, fmt="fp4"), acc)` is acc + (sa . A)(sb . B)^T for A (M, K) and
    B (N, K) fp8 e4m3 row tiles (FP4 in the fp8 container with fmt="fp4"), K a
    multiple of 128, element (r, k) of A counting as A[r][k] *
    2^(sa[r][k // 32] - 127), acc an fp32 col tile (M, N). Both operands are
    scaled, and the variant is mma_ABt."""
    if isinstance(a, Scaled) or isinstance(b, Scaled):
        if not (isinstance(a, Scaled) and isinstance(b, Scaled)):
            raise TypeError(f"{variant}: both operands of a block-scaled MMA carry scales "
                            f"(Tile.scaled), or neither does")
        if variant != "mma_ABt":
            raise TypeError(f"{variant}: the block-scaled MFMA is mma_ABt (A and B row tiles "
                            f"sharing K), not {variant}")
        return _mma_mx(a.tile, b.tile, acc, a.scales, b.scales, into=into, a_fmt=a.fmt,
                       b_fmt=b.fmt, sa_at=a.at, sb_at=b.at)
    spec = _MMA_SPEC[variant]
    for v, what in ((a, "A"), (b, "B"), (acc, "C")):
        if not isinstance(v, Tile):
            raise TypeError(f"{variant} {what} operand is {v!r}, expected a tile")
        _no_strip(variant, v.ty)
    fp8 = a.ty.dtype in FP8 and b.ty.dtype in FP8
    if a.ty.dtype != b.ty.dtype and not fp8:
        raise TypeError(f"{variant}: A is {a.ty.dtype} and B is {b.ty.dtype}. "
                        f"WMMA takes one operand type, not a mixed pair "
                        f"(fp8 e4m3/e5m2 excepted).")
    arch = _t().arch
    if fp8 and a.ty.dtype.name not in get_target(arch).mma:
        raise TypeError(f"{variant}: fp8 operands need an fp8 WMMA, and {arch} "
                        f"has none -- gfx1201 (RDNA4) does. On {arch} keep fp8 "
                        f"as a storage type and dequantize to bf16 before the "
                        f"mma.")
    int8 = a.ty.dtype.name == "i8" and b.ty.dtype.name == "i8"
    if int8:
        if "i8" not in get_target(arch).mma:
            raise TypeError(f"{variant}: i8 operands need an int8 matrix instruction "
                            f"hk uses, and {arch} has none in hk (gfx1100 and gfx1201 do).")
        if variant != "mma_ABt":
            raise TypeError(f"{variant}: i8 operands are row-layout -- C += A @ B.T only")
        if acc.ty.dtype != i32 or acc.ty.layout != "col":
            raise TypeError(f"{variant}: i8 x i8 accumulates into a col-layout i32 "
                            f"tile, not {acc.ty}.")
    elif not fp8 and a.ty.dtype.bits != 16:
        raise TypeError(f"{variant}: operands are {a.ty.dtype}; WMMA takes bf16 "
                        f"or fp16 (fp8 on gfx1201, i8 on gfx1100 and gfx1201).")
    if not int8 and (acc.ty.dtype != fp32 or acc.ty.layout != "col"):
        raise TypeError(f"{variant}: the accumulator is {acc.ty}; WMMA "
                        f"accumulates into a col-layout fp32 tile.")
    for v, want, what in ((a, spec[0], "A"), (b, spec[1], "B")):
        if v.ty.layout != want:
            other = next(n for n, sp in _MMA_SPEC.items()
                         if (a.ty.layout, b.ty.layout) == sp[:2])
            raise TypeError(
                f"{variant} wants {what} in '{want}' layout and it is "
                f"'{v.ty.layout}'. With the layouts you have, the variant that "
                f"applies is {other} -- which computes a different product, so "
                f"if it is not the one you want, the fix is to stage the operand "
                f"the other way round, not to relabel it. swap_layout moves "
                f"data on this architecture.")
    if getattr(a.ty, spec[4]) != getattr(b.ty, spec[5]):
        raise TypeError(f"{variant}: reduction extents disagree -- A.{spec[4]}="
                        f"{getattr(a.ty, spec[4])}, B.{spec[5]}="
                        f"{getattr(b.ty, spec[5])}.")
    # An accumulator declared in 32x32 fragments is gfx950's 32x32x16 MFMA;
    # hk-fragments lays its operands out to match.
    big = acc.ty.frag == (32, 32)
    if big and get_target(arch).wave != 64:
        raise TypeError(f"{variant}: a 32x32-fragment accumulator is gfx950's "
                        f"32x32x16 MFMA; {arch} has none")
    ty = RegTileType(i32 if int8 else fp32, getattr(a.ty, spec[2]), getattr(b.ty, spec[3]),
                     "col", (32, 32) if big else None)
    if acc.ty != ty:
        raise TypeError(f"{variant}: A {a.ty} times B {b.ty} is {ty}, but the "
                        f"accumulator is {acc.ty}.")
    return _make(_MMA_CLS[variant], ty, (a.v, b.v, acc.v), into=into, name="mma")


def _tile_iadd(self: Tile, o):
    if isinstance(o, MatMul):
        variant, a, b = o.variant()
        mma(variant, a, b, self, into=self)
        return self
    return _Elementwise._inplace(self, "add", o)


Tile.__iadd__ = _tile_iadd  # type: ignore[assignment]


class Vec(_Elementwise):
    """A register vector: one value per row (col kind) or column (row kind)
    of the tile it came off."""

    __slots__ = ()

    def __iadd__(self, o):
        if isinstance(o, Reduction) and o.kind == "sum":
            o.value(into=self, accumulate=True)
            return self
        return self._inplace("add", o)

    def __imul__(self, o):
        if isinstance(o, Reduction) and o.kind == "prod":
            o.value(into=self, accumulate=True)
            return self
        return self._inplace("mul", o)

    def max_(self, r: Reduction) -> "Vec":
        """Fold `t.max(axis=...)`, or another partial max (a vector), into this
        vector: the online max. A reduction's NaN and signed-zero rules either
        way (docs/language.md); `maximum` is the elementwise op."""
        if not isinstance(r, Reduction) or r.kind != "max":
            r = _vec_of(r)
            return binary("max", self, r, into=self, reduce=isinstance(r, Vec))
        r.value(into=self, accumulate=True)
        return self

    def min_(self, r: Reduction) -> "Vec":
        if not isinstance(r, Reduction) or r.kind != "min":
            r = _vec_of(r)
            return binary("min", self, r, into=self, reduce=isinstance(r, Vec))
        r.value(into=self, accumulate=True)
        return self

    def __getitem__(self, key) -> "Vec":
        """`v[:, None]` / `v[None, :]`: the axis a broadcast runs along,
        spelled out. A vector broadcasts along the axis it came off anyway;
        this says so at the use and checks it."""
        def whole(k):
            return isinstance(k, slice) and k == slice(None)

        want = None
        if isinstance(key, tuple) and len(key) == 2:
            if whole(key[0]) and key[1] is None:
                want = "col"
            elif key[0] is None and whole(key[1]):
                want = "row"
        if want is None:
            raise TypeError("index a register vector only as v[:, None] (one "
                            "value per row) or v[None, :] (one per column)")
        if self.ty.kind != want:
            raise TypeError(
                f"{'v[:, None]' if want == 'col' else 'v[None, :]'} is a "
                f"{'column' if want == 'col' else 'row'} vector, and this one is "
                f"{self.ty} -- one value per "
                f"{'column' if self.ty.kind == 'row' else 'row'} of its tile.")
        return self

    def any_ne(self, other: "Vec") -> Int:
        """Wave-level: did any entry differ from `other`. A uniform branch."""
        if not isinstance(other, Vec) or self.ty.plain != other.ty.plain:
            raise TypeError(
                f"any_ne: operands are {self.ty} and "
                f"{getattr(other, 'ty', other)}. The comparison is entry by "
                f"entry, so the two have to be the same shape.")
        t = _t()
        return Int(hkd.AnyNeOp(_i32(), self.v, other.v, loc=t.loc("grew")).result,
                   ScalarType(i32))

    def any_above(self, other: "Vec", threshold: float) -> Int:
        """Wave-level: did any entry exceed `other`'s by more than
        `threshold`. A uniform branch. The online softmax's lazy rescale:
        keep taking exp2 against a stale running max until the new one is
        more than `threshold` (log2 units) above it."""
        if (not isinstance(other, Vec) or self.ty.plain != other.ty.plain
                or self.ty.dtype is not fp32):
            raise TypeError(
                f"any_above: operands are {self.ty} and "
                f"{getattr(other, 'ty', other)}; two fp32 vectors of one type")
        t = _t()
        return Int(hkd.AnyAboveOp(_i32(), self.v, other.v, float(threshold),
                                  loc=t.loc("grew")).result, ScalarType(i32))


# ---------------------------------------------------------------- creation


def zeros(ty, *, into=None):
    """A register tile or vector of type `ty` filled with 0, or `into=` an
    existing one: `acc = hk.zeros(hk.RT(hk.fp32, 16, 64))`."""
    name = "z" if isinstance(ty, RegTileType) else "zv"
    return _make(hkd.ZeroOp, ty, (), into=into, name=name)


def load_w4(ty, qw: "Global", s: "Global", zs: "Global", n0, k0, *,
            group: int = 128, into=None) -> "Tile":
    """Rows n0.. and columns k0.. of W, dequantized from 4-bit codes into a
    bf16 row-layout tile: element (n, k) is q * s - zs, q being nibble k % 8
    of row n's word k // 8 (int32 [N, K/8], interleaved [N/16][K/32][16][4]),
    s and zs the group's scale and zero-times-scale `s[n][k // group]`,
    `zs[n][k // group]` (bf16 [N, K/group]). The layout
    `hk.ops.w4a16.repack_*` writes. The tile is a whole number of 32-column
    chunks and k0 a multiple of 32, and of the tile's width when the group
    spans it."""
    if ty.cols % 32:
        raise ValueError(f"load_w4 reads whole 32-column chunks; the tile is "
                         f"{ty.cols} wide")
    if (not isinstance(ty, RegTileType) or ty.layout != "row"
            or ty.dtype.name not in ("bf16", "fp16")):
        raise TypeError(f"load_w4 dequantizes into a bf16 or fp16 row-layout tile, "
                        f"not {ty!r}")
    if group <= 0 or group % 16:
        raise ValueError(f"load_w4: group {group} must be a positive multiple of 16")
    for name, g, want in (("qw", qw, i32), ("s", s, None), ("zs", zs, None)):
        if not isinstance(g, Global):
            raise TypeError(f"load_w4: {name} must be a tensor parameter, got {g!r}")
        if want is not None and g.dtype is not want:
            raise TypeError(f"load_w4: {name} holds int32 words of eight codes, "
                            f"not {g.dtype.name}")
        if want is None and g.dtype.name != ty.dtype.name:
            raise TypeError(f"load_w4: {name} is the tile's {ty.dtype.name}, "
                            f"not {g.dtype.name}")
    return _make(hkd.LoadW4Op, ty, (qw.v, s.v, zs.v, as_int(n0).v, as_int(k0).v),
                 into=into, name="w4", group=group)


def rand(ty, seed, offset, row_stride, *, into=None) -> "Tile":
    """Uniform [0, 1) draws into an fp32 tile: element (r, c) is the draw
    `triton.language.rand(seed, offset + r * row_stride + c)` makes, bit for
    bit -- Philox4x32-10 with Triton's constants and conversion -- so a
    dropout written here keeps exactly the elements Triton's keeps."""
    if not isinstance(ty, RegTileType) or ty.dtype is not fp32:
        raise TypeError(f"rand draws into an fp32 tile type, not {ty!r}")
    _no_strip("rand", ty)
    return _make(hkd.PhiloxOp, ty, (as_int(seed).v, as_int(offset).v,
                                    as_int(row_stride).v),
                 into=into, name="rnd")


def full(ty, value: float, *, into=None):
    """A register tile or vector of type `ty` filled with `value`."""
    name = "f" if isinstance(ty, RegTileType) else "fv"
    return _make(hkd.FullOp, ty, (float(value),), into=into, name=name)


def neg_infty(ty, *, into=None):
    """A register tile or vector of type `ty` filled with -inf: a running
    max's start."""
    name = "ninf" if isinstance(ty, RegTileType) else "ninfv"
    return _make(hkd.NegInftyOp, ty, (), into=into, name=name)


def transpose(t: Tile, *, into=None) -> Tile:
    """Transpose by relabelling: mirrored shape, opposite layout, no data
    moved beyond the register shuffle."""
    _no_strip("transpose", t.ty)
    ty = t.ty.transposed()
    return _make(hkd.TransposeOp, ty, (t.v,), into=into, name="tr")


def transpose_sep(t: Tile, *, into=None) -> Tile:
    """Transpose keeping the layout. Moves data; needs separate registers."""
    _no_strip("transpose_sep", t.ty)
    if into is t:
        raise TypeError("transpose_sep needs separate registers for source "
                        "and destination; use hk.transpose to relabel instead")
    f = (t.ty.frag[1], t.ty.frag[0], *t.ty.frag[2:]) if t.ty.frag else None
    ty = RegTileType(t.ty.dtype, t.ty.cols, t.ty.rows, t.ty.layout, f)
    return _make(hkd.TransposeSepOp, ty, (t.v,), into=into, name="trs")


def broadcast_row(ty: RegTileType, vec: Vec) -> Tile:
    """A tile whose every row is vec[row]."""
    return _broadcast_fill(hkd.BroadcastRowOp, ty, vec, "col")


def broadcast_col(ty: RegTileType, vec: Vec) -> Tile:
    """A tile whose every column is vec[col]: element (r, c) is vec[c], for
    a row_vec `vec` (one value per column)."""
    return _broadcast_fill(hkd.BroadcastColOp, ty, vec, "row")


def _broadcast_fill(cls, ty, vec, want):
    _no_strip("broadcast_row/broadcast_col", ty)
    if not isinstance(vec, Vec) or vec.ty.kind != want:
        raise TypeError(f"takes a {want} vector, got {getattr(vec, 'ty', vec)}")
    if vec.ty.tile.rows != ty.rows or vec.ty.tile.cols != ty.cols:
        raise TypeError(f"the vector came off a {vec.ty.tile}, not a {ty}")
    return _make(cls, ty, (vec.v,), name="bc")


def as_vec(vec: Vec, tile: RegTileType, *, uniform: bool = False) -> Vec:
    """Re-attach a register vector to a tile of another shape, where the C++
    types are the same: element type, length and layout."""
    want = RegVecType(tile, vec.ty.kind, vec.ty.uniform or uniform)
    if want == vec.ty and want.tile == vec.ty.tile:
        return vec
    if want.plain._key() != vec.ty.plain._key():
        raise TypeError(f"as_vec cannot retype {vec.ty} (off {vec.ty.tile}) as "
                        f"{want} (off {tile}); route it through LDS instead.")
    return _make(hkd.RetypeVecOp, want, (vec.v,), name="rv")


# ---------------------------------------------------------------- LDS


class SharedTile(Value):
    """An LDS tile, or a stack of them (`As[i]` picks one)."""

    __slots__ = ()

    def __getitem__(self, i: Scalar) -> "SharedTile":
        if self.ty.count == 1:
            raise TypeError(f"{self.ty} is a single LDS tile, not a stack; pass "
                            f"count= to hk.lds.alloc for a multi-buffer")
        if isinstance(i, int) and not 0 <= i < self.ty.count:
            raise IndexError(f"buffer {i} of {self.ty}")
        t = _t()
        ty = self.ty.element()
        return SharedTile(hkd.SharedAtOp(mlir_type(ty), self.v, as_int(i).v,
                                         loc=t.loc("sb")).result, ty)

    def load_elems(self, idx: "Tile", dtype: Optional[DType] = None, *, runs: bool = False) -> "Tile":
        """Element by element from this fp32 LDS tile (gfx950): element (r, c) of the result is
        element `idx[r, c]` (row-major, through the tile's swizzle) -- fp32, or its bits as i32 with
        dtype=hk.i32; below zero or past the tile, 0. A CTA's own exchange between warps without a
        round trip to global memory (whose drain would also wait out every load in flight). Another
        warp's writes need hk.arch.barrier(drain=True) between.

        `runs=True`: the kernel says each lane's run (a multiple of 4) is consecutive elements from
        16-byte-aligned indices; each 4 is one 16-byte read (b32 reads of a lane's 4 consecutive
        elements conflict 4 ways across the wave). Index flat into a tile 16 columns wide: a wider
        one lays every 16th element a subtile (1 KB) on, on the same banks."""
        if not _is_strip_tile(idx.ty) or idx.ty.dtype.name != "i32":
            raise TypeError(f"load_elems takes an i32 strip of indices, not {idx.ty}")
        if runs and idx.ty.run % 4:
            raise TypeError(f"load_elems(runs=True): runs of 4 elements, not {idx.ty.run}")
        dtype = fp32 if dtype is None else dtype
        ty = RegTileType(dtype, idx.ty.rows, idx.ty.cols, idx.ty.layout, idx.ty.frag, run=idx.ty.run)
        t = _t()
        return wrap(hkd.LdsLoadElemsOp(mlir_type(ty), self.v, idx.v, runs=bool(runs), loc=t.loc("ldse")).result, ty)

    def store_elems(self, val: "Tile", idx: "Tile") -> None:
        """Element by element into this fp32 LDS tile (gfx950): val's element (r, c) (fp32, or i32
        bits) to element `idx[r, c]` (row-major); below zero or past the tile, nothing. Before
        another warp reads it: hk.arch.barrier(drain=True). A bf16 tile takes the fp32 value
        rounded to bf16 (round to nearest even) -- an MMA operand built element by element, read
        back with `.load` / `.subtile(...).load`."""
        if not _is_strip_tile(idx.ty) or idx.ty.dtype.name != "i32":
            raise TypeError(f"store_elems takes an i32 strip of indices, not {idx.ty}")
        t = _t()
        hkd.LdsStoreElemsOp(self.v, val.v, idx.v, loc=t.loc())

    def atomic_add_elems(self, val: "Tile", idx: "Tile") -> None:
        """Element by element, atomically (gfx950): val's element (r, c), an i32, is added to element
        `idx[r, c]` of this fp32 LDS tile taken as an i32 (ds_add_u32); below zero or past the tile,
        nothing. Every element to one index lands: a histogram's counts. Read the counts with
        load_elems(..., i32) after hk.arch.barrier(drain=True)."""
        if getattr(idx.ty, "run", None) is None or idx.ty.dtype.name != "i32":
            raise TypeError(f"atomic_add_elems takes an i32 strip of indices, not {idx.ty}")
        if val.ty.dtype.name != "i32":
            raise TypeError(f"atomic_add_elems adds i32 values, not {val.ty.dtype.name}")
        t = _t()
        hkd.LdsAtomicAddElemsOp(self.v, val.v, idx.v, loc=t.loc())

    def subtile(self, rows: int, cols: int, row: Scalar = 0,
                col: Scalar = 0) -> "SharedTile":
        """A rows x cols window, indexed in units of itself; a view, so the
        swizzle still applies and loads still vectorise."""
        st = self.ty
        if st.count != 1:
            raise TypeError(f"subtile of {st}, a stack; index it first")
        for n, v, whole in (("rows", rows, st.rows), ("cols", cols, st.cols)):
            if v <= 0 or v % 16:
                raise ValueError(f"subtile {n}={v} must be a positive multiple of 16")
            if whole % v:
                raise ValueError(
                    f"subtile {n}={v} does not divide the {whole} of {st}. A "
                    f"subtile is indexed in units of itself, so a window that "
                    f"does not tile the parent has no index to give.")
        t = _t()
        ty = SharedTileType(st.dtype, rows, cols)
        return SharedTile(hkd.SubtileOp(mlir_type(ty), self.v, as_int(row).v,
                                        as_int(col).v, loc=t.loc("sub")).result, ty)

    def load(self, tile: RegTileType, *, wait: bool = True, into=None) -> Tile:
        """LDS -> register tile. `wait=False` leaves the reads in flight; the
        tile must then be named in hk.lds.wait_for before it is read."""
        st = self.ty
        _no_strip("an LDS tile load", tile)
        if (tile.rows, tile.cols) != (st.rows, st.cols):
            raise TypeError(f"load: {tile} from {st}; register and shared tiles "
                            f"must have the same extents")
        n = lds_loads(tile, st, _t().arch)
        if not wait and not n:
            raise TypeError(f"load(wait=False) from {st} into {tile}: this pair "
                            f"does not take the vectorised path, so there is "
                            f"nothing to keep in flight. Asynchronous reads "
                            f"need a 'row' layout 16-bit tile of the shared "
                            f"tile's element type.")
        return _make(hkd.LoadSharedOp, tile, (self.v, n), into=into, name="lds",
                     wait=bool(wait))

    def store(self, value: Tile) -> None:
        """Register tile -> LDS."""
        _no_strip("an LDS tile store", value.ty)
        t = _t()
        hkd.StoreSharedOp(self.v, value.v, loc=t.loc())

    def granules(self, name: str = "gr") -> "Granules":
        """Factor this tile's addressing into swizzle_bytes/16 registers, so a
        loop reading many fragments holds those and not one address each."""
        t = _t()
        ty = GranuleSetType(self.ty)
        return Granules(hkd.GranulesOp(mlir_type(ty), self.v,
                                       loc=t.loc(name)).result, ty)


def lds_loads(tile: RegTileType, src: SharedTileType, arch: str = "gfx1100") -> int:
    """ds_read instructions load_shared issues for this pair; 0 when it
    misses the vectorised path.

    On a wave64 (CDNA) target the count depends on the matrix-core fragment,
    which hk-fragments assigns after tracing, so this only says whether the
    pair is asynchronous at all; hk-lds-pipeline sets the real count."""
    if src.dtype != tile.dtype:
        return 0
    tgt = get_target(arch)
    if tgt.wave == 32:
        # A row-layout operand reads its lane's share of each base tile with
        # 16-byte ds_read_b128s, or one 8-byte read when the share is 8 bytes
        # (include/rdna{3,4} tile/shared_to_register.cuh). gfx1100's 16-bit
        # operand is mirrored, 32 bytes a lane: two; its int8 one, 16: one.
        # gfx1201 splits K between the halves: 16 bytes for bf16, 8 for fp8
        # and int8, one read each.
        shape = tgt.mma.get(tile.dtype.name)
        if shape is None or tile.layout != "row":
            return 0
        return (tile.rows // 16) * (tile.cols // 16) * max(1, shape.operand_vgprs * 4 // 16)
    if tile.dtype.bits == 8:
        # gfx950's fp8 operand (16x128 row fragments): two ds_read_b128 each;
        # hk-lds-pipeline sets the count.
        return ((tile.rows // 16) * (tile.cols // 128) * 2
                if get_target(arch).arch == "gfx950" and tile.layout == "row" else 0)
    if tile.dtype.bits != 16:
        return 0
    if tile.layout == "col" and tgt.arch == "gfx942":
        return 0                        # ordinary loads the compiler waits for
    return (tile.rows // 16) * (tile.cols // 16)


class MxScales(Value):
    """E8M0 block scales of an fp8 operand tile, in the lanes the scaled MFMA
    reads them from (`load_mx_scales`). An operand takes them with
    `Tile.scaled`."""

    __slots__ = ()

    def copy(self, *, into: "MxScales") -> "MxScales":
        """`into` = these scales, register to register -- how a one-stage loop body hands
        the next K-tile's prefetched scales to the current set. Returns `into`."""
        if not isinstance(into, MxScales) or into.ty != self.ty:
            raise TypeError(f"MxScales.copy: into is MxScales of this shape ({self.ty}), "
                            f"got {into!r}")
        hkd.MxScalesCopyOp(into.v, self.v, loc=_t().loc("mxs_copy"))
        return into


class Scaled:
    """An MX operand: an fp8 e4m3 row tile (FP4 in its container with fmt="fp4") and its E8M0
    scales, made by `Tile.scaled`. `hk.mma("mma_ABt", a.scaled(sa), b.scaled(sb), acc)` is
    gfx950's block-scaled MFMA; nothing else takes one."""

    __slots__ = ("tile", "scales", "fmt", "at")

    def __init__(self, tile: "Tile", scales: "MxScales", fmt: str, at):
        if fmt not in ("fp8", "fp4"):
            raise TypeError(f"Tile.scaled: fmt is 'fp8' or 'fp4', not {fmt!r}")
        if not isinstance(scales, MxScales):
            raise TypeError(f"Tile.scaled: the scales are MxScales (hk.load_mx_scales), "
                            f"got {scales!r}")
        self.tile, self.scales, self.fmt, self.at = tile, scales, fmt, (int(at[0]), int(at[1]))

    def __repr__(self) -> str:
        return f"Scaled({self.tile!r}, {self.scales!r}, fmt={self.fmt!r}, at={self.at})"


_MX_LAYOUTS = {"ocp": 0, "aiter_gate_up": 1, "aiter": 2}


def load_mx_scales(src: "Global", rows: int, kblocks: int, row0, kb0, *,
                   into: "MxScales" = None, layout: str = "ocp",
                   group_rows: int = 0, loose: bool = False) -> MxScales:
    """Rows row0.. and 32-wide blocks kb0.. of an E8M0 scale tensor (uint8, one
    byte a row and block: [R, K/32], the OCP MX layout -- or packed, i32, as
    hk.mx.pack_scales lays them out: one coalesced dword a lane per 64
    rows and K step) for an fp8 operand tile of `rows` rows and `kblocks * 32`
    columns. gfx950 only.

    `layout="aiter_gate_up"` / `"aiter"`: AITER's shuffled MoE weight scales
    (uint8 [E * group_rows, K/32], `shuffle_scale(is_guinterleave=True)`, with
    and without gate_up) -- the scales of a `load_fp4_frags` tile of the same
    rows: row0 the tile's first row in [E * group_rows], base tiles alternating
    gate and up with "aiter_gate_up". AITER scales are kept as loaded (fmt 2:
    the MFMA's opsel picks the byte; kb0 a multiple of 8 when kblocks > 4).
    `loose` (byte scales): a register a 16-row tile and K step, the byte as
    loaded (fmt 1). Both leave nothing for the VALU to do with a loaded
    scale, so its wait sits at the MFMA instead of right after the load --
    the packed form's byte insertion waited there, draining every load
    issued before it (K1 c8)."""
    if layout not in _MX_LAYOUTS:
        raise TypeError(f"load_mx_scales: layout is one of {sorted(_MX_LAYOUTS)}, not {layout!r}")
    lay = _MX_LAYOUTS[layout]
    if lay and (src.dtype is not i8 or group_rows <= 0 or group_rows % 32
                or (lay == 1 and rows % 32) or kblocks % 4):
        raise TypeError(f"load_mx_scales: AITER scales are uint8, need group_rows (a "
                        f"multiple of 32), 128-wide K steps, and with gate|up whole "
                        f"(gate, up) 32-row pairs; got {src.dtype.name}, group_rows "
                        f"{group_rows}, {rows} rows, {kblocks} blocks")
    if not isinstance(src, Global):
        raise TypeError(f"load_mx_scales: src must be a tensor parameter, got {src!r}")
    if src.dtype is i32:
        if rows % 64 or kblocks % 4:
            raise TypeError(f"load_mx_scales: packed (i32) scales come in 64-row groups "
                            f"and 128-wide K steps; got {rows} rows, {kblocks} blocks")
    elif src.dtype is not i8:
        raise TypeError(f"load_mx_scales: E8M0 scales are one byte each (an i8/uint8 "
                        f"tensor) or packed dwords (i32, hk.mx.pack_scales), "
                        f"not {src.dtype.name}")
    if get_target(_t().arch).arch != "gfx950":
        raise TypeError(f"load_mx_scales: block-scaled MFMA is gfx950's; {_t().arch} has none")
    if loose and (lay or src.dtype is not i8):
        raise TypeError("load_mx_scales(loose=True): byte (OCP) scales only")
    ty = MxScalesType(rows, kblocks, 2 if lay else (1 if loose else 0))
    t = _t()
    if into is not None:
        if not isinstance(into, MxScales) or into.ty != ty:
            raise TypeError(f"load_mx_scales: into is {into!r}, the result is {ty}")
        hkd.LoadMxScalesOp(None, src.v, as_int(row0).v, as_int(kb0).v, dst=into.v,
                           layout=lay, group_rows=group_rows, loc=t.loc("mxs"))
        return into
    op = hkd.LoadMxScalesOp(mlir_type(ty), src.v, as_int(row0).v, as_int(kb0).v,
                            layout=lay, group_rows=group_rows, loc=t.loc("mxs"))
    return MxScales(op.result, ty)


def copy_mx_scales(dst: "MxScales", src: "MxScales") -> None:
    """Deprecated: `src.copy(into=dst)`. dst = src, E8M0 scales copied register to register."""
    _deprecated("hk.copy_mx_scales(dst, src)", "src.copy(into=dst)")
    if not isinstance(dst, MxScales) or not isinstance(src, MxScales) or dst.ty != src.ty:
        raise TypeError(f"copy_mx_scales: two MxScales of one shape, got {dst!r} and {src!r}")
    src.copy(into=dst)


def load_fp4_frags(ty, src, row0, kc0, *, group_rows: Optional[int] = None,
                   gate_up: bool = False, into=None) -> "Tile":
    """Rows row0.. and 128-wide K steps kc0.. of AITER's shuffled MXFP4 MoE weights (gfx950,
    `shuffle_weight(is_guinterleave=True)`; src uint8 [E * group_rows, K/2], the packed bytes)
    as an FP4 operand, `w.scaled(sw, fmt="fp4")` in `hk.mma`. `ty` is an fp8 e4m3 row tile of
    16x128 fragments -- the container: each fragment's first four dwords hold the FP4 operand, the
    rest stay undefined. With `gate_up` (a stage-1 gate|up weight) the base tiles alternate the
    gate and up halves of 16-row block (row0 % group_rows) // 16 + i // 2, so gate column j and up
    column j of the product land in the same lanes.

    `src` may instead be an LDS tile of raw AITER weight bytes (fp8-typed, a row a 16-row block
    with gate|up rows interleaved, 1 KB a 128-wide K step: the [E * N/16 * 2, K/2 * 16] view of
    the weights, brought in by a group load): fragment (i, s) is then its row row0 + i and 1 KB
    chunk kc0 + s, and `group_rows` / `gate_up` do not apply."""
    if isinstance(src, SharedTile):
        if group_rows is not None or gate_up:
            raise TypeError("load_fp4_frags from an LDS tile: group_rows and gate_up are the "
                            "global layout's; the LDS rows are already in operand order")
        return _load_fp4_frags_lds(ty, src, row0, kc0, into=into)
    if group_rows is None:
        raise TypeError("load_fp4_frags: group_rows (rows an expert) is needed for global "
                        "weights")
    if (not isinstance(ty, RegTileType) or ty.dtype is not fp8e4m3 or ty.layout != "row"
            or ty.rows % 16 or ty.cols % 128):
        raise TypeError(f"load_fp4_frags: an fp8 e4m3 row tile of 16x128 fragments, not {ty!r}")
    if gate_up and ty.rows % 32:
        raise TypeError(f"load_fp4_frags: a gate|up tile is whole (gate, up) 32-row pairs; "
                        f"{ty.rows} rows")
    if not isinstance(src, Global) or src.dtype is not i8:
        raise TypeError(f"load_fp4_frags: src is the packed FP4 bytes, a uint8 tensor, not {src!r}")
    if group_rows <= 0 or group_rows % 32:
        raise TypeError(f"load_fp4_frags: group_rows (rows an expert) is a multiple of 32, "
                        f"got {group_rows}")
    if get_target(_t().arch).arch != "gfx950":
        raise TypeError(f"load_fp4_frags: block-scaled MFMA is gfx950's; {_t().arch} has none")
    return _make(hkd.LoadFp4FragsOp, ty, (src.v, as_int(row0).v, as_int(kc0).v),
                 into=into, name="w4f", gate_up=int(bool(gate_up)), group_rows=group_rows)


def mx_quant_store(src: "SharedTile", q: "Global", s: "Global", row0, col0) -> None:
    """MX-quantize an LDS tile's rows and store them -- the epilogue that hands a GEMM's output
    to an MXFP8 consumer without a separate quant pass. Every thread of the workgroup takes
    (row, 32-column group) pairs of `src` (bf16 or fp32: store it, then barrier, first). For each
    group it computes AITER's E8M0 exponent exactly, e = ceil(log2(amax / 448)) from amax's bits.
    It writes the FP8 e4m3 values to q[row0 + r, col0 + 32 g ..] and the scale byte to `s`:
    a uint8 `s` gets AITER's 32-row layout (hk.mx.aiter_scales of q's [R, K/32], K/32 padded to
    8), an int32 `s` the packed layout hk.load_mx_scales reads ([ceil(R/64), K/128 * 64],
    hk.mx.pack_scales; rows past q's last get the unit scale, as pack_scales pads them). The
    result is bit-identical to hk.mx.quant followed by hk.mx.aiter_scales or hk.mx.pack_scales."""
    if not isinstance(src, SharedTile) or src.ty.dtype not in (bf16, fp32) or src.ty.cols % 32:
        raise TypeError(f"mx_quant_store: a bf16 or fp32 LDS tile of 32-column groups, not {src!r}")
    if not isinstance(q, Global) or q.dtype is not fp8e4m3:
        raise TypeError(f"mx_quant_store: q is an fp8 e4m3 tensor, not {q!r}")
    if not isinstance(s, Global) or s.dtype not in (i8, i32):
        raise TypeError(f"mx_quant_store: s holds E8M0 bytes, a uint8 tensor (AITER's 32-row "
                        f"layout) or an int32 one (packed), not {s!r}")
    if get_target(_t().arch).arch != "gfx950":
        raise TypeError(f"mx_quant_store: FP8 e4m3 conversion is gfx950's here, not {_t().arch}")
    hkd.MxQuantStoreOp(src.v, q.v, s.v, as_int(row0).v, as_int(col0).v, loc=_t().loc("mxq"))


def load_fp4_frags_lds(ty, src: "SharedTile", row0, chunk0, *, into=None) -> "Tile":
    """Deprecated: `hk.load_fp4_frags(ty, src, row0, chunk0)` with `src` an LDS tile."""
    _deprecated("hk.load_fp4_frags_lds(...)", "hk.load_fp4_frags(ty, lds_tile, row0, chunk0)")
    return _load_fp4_frags_lds(ty, src, row0, chunk0, into=into)


def _load_fp4_frags_lds(ty, src: "SharedTile", row0, chunk0, *, into=None) -> "Tile":
    if (not isinstance(ty, RegTileType) or ty.dtype is not fp8e4m3 or ty.layout != "row"
            or ty.rows % 16 or ty.cols % 128):
        raise TypeError(f"load_fp4_frags_lds: an fp8 e4m3 row tile of 16x128 fragments, not {ty!r}")
    if not isinstance(src, SharedTile) or src.ty.dtype is not fp8e4m3 or src.ty.cols % 1024:
        raise TypeError(f"load_fp4_frags_lds: an fp8-typed LDS tile of whole 1 KB rows, not {src!r}")
    return _make(hkd.LoadFp4FragsLdsOp, ty, (src.v, as_int(row0).v, as_int(chunk0).v),
                 into=into, name="w4l")


def fp4_operand(words: Tile) -> Tile:
    """An MXFP4 operand fragment from a lane's four dwords (gfx950): `words`, an i32 strip of 1 x 256
    with run 4, becomes a 16x128 fp8 e4m3 row tile -- the FP4 container `mma_ABt_mx(..., a_fmt /
    b_fmt="fp4")` reads, lane l's 32 values being row l % 16, columns 32 (l / 16) .. + 32. For
    operands gathered with load_elems(runs=True) from a layout no tile load knows (a preshuffled
    paged cache: a lane's 16 bytes are one token's 32 dims)."""
    if (getattr(words.ty, "run", None) != 4 or words.ty.dtype is not i32 or words.ty.rows != 1
            or words.ty.cols != 256):
        raise TypeError(f"fp4_operand: an i32 strip of 1 x 256, run 4 (four dwords a lane), not {words.ty}")
    t = _t()
    ty = RegTileType(fp8e4m3, 16, 128, "row")
    return wrap(hkd.Fp4OperandOp(mlir_type(ty), words.v, loc=t.loc("fp4")).result, ty)


def fp8_operand(words: Tile) -> Tile:
    """An MXFP8 operand fragment from a lane's eight dwords (gfx950): `words`, an i32 strip of 1 x 512
    with run 4 (two runs a lane), becomes a 16x128 fp8 e4m3 row tile -- run 0 the K step's 16 bytes at
    K 16 (l / 16) .., run 1 those at 64 + 16 (l / 16) .., of row l % 16: the fragment a tile load
    makes. For operands kept in LDS as words (an fp8 shared tile has no register path) or gathered."""
    if (getattr(words.ty, "run", None) != 4 or words.ty.dtype is not i32 or words.ty.rows != 1
            or words.ty.cols != 512):
        raise TypeError(f"fp8_operand: an i32 strip of 1 x 512, run 4 (eight dwords a lane), not {words.ty}")
    t = _t()
    ty = RegTileType(fp8e4m3, 16, 128, "row")
    return wrap(hkd.Fp8OperandOp(mlir_type(ty), words.v, loc=t.loc("fp8")).result, ty)


def mx_scales_loose(bytes_: Tile) -> MxScales:
    """Loose E8M0 scales of a 16x128 operand from a lane's element (gfx950): `bytes_`, an i32 strip of
    1 x 64 with run 1 holding the byte in bits 0..7, is lane l's scale -- row l % 16, block l / 16."""
    if (getattr(bytes_.ty, "run", None) != 1 or bytes_.ty.dtype is not i32 or bytes_.ty.rows != 1
            or bytes_.ty.cols != 64):
        raise TypeError(f"mx_scales_loose: an i32 strip of 1 x 64, run 1, not {bytes_.ty}")
    t = _t()
    ty = MxScalesType(16, 4, 1)
    op = hkd.MxScalesLooseOp(mlir_type(ty), bytes_.v, loc=t.loc("mxs"))
    return MxScales(op.result, ty)


def mma_ABt_mx(a: Tile, b: Tile, acc: Tile, sa: MxScales, sb: MxScales, *, into=None,
               a_fmt: str = "fp8", b_fmt: str = "fp8", sa_at=(0, 0), sb_at=(0, 0)) -> Tile:
    """Deprecated: `hk.mma("mma_ABt", a.scaled(sa, fmt=a_fmt, at=sa_at),
    b.scaled(sb, fmt=b_fmt, at=sb_at), acc)`, the same instruction."""
    _deprecated("hk.mma_ABt_mx(a, b, acc, sa, sb, ...)",
                'hk.mma("mma_ABt", a.scaled(sa, ...), b.scaled(sb, ...), acc)')
    return _mma_mx(a, b, acc, sa, sb, into=into, a_fmt=a_fmt, b_fmt=b_fmt, sa_at=sa_at,
                   sb_at=sb_at)


def _mma_mx(a: Tile, b: Tile, acc: Tile, sa: MxScales, sb: MxScales, *, into=None,
            a_fmt: str = "fp8", b_fmt: str = "fp8", sa_at=(0, 0), sb_at=(0, 0)) -> Tile:
    """acc + (sa . A)(sb . B)^T on gfx950's v_mfma_scale_f32_16x16x128_f8f6f4 (hk.mma's scaled
    form; the checks below name it mma_ABt_mx, the op it emits)."""
    for v, what in ((a, "A"), (b, "B")):
        if not isinstance(v, Tile) or v.ty.dtype is not fp8e4m3 or v.ty.layout != "row":
            raise TypeError(f"mma_ABt_mx: {what} is an fp8 e4m3 row-layout tile, got {v!r}")
    if not isinstance(acc, Tile) or acc.ty.dtype != fp32 or acc.ty.layout != "col":
        raise TypeError(f"mma_ABt_mx: the accumulator is an fp32 col tile, got {acc!r}")
    if a.ty.cols != b.ty.cols or a.ty.cols % 128:
        raise TypeError(f"mma_ABt_mx: A and B share K, a multiple of 128; got "
                        f"{a.ty.cols} and {b.ty.cols}")
    for s, v, what, (ti, st) in ((sa, a, "sa", sa_at), (sb, b, "sb", sb_at)):
        ti, st = int(ti), int(st)
        if not isinstance(s, MxScales) or ti < 0 or st < 0:
            raise TypeError(f"mma_ABt_mx: {what} is MxScales and its offsets >= 0, got {s!r} at "
                            f"{(ti, st)}")
        ok = s.ty.rows >= v.ty.rows + 16 * ti and s.ty.kblocks * 32 >= v.ty.cols + 128 * st
        if not ok:
            raise TypeError(f"mma_ABt_mx: {what} holds the scales of a {v.ty.rows}x"
                            f"{v.ty.cols} operand (from tile {ti}, step {st}), got {s!r}")
    ty = RegTileType(fp32, a.ty.rows, b.ty.rows, "col")
    if (acc.ty.rows, acc.ty.cols) != (ty.rows, ty.cols):
        raise TypeError(f"mma_ABt_mx: acc is {acc.ty}, expected {a.ty.rows}x{b.ty.rows}")
    for f, what in ((a_fmt, "a_fmt"), (b_fmt, "b_fmt")):
        if f not in ("fp8", "fp4"):
            raise TypeError(f"mma_ABt_mx: {what} is 'fp8' or 'fp4', not {f!r}")
    return _make(hkd.MmaABtMxOp, acc.ty, (a.v, b.v, acc.v, sa.v, sb.v), into=into, name="mx",
                 a_fmt=4 if a_fmt == "fp4" else 0, b_fmt=4 if b_fmt == "fp4" else 0,
                 sa_tile=int(sa_at[0]), sa_step=int(sa_at[1]), sb_tile=int(sb_at[0]),
                 sb_step=int(sb_at[1]))


class Granules(Value):
    """One shared tile's per-lane LDS addresses."""

    __slots__ = ()

    def load_frag(self, row: int, col: int, tile: RegTileType, *, into=None) -> Tile:
        """One 16x16 fragment at (row, col), in units of 16. Never waits."""
        st = self.ty.tile
        if (tile.rows, tile.cols) != (16, 16) or tile.layout != "row" or tile.dtype != st.dtype:
            raise TypeError(f"load_frag into {tile} from {st}: a 16x16 'row' "
                            f"tile of the shared tile's own dtype")
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("load_frag coordinates are Python ints: they become "
                            "ds_read_b128's immediate")
        return _make(hkd.LoadFragOp, tile, (self.v, row, col, 2), into=into,
                     name="frag", wait=False)

    def store_frag(self, value: Tile, row: Scalar, col: int) -> None:
        """One 16x16 fragment into the tile; `row` may be a runtime value."""
        if not isinstance(col, int):
            raise TypeError("store_frag col is a Python int (it picks the granule)")
        t = _t()
        if isinstance(row, int):
            hkd.StoreFragOp(self.v, value.v, col, row=row, loc=t.loc())
        else:
            hkd.StoreFragOp(self.v, value.v, col, row_index=as_int(row).v,
                            loc=t.loc())


class SharedVec(Value):
    """A stack of LDS vectors: the only way across the warps of a workgroup."""

    __slots__ = ()

    def store(self, vec: Vec, index: Scalar = 0) -> None:
        """Write a register vector into slot `index` of the stack (no
        barrier: put one before another warp reads it)."""
        _strip_vec_lds(vec.ty)
        t = _t()
        hkd.StoreSharedVecOp(self.v, vec.v, as_int(index).v, loc=t.loc())

    def load(self, vt: RegVecType, index: Scalar = 0) -> Vec:
        """Read slot `index` of the stack as a register vector of type `vt`."""
        _strip_vec_lds(vt)
        t = _t()
        return Vec(hkd.LoadSharedVecOp(mlir_type(vt), self.v, as_int(index).v,
                                       loc=t.loc("ldv")).result, vt)


def _strip_vec_lds(vt) -> None:
    if _is_strip(vt) and vt.kind != "col":
        raise TypeError(f"{vt!r} is a strip's row_vec, a strip itself; only a "
                        f"strip's col_vec (one value a row) goes through an "
                        f"LDS vector")


class Stage(Value):
    """float4 buf[depth][calls]: the VGPR half of a global->LDS copy."""

    __slots__ = ("dst",)


# ---------------------------------------------------------------- docstrings
# The one-line methods' docstrings, set here rather than inline so the
# methods stay one line each. docs/api/hk.md is generated from them
# (scripts/gen_api_docs.py).

def _doc(cls, name: str, text: str) -> None:
    obj = cls.__dict__[name]
    if isinstance(obj, property):  # a property copies its doc when made
        setattr(cls, name, property(obj.fget, obj.fset, obj.fdel, text))
    else:
        obj.__doc__ = text


for _n in ("exp", "exp2", "log", "log2", "abs", "relu", "gelu", "dgelu", "sqrt",
           "rsqrt", "silu"):
    _doc(_Elementwise, _n, f"`hk.math.{_n}` of this value: a new value, or "
                           f"`into=` an existing one. See docs/api/hk.math.md for "
                           f"the dtypes each takes.")
for _n in ("exp", "exp2", "rsqrt"):
    _doc(_Elementwise, _n + "_", f"`{_n}` in place: this value's registers.")
_doc(_Elementwise, "maximum", "Elementwise max with `o` (a value of this type "
                              "or a number); `hk.max(a, b)`.")
_doc(_Elementwise, "minimum", "Elementwise min with `o` (a value of this type "
                              "or a number); `hk.min(a, b)`.")
_doc(_Elementwise, "copy", "A copy in new registers, or `into=` an existing value.")
_doc(_Elementwise, "zero_", "Fill with 0, in place.")
_doc(_Elementwise, "fill_", "Fill with `v`, in place.")
_doc(_Elementwise, "neg_infty_", "Fill with -inf, in place.")
for _n, _ax in (("sum", "sum"), ("max", "max"), ("min", "min"), ("prod", "product")):
    _doc(Tile, _n, f"The {_ax} along `axis`: 1 (default) along each row, one "
                   f"value per row (a col_vec); 0 along each column (a row_vec). "
                   f"Lazy: `v = t.{_n}()` makes the vector, "
                   + ("`acc += t.sum()` accumulates into one, " if _n == "sum" else "")
                   + ("`m.max_(t.max())` folds into one (the online max), "
                      if _n == "max" else "")
                   + "and `.value(into=v)` writes an existing one.")
_doc(Tile, "dtype", "The element dtype.")
_doc(Tile, "T", "This tile transposed, as an MMA operand (`acc += a @ b.T`) or "
                "for `.value()`, which makes the transposed tile.")
for _n, _side in (("left_fill_", "columns < idx"), ("right_fill_", "columns >= idx"),
                  ("upper_fill_", "rows < idx"), ("lower_fill_", "rows >= idx")):
    _doc(Tile, _n, f"Set the elements in {_side} to `value` (default 0), in place "
                   f"or `into=` a copy: the mask for a ragged last tile. `idx` may "
                   f"be a device Int." + (" Strips take left_fill_ and right_fill_ "
                                          "only." if _n in ("left_fill_", "right_fill_")
                                          else " Not on strips."))
_doc(Tile, "triu_", "Keep the upper triangle from diagonal `idx`, the rest set to "
                    "`value` (default 0), in place. Not on strips.")
_doc(Tile, "tril_", "Keep the lower triangle from diagonal `idx`, the rest set to "
                    "`value` (default 0), in place. Not on strips.")
_doc(Vec, "min_", "Fold `t.min(axis=...)` into this vector: the running min.")
_doc(Global, "batch", "The tensor's extent along axis 0 of (batch, depth, rows, "
                      "cols), a device Int.")
_doc(Global, "depth", "The tensor's extent along axis 1, a device Int.")
_doc(Global, "rows", "The tensor's rows (axis 2), a device Int. A 2-D tensor is "
                     "(rows, cols).")
_doc(Global, "cols", "The tensor's columns (axis 3), a device Int.")
_doc(Global, "dtype", "The element dtype the parameter was annotated with.")
_doc(Global, "at", "Element indexing: `x.at[r, c]` (or `[b, d, r, c]`) is a View "
                   "at that element; a load reads the tile that starts there.")
_doc(Global, "tile", "Tile indexing: `x.tile[r, c]` is a View at tile (r, c) of "
                     "the tile type the load names.")
_doc(Coord, "unit", "\"tile\" or \"element\".")
del _n, _ax, _side


__all__ = [
    "Value", "Int", "Bool", "Coord", "Global", "View", "Tile", "Vec",
    "SharedTile", "SharedVec", "Granules", "Stage", "Reduction", "MatMul",
    "Transposed", "const", "as_int", "scalar_op", "compare", "i1", "coord",
    "tile_coord", "elem_coord", "binary", "unary", "broadcast", "mma", "zeros",
    "full", "neg_infty", "rand", "load_w4", "transpose", "transpose_sep", "broadcast_row",
    "broadcast_col", "as_vec", "lds_loads", "wrap",
]


def map_cpp(expr: str, *args, dtype: Optional[DType] = None, name: str = "map") -> Tile:
    """An elementwise C++ expression over strips (gfx950): element (r, c) of the result is `expr`
    with `x0`, `x1`, ... bound to element (r, c) of each strip argument -- or to the argument
    itself for an `Int` or a Python int -- converted to `dtype` (default: the strips' one dtype;
    strips of fp32 and i32 together must say it).

    For what the operators cannot say and a kernel must reproduce bit for bit: bit manipulation,
    a fused multiply-add, one particular instruction. E.g. the biased exponent of the smallest
    power of two >= x:

        e = hk.map_cpp("((__builtin_bit_cast(int, x0) >> 23) & 0xFF)"
                       " + ((__builtin_bit_cast(int, x0) & 0x7FFFFF) != 0)", x, dtype=hk.i32)

    The strips have one shape and run; strips and result are fp32 or i32 (convert a 16-bit strip
    with `.to` first). The expression is one C++ expression, compiled as written -- with
    floating-point contraction off, so `a * b + c` is two roundings and a fused multiply-add is
    spelled `__builtin_fmaf` -- no `;` or braces, and the compiler, not hk, checks it."""
    from .arch import target  # noqa: PLC0415

    if target().wave != 64:
        raise TypeError(f"hk.map_cpp is gfx950's (its strips), not {target().arch}'s")
    if not isinstance(expr, str) or not expr.strip() or any(ch in expr for ch in ";{}"):
        raise ValueError(f"hk.map_cpp: one C++ expression, no ';' or braces, got {expr!r}")
    strips = [a for a in args if isinstance(a, Tile)]
    if not strips:
        raise TypeError("hk.map_cpp needs at least one strip argument (its shape is the result's)")
    s0 = strips[0].ty
    if getattr(s0, "run", None) is None:
        raise TypeError(f"hk.map_cpp maps strips (hk.RS), not {s0}")
    if dtype is None:
        kinds = {a.ty.dtype.name for a in strips}
        if len(kinds) > 1:
            raise TypeError(f"hk.map_cpp over strips of {sorted(kinds)}: say the result's dtype= -- "
                            f"the first strip's would silently round an fp32 result to an integer")
        dtype = s0.dtype
    if dtype.name not in ("fp32", "i32"):
        raise TypeError(f"hk.map_cpp makes an fp32 or i32 strip, not {dtype.name}")
    ops = []
    for i, a in enumerate(args):
        if isinstance(a, Tile):
            if (a.ty.rows, a.ty.cols, getattr(a.ty, "run", None)) != (s0.rows, s0.cols, s0.run):
                raise TypeError(f"hk.map_cpp: x{i} is {a.ty}, every strip has {s0}'s shape and run")
            if a.ty.dtype.name not in ("fp32", "i32"):
                raise TypeError(f"hk.map_cpp: x{i} is {a.ty.dtype.name}; strip arguments are fp32 "
                                f"or i32 (convert with .to)")
            ops.append(a.v)
        elif isinstance(a, Int):
            ops.append(a.v)
        elif isinstance(a, int) and not isinstance(a, bool):
            ops.append(as_int(a).v)                       # a Python int: an i32 constant
        else:
            raise TypeError(f"hk.map_cpp: x{i} is {a!r}: a strip, an hk.Int or a Python int; a "
                            f"float belongs in the expression itself")
    ty = RegTileType(dtype, s0.rows, s0.cols, s0.layout, s0.frag, run=s0.run)
    t = _t()
    op = hkd.MapCppOp(mlir_type(ty), ops, expr, loc=t.loc(name))
    return wrap(op.result, ty)


def iota(ty) -> Tile:
    """An i32 strip of type `ty`'s shape whose element (r, c) is c: each element's own column, for
    index arithmetic (hk.map_cpp, load_elems / store_elems) and per-column choices (gfx950)."""
    from .arch import target  # noqa: PLC0415

    if not _is_strip_tile(ty):
        raise TypeError(f"hk.iota makes a strip (hk.RS), not {ty}")
    if target().wave != 64:
        raise TypeError(f"hk.iota is gfx950's, not {target().arch}'s")
    ty = RegTileType(i32, ty.rows, ty.cols, ty.layout, ty.frag, run=ty.run)
    t = _t()
    return wrap(hkd.IotaOp(mlir_type(ty), loc=t.loc("iota")).result, ty)


def _lane_strip_type(ty, run: int, dtype) -> RegTileType:
    if not isinstance(ty, RegTileType):
        raise TypeError(f"a lane strip is of a register tile, not {ty}")
    n = ty.rows * ty.cols // 64
    if run not in (1, 2, 4, 8, 16) or n % run:
        raise ValueError(f"a lane strip of {ty} ({n} elements a lane) in runs of {run}: a run is "
                         f"1, 2, 4, 8 or 16 and divides {n}")
    return RegTileType(dtype, 1, 64 * n, "strip", None, run=run)


def lane_coords(tile, run: int = 8) -> tuple:
    """(rows, cols): i32 strips of `tile.lane_strip(run)`'s shape, element by element the row and the
    column of `tile` that element is (gfx950). `tile` is a register tile. Index arithmetic for an
    MMA result taken as a strip: its global rows and columns, MX groups, masks."""
    from .arch import target  # noqa: PLC0415

    if target().wave != 64:
        raise TypeError(f"hk.lane_coords is gfx950's, not {target().arch}'s")
    if not isinstance(tile, Tile):
        raise TypeError(f"hk.lane_coords takes a register tile, not {tile!r}")
    sty = _lane_strip_type(tile.ty, run, i32)
    t = _t()
    return tuple(wrap(hkd.LaneCoordOp(mlir_type(sty), tile.v, axis, loc=t.loc("lc")).result, sty)
                 for axis in (0, 1))


def mfma_strip(kind: str, a: Tile, b: Tile, acc: Tile) -> Tile:
    """One raw MFMA on strip registers (gfx950), for a result that must match another kernel's
    MFMA bit for bit. "f32_16x16x4": `acc` (hk.RS(fp32, 1, 256, 4): lane l's four values are C
    rows 4 (l // 16) + i, column l % 16) + A B^T over K = 4, where lane l of `a` and `b`
    (hk.RS(fp32, 1, 64, 1)) holds A[l % 16][l // 16] and B[l % 16][l // 16] --
    v_mfma_f32_16x16x4_f32, nothing reordered. Returns the new accumulator."""
    if kind != "f32_16x16x4":
        raise ValueError(f"hk.mfma_strip({kind!r}): the one kind is \"f32_16x16x4\"")

    def shaped(v, cols, run):
        ty = v.ty
        return (_is_strip_tile(ty) and ty.dtype.name == "fp32" and (ty.rows, ty.cols, ty.run) == (1, cols, run))
    if not (shaped(a, 64, 1) and shaped(b, 64, 1) and shaped(acc, 256, 4)):
        raise TypeError(f"hk.mfma_strip(\"f32_16x16x4\") takes a, b: hk.RS(fp32, 1, 64, 1) and acc: "
                        f"hk.RS(fp32, 1, 256, 4), not {a.ty}, {b.ty}, {acc.ty}")
    t = _t()
    return wrap(hkd.MfmaStripOp(mlir_type(acc.ty), a.v, b.v, acc.v, kind, loc=t.loc("mfma")).result, acc.ty)

