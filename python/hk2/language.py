"""The kernel language: what a kernel body calls. Everything here records ops into the tile IR."""
from __future__ import annotations

import builtins
import math
import sys
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from . import ir
from . import types as T
from .ir import PtrType, ScalarType, TileType, Value

__all__ = ["print", "program_id", "zeros", "full", "arange", "load", "store", "dot", "sum", "max", "min", "exp", "exp2",
           "log", "log2", "sqrt", "rsqrt", "abs", "maximum", "minimum", "where", "cdiv", "static_range", "constexpr",
           "Tile", "Scalar", "TensorArg", "HKError"]


class HKError(Exception):
    """A kernel-language error, reported at the kernel source line that caused it."""


class constexpr:
    """Annotation: ``BM: hk.constexpr = 128`` makes a parameter a compile-time value."""


# ---------------------------------------------------------------- tracing state
class _State:
    builder: Optional[ir.Builder] = None
    files: set = set()


def _b() -> ir.Builder:
    if _State.builder is None:
        raise HKError("hk2 language functions can only be called inside a kernel being compiled")
    return _State.builder


def _loc() -> Optional[Tuple[str, int]]:
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename in _State.files:
            return (f.f_code.co_filename, f.f_lineno)
        f = f.f_back
    return None


def _op(kind, args, result_types=(), attrs=None, blocks=()):
    b = _b()
    b.loc = _loc()
    return b.op(kind, args, result_types, attrs, blocks)


class _Undef:
    def __repr__(self):
        return "<undefined>"

    def __getattr__(self, name):
        raise HKError("a variable defined only inside a device loop or branch is used outside it")


UNDEF = _Undef()


def _err(msg: str):
    loc = _loc()
    if loc:
        msg = f"{loc[0]}:{loc[1]}: {msg}"
    return HKError(msg)


def _check_shape(shape):
    shape = tuple(int(s) for s in shape)
    for s in shape:
        if s <= 0 or s & (s - 1):
            raise _err(f"tile shapes must be powers of two, got {shape}")
    if not 1 <= len(shape) <= 2:
        raise _err(f"tiles are 1-D or 2-D, got {shape}")
    return shape


# ---------------------------------------------------------------- scalars
def _const(v, dtype: Optional[T.DType] = None) -> "Scalar":
    if dtype is None:
        dtype = T.i1 if isinstance(v, bool) else T.i32 if isinstance(v, int) else T.f32
    return Scalar(_op("const", [], [ScalarType(dtype)], {"value": v}).result)


def _as_scalar(v) -> "Scalar":
    if isinstance(v, Scalar):
        return v
    if isinstance(v, (bool, int, float)):
        return _const(v)
    raise _err(f"expected a scalar, got {type(v).__name__}")


_CMP = {"lt", "le", "gt", "ge", "eq", "ne"}


def _result_dtype(fn: str, dt: T.DType) -> T.DType:
    if fn in _CMP:
        return T.i1
    if fn == "truediv" and dt.is_int:
        return T.f32
    return dt


def _scalar_binary(fn, a, b):
    a, b = _as_scalar(a), _as_scalar(b)
    dt = T.promote(a.dtype, b.dtype)
    a, b = a.to(dt), b.to(dt)
    return Scalar(_op("sop", [a.v, b.v], [ScalarType(_result_dtype(fn, dt))], {"fn": fn}).result)


class Scalar:
    """A value uniform across the workgroup: an argument, a program id, a loop index, and math on them."""
    __slots__ = ("v",)

    def __init__(self, v: Value):
        self.v = v

    @property
    def dtype(self) -> T.DType:
        return self.v.type.dtype

    def to(self, dtype: T.DType) -> "Scalar":
        if dtype == self.dtype:
            return self
        return Scalar(_op("scast", [self.v], [ScalarType(dtype)]).result)

    def __bool__(self):
        raise _err("a runtime value has no Python truth value here; this condition must be decided on the GPU "
                   "(use `if` directly on it in the kernel body, or & | instead of and / or)")

    def __index__(self):
        raise _err("a runtime value cannot be used where Python needs an int (use hk.static_range for unrolled "
                   "loops, or a constexpr parameter)")

    def _bin(fn, rev=False):
        def f(self, other):
            if isinstance(other, Tile):
                return NotImplemented
            return _scalar_binary(fn, other, self) if rev else _scalar_binary(fn, self, other)
        return f

    __add__, __radd__ = _bin("add"), _bin("add", True)
    __sub__, __rsub__ = _bin("sub"), _bin("sub", True)
    __mul__, __rmul__ = _bin("mul"), _bin("mul", True)
    __truediv__, __rtruediv__ = _bin("truediv"), _bin("truediv", True)
    __floordiv__, __rfloordiv__ = _bin("floordiv"), _bin("floordiv", True)
    __mod__, __rmod__ = _bin("mod"), _bin("mod", True)
    __and__, __rand__ = _bin("and"), _bin("and", True)
    __or__, __ror__ = _bin("or"), _bin("or", True)
    __xor__ = _bin("xor")
    __lshift__, __rshift__ = _bin("shl"), _bin("shr")
    __lt__, __le__, __gt__, __ge__ = _bin("lt"), _bin("le"), _bin("gt"), _bin("ge")
    __eq__, __ne__ = _bin("eq"), _bin("ne")
    __hash__ = object.__hash__

    def __neg__(self):
        return _scalar_binary("sub", 0 if self.dtype.is_int else 0.0, self)

    def __repr__(self):
        return f"Scalar({self.dtype.name})"


# ---------------------------------------------------------------- tiles
def _splat(v, shape, dtype) -> "Tile":
    if isinstance(v, Scalar):
        return Tile(_op("splat", [v.to(dtype).v], [TileType(shape, dtype)]).result)
    return Tile(_op("full", [], [TileType(shape, dtype)], {"value": v}).result)


def _broadcast_shapes(a, b):
    if len(a) < len(b):
        a = (1,) * (len(b) - len(a)) + a
    if len(b) < len(a):
        b = (1,) * (len(a) - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            raise _err(f"shapes {a} and {b} do not broadcast")
        out.append(builtins.max(x, y))
    return tuple(out)


def _broadcast_to(t: "Tile", shape) -> "Tile":
    while len(t.shape) < len(shape):
        t = t._expand(0)
    if t.shape == tuple(shape):
        return t
    return Tile(_op("broadcast", [t.v], [TileType(tuple(shape), t.dtype)]).result)


def _weak_dtype(x, other_dt: T.DType) -> T.DType:
    """Python numbers take the other operand's type (a float with an int tile makes f32)."""
    if isinstance(x, (Tile, Scalar)):
        return x.dtype
    if isinstance(x, bool):
        return other_dt
    if isinstance(x, float) and other_dt.is_int:
        return T.f32
    return other_dt


def _tile_args(*xs):
    tiles = [x for x in xs if isinstance(x, Tile)]
    shape = tiles[0].shape
    for t in tiles[1:]:
        shape = _broadcast_shapes(shape, t.shape)
    dt = None
    for x in xs:
        if isinstance(x, (Tile, Scalar)):
            dt = x.dtype if dt is None else T.promote(dt, x.dtype)
    out = []
    for x in xs:
        if isinstance(x, Tile):
            out.append(_broadcast_to(x.to(dt), shape))
        else:
            out.append(_splat(x if isinstance(x, Scalar) else (x if dt.is_float else int(x)), shape, dt))
    return out, shape, dt


def _ew(fn: str, *xs) -> "Tile":
    if fn in ("and", "or", "xor") and builtins.any(isinstance(x, Tile) and x.dtype.is_float for x in xs):
        raise _err(f"bitwise {fn} needs integer or boolean tiles")
    ts, shape, dt = _tile_args(*xs)
    return Tile(_op("ew", [t.v for t in ts], [TileType(shape, _result_dtype(fn, dt))], {"fn": fn}).result)


class Tile:
    """A block of values spread over the workgroup's registers. Its layout is the compiler's choice."""
    __slots__ = ("v",)

    def __init__(self, v: Value):
        self.v = v

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.v.type.shape

    @property
    def dtype(self) -> T.DType:
        return self.v.type.dtype

    def to(self, dtype: T.DType) -> "Tile":
        if dtype == self.dtype:
            return self
        return Tile(_op("cast", [self.v], [TileType(self.shape, dtype)]).result)

    @property
    def T(self) -> "Tile":
        if len(self.shape) != 2:
            raise _err(".T needs a 2-D tile")
        return Tile(_op("trans", [self.v], [TileType(self.shape[::-1], self.dtype)]).result)

    def _expand(self, axis: int) -> "Tile":
        shape = self.shape[:axis] + (1,) + self.shape[axis:]
        if len(shape) > 2:
            raise _err("tiles are at most 2-D")
        return Tile(_op("expand_dims", [self.v], [TileType(shape, self.dtype)], {"axis": axis}).result)

    def __getitem__(self, idx):
        idx = idx if isinstance(idx, tuple) else (idx,)
        if len(self.shape) == 1 and len(idx) == 2:
            if idx[1] is None and idx[0] == slice(None):
                return self._expand(1)
            if idx[0] is None and idx[1] == slice(None):
                return self._expand(0)
        raise _err("tiles support only t[:, None] and t[None, :] indexing")

    def _bin(fn, rev=False):
        def f(self, other):
            return _ew(fn, other, self) if rev else _ew(fn, self, other)
        return f

    __add__, __radd__ = _bin("add"), _bin("add", True)
    __sub__, __rsub__ = _bin("sub"), _bin("sub", True)
    __mul__, __rmul__ = _bin("mul"), _bin("mul", True)
    __truediv__, __rtruediv__ = _bin("truediv"), _bin("truediv", True)
    __floordiv__, __rfloordiv__ = _bin("floordiv"), _bin("floordiv", True)
    __mod__, __rmod__ = _bin("mod"), _bin("mod", True)
    __and__, __rand__ = _bin("and"), _bin("and", True)
    __or__, __ror__ = _bin("or"), _bin("or", True)
    __xor__, __rxor__ = _bin("xor"), _bin("xor", True)
    __lshift__, __rshift__ = _bin("shl"), _bin("shr")
    __lt__, __le__, __gt__, __ge__ = _bin("lt"), _bin("le"), _bin("gt"), _bin("ge")
    __eq__, __ne__ = _bin("eq"), _bin("ne")
    __hash__ = object.__hash__

    def __neg__(self):
        return _ew("sub", 0, self)

    def __bool__(self):
        raise _err("a tile has no truth value; use hk.where for element-wise choices")

    def sum(self, axis: int) -> "Tile":
        return _reduce("add", self, axis)

    def max(self, axis: int) -> "Tile":
        return _reduce("max", self, axis)

    def min(self, axis: int) -> "Tile":
        return _reduce("min", self, axis)

    def __repr__(self):
        return f"Tile({'x'.join(map(str, self.shape))}, {self.dtype.name})"


def _reduce(fn: str, t: Tile, axis: int) -> Tile:
    if not isinstance(t, Tile):
        raise _err("reductions take a tile")
    axis = axis % len(t.shape)
    shape = tuple(s for d, s in enumerate(t.shape) if d != axis)
    if not shape:
        raise _err("reducing a 1-D tile to a scalar is not supported yet; keep a 2-D tile and reduce one axis")
    dt = t.dtype
    if fn == "add" and dt in (T.bf16, T.f16):
        dt = T.f32  # sums of 16-bit floats accumulate in f32
    return Tile(_op("reduce", [t.to(dt).v], [TileType(shape, dt)], {"fn": fn, "axis": axis}).result)


# ---------------------------------------------------------------- tensors
class TensorArg:
    """A torch tensor passed to a kernel: a base address with runtime shape and strides."""

    def __init__(self, name: str, ptr: Value, dtype: T.DType, shape: List[Scalar], strides: List[Union[Scalar, int]],
                 vec: int):
        self.name, self.ptr, self.dtype, self.shape, self._strides, self.vec = name, ptr, dtype, shape, strides, vec

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def stride(self, d: int):
        return self._strides[d]

    def __repr__(self):
        return f"TensorArg({self.name}, {self.dtype.name}, {self.ndim}-D)"


def _offsets(t: TensorArg, offsets) -> List[Scalar]:
    if not isinstance(offsets, (tuple, list)):
        offsets = (offsets,)
    if len(offsets) != t.ndim:
        raise _err(f"{t.name} is {t.ndim}-D; give {t.ndim} offsets, got {len(offsets)}")
    return [_as_scalar(o).to(T.i32) for o in offsets]


def _tensor_operands(t: TensorArg) -> List[Value]:
    strides = [_as_scalar(s).to(T.i32).v for s in t._strides]
    return [t.ptr] + [s.v for s in t.shape] + strides


def load(t: TensorArg, offsets, shape, other=0) -> Tile:
    """The block of ``t`` at ``offsets`` (one per dimension of ``t``) of the given shape; outside ``t`` reads
    ``other``. A tile is 1-D or 2-D: when ``t`` has more dimensions, the leading offsets index them and the
    tile covers the trailing ones, e.g. ``hk.load(q, (b, h, m0, 0), (BM, D))`` on a (B, H, S, D) tensor."""
    if not isinstance(t, TensorArg):
        raise _err("hk.load takes a tensor argument")
    shape = _check_shape(shape if isinstance(shape, (tuple, list)) else (shape,))
    if len(shape) > t.ndim:
        raise _err(f"{t.name} is {t.ndim}-D; a {len(shape)}-D tile does not fit it")
    offs = _offsets(t, offsets)
    return Tile(_op("load", _tensor_operands(t) + [o.v for o in offs], [TileType(shape, t.dtype)],
                    {"ndim": t.ndim, "other": other, "vec": t.vec, "name": t.name}).result)


def store(t: TensorArg, offsets, value) -> None:
    """Write ``value`` into ``t`` at ``offsets``; the parts outside ``t`` are dropped."""
    if not isinstance(t, TensorArg):
        raise _err("hk.store takes a tensor argument")
    offs = _offsets(t, offsets)
    if not isinstance(value, Tile):
        raise _err("hk.store writes a tile")
    if len(value.shape) > t.ndim:
        raise _err(f"{t.name} is {t.ndim}-D; a {len(value.shape)}-D tile does not fit it")
    value = value.to(t.dtype)
    _op("store", _tensor_operands(t) + [o.v for o in offs] + [value.v], [], {"ndim": t.ndim, "vec": t.vec,
                                                                          "name": t.name})


# ---------------------------------------------------------------- creation
def program_id(axis: int) -> Scalar:
    if axis not in (0, 1, 2):
        raise _err("program_id axis is 0, 1 or 2")
    return Scalar(_op("program_id", [], [ScalarType(T.i32)], {"axis": axis}).result)


def zeros(shape, dtype: T.DType = T.f32) -> Tile:
    return full(shape, 0, dtype)


def full(shape, value, dtype: T.DType = T.f32) -> Tile:
    shape = _check_shape(shape if isinstance(shape, (tuple, list)) else (shape,))
    return _splat(value if isinstance(value, Scalar) else (float(value) if dtype.is_float else int(value)), shape, dtype)


def arange(n: int) -> Tile:
    """0, 1, ..., n-1 as an int32 1-D tile."""
    shape = _check_shape((n,))
    return Tile(_op("arange", [], [TileType(shape, T.i32)]).result)


# ---------------------------------------------------------------- math
def dot(a: Tile, b: Tile, acc: Optional[Tile] = None) -> Tile:
    """``a @ b + acc`` on the matrix cores, accumulated in f32. a is MxK, b is KxN."""
    if not (isinstance(a, Tile) and isinstance(b, Tile)) or len(a.shape) != 2 or len(b.shape) != 2:
        raise _err("hk.dot takes two 2-D tiles")
    (m, k), (k2, n) = a.shape, b.shape
    if k != k2:
        raise _err(f"hk.dot: inner dimensions differ, {a.shape} @ {b.shape}")
    if a.dtype != b.dtype or a.dtype not in (T.bf16, T.f16):
        raise _err(f"hk.dot takes bf16 or f16 operands of one type, got {a.dtype.name} and {b.dtype.name}")
    if acc is None:
        acc = zeros((m, n), T.f32)
    if not isinstance(acc, Tile) or acc.shape != (m, n):
        raise _err(f"hk.dot: the accumulator must be a {m}x{n} tile")
    acc = acc.to(T.f32)
    return Tile(_op("dot", [a.v, b.v, acc.v], [TileType((m, n), T.f32)]).result)


def _unary(fn):
    def f(x):
        if isinstance(x, Scalar):
            return Scalar(_op("sunary", [x.to(T.f32).v if fn != "abs" else x.v],
                              [ScalarType(T.f32 if fn != "abs" else x.dtype)], {"fn": fn}).result)
        if not isinstance(x, Tile):
            raise _err(f"hk.{fn} takes a tile or a runtime scalar")
        dt = x.dtype if (fn == "abs" or x.dtype == T.f32) else T.f32
        x = x.to(dt)
        return Tile(_op("ew", [x.v], [TileType(x.shape, dt)], {"fn": fn}).result)
    f.__name__ = fn
    return f


exp, exp2, log, log2, sqrt, rsqrt, abs = (_unary(n) for n in ("exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs"))


def maximum(a, b):
    if isinstance(a, Tile) or isinstance(b, Tile):
        return _ew("max", a, b)
    if isinstance(a, Scalar) or isinstance(b, Scalar):
        return _scalar_binary("max", a, b)
    return builtins.max(a, b)


def minimum(a, b):
    if isinstance(a, Tile) or isinstance(b, Tile):
        return _ew("min", a, b)
    if isinstance(a, Scalar) or isinstance(b, Scalar):
        return _scalar_binary("min", a, b)
    return builtins.min(a, b)


def where(cond, a, b) -> Tile:
    """``a`` where ``cond`` holds, else ``b``, element-wise (with broadcasting)."""
    if not isinstance(cond, Tile):
        raise _err("hk.where takes a boolean tile as its condition")
    if cond.dtype != T.i1:
        cond = cond != 0
    if isinstance(a, Tile) or isinstance(b, Tile):
        (ta, tb), shape, dt = _tile_args(a, b)
    else:
        dts = [x.dtype for x in (a, b) if isinstance(x, Scalar)]
        dt = dts[0] if dts else (T.f32 if builtins.any(isinstance(x, float) for x in (a, b)) else T.i32)
        for d in dts[1:]:
            dt = T.promote(dt, d)
        ta, tb = (_splat(x if isinstance(x, Scalar) else (float(x) if dt.is_float else int(x)), cond.shape, dt)
                  for x in (a, b))
        shape = cond.shape
    shape = _broadcast_shapes(shape, cond.shape)
    ca, ta, tb = (_broadcast_to(t, shape) for t in (cond, ta, tb))
    return Tile(_op("where", [ca.v, ta.v, tb.v], [TileType(shape, dt)]).result)


def sum(x: Tile, axis: int) -> Tile:  # noqa: A001
    return _reduce("add", x, axis)


def max(x: Tile, axis: int) -> Tile:  # noqa: A001
    return _reduce("max", x, axis)


def min(x: Tile, axis: int) -> Tile:  # noqa: A001
    return _reduce("min", x, axis)


def print(msg: str, *values) -> None:  # noqa: A001
    """Show values when the kernel runs in the interpreter (HK2_INTERPRET=1); compiled kernels ignore it."""
    vs = [v.v for v in values if isinstance(v, (Tile, Scalar))]
    _op("print", vs, [], {"msg": str(msg)})


def cdiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------- control flow (targets of the AST rewrite)
class static_range:
    """A loop unrolled while tracing: its index is a Python int."""

    def __init__(self, *args):
        self.r = builtins.range(*args)

    def __iter__(self):
        return iter(self.r)


class DeviceRange:
    def __init__(self, *args):
        if len(args) == 1:
            args = (0, args[0], 1)
        elif len(args) == 2:
            args = (args[0], args[1], 1)
        self.lb, self.ub, self.step = args


def range_(*args):
    return DeviceRange(*args)


def get(scope: dict, name: str):
    return scope.get(name, UNDEF)


def _is_traced(x) -> bool:
    return isinstance(x, (Tile, Scalar))


def _carry_types(xs):
    return [x.v.type if _is_traced(x) else None for x in xs]


def _coerce(x, like):
    """Make a Python number or other into a value of the traced ``like``'s type (inside the current block)."""
    if isinstance(like, Tile):
        if isinstance(x, Tile):
            if x.shape != like.shape:
                raise _err(f"a loop-carried or branch variable changes shape {x.shape} -> {like.shape}")
            return x.to(like.dtype)
        return full(like.shape, x, like.dtype)
    if isinstance(like, Scalar):
        return _as_scalar(x).to(like.dtype)
    return x


def for_loop(it, body: Callable, carried: tuple, names: tuple):
    if not isinstance(it, DeviceRange):
        for x in it:
            carried = body(x, carried)
        return carried
    b = _b()
    lb, ub, step = (_as_scalar(x).to(T.i32) for x in (it.lb, it.ub, it.step))
    neg_step = isinstance(it.step, int) and it.step < 0
    # trace once in a scratch block to learn what the body carries
    scratch = ir.Block()
    b.push(scratch)
    try:
        probe = body(Scalar(Value(ScalarType(T.i32))), carried)
    finally:
        b.pop()
    carry_idx, inits = [], []
    for i, (x0, x1) in enumerate(zip(carried, probe)):
        if x0 is UNDEF:
            continue  # defined in the body: local to it
        if not (_is_traced(x0) or _is_traced(x1)):
            if x1 is not x0 and x1 != x0:
                raise _err(f"'{names[i]}' is a Python value changed inside a device loop; make it a runtime value "
                           f"(hk.full / a scalar) or unroll the loop with hk.static_range")
            continue
        if isinstance(x0, Tile) and isinstance(x1, Tile) and x0.shape != x1.shape:
            raise _err(f"loop-carried '{names[i]}' changes shape {x0.shape} -> {x1.shape}")
        like = x1 if _is_traced(x1) else x0
        init = _coerce(x0, like)
        if _is_traced(x1) and init.dtype != x1.dtype:  # e.g. an int 0 that the body makes f32
            init = init.to(x1.dtype)
        carry_idx.append(i)
        inits.append(init)
    blk = ir.Block()
    iv = Value(ScalarType(T.i32))
    params = [Value(x.v.type) for x in inits]
    blk.params = [iv] + params
    inner = list(carried)
    for j, i in enumerate(carry_idx):
        inner[i] = type(inits[j])(params[j])
    b.push(blk)
    try:
        out = body(Scalar(iv), tuple(inner))
        ys = []
        for j, i in enumerate(carry_idx):
            y = _coerce(out[i], inits[j])
            if not _is_traced(y) or y.v.type != params[j].type:
                raise _err(f"loop-carried '{names[i]}' changes type to {getattr(y, 'v', y)}")
            ys.append(y.v)
        _op("yield", ys)
    finally:
        b.pop()
    op = _op("for", [lb.v, ub.v, step.v] + [x.v for x in inits], [p.type for p in params],
             {"neg_step": neg_step}, [blk])
    res = list(carried)  # body-local names stay undefined, untouched Python values pass through
    for j, i in enumerate(carry_idx):
        res[i] = type(inits[j])(op.results[j])
    return tuple(res)


def if_(cond, then: Callable, other: Callable, carried: tuple, names: tuple):
    if not isinstance(cond, (Scalar, Tile)):
        return then(carried) if cond else other(carried)
    if isinstance(cond, Tile):
        raise _err("`if` on a tile; use hk.where for element-wise choices")
    if cond.dtype != T.i1:
        cond = cond != 0
    b = _b()
    blocks, outs = [], []
    for fn in (then, other):
        blk = ir.Block()
        b.push(blk)
        try:
            outs.append(fn(carried))
        finally:
            b.pop()
        blocks.append(blk)
    res = list(carried)
    merge_idx, merged_types = [], []
    for i, (a, c) in enumerate(zip(*outs)):
        if a is c:
            res[i] = a
            continue
        if a is UNDEF or c is UNDEF:
            res[i] = UNDEF
            continue
        if not (_is_traced(a) or _is_traced(c)):
            if a == c:
                res[i] = a
                continue
        merge_idx.append(i)
    ylists = [[], []]
    for i in merge_idx:
        a, c = outs[0][i], outs[1][i]
        ys = []
        for blk, x in zip(blocks, (a, c)):
            b.push(blk)
            try:
                if _is_traced(a) or _is_traced(c):
                    ys.append(_coerce(x, a if _is_traced(a) else c))
                else:
                    ys.append(_as_scalar(x))
            finally:
                b.pop()
        dt = T.promote(ys[0].dtype, ys[1].dtype)
        for k, blk in enumerate(blocks):
            b.push(blk)
            try:
                ylists[k].append(ys[k].to(dt))
            finally:
                b.pop()
        merged_types.append(ylists[0][-1].v.type)
    for k, blk in enumerate(blocks):
        b.push(blk)
        try:
            _op("yield", [y.v for y in ylists[k]])
        finally:
            b.pop()
    op = _op("if", [cond.v], merged_types, {}, blocks)
    for j, i in enumerate(merge_idx):
        like = ylists[0][j]
        res[i] = type(like)(op.results[j])
    return tuple(res)
