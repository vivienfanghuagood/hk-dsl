"""What the elementwise and rowwise templates share: the operand kinds, the
way a torch tensor becomes the (rows, cols) global a kernel reads, and the
pieces of a generated kernel's signature.

Nothing here imports torch at module level: the templates trace, translate
and build on a machine without it (tests/hk/ir, tests/hk/codegen).
"""

from __future__ import annotations

import contextlib
import inspect
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from ..types import DTYPES, DType, GlobalType, bf16, fp16, fp32, i8, i32

#: The operand kinds. A template sees every operand as a tensor of the
#: problem's (rows, cols), or broadcast to it:
#:   full     the whole (rows, cols): x, the gate, the residual
#:   per_col  one value per column, shape (cols,): a weight, a bias
#:   per_row  one value per row, shape (rows,) or (rows, 1): a token's scale
#:   scalar   one value, shape () or (1,): a tensor-wide scale
KINDS = ("full", "per_col", "per_row", "scalar")


@dataclass(frozen=True)
class Operand:
    """An operand's kind and element type, as a template is told it:
    `hk.bf16` alone is a full operand, `hk.templates.per_col(hk.bf16)` a
    weight. `dtype` None takes the tensor's at call time."""

    kind: str
    dtype: Optional[DType] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"operand kind {self.kind!r}; one of {KINDS}")

    def __repr__(self) -> str:
        d = self.dtype.name if self.dtype else "any"
        return d if self.kind == "full" else f"{self.kind}({d})"


def full(dtype: Optional[DType] = None) -> Operand:
    """An operand of the problem's whole (rows, cols)."""
    return Operand("full", _dtype(dtype))


def per_col(dtype: Optional[DType] = None) -> Operand:
    """One value per column, a (cols,) tensor: a weight or a bias, the same
    for every row (torch's broadcasting of a 1-D tensor)."""
    return Operand("per_col", _dtype(dtype))


def per_row(dtype: Optional[DType] = None) -> Operand:
    """One value per row, a (rows,) or (rows, 1) tensor: a token's scale, a
    row's label. In a rowwise output, a per-row result (a scale, a loss)."""
    return Operand("per_row", _dtype(dtype))


def scalar(dtype: Optional[DType] = None) -> Operand:
    """One value for the whole tensor, a () or (1,) tensor on the device."""
    return Operand("scalar", _dtype(dtype))


def _dtype(d) -> Optional[DType]:
    if d is None or isinstance(d, DType):
        return d
    got = DTYPES.get(str(d)) or _TORCH_TO_HK.get(str(d))
    if got is None:
        raise TypeError(f"{d!r} is not a dtype a template takes: "
                        f"{sorted(n for n in DTYPES if not n.startswith('fp8'))}")
    return got


def as_operand(spec) -> Operand:
    """An Operand from what a user wrote: an Operand, or a bare dtype (a
    full operand), or None (a full operand of the tensor's dtype)."""
    if isinstance(spec, Operand):
        return spec
    return Operand("full", _dtype(spec))


# ---------------------------------------------------------------- torch

_TORCH_TO_HK = {"torch.bfloat16": bf16, "torch.float16": fp16, "torch.float32": fp32,
                "torch.int8": i8, "torch.int32": i32}
_HK_TO_TORCH = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32",
                "i8": "int8", "i32": "int32"}


def hk_dtype(t) -> DType:
    """The hk dtype of a torch tensor (or of anything with a .dtype that is
    a torch dtype or an hk one)."""
    d = t.dtype
    if isinstance(d, DType):
        return d
    got = _TORCH_TO_HK.get(str(d))
    if got is None:
        raise TypeError(f"templates take bf16, fp16, fp32, int8 and int32 tensors, "
                        f"not {d}")
    return got


def torch_dtype(d: DType):
    import torch  # noqa: PLC0415

    return getattr(torch, _HK_TO_TORCH[d.name])


class Meta:
    """A tensor's shape, strides and dtype, without a tensor: what `plan()`,
    `kernel()` and the no-GPU tests hand a template in place of one.

        hk.templates.meta((4096, 14336), hk.bf16, stride=(28672, 1))
    """

    __slots__ = ("shape", "_stride", "dtype")

    def __init__(self, shape, dtype, stride=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = _dtype(dtype)
        if stride is None:
            st, acc = [], 1
            for s in reversed(self.shape):
                st.append(acc)
                acc *= max(s, 1)
            stride = tuple(reversed(st))
        self._stride = tuple(int(s) for s in stride)

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def __repr__(self) -> str:
        return f"meta({self.shape}, {self.dtype}, stride={self._stride})"


def meta(shape, dtype, stride=None) -> Meta:
    """A stand-in for a tensor of this shape, dtype and strides (contiguous
    if not given), for planning and building without a GPU."""
    return Meta(shape, dtype, stride)


def numel(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def rows_cols(shape) -> Tuple[int, int]:
    """(rows, cols) of a tensor of `shape`: the last axis, and the rest."""
    if not shape:
        return 1, 1
    cols = shape[-1]
    return (numel(shape[:-1]) if len(shape) > 1 else 1), cols


def row_stride(shape, stride, what: str) -> int:
    """The one row stride of a tensor read as (rows, cols): the last axis
    contiguous, the leading axes compact over one stride -- a slice of the
    last axis (one half of a fused gate/up output) qualifies. Raises,
    saying why, for anything else."""
    if not shape:
        return 1
    if shape[-1] > 1 and stride[-1] != 1:
        raise ValueError(f"{what}: the last axis must be contiguous (stride 1), "
                         f"got strides {tuple(stride)}; call .contiguous()")
    if len(shape) == 1:
        return shape[-1]
    rs = None
    inner = None
    for size, st in zip(reversed(shape[:-1]), reversed(stride[:-1])):
        if size == 1:
            continue
        if rs is None:
            rs, inner = st, st * size
        elif st != inner:
            raise ValueError(f"{what}: the leading axes of shape {tuple(shape)} "
                             f"(strides {tuple(stride)}) are not one row stride "
                             f"apart, so it is not a (rows, cols) view; call "
                             f".contiguous()")
        else:
            inner = st * size
    return rs if rs is not None else shape[-1]


def is_contiguous(shape, stride) -> bool:
    acc = 1
    for size, st in zip(reversed(shape), reversed(stride)):
        if size != 1 and st != acc:
            return False
        acc *= size
    return True


def classify(shape, out_shape) -> str:
    """The kind of an operand of `shape` in a problem of `out_shape`, by
    torch's broadcasting: the same shape is full; (cols,) or (1, .., cols)
    per_col; (rows,), (..., 1) per_row; () or one element scalar."""
    shape, out_shape = tuple(shape), tuple(out_shape)
    if shape == out_shape:
        return "full"
    n = numel(shape)
    if n == 1:
        return "scalar"
    rows, cols = rows_cols(out_shape)
    if shape[-1:] == out_shape[-1:] and n == cols:
        return "per_col"
    if (shape[-1:] == (1,) and n == rows) or (len(shape) == len(out_shape) - 1
                                              and shape == out_shape[:-1]):
        return "per_row"
    raise ValueError(f"an operand of shape {shape} does not broadcast to {out_shape} "
                     f"as a template takes it: the same shape, one value per column "
                     f"({out_shape[-1:]}), one per row ({out_shape[:-1]} or "
                     f"{(*out_shape[:-1], 1)}), or one value")


# ---------------------------------------------------------------- kernels

#: Constexpr names the templates use for their own knobs; a hook's
#: keyword-only parameters may not take them.
RESERVED_PREFIX = "T_"


def hook_consts(fn) -> Dict[str, Any]:
    """The hook's constexprs: its parameters with a default (EPS=1e-6),
    which a call may override (`k(x, w, EPS=1e-5)`)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    out = {}
    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty:
            continue
        if p.name.startswith(RESERVED_PREFIX):
            raise TypeError(f"{getattr(fn, '__name__', fn)}: a parameter named "
                            f"{p.name!r}; names starting {RESERVED_PREFIX!r} are the "
                            f"template's own")
        if isinstance(p.default, bool) or not isinstance(p.default, (int, float)):
            raise TypeError(f"{getattr(fn, '__name__', fn)}: parameter {p.name!r} "
                            f"defaults to {p.default!r}; a hook's defaulted "
                            f"parameters are constexprs, ints or floats")
        out[p.name] = p.default
    return out


def hook_arity(fn) -> Optional[int]:
    """How many positional operands the hook takes, or None if it takes *args."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD) \
                and p.default is inspect.Parameter.empty:
            n += 1
    return n


def kernel_name(prefix: str, fn) -> str:
    base = re.sub(r"[^0-9A-Za-z_]", "", getattr(fn, "__name__", "fn")) or "fn"
    if base == "lambda":
        base = "fn"
    return f"{prefix}_{base}"[:48]


def signature(tensors: Sequence[Tuple[str, DType]], consts: Dict[str, Any]):
    """The inspect.Signature of a generated kernel body: tensors positional,
    annotated hk.Tensor[dtype]; constexprs keyword-only with defaults."""
    P = inspect.Parameter
    params = [P(n, P.POSITIONAL_OR_KEYWORD, annotation=GlobalType(d)) for n, d in tensors]
    params += [P(k, P.KEYWORD_ONLY, default=v) for k, v in consts.items()]
    return inspect.Signature(params)


def call_key(args, out, consts):
    """What a template's plan depends on in a call: each tensor's shape,
    strides and dtype, and the constexprs. One tuple, built as cheaply as
    Python builds one -- this is in front of every launch."""
    k = [(a.shape, a.stride(), a.dtype) for a in args]
    if out is not None:
        k.append(None)
        k += [(o.shape, o.stride(), o.dtype) for o in
              (out if isinstance(out, (list, tuple)) else (out,))]
    if consts:
        k.append(tuple(consts.items()))
    return tuple(k)


def view_fn(t_shape, rows: int, cols: int, kind: str, flat: bool = False):
    """The function that turns an operand into the 2-D global the kernel
    reads, or None where the tensor already is it."""
    n = rows * cols
    if kind == "full":
        if flat:
            return None if tuple(t_shape) == (1, n) else (lambda t: t.view(1, n))
        if len(t_shape) == 2:
            return None
        return lambda t: t.view(rows, cols)
    if kind == "per_col":
        return lambda t: t.view(1, cols)
    if kind == "per_row":
        return lambda t: t.view(rows, 1)
    if kind == "per_row_out":
        return lambda t: t.view(1, rows)
    return lambda t: t.view(1, 1)


def make_call(launch, args_meta, out_shapes, out_tdtypes, arg_views, out_views,
              out_meta, single: bool):
    """The per-call closure of a prepared template call: allocate the outputs
    (or check the given ones against the inputs for a partial overlap, which
    reads an input the call has already written), view, launch.

    Everything that depends only on shapes and strides -- the views, the
    byte spans, whether an exact alias has the same strides -- is decided
    here, once; the closure does pointer arithmetic and the launch."""
    import torch  # noqa: PLC0415

    spans = [_span_of(s, st) * es for s, st, es in args_meta]
    ospans = None if out_meta is None else [_span_of(s, st) * es for s, st, es in out_meta]
    same_strides = None if out_meta is None else [
        [st == ost for _, st, _ in args_meta] for _, ost, _ in out_meta]
    nargs = len(args_meta)
    empty = torch.empty

    def views(args, outs):
        a = [t if f is None else f(t) for t, f in zip(args, arg_views)]
        a += [t if f is None else f(t) for t, f in zip(outs, out_views)]
        return a

    def call(args, out):
        if out is None:
            dev = args[0].device
            outs = [empty(s, dtype=d, device=dev) for s, d in zip(out_shapes, out_tdtypes)]
        else:
            outs = list(out) if isinstance(out, (list, tuple)) else [out]
            pas = [a.data_ptr() for a in args]
            for j, o in enumerate(outs):
                po = o.data_ptr()
                eo = po + ospans[j]
                for i in range(nargs):
                    pa = pas[i]
                    if pa < eo and po < pa + spans[i] and not (pa == po and same_strides[j][i]):
                        # Each element is read and written by one lane, so
                        # exactly in place is safe; a partial overlap is not.
                        args = tuple(x.clone() for x in args)
                        pas = [a.data_ptr() for a in args]
                        break
        launch(*views(args, outs))
        return outs[0] if single else tuple(outs)

    return call


def _span_of(shape, stride) -> int:
    if not numel(shape):
        return 0
    return 1 + sum((s - 1) * st for s, st in zip(shape, stride))


_ALLOW_REDUCE = [0]


@contextlib.contextmanager
def tile_reductions_refused(message: str):
    """While a hook is traced, `t.sum(axis=1)` and friends raise `message`:
    on a strip they reduce one warp's share of a row, which inside a
    template is a silently wrong answer. The template's own reductions run
    under `tile_reductions_allowed()`."""
    from ..dsl import values as V  # noqa: PLC0415

    orig = V.Tile._reduce
    div, idiv = V.Tile.__truediv__, V.Tile.__itruediv__

    def refuse(self, kind, axis):
        if _ALLOW_REDUCE[0]:
            return orig(self, kind, axis)
        raise TypeError(message.format(kind=kind))

    # A tile divided by a row value (`e / row_sum(e)`) is multiplied by its
    # reciprocal: one division a row, not one an element (v_div_f32 is a
    # dozen instructions). Within 1 ulp of fp32, as hk.ops.softmax computes.
    def recip(v):
        return V.binary("div", V.full(v.ty, 1.0), v)

    def tdiv(self, o):
        o = V._vec_of(o)
        return V.binary("mul", self, recip(o)) if isinstance(o, V.Vec) else div(self, o)

    def tidiv(self, o):
        o = V._vec_of(o)
        if isinstance(o, V.Vec):
            V.binary("mul", self, recip(o), into=self)
            return self
        return idiv(self, o)

    own = {k: V.Tile.__dict__[k] for k in ("_reduce", "__truediv__", "__itruediv__")
           if k in V.Tile.__dict__}
    V.Tile._reduce = refuse
    V.Tile.__truediv__, V.Tile.__itruediv__ = tdiv, tidiv
    try:
        yield
    finally:
        for k in ("_reduce", "__truediv__", "__itruediv__"):
            if k in own:
                setattr(V.Tile, k, own[k])
            else:
                delattr(V.Tile, k)


@contextlib.contextmanager
def tile_reductions_allowed():
    _ALLOW_REDUCE[0] += 1
    try:
        yield
    finally:
        _ALLOW_REDUCE[0] -= 1


def overlaps(a, b) -> bool:
    """Do two tensors' storage spans share any bytes?"""
    pa, pb = a.data_ptr(), b.data_ptr()
    ea = pa + _span(a) * a.element_size()
    eb = pb + _span(b) * b.element_size()
    return pa < eb and pb < ea


def _span(t) -> int:
    if not t.numel():
        return 0
    return 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))


__all__ = ["Operand", "full", "per_col", "per_row", "scalar", "meta", "Meta", "KINDS"]
