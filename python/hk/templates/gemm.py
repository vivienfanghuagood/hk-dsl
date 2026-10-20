"""hk.templates.gemm: a tuned GEMM whose store is yours.

    import hk

    def bias_gelu(acc, ctx):            # acc: the fp32 accumulator tile
        return (acc + ctx.bias).gelu()  # ctx.bias: one value per column of C

    y = hk.templates.gemm(x, w, epilogue=bias_gelu, bias=b)   # gelu(x @ w.T + b)

C = epilogue(A @ Bᵀ accumulated in fp32, ctx), A (M, K) and B (N, K) bf16 as
in hk.ops.matmul, C (M, N) in `out_dtype` (bf16 by default; fp16, fp32, int8).
The GEMM is hk.ops.matmul's, shape for shape: on gfx1100 the WMMA kernel at
`hk.ops.gemm.schedule_for`'s schedule (the tuned records included), on
gfx950 whichever of hk's kernels `hk.ops.gemm_cdna.route` picks -- the
256x256 pingpong, the four-warp blocks, the weight-streaming ones, the decode
GEMV, split-K. The epilogue is traced into the kernel where it stores C.

The epilogue is a traced function, like an `@hk.func` body: `acc` is a
register tile of C (its shape and lane layout are the kernel's, and are not
the epilogue's business), `ctx` what it can read:

    ctx.<name>   an operand passed to gemm() by that name, loaded for this
                 tile in its layout and converted to fp32. What it is follows
                 torch's broadcasting against C (M, N):
                   (N,) or (1, N)   a row vector: one value per column of C
                                    -- a bias, a per-channel scale
                   (M, 1)           a column vector: one value per row -- a
                                    per-token scale
                   (M, N)           a tile of C's shape -- a residual
                   a Python number  baked into the kernel (a constexpr): a
                                    value that changes per call is a tensor
    ctx.row, ctx.col   C's element coordinates of the tile's first element
    ctx.m              C's rows

It returns the tile to store (or None, having written `acc` in place). The
math is the language's: `+ - * /` with tiles, vectors and numbers,
`.gelu()` (tanh), `.silu()`, `.relu()`, `.exp()`, `.maximum()`, ...

Where it runs: once per element of C, after the whole sum over K. A split-K
plan stores fp32 shares and runs the epilogue in the reduction that adds
them, never on a share, so an epilogue that is not linear (a GELU, a bias)
is right whatever the route; a linear one (a scale) would gain nothing from
running earlier, since the reduction is there anyway. `plan()` says, for a
shape, which kernels run and which of them carries the epilogue.

Ready-made epilogues are attributes of this function: `gemm.bias`,
`gemm.gelu`, `gemm.bias_gelu`, `gemm.silu`, `gemm.bias_silu`, `gemm.relu`,
`gemm.bias_relu`, `gemm.residual`, `gemm.bias_residual`, `gemm.scale` (its
operand a number, a row or a column vector), and `gemm.chain(f, g, ...)`.
docs/templates/gemm.md has the plans and the measurements.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import dsl as hk
from ..types import DType, GlobalType, RegTileType, bf16, fp16, fp32, i8

# ---------------------------------------------------------------- operands

#: What an operand is, by its shape against C: see the module docstring.
ROW, COL, TILE, CONST = "row", "col", "tile", "const"

#: Rows a column-vector operand is padded to a multiple of: the tallest
#: block of M any route stores (the gfx950 pingpong's), since a vector load
#: is not bounded and a block of M past C's last row still loads its entries.
COL_PAD = 256

_RESERVED = {"a", "b", "c", "ws", "cnt", "x", "w", "epilogue", "out", "out_dtype"}


def _hk_dtype(t) -> DType:
    import torch  # noqa: PLC0415

    m = {torch.bfloat16: bf16, torch.float16: fp16, torch.float32: fp32,
         torch.int8: i8}
    if t not in m:
        raise TypeError(f"hk.templates.gemm: {t} is not a dtype hk stores or "
                        f"loads here (bf16, fp16, fp32, int8)")
    return m[t]


def _kind(name: str, v, m: int, n: int) -> str:
    import torch  # noqa: PLC0415

    if isinstance(v, bool) or not isinstance(v, (int, float, torch.Tensor)):
        raise TypeError(f"hk.templates.gemm: operand {name}={v!r}: a tensor or "
                        f"a Python number")
    if not isinstance(v, torch.Tensor):
        return CONST
    s = tuple(v.shape)
    if s in ((n,), (1, n)):
        return ROW
    if s == (m, 1):
        return COL
    if s == (m, n):
        return TILE
    raise ValueError(
        f"hk.templates.gemm: operand {name} is {s}; against C ({m}, {n}) it must "
        f"broadcast as torch would and be one of ({n},) or (1, {n}) -- one value "
        f"a column --, ({m}, 1) -- one a row --, or ({m}, {n}). A scalar is a "
        f"Python number (baked in), or a ({n},)-shaped tensor of it.")


# ---------------------------------------------------------------- the hook


class Context:
    """What an epilogue reads, for one tile of C. Operands load on first
    use, so one the epilogue does not read costs nothing."""

    def __init__(self, ep: "Epilogue", ops: Dict[str, Any], ty, r, col, c,
                 bounded):
        self._ep, self._ops, self._ty = ep, ops, ty
        self._r, self._col, self._c, self._bounded = r, col, c, bounded
        self._loaded: Dict[str, Any] = {}

    @property
    def row(self):
        """C's row of the tile's first element (a device Int)."""
        self._ep.uses_coords = True
        return self._r

    @property
    def col(self):
        """C's column of the tile's first element (a device Int)."""
        self._ep.uses_coords = True
        off = self._ops.get("EP_COL0", 0)
        return self._col + off if off else self._col

    @property
    def m(self):
        """C's rows (a device Int)."""
        return self._c.rows

    @property
    def tile(self) -> RegTileType:
        """The accumulator tile's type."""
        return self._ty

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        ep = self._ep
        kind = ep.kinds.get(name)
        if kind is None:
            raise AttributeError(
                f"the epilogue reads ctx.{name}, and gemm() was given no operand "
                f"{name!r} (it has {sorted(ep.kinds) or 'none'}). Pass it as "
                f"gemm(a, b, epilogue=..., {name}=tensor_or_number).")
        if kind == CONST:
            return ep.consts_by_name[name]
        if name not in self._loaded:
            g = self._ops["ep_" + name]
            ty = self._ty
            strip = ty.layout == "strip"
            if kind == ROW:
                v = g.at[0, self._col].load(
                    hk.row_vec(ty), bounded=self._bounded if strip else False)
            elif kind == COL:
                v = g.at[0, self._r].load(hk.col_vec(ty))
            else:
                v = g.at[self._r, self._col].load(ty, bounded=self._bounded)
            self._loaded[name] = v
        return self._loaded[name]


class Epilogue:
    """A user's epilogue bound to its operands' kinds and dtypes and C's
    dtype: the `epilogue=` the GEMM kernels in hk.ops take
    (hk.ops._epilogue). One per distinct binding; kernels key on `tag`."""

    def __init__(self, fn: Callable, kinds: Dict[str, str],
                 dtypes: Dict[str, DType], consts: Dict[str, float],
                 out_dtype: DType):
        self.fn = fn
        self.kinds = dict(kinds)
        self.dtypes = dict(dtypes)
        self.consts_by_name = dict(consts)
        self.out_dtype = out_dtype
        #: Set while tracing if the epilogue read ctx.row / ctx.col: a launch
        #: on a column slice of C then passes the slice's offset.
        self.uses_coords = False
        npar = len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        if npar not in (1, 2):
            raise TypeError(f"an epilogue is fn(acc) or fn(acc, ctx); "
                            f"{getattr(fn, '__name__', fn)} takes {npar} "
                            f"positional parameters")
        self._takes_ctx = npar == 2
        self.tensors = [n for n in sorted(kinds) if kinds[n] != CONST]
        self.params = [("ep_" + n, GlobalType(dtypes[n])) for n in self.tensors]
        self.consts = {"EP_COL0": 0}
        h = hashlib.sha1(repr((_fn_key(fn), sorted(kinds.items()),
                               sorted((k, v.name) for k, v in dtypes.items()),
                               sorted(consts.items()), out_dtype.name)).encode())
        base = re.sub(r"\W", "_", getattr(fn, "__name__", "fn"))[:24]
        tag = f"ep_{base}_{h.hexdigest()[:8]}"
        # Deterministic, so a new process finds the same C++ in the compile
        # cache; unique in this one, since the kernel caches key on it.
        i = 1
        while _TAGS.setdefault(tag + ("" if i == 1 else f"_{i}"), self) is not self:
            i += 1
        self.tag = tag + ("" if i == 1 else f"_{i}")
        self.has_tile = any(k == TILE for k in kinds.values())

    def store(self, c, ops, tile, r, col, bounded=True):
        """Traced where a kernel would store `tile` at (r, col) of C."""
        ty = tile.ty
        # A tile operand is as many registers as the accumulator: on a
        # pingpong quadrant (64x64 fp32, 64 a lane) beside the other live
        # accumulators that spills. Sixteen rows at a time it is 16.
        band = 16 if (self.has_tile and ty.layout != "strip" and ty.rows > 16
                      and ty.rows % 16 == 0 and not ty.frag) else 0
        if band:
            for i in range(ty.rows // band):
                self._one(c, ops, tile.subtile(band, i), r + band * i, col, bounded)
        else:
            self._one(c, ops, tile, r, col, bounded)

    def _one(self, c, ops, tile, r, col, bounded):
        from ..dsl.rewrite import rewrite  # noqa: PLC0415

        fn = rewrite(self.fn)
        if self._takes_ctx:
            got = fn(tile, Context(self, ops, tile.ty, r, col, c, bounded))
        else:
            got = fn(tile)
        got = tile if got is None else got
        if not isinstance(got, hk.Tile) or (got.ty.rows, got.ty.cols, got.ty.layout) != (
                tile.ty.rows, tile.ty.cols, tile.ty.layout):
            raise TypeError(
                f"the epilogue {getattr(self.fn, '__name__', self.fn)} returned "
                f"{getattr(got, 'ty', got)!r}; it returns the tile it was given "
                f"({tile.ty!r}) or one of the same shape -- C's elements, "
                f"computed elementwise")
        c[hk.elem_coord(0, 0, r, col)].store(got, bounded=bounded)


def _fn_key(fn) -> str:
    """A function's identity across processes -- its source, or its byte
    code, and what its closure holds -- not its id()."""
    if fn in _CHAINS:
        return "chain(" + ",".join(_fn_key(f) for f in _CHAINS[fn]) + ")"
    code = getattr(fn, "__code__", None)
    if code is None:
        return repr(fn)
    try:
        text = inspect.getsource(fn)
    except (OSError, TypeError):
        text = code.co_code.hex() + repr(code.co_names)
    cells = []
    for c in getattr(fn, "__closure__", None) or ():
        try:
            v = c.cell_contents
        except ValueError:
            v = None
        cells.append(_fn_key(v) if callable(v) else repr(v) if isinstance(
            v, (int, float, str, bool, type(None), tuple)) else type(v).__name__)
    return f"{fn.__module__}.{fn.__qualname__}:{text}:{cells}"


_TAGS: Dict[str, "Epilogue"] = {}


_EPILOGUES: Dict[Any, Epilogue] = {}


def bind(fn: Callable, operands: Dict[str, Any], m: int, n: int,
         out_dtype: DType) -> Epilogue:
    """The Epilogue for `fn` with these operands (tensors by kind and dtype,
    numbers by value) and C's dtype -- made once per distinct binding."""
    kinds, dtypes, consts = {}, {}, {}
    for name, v in operands.items():
        if name in _RESERVED or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"hk.templates.gemm: {name!r} cannot name an operand")
        k = kinds[name] = _kind(name, v, m, n)
        if k == CONST:
            consts[name] = float(v)
        else:
            dtypes[name] = _hk_dtype(v.dtype)
    key = (fn, tuple(sorted(kinds.items())),
           tuple(sorted((k, d.name) for k, d in dtypes.items())),
           tuple(sorted(consts.items())), out_dtype.name)
    ep = _EPILOGUES.get(key)
    if ep is None:
        ep = _EPILOGUES[key] = Epilogue(fn, kinds, dtypes, consts, out_dtype)
    return ep


def _tensors(ep: Epilogue, operands: Dict[str, Any], m: int):
    """The operand tensors in the order the kernels take them, laid out as
    the loads read them: a row vector (1, N), a column vector (1, M) padded
    to whole blocks, a tile as it is (any row stride)."""
    import torch  # noqa: PLC0415

    out = []
    for name in ep.tensors:
        v = operands[name]
        k = ep.kinds[name]
        if k == ROW:
            v = v.reshape(1, -1)
        elif k == COL:
            v = v.reshape(1, -1)
            pad = -m % COL_PAD
            if pad:
                v = torch.nn.functional.pad(v, (0, pad))
        if v.stride(-1) != 1:
            v = v.contiguous()
        out.append(v)
    return out


def _cols(ep: Epilogue, ts, lo: int, hi: int):
    """The operands of a launch on columns [lo, hi) of C."""
    out = []
    for name, t in zip(ep.tensors, ts):
        out.append(t if ep.kinds[name] == COL else t[:, lo:hi])
    return out


# ---------------------------------------------------------------- plans


@dataclass
class Step:
    """One launch of a plan.

    kernel    "rdna" (hk.ops.gemm), "pp" (gemm_cdna's pingpong), "dp"
              (gemm_cdna_dp's four-warp blocks), "decode" (decode_gemm),
              "reduce" (gemm_cdna_dp's strip reduction of split-K shares)
    sched     the kernel's schedule, as the module that owns it spells it
    cols      the columns [lo, hi) of C it covers
    epilogue  whether the epilogue runs in it
    into      "out" (C) or "scratch" (fp32 split-K shares)
    route     the candidate name the route table gave (gfx950)
    """

    kernel: str
    sched: Dict[str, Any]
    cols: Tuple[int, int]
    epilogue: bool
    into: str = "out"
    route: str = ""

    def __str__(self) -> str:
        e = " +epilogue" if self.epilogue else ""
        r = f" [{self.route}]" if self.route else ""
        return f"{self.kernel}{r} cols {self.cols[0]}:{self.cols[1]} -> {self.into}{e}"


def _check(m: int, n: int, k: int, arch: str) -> None:
    if m < 1 or k < 1:
        raise ValueError(f"hk.templates.gemm: M={m}, K={k}; nothing to multiply")
    if arch == "gfx950":
        from ..ops import gemm_cdna  # noqa: PLC0415

        if k % gemm_cdna.K_STEP or n % gemm_cdna.BLOCK_N_MIN:
            raise ValueError(f"hk.templates.gemm: K={k} must be a multiple of "
                             f"{gemm_cdna.K_STEP} and N={n} of "
                             f"{gemm_cdna.BLOCK_N_MIN} on gfx950; M may be anything")


def _steps_cdna(name: str, m: int, n: int, k: int, lo: int) -> List[Step]:
    from ..ops import decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

    kind, sched = gemm_cdna.CANDIDATES[name]
    cols = (lo, lo + n)
    if kind == "decode":
        return [Step("decode", decode_gemm.schedule_for(m, k), cols, True, route=name)]
    if kind == "ppw":
        cut = gemm_cdna.wave_cut(m, n)
        return (_steps_cdna("pp", m, cut, k, lo)
                + _steps_cdna(sched["rem"], m, n - cut, k, lo + cut))
    if kind == "dp":
        s = {**gemm_cdna_dp.DEFAULT, **sched}
    else:
        s = {**gemm_cdna.schedule_for(m, n, k), **sched}
        wide = n - n % 256
        if s.get("splits", 1) == 1 and n % 256 and wide:
            s = {**gemm_cdna.schedule_for(m, wide, k), **sched}
            return ([Step("pp", s, (lo, lo + wide), True, route=name)]
                    + _steps_cdna(gemm_cdna.tail_for(m, k), m, n - wide, k, lo + wide))
    if s.get("splits", 1) > 1:
        return [Step(kind, s, cols, False, "scratch", route=name),
                Step("reduce", {"splits": s["splits"]}, cols, True, route=name)]
    return [Step(kind, s, cols, True, route=name)]


def plan(m: int, n: int, k: int, arch: Optional[str] = None) -> List[Step]:
    """The launches gemm() makes for an M x N x K problem, in order, and the
    one(s) that carry the epilogue: hk.ops.matmul's kernels, with the store
    of every launch that writes C replaced, and split-K shares reduced by a
    launch that runs the epilogue on the sum. Needs no GPU."""
    from ..target import current_arch  # noqa: PLC0415

    arch = arch or current_arch()
    _check(m, n, k, arch)
    if arch == "gfx950":
        from ..ops import gemm_cdna  # noqa: PLC0415

        return _steps_cdna(gemm_cdna.route(m, n, k), m, n, k, 0)
    from ..ops import gemm as rdna  # noqa: PLC0415

    sched = rdna.schedule_for(m, n, k)
    if k % sched["k_step"] or n % sched["block_n"]:
        raise ValueError(f"hk.templates.gemm: K={k} must be a multiple of "
                         f"{sched['k_step']} and N={n} of {sched['block_n']} on "
                         f"{arch}; M may be anything")
    return [Step("rdna", dict(sched), (0, n), True)]


# ---------------------------------------------------------------- kernels

_KERNELS: Dict[Tuple, Any] = {}


def kernel_for(step: Step, ep: Optional[Epilogue], arch: Optional[str] = None):
    """The kernel a step launches, with the epilogue if the step carries it.
    Traced and built on first use; `kernel_for(...).source()` is its C++."""
    from ..ops import _epilogue, decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415
    from ..ops import gemm as rdna  # noqa: PLC0415

    e = ep if step.epilogue else None
    key = (step.kernel, tuple(sorted(step.sched.items())), e and e.tag, arch)
    got = _KERNELS.get(key)
    if got is not None:
        return got
    s = step.sched
    if step.kernel == "rdna":
        s = {**rdna.DEFAULT, **s}
        k = rdna.gemm_kernel(_epilogue.name(rdna._sched_name(s), e), **s, epilogue=e)
        if arch:
            k._arch = arch
    elif step.kernel == "pp":
        s = {**gemm_cdna.DEFAULT, **s}
        k = gemm_cdna.gemm_kernel(_epilogue.name(gemm_cdna._sched_name(s), e), **s,
                                  epilogue=e, arch=arch)
    elif step.kernel == "dp":
        s = {**gemm_cdna_dp.DEFAULT, **s}
        k = gemm_cdna_dp.gemm_kernel(_epilogue.name(gemm_cdna_dp._sched_name(s), e),
                                     **s, epilogue=e, arch=arch)
    elif step.kernel == "decode":
        s = dict(s)
        nm = (f"decode_gemm_t{s['tokens']}_s{s['slices']}_w{s['warps']}"
              f"_c{s['combine_depth']}_r{s['resident']}"
              + (f"_k{s['splits']}" if s["splits"] > 1 else ""))
        k = decode_gemm.decode_gemm_kernel(_epilogue.name(nm, e), **s, epilogue=e,
                                           arch=arch)
    else:
        raise ValueError(f"no kernel for a {step.kernel} step; the reduction is "
                         f"gemm_cdna_dp.reduce")
    _KERNELS[key] = k
    return k


def _col0(k, ep: Epilogue, lo: int, **consts) -> Dict[str, int]:
    """EP_COL0 for a launch on columns lo.. of C, if the epilogue reads its
    coordinates (a trace, cached, says whether it does)."""
    if not lo:
        return {}
    k.trace(**consts)
    return {"EP_COL0": lo} if ep.uses_coords else {}


def run(steps: List[Step], ep: Epilogue, a, b, out, operands) -> Any:
    """Launch a plan."""
    from ..ops import decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

    m = a.shape[0]
    ws = None
    for st in steps:
        lo, hi = st.cols
        whole = (lo, hi) == (0, out.shape[1])
        bb = b if whole else b[lo:hi]
        oo = out if whole else out[:, lo:hi]
        ops = operands if whole else _cols(ep, operands, lo, hi)
        if st.kernel == "reduce":
            gemm_cdna_dp.reduce(ws, oo, epilogue=ep, operands=ops,
                                **({"EP_COL0": lo} if lo and ep.uses_coords else {}))
            continue
        if st.into == "scratch":
            ws = gemm_cdna_dp.scratch_for(a.device, st.sched["splits"], m, hi - lo)
            mod = gemm_cdna if st.kernel == "pp" else gemm_cdna_dp
            mod.kernel_for(st.sched)(a, bb, ws)
            continue
        k = kernel_for(st, ep)
        if st.kernel == "decode":
            s = st.sched
            extra = (decode_gemm.scratch_for(a.device, s["splits"], s["tokens"], hi - lo)
                     if s["splits"] > 1 else ())
            k(a, bb, oo, *extra, *ops, **_col0(k, ep, lo))
        else:
            k(a, bb, oo, *ops, **_col0(k, ep, lo))
    return out


# ---------------------------------------------------------------- the API


def _out_dtype(out_dtype):
    import torch  # noqa: PLC0415

    if out_dtype is None:
        return torch.bfloat16
    if isinstance(out_dtype, DType):
        rev = {bf16: torch.bfloat16, fp16: torch.float16, fp32: torch.float32,
               i8: torch.int8}
        if out_dtype not in rev:
            raise TypeError(f"hk.templates.gemm: out_dtype {out_dtype}; bf16, "
                            f"fp16, fp32 or int8")
        return rev[out_dtype]
    _hk_dtype(out_dtype)
    return out_dtype


def _identity(acc):
    return acc


def _gemm(a, b, epilogue=None, out=None, out_dtype=None, **operands):
    import torch  # noqa: PLC0415

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"hk.templates.gemm: want A (M, K) and B (N, K), got "
                         f"{tuple(a.shape)} and {tuple(b.shape)}")
    m, k = a.shape
    n, kb = b.shape
    if kb != k:
        raise ValueError(f"hk.templates.gemm: A is (M={m}, K={k}) and B is "
                         f"(N={n}, K={kb}); B is passed (N, K), as to F.linear")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"hk.templates.gemm: A and B are bf16, got {a.dtype} "
                        f"and {b.dtype}")
    tdt = out.dtype if out is not None else _out_dtype(out_dtype)
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=tdt)
    elif tuple(out.shape) != (m, n) or out.stride(1) != 1:
        raise ValueError(f"hk.templates.gemm: out is {tuple(out.shape)}, "
                         f"expected ({m}, {n}) with unit column stride")
    if epilogue is None and not operands and tdt == torch.bfloat16:
        from ..ops import gemm as rdna  # noqa: PLC0415

        return rdna.matmul(a, b, out=out)
    fn = epilogue or _identity
    ep = bind(fn, operands, m, n, _hk_dtype(tdt))
    if not a.is_contiguous():
        a = a.contiguous()
    steps = plan(m, n, k)
    return run(steps, ep, a, b, out, _tensors(ep, operands, m))


# ---------------------------------------------------------------- ready-made


def bias(acc, ctx):
    """acc + bias, bias one value per column of C: F.linear's."""
    return acc + ctx.bias


def gelu(acc):
    """GELU, tanh approximation (F.gelu(..., approximate="tanh"))."""
    return acc.gelu()


def silu(acc):
    """SiLU, x * sigmoid(x)."""
    return acc.silu()


def relu(acc):
    return acc.relu()


def bias_gelu(acc, ctx):
    """gelu(acc + bias), tanh approximation."""
    return (acc + ctx.bias).gelu()


def bias_silu(acc, ctx):
    return (acc + ctx.bias).silu()


def bias_relu(acc, ctx):
    return (acc + ctx.bias).relu()


def residual(acc, ctx):
    """acc + residual, residual (M, N): an o_proj or down_proj's skip."""
    return acc + ctx.residual


def bias_residual(acc, ctx):
    return acc + ctx.bias + ctx.residual


def scale(acc, ctx):
    """acc * scale: a number, a per-channel (N,) or a per-row (M, 1) vector.
    With out_dtype=torch.int8 this is a quantizer: the store rounds to
    nearest even and saturates."""
    return acc * ctx.scale


def chain(*fns: Callable) -> Callable:
    """One epilogue that applies `fns` in order, each to what the last
    returned, all reading the same ctx."""
    takes = [len([p for p in inspect.signature(f).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) == 2
             for f in fns]

    def chained(acc, ctx):
        for f, t in zip(fns, takes):
            got = f(acc, ctx) if t else f(acc)
            acc = acc if got is None else got
        return acc

    chained.__name__ = "_".join(getattr(f, "__name__", "fn") for f in fns)
    chained.__qualname__ = "chain(" + ", ".join(getattr(f, "__qualname__", "?")
                                                for f in fns) + ")"
    _CHAINS[chained] = fns
    return chained


_CHAINS: Dict[Callable, Tuple] = {}

READY = {f.__name__: f for f in (bias, gelu, silu, relu, bias_gelu, bias_silu,
                                 bias_relu, residual, bias_residual, scale)}


class _Gemm:
    """hk.templates.gemm(a, b, epilogue=fn, out=None, out_dtype=None,
    **operands) -> C; with no a and b, the same bound to its keywords, to
    call later. See the module docstring."""

    plan = staticmethod(plan)
    kernel_for = staticmethod(kernel_for)
    chain = staticmethod(chain)
    Context = Context
    Step = Step

    def __init__(self):
        for name, f in READY.items():
            setattr(self, name, f)

    def __call__(self, a=None, b=None, *, epilogue=None, out=None, out_dtype=None,
                 **operands):
        if a is None and b is None:
            def bound(a, b, *, out=out, **more):
                return _gemm(a, b, epilogue=epilogue, out=out, out_dtype=out_dtype,
                             **{**operands, **more})

            bound.__name__ = f"gemm[{getattr(epilogue, '__name__', 'plain')}]"
            return bound
        if a is None or b is None:
            raise TypeError("hk.templates.gemm(a, b, ...) takes both A and B, or "
                            "neither (to bind the epilogue for later)")
        return _gemm(a, b, epilogue=epilogue, out=out, out_dtype=out_dtype,
                     **operands)

    def __repr__(self) -> str:
        return "<hk.templates.gemm>"


gemm = _Gemm()
gemm.__doc__ = __doc__


class _CallableModule(types.ModuleType):
    """`import hk.templates.gemm` binds the package's attribute to this
    module, so the module calls through too, and has the same attributes."""

    def __call__(self, *args, **kwargs):
        return gemm(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["gemm", "plan", "kernel_for", "run", "bind", "Step", "Context",
           "Epilogue", "chain", "READY", *READY]
