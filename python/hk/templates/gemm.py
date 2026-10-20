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
from ..runtime import ResourceError
from ..types import (DType, GlobalType, RegTileType, bf16, fp8e4m3, fp8e5m2, fp16,
                     fp32, i8)

# ---------------------------------------------------------------- operands

#: What an operand is, by its shape against C: see the module docstring.
ROW, COL, TILE, CONST = "row", "col", "tile", "const"

#: Rows a column-vector operand is padded to a multiple of: the tallest
#: block of M any route stores (the gfx950 pingpong's), since a vector load
#: is not bounded and a block of M past C's last row still loads its entries.
COL_PAD = 256

#: On gfx1100, batches of at most this many rows go to hk.ops.decode_gemm
#: (with the epilogue) rather than the WMMA kernel's 16-row schedule, which
#: is what hk.ops.matmul runs there -- and what hk's vLLM routes do. On a
#: W7900D (rocm 7.2.4) it is 1.6-2.7x the WMMA kernel at M = 1-32 on
#: Qwen3-8B's layers (w3 probe, docs/templates/gemm.md); at 33-64 rows its
#: four token tiles leave no registers for an epilogue.
RDNA_DECODE_MAX_M = 32

#: Rows of an accumulator an epilogue with a tile operand read at the store
#: runs on at a time, the next band's operand loaded under this one's math:
#: 32 measured best on an MI355X (16 waits more often, the whole 64-row
#: quadrant spills beside the pingpong's other accumulators).
BAND = 32

#: Load a 16-bit tile operand as a 16-bit tile, converted where it is used:
#: half the registers while it is in flight.
NATIVE_TILE = True


#: Schedule fields merged into every step's (a knob for measuring).
OVERRIDE: Dict[str, Any] = {}

_RESERVED = {"a", "b", "c", "ws", "cnt", "x", "w", "epilogue", "out", "out_dtype"}


def _hk_dtype(t) -> DType:
    import torch  # noqa: PLC0415

    m = {torch.bfloat16: bf16, torch.float16: fp16, torch.float32: fp32,
         torch.int8: i8, torch.float8_e4m3fn: fp8e4m3, torch.float8_e5m2: fp8e5m2}
    if t not in m:
        raise TypeError(f"hk.templates.gemm: {t} is not a dtype hk stores or "
                        f"loads here (bf16, fp16, fp32, int8, float8_e4m3fn, "
                        f"float8_e5m2)")
    return m[t]


def _kind(name: str, v, m: int, n: int) -> str:
    import torch  # noqa: PLC0415

    if isinstance(v, bool) or not isinstance(v, (int, float, torch.Tensor)):
        raise TypeError(f"hk.templates.gemm: operand {name}={v!r}: a tensor or "
                        f"a Python number")
    if not isinstance(v, torch.Tensor):
        return CONST
    s = tuple(v.shape)
    if s == (m, n):
        return TILE                 # (1, N) at M = 1 is C's shape: a tile
    if s in ((n,), (1, n)):
        return ROW
    if s == (m, 1):
        return COL
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
        self._raw: Dict[str, Any] = {}

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

    def _prefetch(self, names):
        """Load these operands now, at the point the context is made (the
        epilogue reads them later): a 16-bit tile as a 16-bit tile
        (NATIVE_TILE), half the registers while it waits, converted where
        the epilogue reads it.

        Only what is loaded here is kept: a value the epilogue loads itself
        may be made inside a branch hk traces twice (the first time into a
        scratch region it then discards), so holding on to it would hand
        the second trace a value that no longer exists. A second ctx.x is a
        second load, which the compiler folds."""
        for name in names:
            ty, dt = self._ty, self._ep.dtypes[name]
            if (self._ep.kinds[name] == TILE and NATIVE_TILE
                    and ty.layout != "strip" and dt.bits == 16):
                nt = RegTileType(dt, ty.rows, ty.cols, ty.layout, ty.frag)
                self._raw[name] = self._ops["ep_" + name].at[self._r, self._col].load(
                    nt, bounded=self._bounded)
            else:
                self._loaded[name] = self._load(name)

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
        if name in self._loaded:
            return self._loaded[name]
        return self._load(name)

    def _load(self, name: str):
        kind = self._ep.kinds[name]
        g = self._ops["ep_" + name]
        ty = self._ty
        strip = ty.layout == "strip"
        if kind == ROW:
            return g.at[0, self._col].load(
                hk.row_vec(ty), bounded=self._bounded if strip else False)
        if kind == COL:
            # (M, 1) with a row stride of 1: the vector's entries are
            # consecutive from row r.
            return g.at[self._r, 0].load(hk.col_vec(ty))
        if name in self._raw:
            return self._raw[name].to(ty.dtype)
        return g.at[self._r, self._col].load(ty, bounded=self._bounded)


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
        #: The tile operand the accumulator starts as (init_from), or None.
        self.init = _INITS[fn][0] if fn in _INITS else None
        if self.init is not None and kinds.get(self.init) != TILE:
            raise ValueError(
                f"the epilogue starts the accumulator as ctx.{self.init} "
                f"(init_from), so gemm() needs {self.init}= an (M, N) tensor; "
                f"it has {self.init} as {kinds.get(self.init, 'nothing')}")
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

    def load_init(self, ops, tile, r, col):
        """Traced where a kernel zeroes accumulator `tile` of C's (r, col):
        load it from the init_from operand instead (rows past C's read 0).
        A kernel that calls this passes inited=True to store()."""
        ops["ep_" + self.init].at[r, col].load(tile.ty, bounded=True, into=tile)

    def early(self, c, ops, ty, r, col, bounded=True) -> "Context":
        """Load every operand for the tile of type `ty` at C's (r, col) now
        (a tile operand as 16-bit if it is), for a store() later that passes
        it as ctx= -- a kernel with a long loop and a small tile (the decode
        GEMV) hides their latency under the loop."""
        ctx = Context(self, ops, ty, r, col, c, bounded)
        ctx._prefetch(self.tensors)
        return ctx

    def store(self, c, ops, tile, r, col, bounded=True, inited=False, ctx=None):
        """Traced where a kernel would store `tile` at (r, col) of C. With
        an init_from operand the kernel did not start the accumulator as
        (inited=False), it is added here first. `ctx`: operands loaded
        earlier (early())."""
        ty = tile.ty
        if ctx is not None:
            self._one(c, ops, tile, r, col, bounded, ctx, inited)
            return
        tiles = [n for n in self.tensors if self.kinds[n] == TILE
                 and (n != self.init or not inited)]
        # A tile operand is as many registers as the accumulator: on a
        # pingpong quadrant (64x64 fp32, 64 a lane) beside the other live
        # accumulators that spills. A band of rows at a time it is a part.
        band = BAND if (tiles and ty.layout != "strip" and ty.rows > BAND
                        and ty.rows % BAND == 0 and not ty.frag) else 0
        if not band:
            self._one(c, ops, tile, r, col, bounded,
                      Context(self, ops, ty, r, col, c, bounded), inited)
            return
        # Band i+1's tile operands are loaded before band i is computed and
        # stored, so their latency hides under it rather than each band
        # waiting out its own: 16 more registers, not the whole tile's.
        nb = ty.rows // band
        subs = [tile.subtile(band, i) for i in range(nb)]
        ctxs = [Context(self, ops, subs[i].ty, r + band * i, col, c, bounded)
                for i in range(nb)]
        ctxs[0]._prefetch(tiles)
        for i in range(nb):
            if i + 1 < nb:
                ctxs[i + 1]._prefetch(tiles)
            self._one(c, ops, subs[i], r + band * i, col, bounded, ctxs[i], inited)

    def _one(self, c, ops, tile, r, col, bounded, ctx, inited):
        from ..dsl.rewrite import rewrite  # noqa: PLC0415

        if self.init is not None and not inited:
            tile += getattr(ctx, self.init)
        fn = rewrite(_INITS[self.fn][1] if self.fn in _INITS else self.fn)
        got = fn(tile, ctx) if self._takes_ctx else fn(tile)
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
    if fn in _INITS:
        return f"init_from({_INITS[fn][0]}, {_fn_key(_INITS[fn][1])})"
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


def _preparer(ep: Epilogue, m: int):
    """name -> what a call does to that operand before the kernels take it.
    A row vector goes as it is ((N,) or (1, N): the load reads row 0); a
    column vector (M, 1) needs a row stride of 1, and rows to the next
    whole COL_PAD, since a vector load is not bounded; a tile needs a column
    stride of 1 (any row stride)."""
    import torch  # noqa: PLC0415

    pad = -m % COL_PAD

    def col(v):
        if v.stride(0) != 1:
            v = v.contiguous()
        return torch.nn.functional.pad(v, (0, 0, 0, pad)) if pad else v

    def other(v):
        return v if v.stride(-1) == 1 else v.contiguous()

    return [(name, col if ep.kinds[name] == COL else other) for name in ep.tensors]


def _cols(ep: Epilogue, ts, lo: int, hi: int):
    """The operands of a launch on columns [lo, hi) of C."""
    return [t if ep.kinds[name] == COL else t[..., lo:hi]
            for name, t in zip(ep.tensors, ts)]


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
    if kind == "pp" and OVERRIDE:
        sched = {**sched, **OVERRIDE}
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
    from ..ops import decode_gemm  # noqa: PLC0415
    from ..ops import gemm as rdna  # noqa: PLC0415

    if m <= RDNA_DECODE_MAX_M and decode_gemm.fits(m, n, k):
        return [Step("decode", decode_gemm.schedule_for(m, k), (0, n), True)]
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


def prepare(steps: List[Step], ep: Epilogue, m: int, n: int, k: int,
            dtype, device) -> Callable:
    """A plan made ready to launch: every kernel traced (so EP_COL0 is
    passed only to an epilogue that reads its coordinates), every scratch
    made. Returns go(a, b, out, tensors) -> out, which does only the
    launches and the column slices."""
    import torch  # noqa: PLC0415

    from ..ops import decode_gemm, gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

    calls = []
    ws = None
    for st in steps:
        lo, hi = st.cols
        whole = (lo, hi) == (0, n)
        if st.kernel == "reduce":
            kern, consts = gemm_cdna_dp.reduce_epilogue_kernel(st.sched["splits"],
                                                               hi - lo, ep, m=m)
            kern.trace(**consts)
            if lo and ep.uses_coords:
                consts = {**consts, "EP_COL0": lo}
            calls.append((kern, (ws, "out"), consts, lo, hi, whole, True))
            continue
        if st.into == "scratch":
            ws = gemm_cdna_dp.scratch_for(device, st.sched["splits"], m, hi - lo)
            mod = gemm_cdna if st.kernel == "pp" else gemm_cdna_dp
            calls.append((mod.kernel_for(st.sched), ("a", "b", ws), {}, lo, hi, whole,
                          False))
            continue
        extra = ()
        if st.kernel == "decode" and st.sched["splits"] > 1:
            s = st.sched
            extra = decode_gemm.scratch_for(device, s["splits"], s["tokens"], hi - lo)
        kern = kernel_for(st, ep)
        try:
            kern.trace()
            kern.build()
        except ResourceError as e:
            # The epilogue does not fit beside this kernel's registers (the
            # build refuses a spill). Store the sum in fp32 and run the
            # epilogue in a pass of its own -- the split-K reduction, of one
            # share: an extra read and write of C, never a wrong answer.
            import warnings  # noqa: PLC0415

            warnings.warn(f"hk.templates.gemm: {ep.tag} spills in {st} "
                          f"({str(e).splitlines()[1].strip()[:80]}...); it runs "
                          f"in a separate pass over C instead", stacklevel=3)
            ws1 = gemm_cdna_dp.scratch_for(device, 1, m, hi - lo)
            k32 = kernel_for(st, bind(_identity, {}, m, n, fp32))
            calls.append((k32, ("a", "b", ws1, *extra), {}, lo, hi, whole, False))
            rk, rc = gemm_cdna_dp.reduce_epilogue_kernel(1, hi - lo, ep, m=m)
            rk.trace(**rc)
            if lo and ep.uses_coords:
                rc = {**rc, "EP_COL0": lo}
            calls.append((rk, (ws1, "out"), rc, lo, hi, whole, True))
            continue
        consts = {"EP_COL0": lo} if lo and ep.uses_coords else {}
        calls.append((kern, ("a", "b", "out", *extra), consts, lo, hi, whole, True))

    decode = any(st.kernel == "decode" for st in steps)

    if len(calls) == 1 and calls[0][5]:
        kern, args, consts, *_ = calls[0]
        extra = args[3:]

        def go1(a, b, out, ts):
            if decode and not a.is_contiguous():
                a = a.contiguous()
            kern(a, b, out, *extra, *ts, **consts)
            return out

        return go1

    def go(a, b, out, ts):
        if decode and not a.is_contiguous():
            a = a.contiguous()
        for kern, args, consts, lo, hi, whole, takes in calls:
            bb = b if whole else b[lo:hi]
            oo = out if whole else out[:, lo:hi]
            vals = [x if not isinstance(x, str) else a if x == "a" else
                    bb if x == "b" else oo for x in args]
            if takes:
                vals += ts if whole else _cols(ep, ts, lo, hi)
            kern(*vals, **consts)
        return out

    return go


# ---------------------------------------------------------------- the API


def _out_dtype(out_dtype):
    import torch  # noqa: PLC0415

    if out_dtype is None:
        return torch.bfloat16
    if isinstance(out_dtype, DType):
        rev = {bf16: torch.bfloat16, fp16: torch.float16, fp32: torch.float32,
               i8: torch.int8, fp8e4m3: torch.float8_e4m3fn,
               fp8e5m2: torch.float8_e5m2}
        if out_dtype not in rev:
            raise TypeError(f"hk.templates.gemm: out_dtype {out_dtype}; bf16, "
                            f"fp16, fp32, int8 or fp8")
        return rev[out_dtype]
    _hk_dtype(out_dtype)
    return out_dtype


def _identity(acc):
    return acc


def _check_out(dt: DType, arch: Optional[str] = None) -> None:
    from ..target import current_arch  # noqa: PLC0415

    arch = arch or current_arch()
    if dt.bits == 8 and arch != "gfx950":
        raise TypeError(f"hk.templates.gemm: an {dt} C is gfx950's; on {arch} "
                        f"include/rdna3's tile store takes 16- and 32-bit "
                        f"elements (store fp16/bf16 and quantize separately)")


def kernels(m: int, n: int, k: int, *, epilogue=None, out_dtype=None,
            arch: Optional[str] = None, **operands):
    """The kernels gemm() would launch for an M x N x K call with these
    operands, as [(step, kernel, constexprs)] -- for k.trace().dump(),
    k.source(**constexprs), k.build(**constexprs), with no GPU. Operands
    may be tensors on the "meta" device: only their shapes and dtypes
    are read."""
    from ..ops import gemm_cdna, gemm_cdna_dp  # noqa: PLC0415

    tdt = _out_dtype(out_dtype)
    _check_out(_hk_dtype(tdt), arch)
    ep = bind(epilogue or _identity, operands, m, n, _hk_dtype(tdt))
    out = []
    for st in plan(m, n, k, arch=arch):
        if st.kernel == "reduce":
            kern, consts = gemm_cdna_dp.reduce_epilogue_kernel(
                st.sched["splits"], st.cols[1] - st.cols[0], ep, arch=arch, m=m)
        elif st.into == "scratch":
            mod = gemm_cdna if st.kernel == "pp" else gemm_cdna_dp
            kern, consts = mod.kernel_for(st.sched), {}
        else:
            kern, consts = kernel_for(st, ep, arch=arch), {}
        out.append((st, kern, consts))
    return out


#: (epilogue, shapes, dtypes, device, operand signatures) -> a prepared
#: launch. A call that hits it does the launches and nothing else: at a
#: decode batch the GEMM is ~10 us, and every microsecond of Python in front
#: of it is a tenth of a percent of the layer.
_LAUNCH: Dict[Tuple, Any] = {}


def _sig(item):
    name, v = item
    try:
        return (name, v.shape, v.dtype, v.stride())
    except AttributeError:
        return (name, type(v), v)


def _gemm(a, b, epilogue=None, out=None, out_dtype=None, **operands):
    key = (epilogue, a.shape, b.shape, a.dtype, b.dtype,
           out_dtype if out is None else (out.shape, out.dtype, out.stride()),
           a.device, tuple(map(_sig, operands.items())))
    try:
        got = _LAUNCH[key]
    except KeyError:
        got = _LAUNCH[key] = _slow(a, b, epilogue, out, out_dtype, operands)
    except TypeError:                       # an unhashable operand: say what
        got = _slow(a, b, epilogue, out, out_dtype, operands)
    go, shape, dtype, prep = got
    if out is None:
        import torch  # noqa: PLC0415

        out = torch.empty(shape, device=a.device, dtype=dtype)
    return go(a, b, out, [f(operands[name]) for name, f in prep])


def _slow(a, b, epilogue, out, out_dtype, operands):
    """Check a call and make its launch: (go, C's shape, C's dtype, the
    operand preparers)."""
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
    if out is not None and (tuple(out.shape) != (m, n) or out.stride(1) != 1):
        raise ValueError(f"hk.templates.gemm: out is {tuple(out.shape)}, "
                         f"expected ({m}, {n}) with unit column stride")
    if epilogue is None and not operands and tdt == torch.bfloat16:
        from ..ops import gemm as rdna  # noqa: PLC0415

        return (lambda a, b, out, ts: rdna.matmul(a, b, out=out)), (m, n), tdt, []
    _check_out(_hk_dtype(tdt))
    ep = bind(epilogue or _identity, operands, m, n, _hk_dtype(tdt))
    go = prepare(plan(m, n, k), ep, m, n, k, tdt, a.device)
    return go, (m, n), tdt, _preparer(ep, m)


# ---------------------------------------------------------------- ready-made


def bias(acc, ctx):
    """acc + bias, bias one value per column of C: F.linear's."""
    return acc + ctx.bias


#: GELU(tanh) is x * sigmoid(2u), u = sqrt(2/pi) (x + 0.044715 x^3), and
#: sigmoid(2u) = 1 / (1 + 2^(x (A x^2 + B))): one exp2 and, since the
#: denominator is at least 1, 1/d = rsqrt(d)^2 -- two native instructions
#: where Tile.gelu()'s tanh form clamps, exponentiates and divides correctly
#: rounded. On an MI355X that took the fused GELU from 9% of a 4096 x 24576
#: x 4096 GEMM to 3%, at the same error against torch's in bf16.
_GELU_B = -2.0 * 1.4426950408889634 * 0.7978845608028654
_GELU_A = _GELU_B * 0.044715


def gelu_tanh(x):
    """GELU, tanh approximation, of a tile, in place; returns it. Equal to
    F.gelu(x, approximate="tanh") to a few fp32 ulps, and about half the
    instructions of Tile.gelu()."""
    q = x * x
    q *= _GELU_A
    q += _GELU_B
    q *= x
    q.exp2_()
    q += 1.0
    q.rsqrt_()
    x *= q
    x *= q
    return x


def gelu(acc):
    """GELU, tanh approximation (F.gelu(..., approximate="tanh"))."""
    return gelu_tanh(acc)


def silu(acc):
    """SiLU, x * sigmoid(x)."""
    return acc.silu()


def relu(acc):
    return acc.relu()


def bias_gelu(acc, ctx):
    """gelu(acc + bias), tanh approximation."""
    acc += ctx.bias
    return gelu_tanh(acc)


def bias_silu(acc, ctx):
    return (acc + ctx.bias).silu()


def bias_relu(acc, ctx):
    return (acc + ctx.bias).relu()


def _residual(acc):
    """acc + residual, residual (M, N): an o_proj or down_proj's skip -- the
    accumulator starts as the residual (init_from)."""
    return acc


def _bias_residual(acc, ctx):
    """acc + bias + residual, the accumulator starting as the residual."""
    return acc + ctx.bias


_residual.__name__ = "residual"
_bias_residual.__name__ = "bias_residual"


def scale(acc, ctx):
    """acc * scale: a number, a per-channel (N,) or a per-row (M, 1) vector.
    With out_dtype=torch.int8 or torch.float8_e4m3fn (gfx950) this is a
    quantizer: the store rounds to nearest even and saturates (fp8 at
    +-448, where torch's cast gives NaN)."""
    return acc * ctx.scale


def chain(*fns: Callable) -> Callable:
    """One epilogue that applies `fns` in order, each to what the last
    returned, all reading the same ctx."""
    if any(f in _INITS for f in fns[1:]):
        raise ValueError("chain: an init_from epilogue starts the accumulator, "
                         "so it can only come first")
    if fns and fns[0] in _INITS:
        name, first = _INITS[fns[0]]
        return init_from(name)(chain(first, *fns[1:]))
    takes = [len([p for p in inspect.signature(f).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) == 2
             for f in fns]

    def chained(acc, ctx):
        from ..dsl.rewrite import rewrite  # noqa: PLC0415

        for f, t in zip(fns, takes):
            f = rewrite(f)
            got = f(acc, ctx) if t else f(acc)
            acc = acc if got is None else got
        return acc

    chained.__name__ = "_".join(getattr(f, "__name__", "fn") for f in fns)
    chained.__qualname__ = "chain(" + ", ".join(getattr(f, "__qualname__", "?")
                                                for f in fns) + ")"
    _CHAINS[chained] = fns
    return chained


_CHAINS: Dict[Callable, Tuple] = {}

#: init_from's epilogues: wrapper -> (operand name, the user's function).
_INITS: Dict[Callable, Tuple[str, Callable]] = {}


def init_from(name: str) -> Callable:
    """Decorator: the accumulator starts as the (M, N) operand `name`
    instead of zero, so C = fn(name + A @ B^T, ctx) -- BLAS's beta * C with
    beta 1. A kernel that can loads it into the accumulator before its K
    loop, where the load hides under the first tiles' copies; one that
    cannot (a split-K share, the decode GEMV) adds it before fn. gemm.residual
    is init_from("residual") of the identity."""
    if not isinstance(name, str) or not name.isidentifier():
        raise TypeError(f"init_from takes an operand's name, not {name!r}")

    def deco(fn):
        n = len([p for p in inspect.signature(fn).parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
        if n == 1:
            def wrapped(acc):
                return fn(acc)
        else:
            def wrapped(acc, ctx):
                return fn(acc, ctx)
        wrapped.__name__ = getattr(fn, "__name__", "fn")
        wrapped.__qualname__ = "init_from(%r)(%s)" % (name, getattr(fn, "__qualname__", "?"))
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        _INITS[wrapped] = (name, fn)
        return wrapped

    return deco

residual = init_from("residual")(_residual)
bias_residual = init_from("residual")(_bias_residual)

READY = {f.__name__: f for f in (bias, gelu, silu, relu, bias_gelu, bias_silu,
                                 bias_relu, residual, bias_residual, scale,
                                 gelu_tanh)}


class _Gemm:
    """hk.templates.gemm(a, b, epilogue=fn, out=None, out_dtype=None,
    **operands) -> C; with no a and b, the same bound to its keywords, to
    call later. See the module docstring."""

    plan = staticmethod(plan)
    kernels = staticmethod(kernels)
    kernel_for = staticmethod(kernel_for)
    chain = staticmethod(chain)
    init_from = staticmethod(init_from)
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

__all__ = ["gemm", "plan", "kernels", "kernel_for", "prepare", "bind", "init_from", "Step", "Context",
           "Epilogue", "chain", "READY", *READY]
