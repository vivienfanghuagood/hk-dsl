"""hk.templates.elementwise: a pointwise kernel from the per-element math.

    gelu_and_mul = hk.templates.elementwise(lambda g, u: hk.math.gelu(g) * u)
    out = gelu_and_mul(x[:, :d], x[:, d:])      # the halves of a gate/up output

The hook is traced once per (operand kinds, dtypes, constexprs): each operand
arrives as an fp32 register strip of a stretch of one row (loads widen,
stores round to nearest even, as torch's arithmetic does), and what it
returns is stored, converted to the output's dtype. The template owns
everything else, per architecture: the layout, the run a lane reads, the
warps a workgroup has and the stretch each covers, the grid and its order,
the bounded accesses of a ragged row, and whether contiguous operands are
read as one flat row. docs/templates/elementwise-rowwise.md is the guide.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .. import dsl as hk
from ..types import fp32
from . import _common as C

#: Past 2 GiB a strip's bounded offsets reach the out-of-range marker
#: (hk.ops.elementwise._STRIP_MAX_BYTES).
_MAX_BYTES = 1 << 31

#: Elements from which contiguous operands are read as one flat row.
_FLAT_MIN = 1 << 20


@dataclass(frozen=True)
class Plan:
    """What an elementwise call runs: `flat` reads contiguous operands as one
    row of rows*cols; otherwise workgroup i is row i // per_row, stretch
    i % per_row. A warp covers `chunks` chunks of wave x `run` contiguous
    elements, a workgroup `warps` such stretches side by side; `exact` says
    they cover the row with nothing over, so nothing is bounded."""

    arch: str
    flat: bool
    warps: int
    chunks: int
    run: int
    exact: bool
    rows: int
    cols: int
    reserve: int = 0
    #: Launch the preloaded twin (hk.Kernel.preloaded: the arguments in
    #: SGPRs at dispatch), as hk.ops' flat strips do past 32 MB on gfx950.
    preload: bool = False

    @property
    def wave(self) -> int:
        from ..target import get_target  # noqa: PLC0415

        return get_target(self.arch).wave

    @property
    def span(self) -> int:
        """Elements one warp covers."""
        return self.chunks * self.wave * self.run

    @property
    def grid(self) -> Tuple[int, int, int]:
        per_row = -(-self.cols // (self.warps * self.span))
        return (self.rows * per_row, 1, 1)

    def __str__(self) -> str:
        return (f"{'flat' if self.flat else 'rows'} {self.rows}x{self.cols} on {self.arch}: "
                f"{self.warps} warp(s) x {self.chunks} chunk(s) of {self.wave}x{self.run}, "
                f"grid {self.grid[0]}, {'exact' if self.exact else 'bounded'}"
                + (f", {self.reserve >> 10} KB LDS reserved" if self.reserve else "")
                + (", preloaded" if self.preload else ""))


def plan_for(arch: str, rows: int, cols: int, flat: bool, moved: int, inputs: int,
             nbytes: int, override: Optional[Dict[str, int]] = None) -> Plan:
    """The plan for a pass over `rows` x `cols` (one flat row when `flat`)
    moving `moved` bytes with `inputs` streamed inputs of `nbytes`-byte
    elements, on `arch`. The numbers are hk.ops' measured ones:

      - the run: hk.ops.norm.strip_run, a 16-byte access where the row's
        length keeps the runs aligned;
      - flat, gfx950: hk.ops.elementwise's STRIP_PLANS by the bytes moved
        (8 warps of one chunk; 2 of 8 for 24-64 MB of a two-input op; 2 of
        one past the cache);
      - rows (and flat on gfx1100): hk.ops.fused.silu_strip_plan, silu_mul's
        plan -- on gfx950 one warp of one chunk a workgroup, on gfx1100
        eight, with an LDS reserve capping the workgroups a WGP holds once
        a call moves more than the 96 MB Infinity Cache.
    `override` replaces any of warps / chunks / run."""
    from ..ops import elementwise as ew, fused  # noqa: PLC0415
    from ..ops.norm import strip_run  # noqa: PLC0415

    n = cols
    run = strip_run(n, nbytes) if n else 16 // nbytes
    reserve = 0
    if arch == "gfx950" and flat:
        warps, chunks = ew.flat_strip_plan(moved, inputs >= 2)
    else:
        got = fused.silu_strip_plan(arch, moved)
        if got is None:
            raise NotImplementedError(f"hk.templates.elementwise runs on register strips, "
                                      f"which {arch} does not have (gfx1100, gfx1201 and gfx950 do)")
        warps, chunks, reserve = got
    if override:
        warps = override.get("warps", warps)
        chunks = override.get("chunks", chunks)
        run = override.get("run", run)
        reserve = override.get("reserve", reserve)
    from ..ops.norm import STRIP_PRELOAD_PAST  # noqa: PLC0415

    past = STRIP_PRELOAD_PAST.get(arch)
    preload = bool(flat and past is not None and moved > past)
    if override and "preload" in override:
        preload = bool(override["preload"])
    p = Plan(arch, flat, warps, chunks, run, False, rows, cols, reserve, preload)
    return replace(p, exact=cols % (warps * p.span) == 0)


class Elementwise:
    """A pointwise kernel built from `fn`. Call it with torch tensors; see
    hk.templates.elementwise."""

    def __init__(self, fn: Callable, inputs: Optional[Sequence] = None,
                 outputs: Any = None, *, name: Optional[str] = None,
                 arch: Optional[str] = None, plan: Optional[Dict[str, int]] = None):
        self.fn = fn
        self.inputs = None if inputs is None else tuple(C.as_operand(s) for s in inputs)
        if isinstance(outputs, int) and not isinstance(outputs, bool):
            # outputs=2: two outputs of the first full operand's dtype.
            outputs = (None,) * outputs if outputs > 1 else None
        if outputs is None or isinstance(outputs, (list, tuple)):
            outs = outputs
        else:
            outs = (outputs,)
        self.outputs = None if outs is None else tuple(C._dtype(d) for d in outs)
        self._single = outputs is None or not isinstance(outputs, (list, tuple))
        self.name = name or C.kernel_name("ew", fn)
        self._arch = arch
        self.override = dict(plan or {})
        bad = set(self.override) - {"warps", "chunks", "run", "flat", "reserve", "preload"}
        if bad:
            raise TypeError(f"plan= overrides warps, chunks, run, flat, reserve and "
                            f"preload, not {sorted(bad)}")
        self.consts = C.hook_consts(fn)
        self._arity = C.hook_arity(fn)
        if self.inputs is not None and self._arity is not None \
                and self._arity != len(self.inputs):
            raise TypeError(f"{self.name}: the hook takes {self._arity} operand(s) and "
                            f"inputs= declares {len(self.inputs)}")
        self._kernels: Dict[Tuple, Any] = {}
        self._calls: Dict[Tuple, Any] = {}

    @property
    def arch(self) -> str:
        if self._arch is None:
            from ..target import current_arch  # noqa: PLC0415

            return current_arch()
        return self._arch

    def with_plan(self, **override) -> "Elementwise":
        """This template with its plan's warps / chunks / run / flat fixed."""
        return Elementwise(self.fn, self.inputs, self._outputs_arg(), name=self.name,
                           arch=self._arch, plan={**self.override, **override})

    def _outputs_arg(self):
        if self.outputs is None:
            return None
        return self.outputs[0] if self._single else self.outputs

    # ------------------------------------------------------------ planning

    def _resolve(self, args):
        """(operands, out shape, out dtypes) for these call arguments."""
        if self._arity is not None and len(args) != self._arity:
            raise TypeError(f"{self.name} takes {self._arity} operand(s), got {len(args)}")
        if self.inputs is not None and len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes {len(self.inputs)} operand(s), got {len(args)}")
        shapes = [tuple(a.shape) for a in args]
        fulls = [s for s, o in zip(shapes, self.inputs or [None] * len(args))
                 if o is None or o.kind == "full"]
        if not fulls:
            raise TypeError(f"{self.name}: no operand of the problem's whole shape")
        out_shape = max(fulls, key=lambda s: (len(s), C.numel(s)))
        ops = []
        for i, (a, s) in enumerate(zip(args, shapes)):
            kind = (self.inputs[i].kind if self.inputs is not None
                    else C.classify(s, out_shape))
            dt = C.hk_dtype(a)
            want = self.inputs[i].dtype if self.inputs is not None else None
            if want is not None and want is not dt:
                raise TypeError(f"{self.name}: operand {i} is {dt.name}, declared {want.name}")
            if kind == "full" and s != out_shape:
                raise ValueError(f"{self.name}: operand {i} is {s}, the problem is {out_shape}")
            if kind in ("full", "per_col"):
                C.row_stride(s, a.stride(), f"{self.name}: operand {i}")
            ops.append(C.Operand(kind, dt))
        first = next(o.dtype for o in ops if o.kind == "full")
        outs = tuple(d or first for d in self.outputs) if self.outputs else (first,)
        return tuple(ops), out_shape, outs

    def plan(self, *args, out=None) -> Plan:
        """The plan a call with these operands (tensors, or hk.templates.meta)
        runs."""
        ops, shape, outs = self._resolve(args)
        return self._plan(ops, args, shape, outs, out)

    def _plan(self, ops, args, shape, outs, out) -> Plan:
        rows, cols = C.rows_cols(shape)
        flat_ok = all(o.kind in ("full", "scalar") for o in ops) and all(
            C.is_contiguous(tuple(a.shape), a.stride())
            for a, o in zip(args, ops) if o.kind == "full")
        if out is not None:
            for o in (out if isinstance(out, (list, tuple)) else (out,)):
                flat_ok = flat_ok and C.is_contiguous(tuple(o.shape), o.stride())
        # Flat only with enough of it: one row of 14336 flat is four
        # workgroups of eight warps, where by rows it is 28 of one (gelu at
        # 1x14336, MI355X: 2.38 us flat, 2.05 for silu_mul's rows plan).
        flat = self.override.get("flat", flat_ok and rows * cols >= _FLAT_MIN) and flat_ok
        streamed = [o.dtype.bytes for o in ops if o.kind == "full"]
        n = rows * cols
        moved = n * (sum(streamed) + sum(d.bytes for d in outs))
        if flat:
            rows, cols = 1, n
        p = plan_for(self.arch, rows, cols, flat, moved, len(streamed),
                     min(streamed), {k: v for k, v in self.override.items() if k != "flat"})
        C.check_runs(self.name, p.arch, p.run, outs)
        if not p.exact and max(streamed + [d.bytes for d in outs]) * n >= _MAX_BYTES:
            raise ValueError(f"{self.name}: {n} elements past 2 GiB with a ragged row; the "
                             f"bounded accesses address 2 GiB")
        return p

    # ------------------------------------------------------------ kernels

    def _kernel(self, ops, outs, p: Plan):
        key = (p.arch, ops, outs, p.warps)
        k = self._kernels.get(key)
        if k is None:
            k = self._kernels[key] = _build(self.fn, ops, outs, p.warps, p.arch,
                                            f"{self.name}_w{p.warps}", self.consts)
        return k.preloaded() if p.preload else k

    def kernel(self, *args, out=None, **consts):
        """(hk.Kernel, constexprs) for a call with these operands: `k.trace(**c)`,
        `k.source(**c)` and `k.build(**c)` need no GPU."""
        ops, shape, outs = self._resolve(args)
        p = self._plan(ops, args, shape, outs, out)
        return self._kernel(ops, outs, p), self._consts(p, consts)

    def _consts(self, p: Plan, consts) -> Dict[str, Any]:
        unknown = set(consts) - set(self.consts)
        if unknown:
            raise TypeError(f"{self.name}: no constexpr {sorted(unknown)}; the hook "
                            f"takes {sorted(self.consts) or 'none'}")
        return {"T_CW": p.span, "T_RUN": p.run, "T_EXACT": int(p.exact),
                "T_FLAT": int(p.flat), "T_RESERVE": p.reserve, **consts}

    # ------------------------------------------------------------ calling

    def __call__(self, *args, out=None, **consts):
        key = C.call_key(args, out, consts)
        call = self._calls.get(key)
        if call is None:
            call = self._calls[key] = self._prepare(args, out, consts)
        return call(args, out)

    def _prepare(self, args, out, consts):
        import torch  # noqa: PLC0415

        ops, shape, outs = self._resolve(args)
        p = self._plan(ops, args, shape, outs, out)
        rows, cols = C.rows_cols(shape)
        n = rows * cols
        single = self._single
        nout = len(outs)
        if out is not None:
            given = out if isinstance(out, (list, tuple)) else (out,)
            if len(given) != nout:
                raise TypeError(f"{self.name}: out= has {len(given)} tensor(s), the hook "
                                f"returns {nout}")
            for o, d in zip(given, outs):
                if tuple(o.shape) != shape or C.hk_dtype(o) is not d:
                    raise ValueError(f"{self.name}: out is {tuple(o.shape)}/{o.dtype}, "
                                     f"expected {shape}/{d.name}")
                C.row_stride(tuple(o.shape), o.stride(), f"{self.name}: out")
        tdt = [C.torch_dtype(d) for d in outs]
        if not n:
            def empty(args, out):
                got = out if out is not None else tuple(
                    torch.empty(shape, dtype=d, device=args[0].device) for d in tdt)
                return got[0] if single and isinstance(got, tuple) else got
            return empty
        k = self._kernel(ops, outs, p)
        launch = k._launcher(**self._consts(p, consts))
        arg_views = [C.view_fn(tuple(a.shape), rows, cols, o.kind, p.flat)
                     for a, o in zip(args, ops)]
        meta = [(tuple(a.shape), tuple(a.stride()), C.hk_dtype(a).bytes) for a in args]
        given = None if out is None else (out if isinstance(out, (list, tuple)) else (out,))
        out_views = [C.view_fn(tuple(given[j].shape) if given else shape, rows, cols,
                               "full", p.flat) for j in range(nout)]
        out_meta = None if given is None else [
            (tuple(o.shape), tuple(o.stride()), C.hk_dtype(o).bytes) for o in given]
        return C.make_call(launch, meta, [shape] * nout, tdt, arg_views, out_views,
                           out_meta, single)


def _trace(kw, spec):
    fn, ops, warps, name, user_consts, names = spec
    nin = len(ops)
    cw, run, exact, flat = kw["T_CW"], kw["T_RUN"], kw["T_EXACT"], kw["T_FLAT"]
    if kw["T_RESERVE"]:
        hk.lds.reserve(kw["T_RESERVE"])
    hook = {k: kw[k] for k in user_consts}
    xs = [kw[n] for n in names[:nin]]
    ys = [kw[n] for n in names[nin:]]
    S = hk.RS(fp32, 1, cw, run)
    i = hk.arch.block_idx.x
    if flat:
        row, g = 0, i
    else:
        per_row = hk.cdiv(ys[0].cols, warps * cw)
        row = i // per_row
        g = i - row * per_row
    c0 = (g * warps + hk.arch.warp_id() if warps > 1 else g) * cw
    bounded = not exact
    vals = []
    for x, o in zip(xs, ops):
        if o.kind == "full":
            vals.append(x.at[row, c0].load(S, bounded=bounded))
        elif o.kind == "per_col":
            vals.append(x.at[0, c0].load(S, bounded=bounded))
        else:
            # One value a row (or one value): a column vector, broadcast onto
            # a strip so that the hook sees tiles only.
            v = x.at[row if o.kind == "per_row" else 0, 0].load(hk.col_vec(S))
            t = hk.zeros(S)
            t += v
            vals.append(t)
    with C.tile_reductions_refused(
            "an elementwise hook maps elements; t.{kind}(axis=...) reduces one "
            "warp's share of a row. A reduction along the row is "
            "hk.templates.rowwise's (hk.templates.row_{kind})"):
        res = fn(*vals, **hook)
    res = res if isinstance(res, (tuple, list)) else (res,)
    if len(res) != len(ys):
        raise TypeError(f"{name}: the hook returned {len(res)} value(s), the "
                        f"template stores {len(ys)} output(s)")
    for y, r in zip(ys, res):
        if not isinstance(r, hk.Tile):
            raise TypeError(f"{name}: the hook returned {r!r}; an elementwise hook "
                            f"returns tiles (a value per element). A reduction along "
                            f"the row is hk.templates.rowwise's")
        y.at[row, c0].store(r, bounded=bounded)


def _build(fn, ops, outs, warps: int, arch: str, name: str, user_consts):
    """The strip kernel: workgroup i is row i // per_row (0 when flat), its
    warp w the elements [(g*WARPS + w)*CW, +CW) of it, g = i % per_row."""
    nin = len(ops)
    names = [f"x{i}" for i in range(nin)] + [f"y{j}" for j in range(len(outs))]
    spec = (fn, ops, warps, name, user_consts, names)

    # No control flow of its own, so hk's rewrite has nothing to do here.
    def body(**kw):
        _trace(kw, spec)

    body.__name__ = name
    body.__signature__ = C.signature(
        [(n, o.dtype) for n, o in zip(names, ops)] +
        [(n, d) for n, d in zip(names[nin:], outs)],
        {"T_CW": 512, "T_RUN": 8, "T_EXACT": 1, "T_FLAT": 0, "T_RESERVE": 0,
         **user_consts})
    # noalias: each element is read and written by one lane, an in-place call
    # included (a partial overlap is copied first) -- the flat, __restrict__
    # arguments hk.ops' strips take on CDNA.
    return hk.kernel(body, warps=warps, arch=arch, name=name, noalias=True,
                     grid=lambda p: (_grid_x(p, warps), 1, 1))


def _grid_x(p, warps):
    y0 = p.y0
    per_row = hk.cdiv(y0.cols, warps * p.T_CW)
    return y0.rows * per_row


def elementwise(fn: Optional[Callable] = None, inputs: Optional[Sequence] = None,
                outputs: Any = None, *, name: Optional[str] = None,
                arch: Optional[str] = None, plan: Optional[Dict[str, int]] = None):
    """A pointwise kernel from `fn`, the math of one element.

        gelu_and_mul = hk.templates.elementwise(lambda g, u: hk.math.gelu(g) * u)
        y = gelu_and_mul(gate_up[:, :d], gate_up[:, d:])

    `fn` takes one tile per operand -- fp32 register strips; a load widens
    and a store rounds -- and returns a tile, or a tuple of them for several
    outputs. Its parameters with defaults are constexprs, overridable per
    call: `fn = lambda x, b, *, ALPHA=1.0: ...`, `k(x, b, ALPHA=0.5)`.

    `inputs`: the operands' kinds and dtypes, if you want them checked --
    a dtype is a full operand, `per_col(d)` one value per column (a bias),
    `per_row(d)` one per row, `scalar(d)` one value. Left out, they follow
    from the tensors by torch's broadcasting. `outputs`: the output dtype
    (the first full operand's by default), or a tuple of them. Operands may
    be any (rows, cols) view whose last axis is contiguous -- a slice of the
    last axis is read in place -- with any leading axes.

    The plan (layout, warps, the stretch a warp covers, the grid) is picked
    per architecture; `k.plan(*args)` shows it and `plan={"warps": ...,
    "chunks": ..., "run": ..., "flat": ...}` (or `k.with_plan(...)`)
    overrides it. Usable as a decorator: `@hk.templates.elementwise`."""
    if fn is None:
        return lambda f: Elementwise(f, inputs, outputs, name=name, arch=arch, plan=plan)
    return Elementwise(fn, inputs, outputs, name=name, arch=arch, plan=plan)


class _CallableModule(types.ModuleType):
    """`import hk.templates.elementwise` binds the package's attribute to this
    module, so the module calls through too, as hk.templates.gemm's does:
    `hk.templates.elementwise(...)` is the template whichever was imported first."""

    def __call__(self, *args, **kwargs):
        return elementwise(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["elementwise", "Elementwise", "Plan", "plan_for"]
