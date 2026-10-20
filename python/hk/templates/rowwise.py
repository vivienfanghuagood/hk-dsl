"""hk.templates.rowwise: a kernel that reduces along each row, then maps it.

    @hk.templates.rowwise
    def rms_norm(x, w, *, EPS=1e-6):
        ms = hk.templates.row_mean(x * x)
        return x * (ms + EPS).rsqrt() * w

    y = rms_norm(x, weight)                  # weight (cols,): one value a column

The hook gets one fp32 register strip per full operand (a stretch of one
row: the warp's share of it), a per-column operand (a weight) the same way,
and a per-row operand as a column vector (one value for the row). Inside
it, `row_sum`, `row_max`, `row_min` and `row_mean` reduce the WHOLE row --
the template masks a ragged tail with the reduction's identity, combines
the lanes and the warps -- and give a column vector that broadcasts against
the strips. What the hook returns is stored: a tile as a full output, a
column vector as a per-row one (a scale, a loss).

The template picks, per architecture and shape:
  - one workgroup a row, the row held in registers and read once, with the
    warps, the run and the chunks a warp holds from hk.ops.norm's measured
    strip plans (and its occupancy cap past the cache);
  - for a row too wide to hold (a vocabulary's logits), a multi-pass
    schedule: the row streamed in steps, once per reduction the hook
    depends on, and once more to write.
docs/templates/elementwise-rowwise.md is the guide.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import dsl as hk
from ..dsl.collective import cross_warp
from ..dsl.values import Vec, as_vec
from ..types import fp32, i32, i8
from . import _common as C

_NEG_INF = float("-inf")
_IDENT = {"sum": 0.0, "max": _NEG_INF, "min": float("inf")}
_COMBINE = {"sum": "add", "max": "max", "min": "min"}

#: The kinds of hk.ops' strip plans a rowwise call can be planned like.
LIKE = ("rms", "rmsw", "addrmsw", "layer", "softmax", "quantize")


# ---------------------------------------------------------------- reductions


class _Stop(Exception):
    """A pass of a multi-pass schedule has what it came for."""


class _Ctx:
    """The rowwise trace in progress: how a row reduction is done here."""

    def __init__(self, S, warps: int, steps: int):
        self.S = S
        self.warps = warps
        self.steps = steps            # 0: single pass
        self.tail = None              # columns of the warp's strip in the row
        self.n = None                 # the row's length
        self.known: List[Vec] = []    # multi-pass: reductions done
        self.k = 0                    # multi-pass: the reduction this pass makes
        self.count = 0                # reductions met in this hook call
        self.acc = None               # multi-pass: (op, per-warp partial)
        self.dry = False              # multi-pass: probing for the next reduction

    def reduce(self, t, op: str) -> Vec:
        if not isinstance(t, hk.Tile):
            raise TypeError(f"row_{op} reduces a tile (a strip of the row), got {t!r}")
        j = self.count
        self.count += 1
        if self.steps and j < len(self.known):
            return self.known[j]
        if self.dry:
            raise _Stop()
        if self.tail is not None:
            # In place: a tail column is never stored, and every reduction
            # fills it again with its own identity.
            t.right_fill_(self.tail, _IDENT[op])
        with C.tile_reductions_allowed():
            part = getattr(t, op)(axis=1)
            if self.steps:
                if self.acc is None:
                    self.acc = (op, hk.full(hk.col_vec(self.S), _IDENT[op]))
                acc = self.acc[1]
                if op == "sum":
                    acc += part
                elif op == "max":
                    acc.max_(part)
                else:
                    acc.min_(part)
                raise _Stop()
            v = part.value()
        v = cross_warp(v, _COMBINE[op], self.warps, name=f"row_{op}")
        return as_vec(v, self.S, uniform=True)

    def finish_pass(self):
        op, acc = self.acc
        v = cross_warp(acc, _COMBINE[op], self.warps, name=f"row_{op}")
        self.known.append(as_vec(v, self.S, uniform=True))
        self.acc = None


_CTX: List[_Ctx] = []


def _ctx(what: str) -> _Ctx:
    if not _CTX:
        raise RuntimeError(f"hk.templates.{what} is a row reduction inside a "
                           f"hk.templates.rowwise hook, while it is traced")
    return _CTX[-1]


def row_sum(t) -> Vec:
    """The sum of each row of `t` -- the whole row, across every lane and
    warp that holds a part of it: one value per row, broadcast against the
    row's strips (`x - row_sum(x) / n`)."""
    return _ctx("row_sum").reduce(t, "sum")


def row_max(t) -> Vec:
    """The max of each row of `t`, the whole row."""
    return _ctx("row_max").reduce(t, "max")


def row_min(t) -> Vec:
    """The min of each row of `t`, the whole row."""
    return _ctx("row_min").reduce(t, "min")


def row_mean(t) -> Vec:
    """The mean of each row of `t`: row_sum(t) over the row's length."""
    c = _ctx("row_mean")
    return c.reduce(t, "sum") / c.n


def row_length():
    """The row's length (a device int): what row_mean divides by."""
    return _ctx("row_length").n


# ---------------------------------------------------------------- plan


@dataclass(frozen=True)
class Plan:
    """What a rowwise call runs: one workgroup a row, `warps` warps, each
    holding `chunks` chunks of wave x `run` contiguous elements of it.
    `steps` 0 holds the row and reads it once; `steps` > 0 streams a row
    too wide to hold in that many steps, a pass per reduction and one to
    write. `reserve` is LDS reserved to cap the workgroups a CU/WGP holds
    (past the cache); `exact` says nothing is bounded."""

    arch: str
    like: str
    warps: int
    chunks: int
    run: int
    exact: bool
    reserve: int
    steps: int
    rows: int
    cols: int

    @property
    def wave(self) -> int:
        from ..target import get_target  # noqa: PLC0415

        return get_target(self.arch).wave

    @property
    def span(self) -> int:
        return self.chunks * self.wave * self.run

    def __str__(self) -> str:
        how = (f"{self.steps} steps a pass" if self.steps else "held, read once")
        cap = f", {self.reserve >> 10} KB LDS reserved" if self.reserve else ""
        return (f"{self.rows}x{self.cols} like {self.like} on {self.arch}: one workgroup a "
                f"row, {self.warps} warp(s) x {self.chunks} chunk(s) of {self.wave}x"
                f"{self.run} ({self.chunks * self.run} a lane), {how}, "
                f"{'exact' if self.exact else 'bounded'}{cap}")


#: Per-lane elements of a multi-pass step, and its warps: 16 warps of two
#: 16-byte runs a lane.
_STREAM_WARPS = 16
_STREAM_PER_LANE = 16


def plan_for(arch: str, like: str, rows: int, cols: int, nbytes: int,
             override: Optional[Dict[str, int]] = None) -> Plan:
    """The plan for `rows` rows of `cols` `nbytes`-byte elements, planned
    like hk.ops' `like` kernel ("rms", "rmsw", "addrmsw", "layer",
    "softmax", "quantize") on `arch`: hk.ops.norm.strip_plan_for (and
    hk.ops.quant.strip_plan_for for "quantize") where the row fits a
    strip, else a multi-pass stream. `override` replaces warps, chunks,
    run, steps or reserve."""
    from ..ops import norm, quant  # noqa: PLC0415
    from ..target import get_target  # noqa: PLC0415

    tgt = get_target(arch)
    if not tgt.strips:
        raise NotImplementedError(f"hk.templates.rowwise runs on register strips, which "
                                  f"{arch} does not have (gfx1100 and gfx950 do)")
    if like == "quantize":
        p = quant.strip_plan_for(nbytes, cols, rows, arch)
        p = None if p is None else (*p, 0)
    else:
        p = norm.strip_plan_for(like, nbytes, cols, rows, arch)
    override = dict(override or {})
    if p is not None and not override.get("steps"):
        warps, chunks, run, exact, reserve = p
        steps = 0
    else:
        run = norm.strip_run(cols, nbytes)
        warps = _STREAM_WARPS
        chunks = max(1, _STREAM_PER_LANE // run)
        reserve = 0
        steps = -(-cols // (warps * chunks * tgt.wave * run))
        exact = False
    warps = override.get("warps", warps)
    chunks = override.get("chunks", chunks)
    run = override.get("run", run)
    reserve = override.get("reserve", reserve)
    plan = Plan(arch, like, warps, chunks, run, False, reserve, 0, rows, cols)
    step = warps * plan.span
    if steps or override.get("steps"):
        steps = override.get("steps") or -(-cols // step)
        if steps * step < cols:
            raise ValueError(f"plan: {steps} steps of {step} do not cover {cols} columns")
        return replace(plan, steps=steps, exact=cols % step == 0 and steps * step == cols)
    if step < cols:
        raise ValueError(f"plan: {warps} warps of {plan.span} do not hold {cols} columns; "
                         f"give steps=")
    return replace(plan, exact=step == cols)


# ---------------------------------------------------------------- template


class Rowwise:
    """A row-reduce-then-map kernel built from `fn`; see hk.templates.rowwise."""

    def __init__(self, fn: Callable, inputs: Optional[Sequence] = None,
                 outputs: Any = None, *, like: Optional[str] = None,
                 name: Optional[str] = None, arch: Optional[str] = None,
                 plan: Optional[Dict[str, int]] = None):
        self.fn = fn
        self.inputs = None if inputs is None else tuple(C.as_operand(s) for s in inputs)
        if isinstance(outputs, int) and not isinstance(outputs, bool):
            # outputs=2: two full outputs of the first full operand's dtype.
            outputs = (None,) * outputs if outputs > 1 else None
        self._single = outputs is None or not isinstance(outputs, (list, tuple))
        outs = outputs if isinstance(outputs, (list, tuple)) else (
            None if outputs is None else (outputs,))
        self.outputs = None if outs is None else tuple(C.as_operand(o) for o in outs)
        if self.outputs is not None:
            for o in self.outputs:
                if o.kind not in ("full", "per_row"):
                    raise TypeError(f"a rowwise output is full or per_row, not {o.kind}")
        if like is not None and like not in LIKE:
            raise ValueError(f"like= is one of {LIKE}, not {like!r}")
        self.like = like
        self.name = name or C.kernel_name("rw", fn)
        self._arch = arch
        self.override = dict(plan or {})
        bad = set(self.override) - {"warps", "chunks", "run", "steps", "reserve"}
        if bad:
            raise TypeError(f"plan= overrides warps, chunks, run, steps and reserve, "
                            f"not {sorted(bad)}")
        self.consts = C.hook_consts(fn)
        self._arity = C.hook_arity(fn)
        self._kernels: Dict[Tuple, Any] = {}
        self._calls: Dict[Tuple, Any] = {}

    @property
    def arch(self) -> str:
        if self._arch is None:
            from ..target import current_arch  # noqa: PLC0415

            return current_arch()
        return self._arch

    def with_plan(self, **override) -> "Rowwise":
        """This template with its plan's warps / chunks / run / steps /
        reserve fixed."""
        outs = None if self.outputs is None else (
            self.outputs[0] if self._single else self.outputs)
        return Rowwise(self.fn, self.inputs, outs, like=self.like, name=self.name,
                       arch=self._arch, plan={**self.override, **override})

    def _resolve(self, args):
        if self._arity is not None and len(args) != self._arity:
            raise TypeError(f"{self.name} takes {self._arity} operand(s), got {len(args)}")
        if self.inputs is not None and len(args) != len(self.inputs):
            raise TypeError(f"{self.name} takes {len(self.inputs)} operand(s), got {len(args)}")
        shapes = [tuple(a.shape) for a in args]
        fulls = [s for s, o in zip(shapes, self.inputs or [None] * len(args))
                 if o is None or o.kind == "full"]
        if not fulls:
            raise TypeError(f"{self.name}: no operand of the problem's whole shape")
        shape = max(fulls, key=lambda s: (len(s), C.numel(s)))
        ops = []
        for i, (a, s) in enumerate(zip(args, shapes)):
            kind = (self.inputs[i].kind if self.inputs is not None
                    else C.classify(s, shape))
            dt = C.hk_dtype(a)
            want = self.inputs[i].dtype if self.inputs is not None else None
            if want is not None and want is not dt:
                raise TypeError(f"{self.name}: operand {i} is {dt.name}, declared {want.name}")
            if kind == "full" and s != shape:
                raise ValueError(f"{self.name}: operand {i} is {s}, the problem is {shape}")
            if kind in ("full", "per_col"):
                C.row_stride(s, a.stride(), f"{self.name}: operand {i}")
            if dt is i8 or (dt is i32 and kind != "per_row"):
                raise TypeError(f"{self.name}: operand {i} is {kind} {dt.name}; a rowwise "
                                f"operand is a float tensor, or a per-row int32 (a label)")
            ops.append(C.Operand(kind, dt))
        first = next(o.dtype for o in ops if o.kind == "full")
        outs = self.outputs or (C.Operand("full", first),)
        outs = tuple(C.Operand(o.kind, o.dtype or first) for o in outs)
        return tuple(ops), shape, outs

    def _like(self, ops, outs) -> str:
        if self.like:
            return self.like
        nin = sum(o.kind == "full" for o in ops)
        nout = sum(o.kind == "full" for o in outs)
        if nin >= 2 and nout >= 2:
            return "addrmsw"
        if any(o.kind == "full" and o.dtype is i8 for o in outs):
            return "quantize"
        if any(o.kind == "per_col" for o in ops):
            return "rmsw"
        return "rms"

    def plan(self, *args) -> Plan:
        """The plan a call with these operands (tensors or hk.templates.meta)
        runs."""
        ops, shape, outs = self._resolve(args)
        return self._plan(ops, shape, outs)

    def _plan(self, ops, shape, outs) -> Plan:
        rows, cols = C.rows_cols(shape)
        nbytes = next(o.dtype for o in ops if o.kind == "full").bytes
        return plan_for(self.arch, self._like(ops, outs), rows, cols, nbytes, self.override)

    def _kernel(self, ops, outs, p: Plan):
        key = (p.arch, ops, outs, p.warps)
        k = self._kernels.get(key)
        if k is None:
            k = self._kernels[key] = _build(self.fn, ops, outs, p.warps, p.arch,
                                            f"{self.name}_w{p.warps}", self.consts)
        return k

    def _consts(self, p: Plan, consts):
        unknown = set(consts) - set(self.consts)
        if unknown:
            raise TypeError(f"{self.name}: no constexpr {sorted(unknown)}; the hook "
                            f"takes {sorted(self.consts) or 'none'}")
        return {"T_CW": p.span, "T_RUN": p.run, "T_EXACT": int(p.exact),
                "T_STEPS": p.steps, "T_RESERVE": p.reserve, **consts}

    def kernel(self, *args, **consts):
        """(hk.Kernel, constexprs) for these operands; trace/source/build
        need no GPU."""
        ops, shape, outs = self._resolve(args)
        p = self._plan(ops, shape, outs)
        return self._kernel(ops, outs, p), self._consts(p, consts)

    def __call__(self, *args, out=None, **consts):
        key = C.call_key(args, out, consts)
        call = self._calls.get(key)
        if call is None:
            call = self._calls[key] = self._prepare(args, out, consts)
        return call(args, out)

    def _prepare(self, args, out, consts):
        import torch  # noqa: PLC0415

        ops, shape, outs = self._resolve(args)
        p = self._plan(ops, shape, outs)
        rows, cols = p.rows, p.cols
        lead = shape[:-1]
        single = self._single
        shapes_out = [shape if o.kind == "full" else lead for o in outs]
        if out is not None:
            given = out if isinstance(out, (list, tuple)) else (out,)
            if len(given) != len(outs):
                raise TypeError(f"{self.name}: out= has {len(given)} tensor(s), the hook "
                                f"returns {len(outs)}")
            for g, o, s in zip(given, outs, shapes_out):
                if C.numel(tuple(g.shape)) != C.numel(s) or C.hk_dtype(g) is not o.dtype:
                    raise ValueError(f"{self.name}: out is {tuple(g.shape)}/{g.dtype}, "
                                     f"expected {s}/{o.dtype.name}")
                if o.kind == "full":
                    C.row_stride(tuple(g.shape), g.stride(), f"{self.name}: out")
        tdt = [C.torch_dtype(o.dtype) for o in outs]

        if not rows or not cols:
            def empty(args, out):
                got = tuple(out) if isinstance(out, (list, tuple)) else (
                    (out,) if out is not None else tuple(
                        torch.empty(s, dtype=d, device=args[0].device)
                        for s, d in zip(shapes_out, tdt)))
                return got[0] if single else got
            return empty
        k = self._kernel(ops, outs, p)
        launch = k._launcher(**self._consts(p, consts))
        arg_views = [C.view_fn(tuple(a.shape), rows, cols, o.kind) for a, o in zip(args, ops)]
        meta = [(tuple(a.shape), tuple(a.stride()), C.hk_dtype(a).bytes) for a in args]
        given = None if out is None else (out if isinstance(out, (list, tuple)) else (out,))
        out_views = [C.view_fn(tuple(given[j].shape) if given else s, rows, cols,
                               "full" if o.kind == "full" else "per_row_out")
                     for j, (o, s) in enumerate(zip(outs, shapes_out))]
        out_meta = None if given is None else [
            (tuple(o.shape), tuple(o.stride()), C.hk_dtype(o).bytes) for o in given]
        return C.make_call(launch, meta, shapes_out, tdt, arg_views, out_views,
                           out_meta, single)


def _trace(kw, spec):
    fn, ops, outs, warps, name, user_consts, names = spec
    nin = len(ops)
    cw, run, exact = kw["T_CW"], kw["T_RUN"], kw["T_EXACT"]
    steps, reserve = kw["T_STEPS"], kw["T_RESERVE"]
    hook = {k: kw[k] for k in user_consts}
    xs = [kw[n] for n in names[:nin]]
    ys = [kw[n] for n in names[nin:]]
    if reserve:
        hk.lds.reserve(reserve)
    S = hk.RS(fp32, 1, cw, run)
    row = hk.arch.block_idx.x
    w = hk.arch.warp_id() if warps > 1 else 0
    ref = xs[next(i for i, o in enumerate(ops) if o.kind == "full")]
    n = ref.cols
    bounded = not exact
    ctx = _Ctx(S, warps, steps)
    ctx.n = n

    # The per-row operands once: a float as a column vector, an int32 (a
    # label) as a scalar.
    rowvals = {}
    for i, (x, o) in enumerate(zip(xs, ops)):
        if o.kind in ("per_row", "scalar"):
            at = x.at[row if o.kind == "per_row" else 0, 0]
            rowvals[i] = at.load_scalar() if o.dtype is i32 else at.load(hk.col_vec(S))

    def operands(c0):
        vals = []
        for i, (x, o) in enumerate(zip(xs, ops)):
            if o.kind == "full":
                vals.append(x.at[row, c0].load(S, bounded=bounded))
            elif o.kind == "per_col":
                vals.append(x.at[0, c0].load(S, bounded=bounded))
            else:
                vals.append(rowvals[i])
        return vals

    def zeros():
        return [rowvals[i] if i in rowvals else hk.zeros(S) for i in range(nin)]

    def run_hook(vals):
        ctx.count = 0
        with C.tile_reductions_refused(
                "inside a rowwise hook a row's reduction is "
                "hk.templates.row_{kind}(t); t.{kind}(axis=1) would reduce one "
                "warp's share of the row"):
            res = fn(*vals, **hook)
        res = res if isinstance(res, (tuple, list)) else (res,)
        if len(res) != len(ys):
            raise TypeError(f"{name}: the hook returned {len(res)} value(s), the "
                            f"template stores {len(ys)} output(s)")
        return res

    def store(res, c0, full: bool, scalars: bool):
        for j, (y, r, o) in enumerate(zip(ys, res, outs)):
            if o.kind == "full":
                if not isinstance(r, hk.Tile):
                    raise TypeError(
                        f"{name}: output {j} is full and the hook returned {r!r}; "
                        f"a per-row result is declared outputs=(..., "
                        f"hk.templates.per_row(hk.fp32))")
                if full:
                    y.at[row, c0].store(r, bounded=bounded)
            elif scalars:
                if not isinstance(r, Vec):
                    raise TypeError(f"{name}: output {j} is per_row and the hook "
                                    f"returned {r!r}, not a row value")
                # One value a row, the same bits in every lane after the
                # butterfly and the cross-warp combine: uniform.
                y[hk.elem_coord(0, 0, 0, row)].store_scalar(as_vec(r, S, uniform=True))

    _CTX.append(ctx)
    try:
        if not steps:
            c0 = w * cw if warps > 1 else 0
            ctx.tail = None if exact else n - c0
            store(run_hook(operands(c0)), c0, True, True)
            return
        # Multi-pass. Before each pass, a dry run of the hook on zeros finds
        # the next reduction it needs (its ops are dead code); pass k then
        # streams the row and folds reduction k. When the dry run reaches
        # the hook's end, every reduction is known: its per-row results are
        # stored, and a last pass writes the full outputs, if there are any.
        fulls = any(o.kind == "full" for o in outs)
        while True:
            ctx.dry = True
            try:
                res = run_hook(zeros())
            except _Stop:
                res = None
            ctx.dry = False
            if res is not None:
                store(res, 0, False, True)
                if fulls:
                    for s in range(steps):
                        c0 = (s * warps + w) * cw if warps > 1 else s * cw
                        ctx.tail = None if exact else n - c0
                        store(run_hook(operands(c0)), c0, True, False)
                return
            for s in range(steps):
                c0 = (s * warps + w) * cw if warps > 1 else s * cw
                ctx.tail = None if exact else n - c0
                try:
                    run_hook(operands(c0))
                except _Stop:
                    pass
            ctx.finish_pass()
            ctx.k += 1
    finally:
        _CTX.pop()


def _build(fn, ops, outs, warps: int, arch: str, name: str, user_consts):
    """The strip kernel: workgroup i is row i, its warp w the columns
    [w*CW, +CW) (single pass) or [(s*WARPS + w)*CW, +CW) at step s."""
    nin = len(ops)
    names = [f"x{i}" for i in range(nin)] + [f"y{j}" for j in range(len(outs))]
    grid_of = next(n for n, o in zip(names, ops) if o.kind == "full")
    spec = (fn, ops, outs, warps, name, user_consts, names)

    # No control flow of its own: hk's rewrite of `for`/`if`/`while` has
    # nothing to do here, and the schedule below is Python at trace time.
    def body(**kw):
        _trace(kw, spec)

    body.__name__ = name
    body.__signature__ = C.signature(
        [(n, o.dtype) for n, o in zip(names, ops)] +
        [(n, o.dtype) for n, o in zip(names[nin:], outs)],
        {"T_CW": 512, "T_RUN": 8, "T_EXACT": 1, "T_STEPS": 0, "T_RESERVE": 0,
         **user_consts})
    defines = None
    if sum(o.kind == "full" for o in ops) >= 2 and any(o.kind == "full" for o in outs):
        # A sum of two 16-bit values is exact in fp32 and ties often: rounded
        # to nearest even, as torch does (hk.ops.norm's fused add says why).
        defines = {"HK_BF16_ROUND": 2}
    return hk.kernel(body, warps=warps, arch=arch, name=name, defines=defines,
                     grid=lambda p: (getattr(p, grid_of).rows, 1, 1))


def rowwise(fn: Optional[Callable] = None, inputs: Optional[Sequence] = None,
            outputs: Any = None, *, like: Optional[str] = None,
            name: Optional[str] = None, arch: Optional[str] = None,
            plan: Optional[Dict[str, int]] = None):
    """A kernel that reduces each row and maps it, from `fn`, the math of
    one row.

        @hk.templates.rowwise
        def rms_norm(x, w, *, EPS=1e-6):
            return x * (hk.templates.row_mean(x * x) + EPS).rsqrt() * w

    `fn` takes one value per operand: a full or per-column operand as an
    fp32 strip of the row, a per-row float as a column vector, a per-row
    int32 as a scalar. `hk.templates.row_sum / row_max / row_min /
    row_mean` reduce the whole row. It returns a tile (a full output) or a
    column vector (a per-row output), or a tuple. Its parameters with
    defaults are constexprs, overridable per call (`k(x, w, EPS=1e-5)`).

    `inputs` / `outputs` as hk.templates.elementwise's, plus per-row
    outputs: `outputs=(hk.i8, hk.templates.per_row(hk.fp32))`. `like`
    names the hk.ops kernel whose measured plan to take ("rms", "rmsw",
    "addrmsw", "layer", "softmax", "quantize"); left out it follows from
    the operands. `plan={...}` / `k.with_plan(...)` overrides warps,
    chunks, run, steps (a multi-pass stream) and reserve; `k.plan(*args)`
    shows what a call runs."""
    if fn is None:
        return lambda f: Rowwise(f, inputs, outputs, like=like, name=name, arch=arch,
                                 plan=plan)
    return Rowwise(fn, inputs, outputs, like=like, name=name, arch=arch, plan=plan)


__all__ = ["rowwise", "Rowwise", "Plan", "plan_for", "row_sum", "row_max", "row_min",
           "row_mean", "row_length", "LIKE"]
