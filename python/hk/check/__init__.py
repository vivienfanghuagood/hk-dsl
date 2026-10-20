"""hk.check: is the kernel right, how fast is it, what does it cost -- one call.

    import hk, torch
    import torch.nn.functional as F

    @hk.kernel(...)
    def gelu_mul(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]): ...

    r = hk.check(gelu_mul,                                         # what is checked
                 lambda a, b: F.gelu(a, approximate="tanh") * b,   # the torch reference
                 lambda device: [torch.randn(4096, 14336, device=device,
                                             dtype=torch.bfloat16) for _ in range(3)])
    print(r)        # the human summary
    r.json()        # the same as a dict, for an agent

or from a shell (docs/check.md):

    python -m hk.check my_kernel.py:gelu_mul --ref "F.gelu(a, approximate='tanh') * b" \\
        --inputs "bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]"

What is checked can be an `@hk.kernel` (the reference then takes the tensors
the kernel only reads, and returns the ones it writes), an `hk.templates`
product or `hk.ops` function, or any Python function that calls them (it
takes the inputs and returns the output(s), as the reference does).

The steps, each one skipped with a note when the machine cannot do it:

1. **build** -- no GPU. Every kernel instantiation the call launches is
   traced and compiled; its resources (VGPRs, occupancy by registers and by
   LDS, LDS bytes, scratch, spills, the trace's warnings) come from
   `k.build().kernels`. A spilling or failing build is an error, with the
   library's message.
2. **correct** -- a GPU. The outputs against the reference, the reference run
   in fp32 on fp32 copies of the inputs by default, at a tolerance for the
   output's dtype (TOLERANCES): the max and mean error, the mismatch count,
   the first mismatching index. Outputs a raw kernel writes are filled with
   NaN first, so an element it never wrote is a mismatch. torch eager's and
   torch.compile's own error against the same reference is reported beside it.
3. **time** -- a GPU. The kernel against torch eager, torch.compile and any
   extra callables, interleaved in one process: warmup, then rounds that
   rotate the contenders, each a burst of back-to-back calls; the median of
   the rounds, its noise, the per-round ratios' median and a verdict inside a
   noise band. The first call of each, which includes the compile, is timed
   on its own.
"""

from __future__ import annotations

import contextlib
import inspect
import json as _json
import math
import os
import platform
import statistics
import sys
import time as _time
import traceback
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Version of the JSON report's layout. Bumped when a key changes meaning.
SCHEMA = 1

#: Output dtype -> (rtol, atol as a fraction of max|reference|). An element
#: passes when |got - want| <= atol * max|want| + rtol * |want|. About two
#: units in the last place for the 16-bit types: a kernel that computes in
#: fp32 and rounds once is within one of the fp32 reference, and torch eager,
#: which rounds every intermediate, typically within two.
TOLERANCES: Dict[str, Tuple[float, float]] = {
    "float64": (1e-7, 1e-9),
    "float32": (1e-4, 1e-5),
    "float16": (2e-3, 1e-4),
    "bfloat16": (1.6e-2, 1e-3),
    "float8_e4m3fn": (1.25e-1, 1e-2),
    "float8_e4m3fnuz": (1.25e-1, 1e-2),
    "float8_e5m2": (2.5e-1, 1e-2),
    "float8_e5m2fnuz": (2.5e-1, 1e-2),
}

#: Each card's ceilings, for the bandwidth and FLOP rates the summary prints
#: (tools/hk-bench/_rounds.py's CARDS, where each is sourced): DRAM bytes/s,
#: the fastest cache-resident bytes/s measured, the cache's size, bf16 FLOP/s.
CARDS: Dict[str, Dict[str, float]] = {
    "gfx1100": {"dram_bytes_s": 864e9, "cache_bytes_s": 2.43e12,
                "cache_bytes": 96 * 2**20, "bf16_flops": 122.6e12},
    "gfx950": {"dram_bytes_s": 8.0e12, "cache_bytes_s": 9.1e12,
               "cache_bytes": 256 * 2**20, "bf16_flops": 2.52e15},
}

#: A ratio inside this band is a tie whatever the measured noise
#: (docs/agents.md, "a gap under ~2-5%").
BAND_FLOOR = 0.02
#: A band wider than this cannot tell a tie from a real difference.
BAND_CAP = 0.15


# --------------------------------------------------------------- the report


@dataclass
class Report:
    """What `hk.check` found. `str(report)` is the human summary,
    `report.json()` the same as a dict (`SCHEMA`), `report.ok` whether every
    step that ran passed."""

    name: str
    arch: str
    mode: str                       # "gpu" | "build-only" | "trace-only"
    device: Optional[str] = None
    versions: Dict[str, str] = field(default_factory=dict)
    builds: List[Dict[str, Any]] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """"error" (something did not build or run), "fail" (a wrong
        answer), "pass" (correct at every case), or "built" (nothing ran on
        a GPU and everything that could be done without one worked)."""
        if self.errors:
            return "error"
        if any(c.get("correct") and not c["correct"]["ok"] for c in self.cases):
            return "fail"
        if any(c.get("correct") for c in self.cases):
            return "pass"
        return "built"

    @property
    def hints(self) -> List[str]:
        """What to try next, read off the numbers (`hints()`)."""
        return hints(self)

    @property
    def ok(self) -> bool:
        """True for "pass" and "built"."""
        return self.status in ("pass", "built")

    @property
    def exit_code(self) -> int:
        """The CLI's: 0 pass or built, 1 a wrong answer, 2 an error."""
        return {"pass": 0, "built": 0, "fail": 1}.get(self.status, 2)

    def json(self) -> Dict[str, Any]:
        """The report as a dict of JSON types (layout version `SCHEMA`)."""
        return {"schema": SCHEMA, "name": self.name, "status": self.status,
                "arch": self.arch, "mode": self.mode, "device": self.device,
                "versions": self.versions, "builds": self.builds,
                "cases": self.cases, "errors": self.errors, "notes": self.notes,
                "hints": self.hints}

    def dumps(self, **kw) -> str:
        """`json()` as text."""
        return _json.dumps(self.json(), indent=kw.pop("indent", 1),
                           default=_jsonable, **kw)

    def __str__(self) -> str:
        return summary(self)

    def raise_for_status(self) -> "Report":
        """Raise AssertionError unless ok; returns self, for chaining in a
        test: `hk.check(k, ref, make).raise_for_status()`."""
        if not self.ok:
            raise AssertionError(str(self))
        return self


def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return str(x)


# --------------------------------------------------------------- versions


def versions() -> Dict[str, str]:
    """hk's version (and the checkout's commit), torch's, HIP's, Triton's,
    Python's: what a number needs beside it to mean anything."""
    out: Dict[str, str] = {"python": platform.python_version()}
    try:
        import hk  # noqa: PLC0415

        out["hk"] = hk.__version__
        here = os.path.dirname(os.path.abspath(hk.__file__))
        head = os.path.join(here, "..", "..", ".git")
        if os.path.exists(head):
            import subprocess  # noqa: PLC0415

            sha = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if sha:
                out["hk_commit"] = sha
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415

        out["torch"] = torch.__version__
        if getattr(torch.version, "hip", None):
            out["hip"] = str(torch.version.hip)
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton  # noqa: PLC0415

        out["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------- what launches


class _Recorder:
    """Which kernels a call launches, with the constexprs it passes: every
    `Kernel.__call__` while active. With `launch=False` nothing is launched,
    which is how a function's kernels are found on a machine with no GPU
    (its inputs then live on the CPU, and the outputs are garbage)."""

    def __init__(self, launch: bool = True):
        self.launch = launch
        self.calls: List[Tuple[Any, Dict[str, Any]]] = []
        self._seen: set = set()

    @contextlib.contextmanager
    def active(self):
        from ..dsl.kernel import Kernel  # noqa: PLC0415

        orig = Kernel.__call__
        rec = self

        def call(k, *tensors, **overrides):
            try:
                key = (id(k), tuple(sorted(overrides.items())))
            except TypeError:
                key = (id(k), repr(sorted(overrides.items())))
            if key not in rec._seen:
                rec._seen.add(key)
                rec.calls.append((k, dict(overrides)))
            if rec.launch:
                return orig(k, *tensors, **overrides)
            return None

        Kernel.__call__ = call
        try:
            yield self
        finally:
            Kernel.__call__ = orig


def _is_kernel(fn) -> bool:
    from ..dsl.kernel import Kernel  # noqa: PLC0415

    return isinstance(fn, Kernel)


def _resources(k, overrides: Dict[str, Any], do_build: bool) -> Dict[str, Any]:
    """One build row: the trace's facts, and with hipcc the compiler's."""
    from ..target import get_target  # noqa: PLC0415

    row: Dict[str, Any] = {"kernel": k.name, "consts": dict(overrides)}
    t0 = _time.perf_counter()
    tr = k.trace(**overrides)
    row.update(arch=tr.arch, warps=tr.warps, threads=tr.threads,
               lds_bytes=tr.lds_bytes, warnings=list(tr.warnings),
               trace_s=round(_time.perf_counter() - t0, 3))
    if not do_build:
        return row
    built_before = k._key(k._resolve_consts(overrides)) in k._build_cache
    t0 = _time.perf_counter()
    b = k.build(**overrides)
    if built_before:
        # Built by the call that launched it: its time is in first_call_s.
        row["build_s"] = None
    else:
        row["build_s"] = round(_time.perf_counter() - t0, 3)
    #: True when hipcc did not run: the .so came from the disk cache.
    row["cached"] = bool(b.cached)
    row["source"] = str(b.source_path)
    target = get_target(tr.arch)
    insts = []
    for r in b.kernels:
        charged = target.charged_vgprs(r.vgpr, r.agpr)
        occ_reg = r.occ_by_reg if r.occ_by_reg is not None else r.occupancy
        occ_lds = target.occupancy_by_lds(r.lds or tr.lds_bytes, tr.warps)
        insts.append({
            "name": r.name, "vgpr": r.vgpr, "agpr": r.agpr, "sgpr": r.sgpr,
            "vgpr_charged": target.vgpr_rounded(charged) if charged else 0,
            "vgpr_headroom": r.vgpr_headroom, "lds": r.lds,
            "scratch": r.scratch, "spill": r.vgpr_spill + r.sgpr_spill,
            "occupancy_by_regs": occ_reg, "occupancy_by_lds": occ_lds,
            "occupancy": min(x for x in (occ_reg, occ_lds) if x is not None)
            if occ_reg is not None else occ_lds,
            "max_waves_per_simd": target.max_waves_per_simd,
        })
    row["instantiations"] = insts
    return row


# --------------------------------------------------------------- inputs


def _call_maker(make, device: str, shape=None):
    """Call an input generator with what it asks for: `device` and/or
    `shape`, by name or as the first positional parameter."""
    if not callable(make):
        return make
    try:
        params = inspect.signature(make).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if "device" in params:
        kw["device"] = device
    if "shape" in params and shape is not None:
        kw["shape"] = shape
    if kw:
        return make(**kw)
    positional = [p for p in params.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
    if positional:
        return make(device)
    return make()


def _as_tuple(x) -> tuple:
    if x is None:
        return ()
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def _is_tensor(x) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "dtype") \
        and hasattr(x, "shape")


def describe(args) -> str:
    """"bf16[4096,14336], fp32[64]": what a case's inputs are."""
    out = []
    for a in args:
        if _is_tensor(a):
            dt = str(a.dtype).replace("torch.", "")
            dt = _SHORT.get(dt, dt)
            s = f"{dt}[{','.join(str(d) for d in a.shape)}]"
            if a.dim() and not a.is_contiguous():
                s += f" strides({','.join(str(d) for d in a.stride())})"
            out.append(s)
        else:
            out.append(repr(a))
    return ", ".join(out)


_SHORT = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
          "float64": "fp64", "int32": "i32", "int64": "i64", "int8": "i8",
          "uint8": "u8", "int16": "i16"}


def _clone(args, *, fp32: bool = False):
    import torch  # noqa: PLC0415

    out = []
    for a in args:
        if _is_tensor(a):
            if fp32 and a.is_floating_point() and a.dtype != torch.float64:
                out.append(a.detach().float())
            else:
                out.append(a.detach().clone())
        else:
            out.append(a)
    return out


# --------------------------------------------------------------- correctness


def tolerance(dtype, rtol=None, atol=None, scale: float = 1.0) -> Tuple[float, float]:
    """(rtol, absolute atol) for comparing an output of `dtype` against a
    reference whose largest magnitude is `scale`."""
    name = str(dtype).replace("torch.", "")
    d_rtol, d_atol = TOLERANCES.get(name, (0.0, 0.0))
    return (d_rtol if rtol is None else rtol,
            d_atol * scale if atol is None else atol)


def compare(got, want, *, rtol=None, atol=None, exact=False) -> Dict[str, Any]:
    """One output against its reference. `want` may be any dtype (the fp32
    reference); the tolerance is for `got`'s dtype."""
    import torch  # noqa: PLC0415

    res: Dict[str, Any] = {"dtype": str(got.dtype).replace("torch.", ""),
                           "shape": list(got.shape)}
    if tuple(got.shape) != tuple(want.shape):
        res.update(ok=False, why=f"shape {tuple(got.shape)}, the reference's "
                                 f"{tuple(want.shape)}")
        return res
    g = got.detach().to(torch.float64) if got.is_floating_point() or got.is_complex() \
        else got.detach().to(torch.int64)
    w = want.detach().to(g.dtype)
    n = g.numel()
    res["numel"] = n
    if n == 0:
        res.update(ok=True, max_abs=0.0, mean_abs=0.0, max_rel=0.0, mismatches=0)
        return res
    scale = float(w.abs().max()) if n else 0.0
    if not math.isfinite(scale):
        finite = w[torch.isfinite(w)]
        scale = float(finite.abs().max()) if finite.numel() else 0.0
    if exact or not (got.is_floating_point()):
        rt, at = 0.0, 0.0
    else:
        rt, at = tolerance(got.dtype, rtol, atol, scale)
    diff = (g - w).abs()
    both_nan = torch.isnan(g) & torch.isnan(w)
    same_inf = torch.isinf(g) & torch.isinf(w) & (g == w)
    diff = torch.where(both_nan | same_inf, torch.zeros_like(diff), diff)
    bad = ~(diff <= at + rt * w.abs())          # NaN compares False: a mismatch
    bad &= ~(both_nan | same_inf)
    nbad = int(bad.sum())
    finite = diff[torch.isfinite(diff)]
    res.update(
        ok=nbad == 0, rtol=rt, atol=at, scale=scale,
        max_abs=float(finite.max()) if finite.numel() else float("nan"),
        mean_abs=float(finite.mean()) if finite.numel() else float("nan"),
        max_rel=(float(finite.max()) / scale) if finite.numel() and scale else 0.0,
        mismatches=nbad,
        nan=int(torch.isnan(g).sum()) - int(both_nan.sum()),
    )
    if nbad:
        flat = int(bad.flatten().to(torch.uint8).argmax())
        idx = []
        for d in reversed(got.shape):
            idx.append(flat % d)
            flat //= d
        idx = idx[::-1]
        at_idx = tuple(idx)
        res["first_mismatch"] = {"index": idx, "got": float(g[at_idx]),
                                 "want": float(w[at_idx])}
        res["why"] = (f"{nbad} of {n} elements outside |got - want| <= "
                      f"{at:.3g} + {rt:.3g} * |want|")
    return res


def _outputs_of(res) -> List[Any]:
    return [x for x in _as_tuple(res) if _is_tensor(x)]


# --------------------------------------------------------------- timing


def _sync():
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()


def burn_in(seconds: float, device: str = "cuda") -> None:
    """Sustained memory traffic until the clocks stop climbing
    (tools/hk-bench/_timing.py says why: an idle card's first second of
    work measures the ramp)."""
    if seconds <= 0:
        return
    import torch  # noqa: PLC0415

    a = torch.empty(4096, 8192, device=device, dtype=torch.bfloat16).fill_(1)
    b = torch.empty_like(a)
    end = _time.time() + seconds
    while _time.time() < end:
        for _ in range(20):
            b.copy_(a)
        _sync()
    del a, b


def _event_ms(fn, n: int) -> float:
    import torch  # noqa: PLC0415

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n


def interleave(thunks: Dict[str, Callable[[], Any]], *, rounds: int = 15,
               warmup: int = 3, batch_ms: float = 1.0) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    """{name: thunk} -> ({name: ms a call, one per round}, {name: calls a
    burst}). Every round runs every contender once, the order rotated round
    by round; a contender's burst is enough back-to-back calls to fill
    `batch_ms`, so a microsecond kernel is not timed against the launch gap
    and the event resolution."""
    for _ in range(warmup):
        for fn in thunks.values():
            fn()
    _sync()
    calls = {}
    for k, fn in thunks.items():
        t = min(_event_ms(fn, 1) for _ in range(3))
        calls[k] = int(max(1, min(2000, math.ceil(batch_ms / max(t, 1e-4)))))
    names = list(thunks)
    times: Dict[str, List[float]] = {k: [] for k in names}
    for r in range(rounds):
        sh = r % len(names)
        for k in names[sh:] + names[:sh]:
            times[k].append(_event_ms(thunks[k], calls[k]))
    return times, calls


def _spread(xs: Sequence[float]) -> float:
    """Median absolute deviation over the median: the rounds' noise."""
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    if not m:
        return 0.0
    return statistics.median(abs(x - m) for x in xs) / m


def verdict(ratio: float, band: float) -> str:
    """win / loss past the band, tie inside it, noise when the band is too
    wide to tell (docs/agents.md, `ab`'s rules)."""
    if not math.isfinite(ratio):
        return "noise"
    if band > BAND_CAP:
        return "noise"
    if ratio > 1 + band:
        return "win"
    if ratio < 1 - band:
        return "loss"
    return "tie"


def _ratios(times: Dict[str, List[float]], subject: str) -> Dict[str, Dict[str, Any]]:
    """Each contender against the subject: the median of the per-round
    ratios (contender / subject: above 1 the subject is faster), and a band
    of max(2%, the ratios' spread, the two contenders' summed spread)."""
    out = {}
    mine = times[subject]
    for k, ts in times.items():
        if k == subject:
            continue
        rs = [b / a for a, b in zip(mine, ts) if a > 0]
        if not rs:
            continue
        x = statistics.median(rs)
        band = max(BAND_FLOOR, _spread(rs), _spread(mine) + _spread(ts))
        out[k] = {"x": round(x, 4), "band": round(band, 4), "verdict": verdict(x, band)}
    return out


# --------------------------------------------------------------- check


def _gpu_state(want: Optional[bool]) -> Tuple[bool, Optional[str], Optional[str]]:
    """(usable, device name, device arch)."""
    if want is False:
        return False, None, None
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False, None, None
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        return True, p.name, (getattr(p, "gcnArchName", "") or "").split(":")[0] or None
    except Exception:  # noqa: BLE001
        return False, None, None


def _err(report: Report, stage: str, e: BaseException) -> None:
    msg = str(e).strip() or type(e).__name__
    report.errors.append({"stage": stage, "type": type(e).__name__, "message": msg,
                          "where": _where(e)})


def _where(e: BaseException) -> str:
    """The innermost frame outside hk.check, as file:line."""
    tb = traceback.extract_tb(e.__traceback__)
    for fr in reversed(tb):
        if os.sep + "check" + os.sep not in fr.filename:
            return f"{fr.filename}:{fr.lineno}"
    return ""


def _name_of(fn) -> str:
    return getattr(fn, "name", None) or getattr(fn, "__qualname__", None) \
        or getattr(fn, "__name__", None) or type(fn).__name__


def _kernel_outputs(k, overrides, n_args: int) -> List[int]:
    names = [p.name for p in k.trace(**overrides).tensors()]
    written = set(k.trace(**overrides).written_tensors())
    return [i for i, nm in enumerate(names) if nm in written and i < n_args]


def _fill_nan(args, idx):
    import torch  # noqa: PLC0415

    for i in idx:
        a = args[i]
        if _is_tensor(a) and a.is_floating_point():
            a.fill_(float("nan"))
        elif _is_tensor(a):
            a.fill_(torch.iinfo(a.dtype).min if a.dtype != torch.bool else False)


def check(fn, reference: Optional[Callable] = None, inputs=None, *,
          cases: Optional[Sequence] = None,
          consts: Optional[Dict[str, Any]] = None,
          outputs: Optional[Sequence] = None,
          baselines: Sequence[str] = ("eager", "compile"),
          extra: Optional[Dict[str, Callable]] = None,
          rtol: Optional[float] = None, atol: Optional[float] = None,
          exact: bool = False, ref_precision: str = "fp32",
          time: bool = True, rounds: int = 15, warmup: int = 3,
          batch_ms: float = 1.0, burn: float = 0.5,
          compile_mode: Optional[str] = None,
          bytes: Optional[float] = None, flops: Optional[float] = None,  # noqa: A002
          name: Optional[str] = None, gpu: Optional[bool] = None,
          build: bool = True, seed: int = 0) -> Report:
    """Check `fn` against `reference` on `inputs`; see the module docstring
    and docs/check.md.

    fn         an `@hk.kernel`, or a callable that launches hk kernels
    reference  the torch reference: called with the inputs `fn` reads (for a
               raw kernel, the tensors it does not write) and returning the
               output(s), in the order `fn` returns or writes them
    inputs     a sequence of tensors, or a generator called as
               `inputs(device)` (or `inputs(device=..., shape=...)`)
    cases      several `inputs`, one case each (shapes to sweep)
    consts     constexprs for a raw kernel, passed as `k(*tensors, **consts)`
    outputs    which arguments are outputs (indices or a raw kernel's
               parameter names): for a function that writes into an
               argument instead of returning. A raw kernel's default is the
               tensors it stores into (`TracedKernel.written_tensors`)
    baselines  which of "eager" (the reference itself) and "compile"
               (`torch.compile(reference, mode=compile_mode)`) to time
    extra      more contenders, {name: callable taking the reference's
               arguments}; checked against the reference like `fn`
    rtol, atol override TOLERANCES (atol is then absolute); `exact` asks
               for bit equality
    ref_precision  "fp32" runs the reference on fp32 copies of the 16-bit
               inputs (the default); "native" on the inputs as they are
    time, rounds, warmup, batch_ms, burn   the timing (burn: seconds of
               memory traffic first, to settle the clocks)
    bytes, flops   work a call does, for the rates (default bytes: every
               input and output once)
    gpu        False forces build-only; None uses a GPU when one is there
    build      False skips hipcc (trace only)
    """
    from .. import target as _target  # noqa: PLC0415

    consts = dict(consts or {})
    arch = _target.current_arch()
    use_gpu, dev_name, dev_arch = _gpu_state(gpu)
    if use_gpu and dev_arch and dev_arch != arch:
        use_gpu = False
        mismatch = (f"the GPU is {dev_arch} and kernels build for {arch} "
                    f"(HK_ARCH): built only")
    else:
        mismatch = None
    mode = "gpu" if use_gpu else ("build-only" if build else "trace-only")
    report = Report(name=name or _name_of(fn), arch=arch, mode=mode,
                    device=dev_name if use_gpu else None, versions=versions())
    if mismatch:
        report.notes.append(mismatch)
    elif not use_gpu and gpu is not False:
        report.notes.append("no GPU: traced and built only; correctness and "
                            "timing need one")
    if build:
        try:
            from ..runtime.compile import hipcc  # noqa: PLC0415

            hipcc()
        except Exception as e:  # noqa: BLE001
            build = False
            report.mode = mode = "trace-only" if not use_gpu else mode
            report.notes.append(f"no hipcc ({e}): traced only")

    all_cases = list(cases) if cases is not None else [inputs]
    is_kernel = _is_kernel(fn)
    seen_builds: Dict[Tuple, Dict[str, Any]] = {}

    def add_builds(calls):
        for k, ov in calls:
            try:
                key = (id(k), tuple(sorted(ov.items())))
            except TypeError:
                key = (id(k), repr(ov))
            if key in seen_builds:
                continue
            try:
                k.trace(**ov)
            except Exception as e:  # noqa: BLE001
                _err(report, "trace", e)
                seen_builds[key] = {}
                continue
            try:
                row = _resources(k, ov, build)
            except Exception as e:  # noqa: BLE001
                _err(report, "build", e)
                seen_builds[key] = {}
                continue
            seen_builds[key] = row
            report.builds.append(row)

    # 1. build: a raw kernel needs only its constexprs; a function is run
    # once with launching off to find what it launches.
    built_s = 0.0
    if is_kernel:
        t0 = _time.perf_counter()
        add_builds([(fn, consts)])
        built_s = _time.perf_counter() - t0
    elif not use_gpu:
        for ci, make in enumerate(all_cases):
            if make is None:
                continue
            rec = _Recorder(launch=False)
            try:
                args = _as_tuple(_call_maker(make, "cpu"))
                with rec.active():
                    fn(*args)
            except Exception as e:  # noqa: BLE001
                report.notes.append(
                    f"case {ci}: could not find the kernels without a GPU "
                    f"({type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}); "
                    f"check an @hk.kernel directly to build it here")
            add_builds(rec.calls)
    if report.errors or not use_gpu:
        return report

    # 2-3. on the GPU, case by case.
    import torch  # noqa: PLC0415

    if reference is None:
        report.notes.append("no reference: nothing to check correctness "
                            "against; timing only")
    burned = False
    for ci, make in enumerate(all_cases):
        case: Dict[str, Any] = {"case": ci}
        report.cases.append(case)
        try:
            torch.manual_seed(seed + ci)
            args = list(_as_tuple(_call_maker(make, "cuda")))
        except Exception as e:  # noqa: BLE001
            _err(report, "inputs", e)
            continue
        case["inputs"] = describe(args)
        # Which arguments are outputs.
        out_idx: List[int] = []
        if outputs is not None:
            names = []
            if is_kernel:
                names = [p.name for p in fn.trace(**consts).tensors()]
            for o in outputs:
                out_idx.append(names.index(o) if isinstance(o, str) else int(o))
        elif is_kernel:
            try:
                out_idx = _kernel_outputs(fn, consts, len(args))
            except Exception as e:  # noqa: BLE001
                _err(report, "trace", e)
                continue
        ref_args = [a for i, a in enumerate(args) if i not in out_idx]

        # The reference, in fp32 unless asked otherwise.
        want = None
        if reference is not None:
            try:
                with torch.no_grad():
                    want = _outputs_of(reference(*_clone(ref_args, fp32=ref_precision == "fp32")))
                _sync()
            except Exception as e:  # noqa: BLE001
                _err(report, "reference", e)
                continue

        # The subject: first call timed (compile included), launches recorded.
        mine = _clone(args)
        _fill_nan(mine, out_idx)
        rec = _Recorder(launch=True)
        try:
            _sync()
            t0 = _time.perf_counter()
            with rec.active():
                res = fn(*mine, **consts)
            _sync()
            # A raw kernel was traced and built in step 1: its first call is
            # that and this launch.
            first = {"hk": round(_time.perf_counter() - t0 + (built_s if ci == 0 else 0), 4)}
        except Exception as e:  # noqa: BLE001
            _err(report, "run", e)
            add_builds(rec.calls)
            continue
        add_builds(rec.calls)
        if report.errors:
            continue
        got = [mine[i] for i in out_idx] if out_idx else _outputs_of(res)
        case["first_call_s"] = first

        if want is not None:
            if len(got) != len(want):
                case["correct"] = {"ok": False, "why":
                                   f"{len(got)} output(s), the reference returned {len(want)}",
                                   "outputs": []}
            else:
                outs = [compare(g, w, rtol=rtol, atol=atol, exact=exact)
                        for g, w in zip(got, want)]
                case["correct"] = {"ok": all(o["ok"] for o in outs), "outputs": outs}

        # Baselines: built, checked, then timed with the subject.
        contenders: Dict[str, Callable] = {}
        if is_kernel:
            contenders["hk"] = lambda f=fn, a=args, c=consts: f(*a, **c)
        else:
            contenders["hk"] = lambda f=fn, a=args, c=consts: f(*a, **c)
        others: Dict[str, Callable] = {}
        if reference is not None and "eager" in baselines:
            others["eager"] = reference
        if reference is not None and "compile" in baselines:
            try:
                others["compile"] = torch.compile(reference, mode=compile_mode) \
                    if compile_mode else torch.compile(reference)
            except Exception as e:  # noqa: BLE001
                report.notes.append(f"torch.compile unavailable: {e}")
        for k, f in (extra or {}).items():
            others[k] = f
        errs: Dict[str, Any] = {}
        for k, f in list(others.items()):
            try:
                a = _clone(ref_args)
                _sync()
                t0 = _time.perf_counter()
                with torch.no_grad():
                    out = _outputs_of(f(*a))
                _sync()
                first[k] = round(_time.perf_counter() - t0, 4)
                if want is not None and len(out) == len(want):
                    cmp = [compare(g, w, rtol=rtol, atol=atol, exact=exact)
                           for g, w in zip(out, want)]
                    errs[k] = {"max_abs": max(c.get("max_abs", 0.0) for c in cmp),
                               "ok": all(c["ok"] for c in cmp)}
                contenders[k] = (lambda f=f, a=a: f(*a))
            except Exception as e:  # noqa: BLE001
                report.notes.append(f"case {ci}: baseline {k} failed: "
                                    f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
        if errs and "correct" in case:
            case["correct"]["baselines"] = errs

        if not time:
            continue
        if not burned:
            burn_in(burn)
            burned = True
        try:
            with torch.no_grad():
                times, calls = interleave(contenders, rounds=rounds, warmup=warmup,
                                          batch_ms=batch_ms)
        except Exception as e:  # noqa: BLE001
            _err(report, "time", e)
            continue
        nbytes = bytes
        if nbytes is None:
            nbytes = sum(a.numel() * a.element_size() for a in args if _is_tensor(a))
            if not out_idx:
                nbytes += sum(g.numel() * g.element_size() for g in got)
        tm: Dict[str, Any] = {"rounds": rounds, "calls_per_round": calls,
                              "bytes": nbytes, "flops": flops, "ms": {}}
        card = CARDS.get(arch, {})
        for k, ts in times.items():
            med = statistics.median(ts)
            row = {"median": round(med, 5), "min": round(min(ts), 5),
                   "noise": round(_spread(ts), 4)}
            if nbytes and med > 0:
                row["gb_s"] = round(nbytes / med / 1e6, 1)
            if flops and med > 0:
                row["tflop_s"] = round(flops / med / 1e9, 2)
            tm["ms"][k] = row
        tm["vs"] = _ratios(times, "hk")
        hk_ms = tm["ms"]["hk"]["median"]
        if card and nbytes and hk_ms > 0:
            rate = nbytes / (hk_ms * 1e-3)
            in_cache = nbytes <= card["cache_bytes"]
            ceiling = card["cache_bytes_s"] if in_cache else card["dram_bytes_s"]
            tm["of_dram_peak"] = round(rate / card["dram_bytes_s"], 3)
            if rate > ceiling * 1.15:
                report.notes.append(
                    f"case {ci}: hk moved {rate / 1e12:.2f} TB/s, past this card's "
                    f"{'cache' if in_cache else 'DRAM'} ceiling: the byte count "
                    f"overstates the traffic, or the kernel skipped work")
        case["time"] = tm
    return report


# --------------------------------------------------------------- hints


def hints(r: Report) -> List[str]:
    """Rules of thumb over a report, each naming the number it read and the
    page that says more. Heuristics, not verdicts: the timing is the verdict."""
    out: List[str] = []
    # Occupancy is worth a hint where nothing was timed, or where the kernel
    # lost: a GEMM's LDS-bound occupancy is its design, not its problem.
    lost = any((c.get("time") or {}).get("vs", {}).get("compile", {}).get("verdict")
               == "loss" for c in r.cases)
    timed = any(c.get("time") for c in r.cases)
    for b in r.builds:
        for i in b.get("instantiations", []):
            occ, top = i.get("occupancy"), i.get("max_waves_per_simd")
            if occ and top and occ * 2 <= top and (lost or not timed):
                by_lds = i.get("occupancy_by_lds") or top
                by_regs = i.get("occupancy_by_regs") or top
                binding = "LDS" if by_lds < by_regs else "registers"
                out.append(f"{b['kernel']}: occupancy {occ}/{top}, bound by {binding}; "
                           f"a memory-bound kernel wants more waves (fewer live tiles, "
                           f"less LDS) -- docs/runtime.md#the-resource-gates")
    for c in r.cases:
        t = c.get("time")
        if not t:
            continue
        comp = t.get("vs", {}).get("compile")
        peak = t.get("of_dram_peak")
        card = CARDS.get(r.arch, {})
        big = card and t.get("bytes") and t["bytes"] > card.get("cache_bytes", 0)
        if comp and comp["verdict"] == "loss":
            msg = (f"case {c['case']}: {1 / comp['x']:.2f}x slower than torch.compile "
                   f"({c.get('inputs', '')})")
            if peak is not None and big and peak < 0.6:
                msg += (f"; at {peak * 100:.0f}% of the DRAM peak a streaming kernel "
                        f"is not moving bytes fast enough: start from hk.templates "
                        f"(docs/templates/), or put rows on register strips "
                        f"(hk.RS, docs/language.md#register-strips) with 16-byte runs")
            out.append(msg)
        fc = c.get("first_call_s", {}).get("hk")
        if fc and fc > 20:
            out.append(f"case {c['case']}: the first call took {fc:.0f} s (hipcc); "
                       f"`python -m hk.runtime.warm` builds ahead of time "
                       f"(docs/runtime.md#warming-the-cache)")
    return out


# --------------------------------------------------------------- summary


def _fmt_ms(ms: float) -> str:
    return f"{ms * 1e3:.1f} us" if ms < 1 else f"{ms:.3f} ms"


def summary(r: Report) -> str:
    """The human report: what was built, what was checked, how fast."""
    v = r.versions
    head = (f"hk.check {r.name}  {r.arch}"
            + (f" ({r.device})" if r.device else "")
            + f"  [{r.mode}]  hk {v.get('hk', '?')}"
            + (f"+{v['hk_commit']}" if v.get("hk_commit") else "")
            + (f"  torch {v['torch']}" if v.get("torch") else ""))
    lines = [head]
    for b in r.builds:
        c = ", ".join(f"{k}={x}" for k, x in b.get("consts", {}).items())
        title = f"{b['kernel']}" + (f"({c})" if c else "")
        lines.append(f"  build  {title}: warps={b.get('warps')} lds={b.get('lds_bytes')} B"
                     + (f"  {b['build_s']:.1f} s{' (from the disk cache)' if b.get('cached') else ''}"
                        if b.get("build_s") is not None else
                        ("  built by the first call" + (" (from the disk cache)"
                                                        if b.get("cached") else ""))
                        if "build_s" in b else ""))
        for i in b.get("instantiations", []):
            vg = f"{i['vgpr']}" + (f"+{i['agpr']}a" if i.get("agpr") else "")
            lines.append(
                f"         vgpr={vg} (charged {i['vgpr_charged']}, headroom "
                f"{i['vgpr_headroom']})  occupancy={i['occupancy']}/{i['max_waves_per_simd']} "
                f"(regs {i['occupancy_by_regs']}, lds {i['occupancy_by_lds']})  "
                f"scratch={i['scratch']} spill={i['spill']}")
        for w in b.get("warnings", []):
            lines.append(f"         warning: {w}")
    for c in r.cases:
        lines.append(f"  case {c['case']}: {c.get('inputs', '?')}")
        cor = c.get("correct")
        if cor:
            for j, o in enumerate(cor.get("outputs", [])):
                tag = "ok  " if o["ok"] else "FAIL"
                if "max_abs" in o:
                    s = (f"    correct {tag} out{j} {o['dtype']}: max |err| {o['max_abs']:.3g} "
                         f"({o['max_rel']:.2g} of max|ref|), mean {o['mean_abs']:.3g}; "
                         f"allowed {o['atol']:.3g} + {o['rtol']:.3g}*|ref|")
                else:
                    s = f"    correct {tag} out{j}: {o.get('why')}"
                lines.append(s)
                if not o["ok"] and o.get("first_mismatch"):
                    fm = o["first_mismatch"]
                    lines.append(f"            {o['mismatches']} mismatches; first at "
                                 f"{tuple(fm['index'])}: got {fm['got']:.6g}, want "
                                 f"{fm['want']:.6g}")
            if not cor.get("outputs") and cor.get("why"):
                lines.append(f"    correct FAIL: {cor['why']}")
            for k, e in cor.get("baselines", {}).items():
                lines.append(f"            {k}'s own max |err| {e['max_abs']:.3g}"
                             + ("" if e["ok"] else " (outside the tolerance too)"))
        t = c.get("time")
        if t:
            parts = []
            for k, m in t["ms"].items():
                s = f"{k} {_fmt_ms(m['median'])}"
                if m.get("noise"):
                    s += f" ±{m['noise'] * 100:.1f}%"
                parts.append(s)
            lines.append("    time    " + "   ".join(parts))
            vs = "   ".join(f"{x['x']:.2f}x {k} ({x['verdict']}, band {x['band'] * 100:.0f}%)"
                           for k, x in t["vs"].items())
            if vs:
                lines.append("    speed   hk is " + vs)
            hk_m = t["ms"]["hk"]
            if hk_m.get("gb_s"):
                peak = (f" = {t['of_dram_peak'] * 100:.0f}% of DRAM peak"
                        if t.get("of_dram_peak") is not None else "")
                lines.append(f"    rate    hk {hk_m['gb_s']:.0f} GB/s{peak} "
                             f"(inputs + outputs once)")
        fc = c.get("first_call_s")
        if fc:
            lines.append("    first   " + "   ".join(f"{k} {s:.2f} s" for k, s in fc.items())
                         + "  (first call: trace, build or cache load, launch)")
    for e in r.errors:
        lines.append(f"  ERROR [{e['stage']}] {e['type']}: {e['message']}"
                     + (f"\n         at {e['where']}" if e.get("where") else ""))
    for n in r.notes:
        lines.append(f"  note: {n}")
    for h in r.hints:
        lines.append(f"  hint: {h}")
    lines.append(r.status.upper())
    return "\n".join(lines)


# --------------------------------------------------------------- hk.check(...)


class _CallableModule(types.ModuleType):
    """So that `hk.check(fn, ref, inputs)` is the call, while `hk.check.compare`,
    `hk.check.TOLERANCES` and `python -m hk.check` stay where a module's are."""

    def __call__(self, *args, **kwargs) -> Report:
        return check(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["check", "Report", "compare", "tolerance", "interleave", "verdict",
           "summary", "versions", "burn_in", "describe", "TOLERANCES", "CARDS",
           "SCHEMA"]
