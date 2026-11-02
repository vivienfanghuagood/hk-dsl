"""What every engine patch here has in common, written once.

A patch puts a generated kernel under a call the engine already makes. The
two so far -- vLLM's linear layers (`vllm_gemm`) and its attention backend
(`vllm_backend`) -- each wrote the same parts by hand:

* an opaque torch custom op with a fake implementation, so that under
  torch.compile the shape-dependent choice of kernel is not a guard and a
  recompile (`Op`);
* a route from shape to kernel, which was a rule transcribed from a
  measurement (`RouteTable`, which *is* the measurement);
* a fallback to what the engine would have run, and counters of which way
  calls went;
* a CUDA-graph contract, which was a comment beside a class attribute.

The contract is now checked. A patch declares `graph="safe"` -- every launch
is on torch's stream and depends on nothing but shapes, so a replay repeats
it -- or `graph="never"` with the reason, for a patch that plans on host
metadata a replay would not re-read. Under capture a "never" patch raises
`GraphCaptureError` with its reason instead of being frozen at the capture
batch's values, and a "safe" patch runs with torch's sync debug mode set to
error, so a host sync inside a wrapper -- `.item()`, `.tolist()`, a blocking
copy -- raises at the line that does it, naming the patch, instead of failing
the capture somewhere else or, worse, succeeding.
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..runtime import hostpath as _hostpath

GRAPH_CONTRACTS = ("safe", "never")

#: Every patch constructed, by name, so a status line can report them all.
PATCHES: Dict[str, "Patch"] = {}


class GraphCaptureError(RuntimeError):
    """A patch did, under CUDA-graph capture, something a replay cannot repeat."""


#: torch's capture query once a GPU is known to be there, else a False stub.
#: Every patched call asks twice (`guard`, `count`), and torch.cuda.is_available
#: re-reads an environment variable each time -- 3.5 us a call of the 32 a
#: routed int4 layer spent in Python on the W7900D.
_CAPTURING: List[Callable[[], bool]] = []


def capturing() -> bool:
    """Whether the current stream is capturing a CUDA graph. No sync."""
    if _CAPTURING and _hostpath.ENABLED:
        return _CAPTURING[0]()
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    if not _hostpath.ENABLED:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if not torch.cuda.is_available():
        return False                    # asked again next time: a GPU may yet appear
    _CAPTURING.append(torch.cuda.is_current_stream_capturing)
    return _CAPTURING[0]()


@contextlib.contextmanager
def no_host_sync(what: str):
    """Raise `GraphCaptureError` naming `what` at any host sync inside.

    torch's sync debug mode sees the syncs torch itself makes -- `.item()`,
    `.tolist()`, a blocking copy, `synchronize()` -- and not a raw HIP call
    inside an extension.
    """
    import warnings  # noqa: PLC0415

    import torch  # noqa: PLC0415

    prev = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    except RuntimeError as ex:
        # torch 2.10 reports the sync from its debug mode; on 2.13 the HIP
        # call fails first, "operation not permitted when stream is capturing".
        msg = str(ex)
        if isinstance(ex, GraphCaptureError) or not (
                "synchroniz" in msg or "stream is capturing" in msg):
            raise
        raise GraphCaptureError(
            f"{what} synchronized with the host while a CUDA graph was being "
            f"captured. A replay runs no Python, so whatever the wrapper read "
            f"back is frozen at the capture batch's value. Keep the decision "
            f"on host values the engine already has, or declare the patch "
            f"graph='never'. ({str(ex).splitlines()[0]})") from ex
    finally:
        torch.cuda.set_sync_debug_mode(prev)


class Patch:
    """One place an engine runs a generated kernel: its paths, counted, and
    its CUDA-graph contract, enforced by `guard`.

    The counters are Python, so a CUDA-graph replay -- which runs no Python --
    is not counted. Calls made while capturing are counted apart, in
    `captured`, so a table of where calls went cannot mix the two.
    """

    def __init__(self, name: str, paths: Sequence[str], *, graph: str = "safe",
                 why: str = ""):
        if graph not in GRAPH_CONTRACTS:
            raise ValueError(f"graph contract {graph!r} is not one of {GRAPH_CONTRACTS}")
        if graph != "safe" and not why:
            raise ValueError(f"{name}: a patch that cannot be captured says why")
        self.name = name
        self.paths = tuple(paths)
        self.graph = graph
        self.why = why
        self.stats: Dict[str, int] = {p: 0 for p in self.paths}
        self.captured: Dict[str, int] = {p: 0 for p in self.paths}
        PATCHES[name] = self

    def count(self, path: str) -> None:
        (self.captured if capturing() else self.stats)[path] += 1

    def reset(self) -> None:
        for d in (self.stats, self.captured):
            for p in d:
                d[p] = 0

    def check_capture(self) -> None:
        """Raise if this patch is being captured and has said it must not be."""
        if self.graph == "never" and capturing():
            raise GraphCaptureError(
                f"{self.name} is declared graph='never' and is being captured: "
                f"{self.why}")

    def guard(self, fn: Callable) -> Callable:
        """`fn` under this patch's contract. Free outside capture."""

        @functools.wraps(fn)
        def guarded(*args, **kwargs):
            if not capturing():
                return fn(*args, **kwargs)
            self.check_capture()
            with no_host_sync(self.name):
                return fn(*args, **kwargs)

        return guarded

    def summary(self) -> str:
        def fmt(d):
            return ", ".join(f"{p} {n}" for p, n in d.items())

        cap = sum(self.captured.values())
        return (f"{self.name}: {fmt(self.stats)}"
                + (f"; captured: {fmt(self.captured)}" if cap else ""))


class Op(Patch):
    """A patch that is an opaque torch custom op, `torch.ops.<ns>.<name>`.

    `schema` is the torch.library schema without the namespace; `fake` builds
    the output from shapes alone, which is what torch.compile traces.
    """

    def __init__(self, qualname: str, schema: str, fake: Callable,
                 paths: Sequence[str], **kw):
        super().__init__(qualname, paths, **kw)
        self.ns, self.op_name = qualname.split("::")
        self.schema = schema
        self.fake = fake
        self._lib = None
        self._op = None
        self._direct = None

    @property
    def defined(self) -> bool:
        return self._lib is not None

    def define(self, impl: Callable, dispatch: str = "CUDA") -> "Op":
        """Register `impl`, guarded, and the fake. Idempotent."""
        if self._lib is not None:
            return self
        import torch  # noqa: PLC0415

        lib = torch.library.Library(self.ns, "FRAGMENT")
        lib.define(self.schema)
        guarded = self.guard(impl)
        lib.impl(self.op_name, guarded, dispatch)
        torch.library.register_fake(self.name, self.fake, lib=lib)
        self._lib = lib
        if dispatch == "CUDA":
            self._direct = (guarded, torch.Tensor, torch.compiler.is_compiling)
        return self

    def __call__(self, *args, **kwargs):
        """The op -- or, in eager code, what the op would run: the guarded
        implementation, called without the dispatcher.

        The op is opaque so that torch.compile does not trace into it; eager
        code gains nothing from the dispatcher and pays 7 us a call for it on
        the W7900D, as much as the rest of a decode call's route. So where
        nothing is tracing (torch.compile, or a fake or functional tensor
        subclass in the first argument) and the call is for the GPU, the
        implementation runs directly, under the same capture contract
        (`guard`). HK_HOST_FASTPATH=0: always the op."""
        if _hostpath.ENABLED and self._direct is not None and args:
            guarded, tensor, compiling = self._direct
            x = args[0]
            if type(x) is tensor and x.is_cuda and not compiling():
                return guarded(*args, **kwargs)
        op = self._op
        if op is None or not _hostpath.ENABLED:
            import torch  # noqa: PLC0415

            op = getattr(getattr(torch.ops, self.ns), self.op_name)
            if self._lib is not None:
                self._op = op
        return op(*args, **kwargs)


class RouteTable:
    """Measured winners: for each (N, K), which path each measured M took.

    Written by the tuner (`hk.autotune.routes`, through route_tune.py, `python
    -m hk.integration tune` or vLLM's load), never by hand. A shape's entry is
    a list of runs `[lo, hi, path]`: every measured M in [lo, hi] had `path`
    as its fastest by the tuner's margin. An M inside a run takes its path.
    An M between two runs lies between two measured Ms that disagreed, so it
    takes `fallback` -- the table does not guess which side of a tooth it is
    on. Above the largest measured M it takes the path measured there; below
    the smallest, and for a shape never measured, `fallback`. An entry whose
    grid a budget cut short is `partial`, and its `above` is `fallback`.

    **This machine's records.** `shipped_table` overlays the shapes this
    machine measured on the shipped table, each replacing the shipped entry
    for its (N, K) whole; `origin` says which file each shape came from.

    No buckets. GEMM time is a comb in M, not a curve, and a power-of-two
    bucket either routes the teeth or gives up the gaps between them.

    **Optional paths.** A kernel from a library hk does not depend on --
    FlyDSL's GEMM -- is not in the runs above, which are what every install
    gets. It is an overlay, `ent["prefer"][path]`: a list of `[lo, hi,
    variant]`, each saying that at every measured M in [lo, hi] where the
    library picked `variant` (its tile, say) it beat the runs' answer. `hi`
    None means "and above". `preferred` answers which optional paths an M is
    inside a range of, with the variant measured there; the caller takes one
    only if that library is importable and would pick that same variant at
    this M, and otherwise uses `lookup`. So a table with FlyDSL rows routes
    exactly as one without them wherever FlyDSL is missing or would run
    something that was not measured.
    """

    def __init__(self, shapes: Dict[Tuple[int, int], dict], *, fallback: str = "fallback",
                 meta: Optional[dict] = None, source: str = ""):
        self.fallback = fallback
        self.meta = meta or {}
        self.source = source
        self._runs: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
        self._lows: Dict[Tuple[int, int], List[int]] = {}
        self._above: Dict[Tuple[int, int], str] = {}
        self._prefer: Dict[Tuple[int, int], List[Tuple[str, int, float, str]]] = {}
        #: Where each shape's entry came from: `source`, or an overlay's.
        self.origin: Dict[Tuple[int, int], str] = {}
        #: Shapes whose grid was cut short (a tuning budget ran out): they
        #: answer only up to the largest M measured.
        self.partial: set = set()
        for nk, ent in shapes.items():
            self._add(nk, ent, source)
        #: (N, K) asked for and not in the table -- each is a shape a tuner
        #: run would route, and until then takes the engine's path.
        self.misses: set = set()

    def _add(self, nk: Tuple[int, int], ent: dict, source: str) -> None:
        rs = sorted((int(lo), int(hi), str(p)) for lo, hi, p in ent["runs"])
        for (_, hi, _), (lo2, _, _) in zip(rs, rs[1:]):
            if lo2 <= hi:
                raise ValueError(f"{source}: runs for {nk} overlap at M={lo2}")
        self._runs[nk] = rs
        self._lows[nk] = [r[0] for r in rs]
        self._above[nk] = str(ent.get("above", rs[-1][2] if rs else self.fallback))
        pref = [(str(path), int(lo), float("inf") if hi is None else int(hi), str(var))
                for path, ranges in ent.get("prefer", {}).items()
                for lo, hi, var in ranges]
        self._prefer.pop(nk, None)
        if pref:
            self._prefer[nk] = pref
        self.origin[nk] = source
        (self.partial.add if ent.get("partial") else self.partial.discard)(nk)

    def overlay(self, shapes: Dict[Tuple[int, int], dict], source: str) -> "RouteTable":
        """Replace or add entries -- this machine's records over a shipped
        table -- each shape whole: an overlaid shape keeps nothing of the
        entry it replaced."""
        for nk, ent in shapes.items():
            self._add(nk, ent, source)
        return self

    @staticmethod
    def parse(d: dict) -> Dict[Tuple[int, int], dict]:
        """A table file's shapes, keyed by (N, K)."""
        out = {}
        for key, ent in d.get("shapes", {}).items():
            n, k = (int(v) for v in key.split("x"))
            out[(n, k)] = ent
        return out

    @classmethod
    def from_json(cls, path) -> "RouteTable":
        path = Path(path)
        d = json.loads(path.read_text())
        meta = {k: v for k, v in d.items() if k != "shapes"}
        return cls(cls.parse(d), meta=meta, source=str(path))

    def shapes(self) -> Iterable[Tuple[int, int]]:
        return self._runs.keys()

    def complete(self) -> set:
        """The shapes measured over their whole grid."""
        return set(self._runs) - self.partial

    def from_source(self, source: str) -> List[Tuple[int, int]]:
        return [nk for nk, s in self.origin.items() if s == source]

    def paths(self) -> set:
        return ({p for rs in self._runs.values() for _, _, p in rs} | set(self._above.values())
                | self.optional_paths())

    def optional_paths(self) -> set:
        return {p for pref in self._prefer.values() for p, _, _, _ in pref}

    def preferred(self, m: int, n: int, k: int) -> List[Tuple[str, str]]:
        """`(path, variant)` for every optional path whose measured ranges
        cover M at this shape. Empty for most tables and most Ms."""
        pref = self._prefer.get((n, k))
        if not pref:
            return []
        return [(p, var) for p, lo, hi, var in pref if lo <= m <= hi]

    def routes_any(self, n: int, k: int) -> bool:
        """Whether some M of this (N, K) takes a path other than `fallback`: a run, the path
        above the measured grid, or an optional path's range. False for a shape the table never
        measured."""
        rs = self._runs.get((n, k))
        if self._prefer.get((n, k)):
            return True
        if not rs:
            return False
        return (any(p != self.fallback for _, _, p in rs)
                or self._above[(n, k)] != self.fallback)

    def lookup(self, m: int, n: int, k: int) -> str:
        rs = self._runs.get((n, k))
        if not rs:
            self.misses.add((n, k))
            return self.fallback
        i = bisect.bisect_right(self._lows[(n, k)], m) - 1
        if i < 0:
            return self.fallback
        lo, hi, path = rs[i]
        if m <= hi:
            return path
        return self._above[(n, k)] if i == len(rs) - 1 else self.fallback


def shipped_table(kind: str, arch: str, env: str, *,
                  engine: Optional[str] = None) -> Optional[RouteTable]:
    """The routing table for `kind` on `arch`: `$env` if set, and then that
    table alone; else the one in hk/tuned/ with this machine's records over
    it (`hk.autotune.routes`), each measured shape replacing the shipped
    one; else None. A table measured on another arch is not loaded.

    With `engine` -- the name of the path the installed engine runs, as the
    tuner writes it -- a record of this machine's measured against another
    engine is left out (and listed in `meta["stale"]`): the path a kernel
    had to beat is not the one it would be beating. `HK_AUTOTUNE=0` leaves
    every record out, as it does schedule records."""
    override = os.environ.get(env)
    path = Path(override) if override else (
        Path(__file__).resolve().parents[1] / "tuned" / f"routes_{kind}_{arch}.json")
    if override and not path.is_file():
        raise FileNotFoundError(f"${env} names {path}, which does not exist")
    t = RouteTable.from_json(path) if path.is_file() else None
    if t is not None and t.meta.get("arch") not in (None, arch):
        raise ValueError(f"{path} was measured on {t.meta['arch']}, not {arch}")
    if override:
        return t
    from .. import autotune  # noqa: PLC0415
    from ..autotune import routes as _routes  # noqa: PLC0415

    user = _routes.load_user(kind, arch) if autotune.enabled() else None
    if user is None or user.get("arch") not in (None, arch):
        return t
    shapes, stale = {}, []
    for nk, ent in RouteTable.parse(user).items():
        made = ent.get("meta", {}).get("engine")
        if engine is not None and made not in (None, engine):
            stale.append(nk)
            continue
        shapes[nk] = ent
    if t is None:
        if not shapes:
            return None
        t = RouteTable({}, meta={"kind": kind, "arch": arch}, source="")
    t.overlay(shapes, str(_routes.user_path(kind, arch)))
    if stale:
        t.meta = {**t.meta, "stale": stale}
    return t
