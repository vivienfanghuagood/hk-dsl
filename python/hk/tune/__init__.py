"""`hk.tune`: search a GEMM shape's schedule space within a GPU-time budget,
and keep a winner only when the measurement says it is one.

    python -m hk.tune 2048x16384x4096                  # one shape (M x N x K)
    python -m hk.tune 1x4608x9216 512x7168x8192 --budget 90
    python -m hk.tune --model Qwen/Qwen2.5-32B --tp 2 --m 1,16,512,2048
    python -m hk.tune --list                           # this machine's records

From Python:

    import hk.tune
    res = hk.tune.tune(2048, 16384, 4096, budget=60)
    print(res.summary())           # champion, verdict, gain and band, GPU-s
    res.accepted                   # what hk.ops.matmul now runs for this shape

How a shape is tuned (hk.tune.search): the shipped route's candidates and the
generated space's best by an analytic model (hk.tune.cost) are compiled on the
CPU in parallel, checked against fp32, then timed in rungs of successive
halving and improved by local search, all interleaved in one process; the
champion is then measured against the current route (and the baselines) in
sets of rounds on fresh activations and accepted only past a margin drawn
from the measured spread, confirmed on fresh sets (hk.tune.stats). A tie
keeps the route.

What it writes (hk.tune.records): a win goes into this machine's records,
`$HK_TUNE_DIR/routes_gemm_<arch>.json`, which hk.ops.matmul and
hk.templates.gemm read in front of the shipped routes for that shape's M
bucket. Nothing shipped changes. `HK_AUTOTUNE=0` ignores the records.

The space is hk's own GEMMs' knobs (hk.tune.space): on gfx950 the pingpong
GEMM, the four-warp block GEMM and the decode GEMV; on gfx1100 the WMMA GEMM
and the decode GEMV.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["tune", "tune_shapes", "model_shapes", "parse_shape", "explain", "Result",
           "BUDGET"]

#: GPU-seconds a shape's search may spend, by default.
BUDGET = 60.0


def __getattr__(name):
    if name == "Result":
        from .search import Result  # noqa: PLC0415

        return Result
    raise AttributeError(name)


def parse_shape(s: str) -> Tuple[int, int, int]:
    """"MxNxK" -> (M, N, K)."""
    m, n, k = (int(x) for x in s.lower().split("x"))
    return m, n, k


def _meta(arch: str) -> dict:
    out = dict(arch=arch, date=time.strftime("%Y-%m-%d"), by="hk.tune")
    try:
        import torch  # noqa: PLC0415

        out.update(device=torch.cuda.get_device_name(), torch=torch.__version__,
                   hip=torch.version.hip)
    except Exception:  # noqa: BLE001 -- metadata only
        pass
    try:
        from ..runtime.compile import _hipcc_version, toolchain_fingerprint  # noqa: PLC0415

        out.update(toolchain=toolchain_fingerprint(), hipcc=_hipcc_version().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return out


def torch_baselines(names: Iterable[str] = ("torch",)) -> Dict[str, Callable]:
    """Baselines to time beside the decision, by name: "torch" is F.linear
    (hipBLASLt on ROCm); the tuning benchmark's others (tunableop, aiter,
    aiter_triton, flydsl, hipkittens, triton -- bench/tune/baselines.py) when
    this runs from a checkout that has them. An unavailable one is skipped
    with a message."""
    import torch.nn.functional as F  # noqa: N812,PLC0415

    out: Dict[str, Callable] = {}
    rest = []
    for n in names:
        if n == "torch":
            out["torch"] = lambda a, b, o: F.linear(a, b)
        elif n:
            rest.append(n)
    if rest:
        try:
            from bench.tune import baselines as more  # noqa: PLC0415
        except ImportError:
            raise ValueError(f"hk.tune: baselines {rest} need bench/tune (run from a "
                             f"checkout of hk-dsl, with its root on sys.path)") from None
        from ..target import current_arch  # noqa: PLC0415

        got, why = more.available(current_arch(), rest)
        for n, reason in why.items():
            print(f"hk.tune: baseline {n} not timed: {reason}")
        out.update(got)
    return out


def tune(m: int, n: int, k: int, *, budget: float = BUDGET, save: bool = True,
         baselines: Optional[Dict[str, Callable]] = None, workers: int = 32,
         log: Optional[Callable[[str], None]] = print, arch: Optional[str] = None,
         compiler=None, profile: bool = False, **search):
    """Tune one M x N x K bf16 GEMM (C = A @ B.T, B as (N, K)) on the current
    GPU; returns a `Result` (hk.tune.search). A win is written to this
    machine's records unless `save=False`. `baselines` ({name: fn(a, b, out)})
    are timed in the decision's sets; default F.linear.

    `profile=True` then runs hk.profile on what hk.ops.matmul now launches
    for the shape against hipBLASLt (F.linear) -- 30-70 s -- and keeps its
    verdict (`res.profile`): why the result is as fast as it is, and what
    the kernel would have to change to be faster."""
    from ..target import current_arch  # noqa: PLC0415
    from . import records  # noqa: PLC0415
    from .measure import Compiler, GemmBench  # noqa: PLC0415
    from .search import Search  # noqa: PLC0415
    from .space import provider  # noqa: PLC0415

    arch = arch or current_arch()
    sp = provider(arch)
    if baselines is None:
        baselines = torch_baselines()
    own = compiler is None
    compiler = compiler or Compiler(workers)
    # Candidates launch through device-only code objects (HK_LAUNCH=codeobj):
    # a fifth of the compile, and no extension module a kernel -- a process
    # that has loaded a few hundred of those runs out of thread-specific
    # storage keys and aborts. Device code and device time are the same; the
    # host launch is slower, which the queued timing does not see.
    prev = os.environ.get("HK_LAUNCH")
    if prev is None:
        os.environ["HK_LAUNCH"] = "codeobj"
    bench = GemmBench(m, n, k, arch)
    try:
        res = Search(sp, budget=budget, log=log or (lambda s: None), **search).run(
            m, n, k, bench, compiler, baselines=baselines, arch=arch)
    finally:
        bench.close()
        if own:
            compiler.close()
        if prev is None:
            os.environ.pop("HK_LAUNCH", None)
    if save and res.verdict == "win" and res.champion is not None:
        name = "u:" + res.champion.label
        point = {**res.as_dict(), **_meta(arch)}
        point.pop("refused", None)
        res.record = str(records.save(arch, m, n, k, name, sp.spec(res.champion), point))
    if profile:
        res.profile = explain(m, n, k)
    if log:
        log(res.summary())
        if res.profile.get("line"):
            log(f"  hk.profile: {res.profile['line']}")
    return res


def explain(m: int, n: int, k: int, *, vs: str = "torch.nn.functional:linear") -> dict:
    """hk.profile (docs/profile.md) on hk.ops.matmul at M x N x K -- the
    kernel this machine's records give the shape -- against `vs`:
    {"line": the verdict, "bound", "losses", "compare": the interleaved
    comparison's line}; {"error": ...} if the profiler cannot run here."""
    try:
        from .. import profile as _profile  # noqa: PLC0415

        p = _profile.profile("hk.ops:matmul", inputs=f"bf16[{m},{k}], bf16[{n},{k}]",
                             vs={"hipblaslt": vs})
        v = p.verdict or {}
        cmp_ = (p.json() or {}).get("compare") or {}
        return dict(line=v.get("line", ""), bound=v.get("bound"), losses=v.get("losses"),
                    compare=cmp_.get("line", ""), ratio=cmp_.get("ratio"))
    except Exception as e:  # noqa: BLE001 -- an explanation is optional
        return dict(error=f"{type(e).__name__}: {e}")


def tune_shapes(shapes: Sequence[Tuple[int, int, int]], **kw) -> List:
    """tune() each shape in turn, one compiler for all of them (kernels shared
    between shapes compile once)."""
    from .measure import Compiler  # noqa: PLC0415

    comp = Compiler(kw.pop("workers", 32))
    try:
        return [tune(m, n, k, compiler=comp, **kw) for m, n, k in shapes]
    finally:
        comp.close()


def model_shapes(model: str, tp: int = 1, ms: Sequence[int] = (1, 16, 128, 512, 2048, 8192),
                 layers: str = "") -> List[Tuple[int, int, int]]:
    """(M, N, K) of a model's linear layers (qkv, o, gate_up, down; and the LM
    head where asked) at tensor-parallel size `tp`, at each M of `ms`."""
    from ..autotune.routes import linear_shapes, model_config  # noqa: PLC0415

    want = set(layers.split(",")) if layers else {"qkv", "o", "gate_up", "down"}
    out = []
    for name, n, k, _ in linear_shapes(model_config(model), tp):
        if name in want:
            out += [(m, n, k) for m in ms]
    return out


def _each_in_a_process(shapes, args, argv) -> List[dict]:
    """Tune each shape in a child process of its own, and gather the results.

    A search loads every candidate's kernel module, a few hundred a shape;
    a process that has loaded several shapes' worth aborts on gfx1100 (each
    hk extension module takes thread-specific-storage keys, and a process has
    1024: "thread_specific_storage constructor: could not initialize the TSS
    key"). The compile cache is shared, so a child rebuilds nothing."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (m, n, k) in enumerate(shapes):
            out = Path(tmp) / f"{i}.json"
            cmd = [sys.executable, "-m", "hk.tune", f"{m}x{n}x{k}", "--budget", str(args.budget),
                   "--baselines", args.baselines, "-j", str(args.jobs), "--json", str(out)]
            if args.no_save:
                cmd.append("--no-save")
            if args.profile:
                cmd.append("--profile")
            rc = subprocess.call(cmd)
            if rc == 0 and out.exists():
                rows += json.loads(out.read_text())["results"]
            else:
                print(f"hk.tune: {m}x{n}x{k} exited {rc}; its route stands")
                rows.append(dict(m=m, n=n, k=k, verdict="none", error=f"exit {rc}"))
    return rows


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m hk.tune", description=__doc__.splitlines()[0])
    ap.add_argument("shapes", nargs="*", help="MxNxK, e.g. 2048x16384x4096")
    ap.add_argument("--model", default="", help="tune a model's linear layers (HF name or config.json)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--m", default="1,16,128,512,2048,8192", help="Ms for --model")
    ap.add_argument("--layers", default="", help="subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--budget", type=float, default=BUDGET, help="GPU-seconds per shape")
    ap.add_argument("--baselines", default="torch",
                    help="comma list: torch, and from a checkout bench/tune's tunableop, "
                         "aiter, aiter_triton, flydsl, hipkittens, triton; '' for none")
    ap.add_argument("--no-save", action="store_true", help="measure, but write no record")
    ap.add_argument("--json", default="", help="write every result here")
    ap.add_argument("-j", "--jobs", type=int, default=32, help="hipcc processes at once")
    ap.add_argument("--list", action="store_true", help="this machine's records")
    ap.add_argument("--profile", action="store_true",
                    help="then explain each result with hk.profile against hipBLASLt (30-70 s)")
    ap.add_argument("--in-process", action="store_true",
                    help="tune several shapes in this one process (default: a process each)")
    ap.add_argument("--dry-run", action="store_true",
                    help="the space and the pool the search would compile, no GPU")
    args = ap.parse_args(argv)

    from ..target import current_arch  # noqa: PLC0415

    arch = current_arch()
    if args.list:
        from . import records  # noqa: PLC0415

        d = records.load(arch)
        print(f"{records.path(arch)}: {sum(len(v) for v in d['routes'].values())} tuned shape(s)")
        for p in d.get("points", []):
            dec = p.get("decision") or {}
            print(f"  {p['m']}x{p['n']}x{p['k']}: {p.get('champion')} "
                  f"{dec.get('gain', 0) * 100:+.2f}% vs {p.get('route')} ({p.get('date', '?')})")
        return 0
    shapes = [parse_shape(s) for s in args.shapes]
    if args.model:
        shapes += model_shapes(args.model, args.tp, [int(x) for x in args.m.split(",")],
                               args.layers)
    if not shapes:
        ap.error("give shapes (MxNxK), --model, or --list")
    if args.dry_run:
        from .search import Result, Search  # noqa: PLC0415
        from .space import provider  # noqa: PLC0415

        sp = provider(arch)
        for m, n, k in shapes:
            res = Result(m, n, k, arch, sp.current(m, n, k))
            pool = Search(sp).pool(m, n, k, res)
            print(f"{m}x{n}x{k}: route {res.route}; {len(pool)} to compile of "
                  f"{res.counts['legal']} legal ({res.counts['generated']} generated)")
            for c in pool:
                print(f"    {c}  ~{sp.estimate(c, m, n, k) * 1e6:.1f} us")
        return 0
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("hk.tune: no GPU here; --dry-run shows the search without one")
        return 2
    if len(shapes) > 1 and not args.in_process:
        rows = _each_in_a_process(shapes, args, argv)
    else:
        base = torch_baselines([b for b in args.baselines.split(",") if b])
        rows = [r.as_dict() for r in tune_shapes(shapes, budget=args.budget,
                                                 save=not args.no_save, baselines=base,
                                                 workers=args.jobs, profile=args.profile)]
    if args.json:
        from pathlib import Path  # noqa: PLC0415

        Path(args.json).write_text(json.dumps({"meta": _meta(arch), "results": rows},
                                              indent=1) + "\n")
    wins = sum(r.get("verdict") == "win" for r in rows)
    gpu = sum(r.get("gpu_seconds") or 0.0 for r in rows)
    results = rows
    print(f"{len(results)} shape(s): {wins} win(s), the rest keep their route; "
          f"{gpu:.0f} GPU-s ({gpu / 60:.1f} GPU-min)")
    return 0
