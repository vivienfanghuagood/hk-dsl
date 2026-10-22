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
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["tune", "tune_shapes", "model_shapes", "parse_shape", "Result", "BUDGET"]

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
    """Baselines hk.tune can time beside the search without extra packages:
    "torch" is F.linear (hipBLASLt on ROCm)."""
    import torch.nn.functional as F  # noqa: N812,PLC0415

    out: Dict[str, Callable] = {}
    for n in names:
        if n == "torch":
            out["torch"] = lambda a, b, o: F.linear(a, b)
        elif n:
            raise ValueError(f"hk.tune: unknown baseline {n!r} (bench/tune/baselines.py has more)")
    return out


def tune(m: int, n: int, k: int, *, budget: float = BUDGET, save: bool = True,
         baselines: Optional[Dict[str, Callable]] = None, workers: int = 32,
         log: Optional[Callable[[str], None]] = print, arch: Optional[str] = None,
         compiler=None, **search):
    """Tune one M x N x K bf16 GEMM (C = A @ B.T, B as (N, K)) on the current
    GPU; returns a `Result` (hk.tune.search). A win is written to this
    machine's records unless `save=False`. `baselines` ({name: fn(a, b, out)})
    are timed in the decision's sets; default F.linear."""
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
    bench = GemmBench(m, n, k, arch)
    try:
        res = Search(sp, budget=budget, log=log or (lambda s: None), **search).run(
            m, n, k, bench, compiler, baselines=baselines, arch=arch)
    finally:
        bench.close()
        if own:
            compiler.close()
    if save and res.verdict == "win" and res.champion is not None:
        name = "u:" + res.champion.label
        point = {**res.as_dict(), **_meta(arch)}
        point.pop("refused", None)
        res.record = str(records.save(arch, m, n, k, name, sp.spec(res.champion), point))
    if log:
        log(res.summary())
    return res


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


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m hk.tune", description=__doc__.splitlines()[0])
    ap.add_argument("shapes", nargs="*", help="MxNxK, e.g. 2048x16384x4096")
    ap.add_argument("--model", default="", help="tune a model's linear layers (HF name or config.json)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--m", default="1,16,128,512,2048,8192", help="Ms for --model")
    ap.add_argument("--layers", default="", help="subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--budget", type=float, default=BUDGET, help="GPU-seconds per shape")
    ap.add_argument("--baselines", default="torch", help="comma list (torch) or '' for none")
    ap.add_argument("--no-save", action="store_true", help="measure, but write no record")
    ap.add_argument("--json", default="", help="write every result here")
    ap.add_argument("-j", "--jobs", type=int, default=32, help="hipcc processes at once")
    ap.add_argument("--list", action="store_true", help="this machine's records")
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
    base = torch_baselines([b for b in args.baselines.split(",") if b])
    results = tune_shapes(shapes, budget=args.budget, save=not args.no_save, baselines=base,
                          workers=args.jobs)
    if args.json:
        from pathlib import Path  # noqa: PLC0415

        Path(args.json).write_text(json.dumps({"meta": _meta(arch),
                                               "results": [r.as_dict() for r in results]},
                                              indent=1) + "\n")
    wins = sum(r.verdict == "win" for r in results)
    gpu = sum(r.gpu_seconds for r in results)
    print(f"{len(results)} shape(s): {wins} win(s), the rest keep their route; "
          f"{gpu:.0f} GPU-s ({gpu / 60:.1f} GPU-min)")
    return 0
