"""Score T4 solutions: correctness first, then interleaved timing against every
baseline on this GPU, the ratio to the best one per shape, and the pass rule.

    python3 -m bench.t4.score --solution bench/t4/solution.py                 # the public shapes
    python3 -m bench.t4.score --solution S.py --only gpt-like-gate_up-t4096   # some of them
    python3 -m bench.t4.score --baselines-only --out DIR                      # the baselines alone
    python3 -m bench.t4.score --check-only --solution S.py                    # correctness, no timing

Per shape (reference.py has the semantics and the inputs):

 1. **Arms.** Every baseline that imports here (baselines.py) and each
    solution. A solution's optional `prepare(w)` runs once per weight copy,
    untimed. Also `proxy:dense`: the dense hipBLASLt GEMM of the same
    rows x N x K on one expert's weights -- different math, timed only for
    scale (how far the grouped kernels are from a dense GEMM of their size),
    never checked, never a baseline.
 2. **Correctness first.** Three routings no timing uses -- an edge routing
    (empty experts, a one-row expert, one taking half the rows, odd sizes) and
    two drawn like the shape's -- each on fresh activations, against fp32
    (relative max error < reference.TOL), with `out` filled with NaN before
    the call, and the inputs compared after it. A wrong, raising or
    input-modifying arm is not timed; a wrong baseline is not "best".
 3. **Timing**, of the correct arms, all in this one process: S sets (fresh
    activations each) of R rounds; every round draws P new routings (the
    shape's distribution, seeded) that no arm has seen, and runs every arm in a
    rotated order on them, each arm's pass being P back-to-back calls over the
    weight copies (rotated past the Infinity Cache), queued behind a GPU spin
    longer than its host time, between two events: a call's time is device
    time from its first kernel to its last, plus whatever host time it makes
    the GPU wait for (a host synchronisation inside the call is paid for).
 4. **Numbers** (stats.py): an arm's time is the median over sets of each
    set's median round; the best baseline is the fastest correct one; a
    solution's ratio is best / solution per set, median and range over sets.
 5. **Audit** (solutions only): one call under torch.profiler; every GPU
    kernel it launches is classified -- hk's (a kernel hk-dsl built), torch's
    bookkeeping (at::native elementwise/scan/index kernels, copies, fills),
    or foreign (anything else: hipBLASLt, rocBLAS, CK, Triton, aiter...).
    Foreign kernels, or a torch kernel that multiplies matrices, fail the
    audit; so does a solution file that imports a forbidden package.

Writes OUT/<arch>-<label>.json and .md and prints one SCORE line a shape.
Sealed shapes (`--shapes sealed`) are read from $HK_T4_SEALED/shapes.json
and their results go under $HK_T4_SEALED/results, never into the repository.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import reference as R
from . import stats
from .baselines import ALL as BASELINES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: Bytes of weights a pass streams at least: three times the arch's last-level cache.
LLC = {"gfx950": 256 * 2**20, "gfx1100": 96 * 2**20}
MAX_COPIES = 4
CALLS = 4                 # calls (routings) a pass, at least
CHECK_SEEDS = (900001, 900002)

#: Top-level packages a solution file may not import.
FORBIDDEN = ("aiter", "vllm", "triton", "flydsl", "sglang", "bench", "deep_gemm",
             "hipblaslt", "ck4inductor", "tilelang")

_HK_KERNEL = re.compile(r"(_kernel\(globals\)|^hk_flat::|\bhk_flat::|_Z\d+\w+_kernel7globals|_ZN7hk_flat)")
_TORCH_AUX = re.compile(r"(at::native::|at::cuda::|Memcpy|Memset|memcpy|memset|__amd_rocclr_|"
                        r"rocprim|cub::|hipcub)", re.I)
_GEMMISH = re.compile(r"(Cijk|gemm|gemv|mfma|wmma|matmul|_mm_|bmm|ck::|ck_tile|grouped_mm)", re.I)


# ---------------------------------------------------------------- solutions


def import_audit(path: Path) -> List[str]:
    """Forbidden top-level imports in a solution file (static)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "__import__":
            names = [a.value for a in node.args[:1] if isinstance(a, ast.Constant)]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "import_module"):
            names = [a.value for a in node.args[:1] if isinstance(a, ast.Constant)]
        for n in names:
            if str(n).split(".")[0] in FORBIDDEN:
                bad.append(str(n))
    return sorted(set(bad))


def classify_kernel(name: str) -> str:
    """'hk', 'torch' (bookkeeping) or 'foreign'."""
    if _HK_KERNEL.search(name):
        return "hk"
    if _TORCH_AUX.search(name) and not _GEMMISH.search(name):
        return "torch"
    return "foreign"


def kernel_audit(kernels: List[Dict]) -> Dict:
    """kernels: [{"name", "us"}] of one call. -> {ok, hk_us, torch_us,
    foreign_us, foreign: [...]}."""
    by = {"hk": 0.0, "torch": 0.0, "foreign": 0.0}
    foreign = []
    for k in kernels:
        c = classify_kernel(k["name"])
        by[c] += k["us"]
        if c == "foreign":
            foreign.append(k["name"][:120])
    total = sum(by.values()) or 1.0
    return dict(ok=not foreign and by["hk"] > 0, hk_us=round(by["hk"], 1),
                torch_us=round(by["torch"], 1), foreign_us=round(by["foreign"], 1),
                hk_share=round(by["hk"] / total, 4), foreign=sorted(set(foreign))[:12],
                kernels=len(kernels))


def load_solution(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"t4_solution_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise SystemExit(f"{path}: no solve(x, w, offsets, out)")
    return mod


# ---------------------------------------------------------------- GPU side


class Clock:
    """torch.cuda._sleep's cycles a second, measured after a warm-up
    (hk.tune.measure.spin_hz's method)."""

    def __init__(self, torch):
        self.torch = torch
        torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            torch.cuda._sleep(2_000_000)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        self.hz = 2_000_000 / (best * 1e-3)

    def spin(self, seconds: float) -> None:
        self.torch.cuda._sleep(int(seconds * self.hz))


def _profile_call(torch, fn) -> List[Dict]:
    from torch.autograd import DeviceType  # noqa: PLC0415
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out = []
    for ev in prof.events():
        if getattr(ev, "device_type", None) == DeviceType.CUDA:
            try:
                us = ev.time_range.elapsed_us()
            except Exception:  # noqa: BLE001
                us = getattr(ev, "device_time", 0.0) or 0.0
            out.append(dict(name=ev.name, us=float(us)))
    return out


def _arch(torch) -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def score_shape(torch, shape: Dict, arms: Dict[str, Callable], prepares: Dict[str, Callable],
                baselines: List[str], solutions: List[str], args, clock: Clock, arch: str,
                log=print) -> Dict:
    t0 = time.time()
    rows, n = R.rows(shape), shape["N"]
    dev = torch.device("cuda")
    wb = R.weight_bytes(shape)
    copies = max(1, min(MAX_COPIES, -(-3 * LLC.get(arch, LLC["gfx950"]) // wb)))
    calls = max(copies, CALLS)
    ws = R.make_weights(shape, copies, dev)
    out = torch.empty(rows, n, device=dev, dtype=torch.bfloat16)
    row = dict(name=shape["name"], shape=shape, rows=rows, flop=R.flops(shape), copies=copies,
               calls_per_pass=calls, errors={}, first_call_s={}, prepare_s={})

    # prepared weights, per arm
    wp: Dict[str, list] = {}
    for a in list(arms):
        p = prepares.get(a)
        if p is None:
            wp[a] = ws
            continue
        try:
            t = time.perf_counter()
            wp[a] = [p(w) for w in ws]
            torch.cuda.synchronize()
            row["prepare_s"][a] = round(time.perf_counter() - t, 3)
        except Exception as e:  # noqa: BLE001
            row["errors"][a] = f"prepare: {type(e).__name__}: {str(e)[:200]}"
            arms.pop(a)

    # 1. correctness
    w_sum = ws[0].view(torch.int16).sum(dtype=torch.int64).item()
    checks = [("edge", R.edge_sizes(shape))] + [(f"r{s}", R.group_sizes(shape, s)) for s in CHECK_SEEDS]
    ok: Dict[str, Callable] = {}
    worst: Dict[str, float] = {}
    for a, fn in arms.items():
        worst[a] = 0.0
    proxies = [a for a in arms if a.startswith("proxy:")]
    for ci, (label, sizes) in enumerate(checks):
        x = R.make_x(shape, 800000 + ci, dev)
        offs = R.offsets_tensor(sizes, dev)
        ref = R.reference(x, ws[0], offs)
        x0, o0 = x.clone(), offs.clone()
        for a in list(arms):
            if a in row["errors"] or a in proxies:
                continue
            fn = arms[a]
            try:
                out.fill_(float("nan"))
                t = time.perf_counter()
                y = fn(x, wp[a][0], offs, out)
                torch.cuda.synchronize()
                if a not in row["first_call_s"]:
                    row["first_call_s"][a] = round(time.perf_counter() - t, 3)
                y = out if y is None else y
                e = R.error(y, ref)
                if not torch.equal(x, x0) or not torch.equal(offs, o0):
                    raise AssertionError("modified its inputs")
            except Exception as ex:  # noqa: BLE001 -- recorded, the arm is not timed
                row["errors"][a] = f"{label}: {type(ex).__name__}: {str(ex).splitlines()[0][:200] if str(ex) else ''}"
                continue
            worst[a] = max(worst[a], e)
            if not e < R.TOL:
                row["errors"][a] = f"{label}: wrong, relative max error {e:.3g}"
        del ref, x, x0
    if ws[0].view(torch.int16).sum(dtype=torch.int64).item() != w_sum:
        raise RuntimeError("an arm modified the weights; rerun arms separately to find it")
    for a, fn in arms.items():
        if a not in row["errors"]:
            ok[a] = fn
    row["check_error"] = {a: (None if math.isinf(v) else round(v, 6)) for a, v in worst.items()
                          if a not in proxies}
    row["correct"] = {a: a in ok for a in arms}
    for a in arms:
        if a in row["errors"]:
            log(f"  {shape['name']} {a}: {row['errors'][a]}")

    # 2. timing
    sets: Dict[str, List[List[float]]] = {a: [] for a in ok}
    if ok and not args.check_only:
        names = list(ok)
        for s in range(args.sets):
            x = R.make_x(shape, 1000 + s, dev)
            warm = R.offsets_tensor(R.group_sizes(shape, 5000 + s), dev)
            host = {}
            for a in names:
                fn = ok[a]
                fn(x, wp[a][-1], warm, out)
                torch.cuda.synchronize()
                h = time.perf_counter()
                for i in range(calls):
                    fn(x, wp[a][i % copies], warm, out)
                host[a] = time.perf_counter() - h
                torch.cuda.synchronize()
            per = {a: [] for a in names}
            for r in range(args.rounds):
                seeds = [10000 * (s + 1) + 100 * r + i for i in range(calls)]
                offs = [R.offsets_tensor(R.group_sizes(shape, sd), dev) for sd in seeds]
                j = r % len(names)
                order = names[j:] + names[:j]
                if r % 2:
                    order = order[::-1]
                for a in order:
                    fn = ok[a]
                    fn(x, wp[a][-1], warm, out)
                    torch.cuda.synchronize()
                    clock.spin(2 * host[a] + 5e-4)
                    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    ev0.record()
                    for i in range(calls):
                        fn(x, wp[a][i % copies], offs[i], out)
                    ev1.record()
                    torch.cuda.synchronize()
                    per[a].append(ev0.elapsed_time(ev1) / calls)
            for a in names:
                sets[a].append(per[a])
            del x

    # 3. audit
    audits = {}
    if not args.no_audit:
        x = R.make_x(shape, 7, dev)
        offs = R.offsets_tensor(R.group_sizes(shape, 7), dev)
        for a in solutions:
            if a not in ok:
                continue
            try:
                ks = _profile_call(torch, lambda a=a: ok[a](x, wp[a][0], offs, out))
                audits[a] = kernel_audit(ks)
            except Exception as ex:  # noqa: BLE001
                audits[a] = dict(ok=False, error=f"{type(ex).__name__}: {str(ex)[:200]}")
        del x

    floor = stats.floor_ms(R.flops(shape), arch)
    row["sets"] = sets
    base_arms = [b for b in baselines if b in sets]
    row["ms"] = {}
    row["spread"] = {}
    for a, v in sets.items():
        row["ms"][a], row["spread"][a] = stats.arm_time(v, floor)
    best = stats.best_baseline(sets, base_arms, floor)
    row["best_baseline"] = best
    row["solutions"] = {}
    for sname in solutions:
        res = dict(correct=bool(row["correct"].get(sname)), audit=audits.get(sname),
                   ratio=None, band=None)
        if args.no_audit:
            res["audit_ok"] = True
        else:
            res["audit_ok"] = bool(audits.get(sname, {}).get("ok"))
        if sname in sets and best:
            r, lo, hi = stats.ratio_band(sets[best], sets[sname], floor)
            res["ratio"] = None if r is None else round(r, 4)
            res["band"] = None if r is None else [round(lo, 4), round(hi, 4)]
        row["solutions"][sname] = res
    if "triton" in sets:
        try:
            from . import triton_grouped  # noqa: PLC0415

            row["triton_pick"] = triton_grouped.picked(rows, n, shape["K"], shape["E"])
        except Exception:  # noqa: BLE001
            pass
    row["wall_s"] = round(time.time() - t0, 1)
    del ws, out, wp
    torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------- reports


def _us(ms) -> str:
    return "-" if ms is None else f"{ms * 1e3:.1f}"


def _roof(shape: Dict, arch: str) -> Optional[float]:
    try:
        return stats.roofline_us(R.flops(shape), R.min_bytes(shape), arch)
    except Exception:  # noqa: BLE001 -- a partial shape (the tests') has no roofline
        return None


def markdown(doc: Dict) -> str:
    meta = doc["meta"]
    lines = [f"# T4 score: {meta['arch']} ({meta['gpu']}), {meta['label']}", "",
             f"{meta['date']}; hk-dsl at {meta.get('commit') or '?'}; "
             f"{', '.join(f'{k} {v}' for k, v in meta['versions'].items())}; "
             f"{meta['sets']} sets x {meta['rounds']} rounds; shapes: {meta['shapes']}.", ""]
    for m in meta.get("merged") or []:
        lines.append(f"- merged: `{m['file']}` ({m['date']}, {m.get('commit') or '?'})")
    if meta.get("merged"):
        lines.append("")
    if meta.get("missing"):
        lines.append("Baselines not here: " + "; ".join(f"`{k}` ({v})" for k, v in meta["missing"].items()))
        lines.append("")
    sols = meta["solutions"]
    for s in sols:
        v = doc["verdict"].get(s)
        if v:
            lines.append(f"**{s}**: {'PASS' if v['passed'] else 'FAIL'} -- geomean {v['geomean']}, "
                         f"min {v['min']}, {v['reached']}/{v['shapes']} shapes >= {v['target']} "
                         f"({v['why']}). Rule: {v['rule']}.")
    lines.append("")
    seen = []
    for r in doc["rows"]:
        for a in list(r.get("ms", {})) + list(r.get("errors", {})):
            if a not in seen:
                seen.append(a)
    rank = {"base": 0, "proxy": 1, "sol": 2}
    arms = sorted(seen, key=lambda a: (rank.get(a.split(":")[0], 3), seen.index(a)))
    def name(a):
        return a[5:] if a.startswith("base:") else ("dense (proxy)" if a == "proxy:dense" else a)
    head = ("| shape | E | N | K | rows | roofline | " + " | ".join(name(a) for a in arms)
            + " | best | best/roofline | " + " | ".join(f"{s} ratio" for s in sols) + " |")
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for r in doc["rows"]:
        sh = r["shape"]
        cells = []
        for a in arms:
            if r.get("ms", {}).get(a):
                cells.append(_us(r["ms"][a]))
            elif a in r.get("errors", {}):
                cells.append("x")
            else:
                cells.append("-")
        sc = []
        for s in sols:
            res = r["solutions"].get(s, {})
            if not res.get("correct"):
                sc.append("wrong")
            elif res.get("ratio") is None:
                sc.append("-")
            else:
                b = res.get("band") or [res["ratio"], res["ratio"]]
                flag = "" if res.get("audit_ok") else " (audit)"
                sc.append(f"{res['ratio']:.3f} ({b[0]:.2f}-{b[1]:.2f}){flag}")
        roof = _roof(sh, meta["arch"])
        best = r.get("best_baseline")
        best_ms = r.get("ms", {}).get(best) if best else None
        frac = f"{roof / (best_ms * 1e3):.2f}" if roof and best_ms else "-"
        lines.append(f"| {r['name']} | {sh['E']} | {sh['N']} | {sh['K']} | {r['rows']} | "
                     f"{f'{roof:.1f}' if roof else '-'} | " + " | ".join(cells)
                     + f" | {best.split(':', 1)[-1] if best else '-'} | {frac} | " + " | ".join(sc) + " |")
    errs = [(r["name"], a, e) for r in doc["rows"] for a, e in r.get("errors", {}).items()]
    if errs:
        lines += ["", "Arms not timed:", ""]
        lines += [f"- {n} `{a}`: {e}" for n, a, e in errs]
    aud = [(r["name"], s, res.get("audit")) for r in doc["rows"]
           for s, res in r["solutions"].items() if res.get("audit")]
    if aud:
        lines += ["", "Audit (one call under torch.profiler):", ""]
        for n, s, a in aud:
            if "error" in a:
                lines.append(f"- {n} `{s}`: {a['error']}")
            else:
                fk = f" e.g. `{a['foreign'][0][:60]}`" if a.get("foreign") else ""
                lines.append(f"- {n} `{s}`: {'ok' if a['ok'] else 'FAIL'}; hk {a['hk_us']} us "
                             f"({a['hk_share']:.0%}), torch {a['torch_us']} us, foreign "
                             f"{a['foreign_us']} us{fk}")
    lines.append("")
    lines.append("Times are per call, us: the median over sets of each set's median round. "
                 "ratio = best baseline / solution (> 1: faster), the median and range over sets. "
                 "`x` = not timed (see above). roofline = max(FLOPs / bf16 peak, bytes / DRAM "
                 "bandwidth) at the datasheets' peaks, bytes = x + out + the weights of every expert "
                 "with a row; best/roofline = the fraction of it the best baseline reaches. "
                 "`dense` is a proxy (one expert's weights for all rows), not the task's math.")
    return "\n".join(lines) + "\n"


def _commit() -> str:
    if os.environ.get("HK_T4_COMMIT"):
        return os.environ["HK_T4_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _select(shapes: List[Dict], only: str) -> List[Dict]:
    if not only:
        return shapes
    if ":" in only and all(p.strip().lstrip("-").isdigit() or not p.strip() for p in only.split(":")):
        a, b = (int(p) if p.strip() else None for p in only.split(":"))
        return shapes[a:b]
    want = [w.strip() for w in only.split(",") if w.strip()]
    got = [s for s in shapes if any(w == s["name"] or w in s["name"] for w in want)]
    if not got:
        raise SystemExit(f"--only {only}: no shape matches")
    return got


def merge(files: List[str], out_dir: Path, label: str) -> int:
    """One scoreboard from runs of different shapes on one GPU: the rows in
    order (a shape scored twice keeps its last run), the verdict recomputed,
    the first run's metadata with every run's commands and dates."""
    docs = [json.loads(Path(f).read_text()) for f in files]
    archs = {d["meta"]["arch"] for d in docs}
    if len(archs) != 1:
        raise SystemExit(f"--merge: runs of different GPUs ({sorted(archs)})")
    meta = dict(docs[0]["meta"], label=label, merged=[
        dict(file=Path(f).name, date=d["meta"]["date"], commit=d["meta"].get("commit"),
             command=d["meta"].get("command")) for f, d in zip(files, docs)])
    sols = []
    for d in docs:
        sols += [s for s in d["meta"]["solutions"] if s not in sols]
    meta["solutions"] = sols
    rows: Dict[str, Dict] = {}
    for d in docs:
        for r in d["rows"]:
            rows[r["name"]] = r
    rows_l = list(rows.values())
    verdict = {s: stats.verdict([dict(name=r["name"], correct=r["solutions"].get(s, {}).get("correct"),
                                      audit_ok=r["solutions"].get(s, {}).get("audit_ok"),
                                      ratio=r["solutions"].get(s, {}).get("ratio")) for r in rows_l])
               for s in sols}
    doc = dict(meta=meta, rows=rows_l, verdict=verdict)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{meta['arch']}-{label}.json"
    js.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    js.with_suffix(".md").write_text(markdown(doc))
    for s, v in verdict.items():
        print(f"VERDICT {s}: {json.dumps(v)}")
    print(f"wrote {js} and .md")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solution", action="append", default=[], help="a solution file (repeatable)")
    ap.add_argument("--shapes", default="public", help="public, sealed, or a shape file")
    ap.add_argument("--only", default="", help="shape names (comma list, substrings) or a:b")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--baselines-only", action="store_true", help="no solution; time the baselines")
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--check-only", action="store_true", help="correctness only, no timing")
    ap.add_argument("--no-audit", action="store_true", help="skip the kernel audit (not a scored run)")
    ap.add_argument("--no-proxy", action="store_true", help="skip the dense-GEMM proxy (scale only)")
    ap.add_argument("--out", default="", help="output directory (default bench/t4/out)")
    ap.add_argument("--label", default="", help="name of the result files")
    ap.add_argument("--picks", default="", help="Triton picks file, read and rewritten after the run "
                    "(default for the public shapes: results/triton_picks.json, read only)")
    ap.add_argument("--merge", nargs="+", default=None, metavar="JSON",
                    help="combine result files of one GPU (runs of different shapes) into --out/--label")
    args = ap.parse_args(argv)
    if args.merge:
        return merge(args.merge, Path(args.out or HERE / "out"), args.label or "merged")
    if not args.solution and not args.baselines_only:
        ap.error("give --solution FILE (or --baselines-only)")

    os.environ.setdefault("HK_LAUNCH", "codeobj")
    sealed = args.shapes == "sealed"
    if sealed:
        sd = os.environ.get("HK_T4_SEALED")
        if not sd or not Path(sd, "shapes.json").is_file():
            raise SystemExit("--shapes sealed needs $HK_T4_SEALED/shapes.json")
        shape_file = Path(sd, "shapes.json")
        out_dir = Path(args.out) if args.out else Path(sd, "results")
        if REPO in out_dir.resolve().parents or out_dir.resolve() == REPO:
            raise SystemExit("sealed results may not be written inside the repository")
    else:
        shape_file = R.PUBLIC if args.shapes == "public" else Path(args.shapes)
        out_dir = Path(args.out) if args.out else HERE / "out"

    bad = {s: import_audit(Path(s)) for s in args.solution}
    for s, b in bad.items():
        if b:
            print(f"AUDIT {s}: forbidden imports {b}", flush=True)

    import torch  # noqa: PLC0415

    from . import baselines as B  # noqa: PLC0415

    arch = _arch(torch)
    shapes = _select(R.load_shapes(shape_file, arch), args.only)
    fns, missing = B.available(arch, [b for b in args.baselines.split(",") if b])
    arms: Dict[str, Callable] = {f"base:{k}": v for k, v in fns.items()}
    if not args.no_proxy:
        # Not the task's math (one expert's weights for every row): the dense
        # hipBLASLt GEMM of the same size, timed for scale, never a baseline.
        arms["proxy:dense"] = lambda x, w, offsets, out: torch.mm(x, w[0].t(), out=out)
    prepares: Dict[str, Callable] = {f"base:{k}": v for k, v in B.PREPARE.items() if k in fns}
    sol_names = []
    sol_path = {}
    for s in args.solution:
        mod = load_solution(s)
        name = f"sol:{Path(s).stem}"
        sol_names.append(name)
        sol_path[name] = s
        arms[name] = mod.solve
        if hasattr(mod, "prepare"):
            prepares[name] = mod.prepare
    picks_file = Path(args.picks) if args.picks else None
    picks_in = picks_file or (HERE / "results" / "triton_picks.json"
                              if args.shapes == "public" else None)
    if picks_in and picks_in.is_file() and "triton" in fns:
        from . import triton_grouped  # noqa: PLC0415

        for key, pick in json.loads(picks_in.read_text()).get(arch, {}).items():
            r_, n_, k_, e_ = (int(v) for v in key.split("x"))
            triton_grouped.preset(r_, n_, k_, e_, pick)

    label = args.label or ("sealed" if sealed else "public") + \
        (f"-{Path(args.solution[0]).stem}" if args.solution else "-baselines")
    meta = dict(arch=arch, gpu=torch.cuda.get_device_name(0), label=label,
                date=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                commit=_commit(), versions=B.versions(), missing=missing,
                hk_launch=os.environ.get("HK_LAUNCH"), sets=args.sets, rounds=args.rounds,
                shapes="sealed" if sealed else str(shape_file.name), solutions=sol_names,
                baselines=[f"base:{k}" for k in fns], forbidden_imports=bad,
                command=" ".join([Path(sys.argv[0]).name] + sys.argv[1:]))
    print(json.dumps({k: meta[k] for k in ("arch", "gpu", "versions", "missing", "commit")}), flush=True)
    clock = Clock(torch)
    rows = []
    t_all = time.time()
    for shape in shapes:
        row = score_shape(torch, shape, dict(arms), prepares, list(meta["baselines"]), sol_names,
                          args, clock, arch)
        for s in sol_names:
            if bad.get(sol_path[s]):
                row["solutions"][s]["audit_ok"] = False
        rows.append(row)
        best = row.get("best_baseline")
        msg = ", ".join(f"{s[4:]} {_us(row['ms'].get(s))}us {row['solutions'][s]['ratio']}x"
                        for s in sol_names)
        print(f"SCORE {shape['name']}: best {best[5:] if best else '-'} "
              f"{_us(row['ms'].get(best)) if best else '-'}us; {msg} [{row['wall_s']}s]", flush=True)
    verdict = {}
    for s in sol_names:
        verdict[s] = stats.verdict([dict(name=r["name"], correct=r["solutions"][s]["correct"],
                                         audit_ok=r["solutions"][s]["audit_ok"],
                                         ratio=r["solutions"][s]["ratio"]) for r in rows])
        print(f"VERDICT {s}: {json.dumps(verdict[s])}", flush=True)
    meta["wall_s"] = round(time.time() - t_all, 1)
    doc = dict(meta=meta, rows=rows, verdict=verdict)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"{arch}-{label}.json"
    js.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    js.with_suffix(".md").write_text(markdown(doc))
    if picks_file and "triton" in fns:
        from . import triton_grouped  # noqa: PLC0415

        old = json.loads(picks_file.read_text()) if picks_file.is_file() else {}
        mine = old.setdefault(arch, {})
        for r in rows:
            p = triton_grouped.picked(r["rows"], r["shape"]["N"], r["shape"]["K"], r["shape"]["E"])
            if p:
                mine[f"{r['rows']}x{r['shape']['N']}x{r['shape']['K']}x{r['shape']['E']}"] = p
        picks_file.write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
    print(f"wrote {js} and .md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
