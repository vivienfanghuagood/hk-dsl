"""Run the DX suite: check every solution against the reference, then time the
ones that are correct, interleaved in one process, and write a scoreboard.

    python3 -m bench.dx.run                                  # every task, every solution
    python3 -m bench.dx.run --tasks t1,t3 --quick            # a few shapes, fewer rounds
    python3 -m bench.dx.run --solutions torch_compile,docs_only --json out.json
    python3 -m bench.dx.run --list                           # tasks, shapes, solutions

On the workstation: `hkx950 ~/hk-dsl-dxsuite 5 python3 -m bench.dx.run ...` (gfx950)
and `hkx1100 ~/hk-dsl-dxsuite python3 -m bench.dx.run --tasks t1 ...` (gfx1100,
one task per call: the GPU is shared). bench/dx/README.md has the protocol.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--tasks", help="comma list: t1,t2 or full ids (default: all)")
    p.add_argument("--solutions", help="comma list of solution names (default: all)")
    p.add_argument("--kinds", help="comma list of kinds: baseline,hk_ops,proxy,docs_only,template,agent,expert")
    p.add_argument("--shapes", help="comma list of substrings; a shape runs if its name contains one")
    p.add_argument("--quick", action="store_true", help="the task's QUICK shapes, 5 rounds")
    p.add_argument("--small", action="store_true", help="the task's SMALL shapes (the test's), check only")
    p.add_argument("--check-only", action="store_true", help="correctness only, no timing")
    p.add_argument("--rounds", type=int, default=None, help="timing rounds (default 15, quick 5)")
    p.add_argument("--batch-ms", type=float, default=1.0,
                   help="each timed batch runs at least this long (default 1 ms)")
    p.add_argument("--no-graphs", action="store_true",
                   help="skip the CUDA-graph timing (device time without host overhead)")
    p.add_argument("--cold-cache", action="store_true",
                   help="fresh HK_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR, so "
                        "first_call_s is a cold compile")
    p.add_argument("--json", help="write the scoreboard JSON here (and a .md beside it)")
    p.add_argument("--md", help="write the markdown table here (default: beside --json)")
    p.add_argument("--list", action="store_true", help="list tasks, shapes and solutions and exit")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- timing

def _sync():
    import torch
    torch.cuda.synchronize()


def _batch_ms(fn, n: int) -> float:
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


@contextlib.contextmanager
def _graph_launches():
    """hk on RDNA: a kernel's pybind entry point launches on the null stream,
    which a CUDA graph does not capture (docs/runtime.md, "The stream") -- the
    graph would replay without it. While graphs are captured, switch hk to its
    code-object launch (HK_LAUNCH=codeobj: the same device code, launched on
    torch's current stream) by dropping the memoized launchers; both are put
    back afterwards, so host_us is still timed on the default launch."""
    try:
        from hk.codegen.scaffold import takes_stream
        from hk.dsl.kernel import Kernel
        from hk.target import current_arch
        native = takes_stream(current_arch())
    except Exception:  # noqa: BLE001 -- no hk: nothing to switch
        native = True
    if native:
        yield
        return
    import gc
    kernels = [o for o in gc.get_objects() if isinstance(o, Kernel)]
    saved = [(k, dict(k._launch_cache)) for k in kernels]
    old = os.environ.get("HK_LAUNCH")
    os.environ["HK_LAUNCH"] = "codeobj"
    for k in kernels:
        k._launch_cache.clear()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HK_LAUNCH", None)
        else:
            os.environ["HK_LAUNCH"] = old
        for k, cache in saved:
            k._launch_cache.clear()
            k._launch_cache.update(cache)


def _follows_inputs(g, fn, outs, ins) -> str:
    """'' if replaying the graph recomputes from the inputs, else why not: the
    float inputs are halved, the graph replayed and fn called eagerly, and the
    two must agree. A launch the capture missed leaves its output stale."""
    import torch
    from bench.dx.common import as_tuple

    floats = [t for t in ins.values() if isinstance(t, torch.Tensor) and t.is_floating_point()]
    if not floats:
        return ""
    for t in floats:
        t.mul_(0.5)
    try:
        g.replay()
        _sync()
        got = [o.clone() for o in as_tuple(outs)]
        want = as_tuple(fn())
        _sync()
    finally:
        for t in floats:
            t.mul_(2.0)
    for a, b in zip(got, want):
        if not isinstance(a, torch.Tensor) or not a.is_floating_point():
            continue
        tol = 2e-2 * max(1.0, b.float().abs().max().item())
        if not torch.allclose(a.float(), b.float(), rtol=2e-2, atol=tol):
            return "the graph's replay does not follow its inputs (a launch was not captured)"
    return ""


def _graph_for(fn, n: int, ins=None, check_inputs: bool = True):
    """A CUDA graph of n back-to-back calls, or the reason there is none."""
    import torch
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        _sync()
        g = torch.cuda.CUDAGraph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.cuda.graph(g):
                for _ in range(n):
                    outs = fn()
        _sync()
        empty = [w for w in caught if "empty" in str(w.message).lower()]
        if empty:
            return None, f"empty graph: {str(empty[0].message)[:120]}"
        g.replay()
        _sync()
        if check_inputs and ins is not None:
            why = _follows_inputs(g, fn, outs, ins)
            if why:
                return None, why
        return g, ""
    except Exception as e:  # noqa: BLE001
        try:
            _sync()
        except Exception:  # noqa: BLE001
            pass
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"


def _graph_ms(g) -> float:
    import torch
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def time_contenders(cands: List[Dict[str, Any]], rounds: int, batch_ms: float,
                    graphs: bool) -> None:
    """Fill rec['host_us'] / rec['device_us'] for each candidate, interleaved:
    every round runs every contender once, in a rotating order; the median
    round is reported. host_us is back-to-back eager calls (what a Python
    caller pays, host overhead included); device_us replays the same calls in
    a CUDA graph (the GPU's time, as a serving engine's captured step sees it)."""
    if not cands:
        return
    # Warm up, and size the batches so the events' resolution does not matter.
    for c in cands:
        fn = c["fn"]
        t0 = time.perf_counter()
        k = 0
        while k < 3 or (time.perf_counter() - t0 < 0.02 and k < 200):
            fn()
            k += 1
        _sync()
        per = _batch_ms(fn, 3) / 3
        c["n"] = int(min(2000, max(1, math.ceil(batch_ms / max(per, 1e-4)))))
        c["host"], c["dev"] = [], []
        c["graph"] = None
    if graphs:
        with _graph_launches():
            for c in cands:
                c["graph"], why = _graph_for(c["fn"], c["n"], c.get("ins"),
                                             check_inputs=not c.get("mutable"))
                if c["graph"] is None:
                    c["rec"]["graph_error"] = why
    for r in range(rounds):
        order = cands[r % len(cands):] + cands[:r % len(cands)]
        for c in order:
            c["host"].append(_batch_ms(c["fn"], c["n"]) / c["n"] * 1e3)
            if c["graph"] is not None:
                c["dev"].append(_graph_ms(c["graph"]) / c["n"] * 1e3)
    for c in cands:
        rec = c["rec"]
        rec["n_inner"], rec["rounds"] = c["n"], rounds
        rec["host_us"] = statistics.median(c["host"])
        rec["host_us_min"] = min(c["host"])
        if c["dev"]:
            rec["device_us"] = statistics.median(c["dev"])
            rec["device_us_min"] = min(c["dev"])
        c["graph"] = None


# --------------------------------------------------------------------------- one shape

def _clone(inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def _mutated(task, before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    import torch
    allowed = set(getattr(task, "MUTABLE_INPUTS", ()))
    bad = []
    for k, v in before.items():
        if isinstance(v, torch.Tensor) and k not in allowed and not torch.equal(v, after[k]):
            bad.append(k)
    return bad


def _err(e: BaseException) -> str:
    msg = str(e).strip().splitlines()
    head = msg[0] if msg else ""
    if len(msg) > 1:
        # keep the line that names the problem, when the first is generic
        tail = [m for m in msg[1:] if m.strip()][-1:]
        head = head + (" | " + tail[0].strip() if tail else "")
    return f"{type(e).__name__}: {head[:400]}"


def run_shape(task_id: str, task, shape: Dict[str, Any], sols, args, log) -> List[Dict[str, Any]]:
    import torch

    inputs = task.make_inputs(shape, device="cuda", seed=args.seed)
    ref = task.reference(**inputs)
    _sync()
    recs, cands = [], []
    for sol in sols:
        rec: Dict[str, Any] = {
            "task": task_id, "shape": shape["name"],
            "params": {k: v for k, v in shape.items() if k != "name"},
            "solution": sol.name, "kind": sol.kind, "label": sol.label,
            "lines": sol.lines, "proxy": sol.proxy or None,
        }
        recs.append(rec)
        try:
            sup = sol.supports(inputs)
        except Exception as e:  # noqa: BLE001
            sup = f"supports() raised {_err(e)}"
        if sup is not True:
            rec["status"], rec["error"] = "unsupported", str(sup)
            log(f"    {sol.name:<24} unsupported: {sup}")
            continue
        ins = _clone(inputs)
        try:
            _sync()
            t0 = time.perf_counter()
            fn = sol.prepare(ins)
            out = fn()
            _sync()
            rec["first_call_s"] = round(time.perf_counter() - t0, 3)
            if sol.kind == "proxy":
                rec["status"], rec["check"] = "proxy", f"not checked: {sol.proxy}"
            else:
                mut = _mutated(task, inputs, ins)
                if mut:
                    raise AssertionError(f"modified its input(s) {mut}, which the task does not allow")
                # A mutating task's check gets the inputs as they were.
                rec["check"] = task.check(out, ref, inputs) or "ok"
                rec["status"] = "ok"
            del out
            if not args.check_only:
                cands.append({"fn": fn, "rec": rec, "ins": ins,
                              "mutable": bool(getattr(task, "MUTABLE_INPUTS", ()))})
            log(f"    {sol.name:<24} {rec['status']:<6} first call {rec['first_call_s']:.2f}s  {rec.get('check', '')}")
        except torch.cuda.OutOfMemoryError as e:
            rec["status"], rec["error"] = "oom", _err(e)
            log(f"    {sol.name:<24} OOM")
        except AssertionError as e:
            rec["status"], rec["error"] = "wrong", _err(e)
            log(f"    {sol.name:<24} WRONG  {rec['error']}")
        except Exception as e:  # noqa: BLE001
            rec["status"], rec["error"] = "error", _err(e)
            rec["traceback"] = traceback.format_exc()[-2000:]
            log(f"    {sol.name:<24} ERROR  {rec['error']}")
        finally:
            if rec.get("status") in ("oom", "error", "wrong"):
                torch.cuda.empty_cache()
    if cands:
        rounds = args.rounds or (5 if args.quick else 15)
        try:
            time_contenders(cands, rounds, args.batch_ms, graphs=not args.no_graphs)
        except torch.cuda.OutOfMemoryError:
            # time one at a time, dropping the others' captured state
            for c in cands:
                try:
                    time_contenders([c], rounds, args.batch_ms, graphs=False)
                except Exception as e:  # noqa: BLE001
                    c["rec"]["status"], c["rec"]["error"] = "error", "timing: " + _err(e)
        for c in cands:
            r = c["rec"]
            if "host_us" in r:
                dev = f"{r['device_us']:9.2f}" if "device_us" in r else "      n/a"
                log(f"    {r['solution']:<24} host {r['host_us']:9.2f} us  device {dev} us")
    del inputs, ref, cands
    torch.cuda.empty_cache()
    return recs


# --------------------------------------------------------------------------- ratios

def add_ratios(results: List[Dict[str, Any]]) -> None:
    """Per (task, shape): speedups (>1 = faster) against torch.compile, the
    best baseline (torch eager, torch.compile, Triton, hk.ops) and hk.ops.
    Device time where both sides have it, else host time."""
    by = {}
    for r in results:
        by.setdefault((r["task"], r["shape"]), []).append(r)
    for rs in by.values():
        timed = [r for r in rs if r.get("status") == "ok" and "host_us" in r]

        def t(r, metric):
            return r.get(metric)

        for metric, suffix in (("device_us", "dev"), ("host_us", "host")):
            tc = next((t(r, metric) for r in timed if r["solution"] == "torch_compile"), None)
            base = [t(r, metric) for r in timed if r["kind"] in ("baseline", "hk_ops") and t(r, metric)]
            hko = [t(r, metric) for r in timed if r["kind"] == "hk_ops" and t(r, metric)]
            best = min(base) if base else None
            hkb = min(hko) if hko else None
            for r in rs:
                v = t(r, metric)
                if not v or "host_us" not in r:
                    continue
                if tc:
                    r[f"x_tc_{suffix}"] = round(tc / v, 4)
                if best:
                    r[f"x_best_{suffix}"] = round(best / v, 4)
                if hkb:
                    r[f"x_hk_ops_{suffix}"] = round(hkb / v, 4)
            for r in rs:
                if r.get("status") == "ok" and best and t(r, metric) == best and r["kind"] in BASELINE:
                    r[f"best_baseline_{suffix}"] = True


BASELINE = ("baseline", "hk_ops")


def _geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summarize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = {}
    for r in results:
        key = (r["task"], r["solution"])
        s = rows.setdefault(key, {"task": r["task"], "solution": r["solution"], "kind": r["kind"],
                                  "lines": r.get("lines"), "shapes": 0, "ok": 0, "statuses": {},
                                  "x_tc": [], "x_best": [], "x_hk_ops": []})
        s["shapes"] += 1
        st = r.get("status", "?")
        s["statuses"][st] = s["statuses"].get(st, 0) + 1
        if st in ("ok", "proxy"):
            s["ok"] += st == "ok"
            for k in ("x_tc", "x_best", "x_hk_ops"):
                v = r.get(f"{k}_dev") or r.get(f"{k}_host")
                if v:
                    s[k].append(v)
    out = []
    for s in rows.values():
        for k in ("x_tc", "x_best", "x_hk_ops"):
            xs = s.pop(k)
            s[f"{k}_geomean"] = round(_geo(xs), 3) if xs else None
            s[f"{k}_min"] = round(min(xs), 3) if xs else None
            s[f"{k}_max"] = round(max(xs), 3) if xs else None
        out.append(s)
    return out


# --------------------------------------------------------------------------- output

def _fmt(v, nd=2):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def markdown(board: Dict[str, Any]) -> str:
    m = board["meta"]
    L = [f"# DX suite scoreboard: {m.get('gpu_product', '?')} ({m.get('gpu_arch', '?')}), {m['date']}", ""]
    L.append("| | |\n|---|---|")
    for k in ("gpu_product", "gpu", "gpu_arch", "cu", "gpu_mem_gb", "torch", "hip", "triton", "hk", "hk_arch",
              "hk_commit", "python", "host", "visible_devices", "command"):
        if k in m:
            L.append(f"| {k} | `{m[k]}` |")
    L += ["", "Speedups are `other / this` (> 1 is faster), on device time (a CUDA graph of the calls) "
          "where both sides have one, else on host time (back-to-back eager calls). "
          "`x TC` against torch.compile, `x best` against the fastest baseline "
          "(torch eager, torch.compile, Triton, hk.ops) at that shape, `x hk.ops` against the hk.ops kernel "
          "where one exists. Wrong answers are not timed. *proxy* rows compute different math "
          "(said in their label) and are shown for scale only. Lines are code lines of the solution file "
          "(no blanks, comments or docstrings).", ""]
    L += ["## Summary", "",
          "| task | solution | kind | lines | ok/shapes | x TC geomean (min-max) | x best geomean | x hk.ops geomean | other statuses |",
          "|---|---|---|---:|---:|---|---:|---:|---|"]
    for s in board["summary"]:
        other = ", ".join(f"{k} {v}" for k, v in s["statuses"].items() if k != "ok")
        rng = f"{_fmt(s['x_tc_geomean'])} ({_fmt(s['x_tc_min'])}-{_fmt(s['x_tc_max'])})" if s["x_tc_geomean"] else ""
        L.append(f"| {s['task']} | {s['solution']} | {s['kind']} | {_fmt(s['lines'])} | {s['ok']}/{s['shapes']} | "
                 f"{rng} | {_fmt(s['x_best_geomean'])} | {_fmt(s['x_hk_ops_geomean'])} | {other} |")
    tasks = []
    for r in board["results"]:
        if r["task"] not in tasks:
            tasks.append(r["task"])
    for tid in tasks:
        L += ["", f"## {tid}", "",
              "| shape | solution | kind | status | first call s | host us | device us | x TC | x best | x hk.ops | lines | note |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in board["results"]:
            if r["task"] != tid:
                continue
            dev = "dev" if r.get("device_us") else "host"
            note = r.get("error") or r.get("proxy") or ""
            if r.get("graph_error") and r.get("status") == "ok":
                note = (note + " " if note else "") + "no graph: " + r["graph_error"]
            if r.get(f"best_baseline_{dev}"):
                note = ("**best baseline** " + note).strip()
            L.append("| {shape} | {sol} | {kind} | {st} | {fc} | {h} | {d} | {xtc} | {xb} | {xh} | {ln} | {note} |".format(
                shape=r["shape"], sol=r["solution"], kind=r["kind"], st=r.get("status", "?"),
                fc=_fmt(r.get("first_call_s")), h=_fmt(r.get("host_us")), d=_fmt(r.get("device_us")),
                xtc=_fmt(r.get(f"x_tc_{dev}")), xb=_fmt(r.get(f"x_best_{dev}")), xh=_fmt(r.get(f"x_hk_ops_{dev}")),
                ln=_fmt(r.get("lines")), note=note.replace("|", "/")[:200]))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    args = parse(argv)
    if args.cold_cache:
        tmp = tempfile.mkdtemp(prefix="hk-dx-cold-")
        for var in ("HK_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            os.environ[var] = os.path.join(tmp, var.lower())
    from bench.dx import common

    if args.list:
        for tid in common.resolve_tasks(args.tasks):
            task = common.load_task(tid)
            print(tid)
            for s in task.SHAPES:
                print(f"    shape {s['name']}{'  (quick)' if s['name'] in getattr(task, 'QUICK', ()) else ''}")
            for sol in common.solutions_for(tid, task):
                print(f"    solution {sol.name:<24} {sol.kind:<10} {sol.lines or '':>4} lines  {sol.label}")
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    common.configure_torch_compile()
    meta = common.versions()
    meta["date"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["command"] = " ".join(["python3 -m bench.dx.run"] + (argv if argv is not None else sys.argv[1:]))
    meta["cold_cache"] = bool(args.cold_cache)
    meta["protocol"] = ("correctness first (wrong = not timed); per shape all correct contenders "
                        "interleaved in one process, rotating order, median of rounds; host_us = "
                        "back-to-back eager calls, device_us = CUDA-graph replay of the same calls; "
                        "inputs stay resident (hot L2/MALL) across calls")
    print(f"[dx] {meta.get('gpu_arch')} {meta.get('gpu')} torch {meta['torch']} triton {meta['triton']} "
          f"hk {meta['hk']} @ {meta['hk_commit']}", flush=True)

    def log(msg):
        print(msg, flush=True)

    names = args.solutions.split(",") if args.solutions else None
    kinds = args.kinds.split(",") if args.kinds else None
    results: List[Dict[str, Any]] = []
    t_start = time.time()
    for tid in common.resolve_tasks(args.tasks):
        task = common.load_task(tid)
        sols = common.solutions_for(tid, task, names, kinds)
        if args.small:
            shapes = list(task.SMALL)
            args.check_only = True
        elif args.quick:
            shapes = [s for s in task.SHAPES if s["name"] in getattr(task, "QUICK", ())] or task.SHAPES[:2]
        else:
            shapes = list(task.SHAPES)
        if args.shapes:
            subs = args.shapes.split(",")
            shapes = [s for s in shapes if any(x in s["name"] for x in subs)]
        log(f"[dx] {tid}: {len(shapes)} shapes x {len(sols)} solutions")
        for shape in shapes:
            log(f"  {shape['name']}")
            try:
                results += run_shape(tid, task, shape, sols, args, log)
            except torch.cuda.OutOfMemoryError as e:
                log(f"    the reference or inputs ran out of memory: {e}")
                torch.cuda.empty_cache()
    add_ratios(results)
    board = {"meta": meta, "summary": summarize(results), "results": results,
             "wall_s": round(time.time() - t_start, 1)}
    md = markdown(board)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(board, indent=1, default=str) + "\n")
        mdp = Path(args.md) if args.md else Path(args.json).with_suffix(".md")
        mdp.write_text(md)
        log(f"[dx] wrote {args.json} and {mdp}")
    elif args.md:
        Path(args.md).write_text(md)
    else:
        print(md)
    bad = [r for r in results if r.get("status") in ("wrong", "error") and r["kind"] != "proxy"]
    log(f"[dx] {len(results)} results, {len(bad)} wrong or failing, {board['wall_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
