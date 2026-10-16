#!/usr/bin/env python3
"""Why hk's one- and 32-row norms take 2.4 us in one round and 4 in the next,
when Inductor's and Triton's do not move (docs/history/32).

The scoreboard's cases and timing (scoreboard.gpu_rounds: a profiler session
per arm per round, rounds interleaved), but every kernel record kept, so a
slow round can be told apart into slow calls, and many rounds:

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/small_rows.py --json out.json
    HIP_FORCE_DEV_KERNARG=1 ... small_rows.py         # kernel arguments in VRAM
    HK_LAUNCH=codeobj ... small_rows.py               # hk through hipModuleLaunchKernel
    ... small_rows.py --graph                         # hk replayed from a CUDA graph too

Prints, per case and arm, the rounds' per-call medians and every round's
spread of call durations; --json keeps the durations.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoreboard as sb                            # noqa: E402
import torch                                       # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

CASES = {"rms1": lambda: sb.rmsnorm_case(1, False), "rms32": lambda: sb.rmsnorm_case(32, False),
         "fadd1": lambda: sb.fused_add_case(1, False),
         "fadd32": lambda: sb.fused_add_case(32, False)}


def durations(fns, rounds, calls, gap_us=0.0, starts=None, queue_ms=0.0):
    """{arm: [[us per kernel record, in order], one list per round]}; with
    `starts` a dict, the records' start times (us) go there the same way.
    `queue_ms` puts a GPU sleep that long ahead of each round's calls, so the
    host has enqueued them all before the first runs: back to back on the
    GPU whatever the launcher costs."""
    per = {a: [] for a in fns}
    order = list(fns)
    for r in range(rounds):
        k = r % len(order)
        for a in order[k:] + order[:k]:
            fns[a]()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                if queue_ms:
                    torch.cuda._sleep(int(queue_ms * 2.5e6))
                for _ in range(calls):
                    fns[a]()
                    if gap_us:
                        end = time.perf_counter() + gap_us * 1e-6
                        while time.perf_counter() < end:
                            pass
                torch.cuda.synchronize()
            ks = sorted(sb._kernels(p), key=lambda e: e.time_range.start)
            if queue_ms:
                ks = [e for e in ks if "sleep" not in e.name.lower() and "spin" not in e.name.lower()]
            per[a].append([round(e.device_time_total, 3) for e in ks])
            if starts is not None:
                starts.setdefault(a, []).append([round(e.time_range.start, 3) for e in ks])
    return per


def host_us(fn, calls=200, reps=5):
    """What one launch costs the host: `calls` launches behind a GPU sleep
    long enough that none of them waits for the GPU, fastest of `reps`."""
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda._sleep(int(5e-3 * 2.5e9))
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        best = min(best, (time.perf_counter() - t0) / calls * 1e6)
        torch.cuda.synchronize()
    return round(best, 2)


def graph_arm(run, calls):
    """`run` captured `calls` times in one CUDA graph: replayed, the launches
    are back to back on the GPU with no host in between."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            run()
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="rms1,fadd1,rms32,fadd32")
    ap.add_argument("--arms", default="hk,inductor,triton")
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--gap-us", type=float, default=0.0,
                    help="spin the host this long after every call")
    ap.add_argument("--queue-ms", type=float, default=0.0,
                    help="a GPU sleep this long ahead of each round, so its calls "
                    "run back to back")
    ap.add_argument("--graph", action="store_true",
                    help="also time hk's calls replayed from a CUDA graph, by events")
    ap.add_argument("--burn", type=float, default=3.0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    env = {k: os.environ.get(k) for k in ("HIP_FORCE_DEV_KERNARG", "HK_LAUNCH", "HK_TORCH_OP")}
    print(f"pid {os.getpid()}, {env}, gap {args.gap_us} us, queue {args.queue_ms} ms")
    sb.burn_in(args.burn)
    out = {"env": env, "args": vars(args), "cases": {}}
    for cname in args.cases.split(","):
        name, _, _, _, arms, _ = CASES[cname]()
        fns = {a: arms[a][0] for a in args.arms.split(",") if a in arms
               and not isinstance(arms[a], str)}
        for a in fns:
            fns[a]()
        torch.cuda.synchronize()
        st = {}
        d = durations(fns, args.rounds, args.calls, args.gap_us, st, args.queue_ms)
        res = {a: {"rounds": v, "starts": st.get(a), "host_us": host_us(fns[a])}
               for a, v in d.items()}
        if args.graph and "hk" in fns:
            g = graph_arm(fns["hk"], args.calls)
            per = []
            for _ in range(args.rounds):
                g.replay()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                per.append(s.elapsed_time(e) * 1e3 / args.calls)
            res["hk_graph_events"] = {"per_call_us": per}
        out["cases"][name] = res
        print(f"\n{name}")
        for a, v in res.items():
            if "rounds" in v:
                allc = [x for r in v["rounds"] for x in r]
                med = statistics.median(allc)
                slow = sum(x > med + 5 for x in allc) / len(allc)
                print(f"  {a:<10} host {v['host_us']:.1f} us a launch; per call median "
                      f"{med:.2f} us, {slow:.1%} over median + 5 us, mean "
                      f"{statistics.mean(allc):.2f}")
            else:
                print(f"  {a:<10} graph replay, us per call: "
                      f"{' '.join(f'{m:.2f}' for m in v['per_call_us'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(out))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
