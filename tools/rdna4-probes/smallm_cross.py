"""Whether gfx1201's small-M GEMM schedules depend on the compiler: hk.tune's champions from two
toolchains' record files (ROCm 7.2.3's clang 22, the R9700 vLLM image's clang 23), each timed in
this process's toolchain against the other's and the default, with hk.tune's bench (W rotated past
the Infinity Cache, queued interleaved rounds). Usage: smallm_cross.py RECORDS_A RECORDS_B"""
import json
import statistics
import sys

import torch
from hk import schedules
from hk.ops import gemm as G
from hk.schedules.gemm import _runner
from hk.tune.measure import GemmBench

files = sys.argv[1:3]
recs = [json.load(open(f)) for f in files]
tags = ["A", "B"]
shapes = sorted({(int(m), *map(int, nk.split("x"))) for r in recs for nk, by_m in r["routes"].items()
                 for m in by_m}, key=lambda s: (s[1], s[2], s[0]))
for m, n, k in shapes:
    bench = GemmBench(m, n, k, "gfx1201")
    arms = {"default": G.TUNER._kernel_for(G.DEFAULT)}
    names = {}
    for tag, r in zip(tags, recs):
        name = r["routes"].get(f"{n}x{k}", {}).get(str(m))
        if name:
            names[tag] = name
            arms[tag] = _runner(schedules.from_record(r["candidates"][name], strict=False), m, n, k, bench.dev)
    for a, fn in arms.items():
        err = bench.check(fn)
        assert err < 1e-2, (m, n, k, a, err)
    ms = {a: [] for a in arms}
    for _ in range(3):
        bench.refresh()
        for a, ts in bench.time(arms, 10).items():
            ms[a] += ts
    us = {a: statistics.median(v) * 1e3 for a, v in ms.items()}
    print(f"{m}x{n}x{k}: " + "  ".join(f"{a}={u:.1f}" for a, u in us.items())
          + "  | " + "  ".join(f"{t} {names[t]}" for t in names)
          + (f"  | t(B)/t(A) {us['B'] / us['A']:.3f}" if len(names) == 2 else ""), flush=True)
    bench.close()
