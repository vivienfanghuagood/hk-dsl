"""int8 / fp8 GEMM throughput (x @ w.T): hk2 vs torch._int_mm / torch._scaled_mm where they exist."""
import os
import statistics
import sys

import torch

import hk2 as hk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "hk2"))
from test_lowp import mm, mm8  # noqa: E402


def timeit(f, reps=10):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
cfgs = [(256, 256, 128, 8), (256, 128, 128, 8), (128, 128, 128, 4)] if arch == "gfx950" else \
    [(128, 256, 64, 8), (128, 128, 64, 4), (128, 128, 128, 8)]
kinds = ["int8"] + (["fp8"] if arch == "gfx950" else [])
for kind in kinds:
    for n in (4096, 8192):
        if kind == "int8":
            a = torch.randint(-100, 100, (n, n), device="cuda", dtype=torch.int8)
            b = torch.randint(-100, 100, (n, n), device="cuda", dtype=torch.int8)
            c = torch.empty(n, n, device="cuda", dtype=torch.int32)
            kern, base = mm8, (lambda: torch._int_mm(a, b.T))
        else:
            a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
            c = torch.empty(n, n, device="cuda", dtype=torch.float32)
            one = torch.ones((), device="cuda")
            kern, base = mm, (lambda: torch._scaled_mm(a, b.T, one, one, out_dtype=torch.float32))
        arms = {}
        try:
            base()
            arms["torch"] = base
        except Exception as e:  # noqa: BLE001
            print(f"  torch {kind}: {type(e).__name__}: {str(e)[:100]}")
        for bm, bn, bk, w in cfgs:
            def f(bm=bm, bn=bn, bk=bk, w=w):
                kern[(hk.cdiv(n, bm), hk.cdiv(n, bn))](a, b, c, BM=bm, BN=bn, BK=bk, num_warps=w)
            try:
                f()
                arms[f"hk2 {bm}x{bn}x{bk}x{w}"] = f
            except Exception as e:  # noqa: BLE001
                print(f"  {bm}x{bn}x{bk}x{w}: {type(e).__name__}: {str(e)[:100]}")
        ts = {k: [] for k in arms}
        for _ in range(5):
            for k, f in arms.items():
                ts[k].append(timeit(f))
        print(f"{arch} {kind} {n}^3 TOP/s: " + " | ".join(f"{k} {2 * n ** 3 / statistics.median(v) / 1e9:.0f}"
                                                        for k, v in ts.items()), flush=True)
