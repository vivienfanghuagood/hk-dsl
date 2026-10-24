"""gemm_ab.py: F.linear (hipBLASLt) vs hk.ops.matmul at Qwen-Image 2.1's DiT GEMMs, interleaved (GemmBench)."""
import statistics, sys, torch, hk, hk.ops
from hk.tune.measure import GemmBench
shapes = [(4096, 12288, 4096), (4096, 4096, 4096), (4096, 4096, 12288), (4096, 24576, 4096)]
for m, n, k in shapes:
    gb = GemmBench(m, n, k, "gfx1100")
    arms = {"hipblaslt": lambda x, w, out: torch.matmul(x, w.t(), out=out),
            "hk": lambda x, w, out: hk.ops.matmul(x, w, out=out)}
    for a, fn in list(arms.items()):
        e = gb.check(fn)
        if not e < 2e-2:
            print("WRONG", a, e); arms.pop(a)
    res = {a: [] for a in arms}
    for _ in range(3):
        gb.refresh()
        per = gb.time(arms, 7)
        for a in arms:
            res[a].append(statistics.median(per[a]))
    t = {a: statistics.median(v) for a, v in res.items()}
    f = 2 * m * n * k
    print(f"{m}x{n}x{k}: " + "  ".join(f"{a} {t[a]*1e3:.0f} us {f/t[a]/1e9:.0f} TF/s" for a in t)
          + (f"  hk/hipblaslt speedup {t['hipblaslt']/t['hk']:.3f}x" if len(t) == 2 else ""), flush=True)
