"""hk2.kernels.gemm.linear (autotuned) vs torch and Triton on LLM-shaped GEMMs, bf16, same process."""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import triton_ref  # noqa: E402

from hk2.kernels import gemm  # noqa: E402


def timeit(f, reps=10):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
print(f"# {arch} bf16 x @ w.T, TF/s")
shapes = [(8192, 8192, 8192), (4096, 4096, 4096), (16384, 4096, 4096), (4096, 14336, 4096), (4096, 4096, 14336),
          (2048, 8192, 2048), (512, 4096, 4096), (128, 8192, 8192)]
for m, n, k in shapes:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    arms = {"torch": lambda: x @ w.T, "triton": lambda: triton_ref.matmul_nt(x, w), "hk2": lambda: gemm.linear(x, w)}
    for f in arms.values():
        f()
    ts = {a: [] for a in arms}
    for _ in range(7):
        for a, f in arms.items():
            ts[a].append(timeit(f))
    fl = 2 * m * n * k
    t = {a: statistics.median(v) for a, v in ts.items()}
    print(f"{m}x{n}x{k}: " + " | ".join(f"{a} {fl / v / 1e9:.0f}" for a, v in t.items()) +
          f"  (hk2/torch {t['torch'] / t['hk2']:.2f}, hk2/triton {t['triton'] / t['hk2']:.2f})", flush=True)
