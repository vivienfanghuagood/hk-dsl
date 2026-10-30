"""Row-wise kernels: hk2 vs torch (eager) vs Triton, effective bandwidth (read + write bytes / time)."""
import statistics

import torch
import triton
import triton.language as tl

from hk2.kernels import rowwise


@triton.jit
def _tri_rms(x, w, o, n, eps, BN: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BN)
    m = c < n
    t = tl.load(x + r * n + c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(w + c, mask=m, other=0.0).to(tl.float32)
    y = t * tl.math.rsqrt(tl.sum(t * t, 0) / n + eps) * g
    tl.store(o + r * n + c, y.to(o.dtype.element_ty), mask=m)


def tri_rms(x, w, eps=1e-6):
    o = torch.empty_like(x)
    n = x.shape[-1]
    bn = triton.next_power_of_2(n)
    _tri_rms[(x.shape[0],)](x, w, o, n, eps, BN=bn, num_warps=8 if bn >= 4096 else 4)
    return o


@triton.jit
def _tri_softmax(x, o, n, BN: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BN)
    m = c < n
    t = tl.load(x + r * n + c, mask=m, other=float("-inf")).to(tl.float32)
    e = tl.exp(t - tl.max(t, 0))
    tl.store(o + r * n + c, (e / tl.sum(e, 0)).to(o.dtype.element_ty), mask=m)


def tri_softmax(x):
    o = torch.empty_like(x)
    n = x.shape[-1]
    bn = triton.next_power_of_2(n)
    _tri_softmax[(x.shape[0],)](x, o, n, BN=bn, num_warps=8 if bn >= 4096 else 4)
    return o


def timeit(f, reps=20):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def race(arms, rounds=10):
    for f in arms.values():
        f()
    ts = {k: [] for k in arms}
    for _ in range(rounds):
        for k, f in arms.items():
            ts[k].append(timeit(f))
    return {k: statistics.median(v) for k, v in ts.items()}


arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
print(f"# {arch} bf16, GB/s (read + write)")
for m, n in ((4096, 4096), (16384, 5120), (8192, 8192), (65536, 1024)):
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    by = 2 * x.numel() * 2
    t = race({"torch": lambda: torch.nn.functional.rms_norm(x, (n,), w, 1e-6), "triton": lambda: tri_rms(x, w),
              "hk2": lambda: rowwise.rmsnorm(x, w)})
    print(f"rmsnorm {m}x{n}: " + " | ".join(f"{k} {by / v / 1e6:.0f}" for k, v in t.items()))
    t = race({"torch": lambda: torch.softmax(x, -1), "triton": lambda: tri_softmax(x),
              "hk2": lambda: rowwise.softmax(x)})
    print(f"softmax {m}x{n}: " + " | ".join(f"{k} {by / v / 1e6:.0f}" for k, v in t.items()))
