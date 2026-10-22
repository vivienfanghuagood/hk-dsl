"""A standard autotuned Triton matmul, C = A @ B.T with B as (N, K): the
"tuned Triton" baseline of the tuning benchmark.

The kernel is the Triton tutorial's (grouped launch order, masked tiles),
which is what a Triton user starts from; the autotuning space is what such a
user sweeps on AMD: block sizes, warps, and on CDNA the MFMA size
(`matrix_instr_nonkdim`), `kpack` and `waves_per_eu`, pruned where the staged
tiles pass the arch's LDS or the block is far wider than the problem. Each
config is timed the way the scoreboard times: warm, back to back behind a GPU
spin, the median of a few rounds (Triton's own do_bench flushes the cache and
ranks a different regime; tools/hk-bench/triton_ref.py says why).
"""

from __future__ import annotations

import statistics

import torch
import triton
import triton.language as tl

LDS = {"gfx950": 160 * 1024, "gfx1100": 64 * 1024}


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def configs(arch: str):
    out = []
    cdna = not arch.startswith("gfx11")
    for bm in (16, 32, 64, 128, 256):
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                for w in (4, 8):
                    stages = (2,) if cdna else (1, 2)
                    for s in stages:
                        if (bm + bn) * bk * 2 * s > LDS.get(arch, 64 * 1024):
                            continue
                        if bm * bn // (w * 64 if cdna else w * 32) > 256:
                            continue
                        if bm * bn < 64 * 64 and w == 8:
                            continue
                        for wpe in (0, 2):
                            kw = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8,
                                  "waves_per_eu": wpe}
                            if cdna:
                                kw.update(matrix_instr_nonkdim=16, kpack=1)
                            out.append(triton.Config(kw, num_warps=w, num_stages=s))
    return out


def _prune(configs, named_args, **kwargs):
    """Blocks far wider than the problem compute rows nobody keeps; and a
    small problem does not need 256-wide tiles to fill the card."""
    m, n = named_args["M"], named_args["N"]
    keep = [c for c in configs
            if c.kwargs["BLOCK_M"] <= max(16, triton.next_power_of_2(m))
            and c.kwargs["BLOCK_N"] <= max(64, triton.next_power_of_2(n))]
    if m > 64:
        keep = [c for c in keep if c.kwargs["BLOCK_M"] >= 64]
    return keep or configs


def _bench(fn, quantiles=None, rounds=5, calls=10):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        torch.cuda._sleep(5_000_000)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(calls):
            fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) / calls)
    t = statistics.median(ts)
    return [t] * len(quantiles) if quantiles else t


@triton.jit
def _body(a_ptr, b_ptr, c_ptr, M, N, K, s_am, s_ak, s_bn, s_bk, s_cm, s_cn,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
          GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm[:, None] * s_am + rk[None, :] * s_ak
    b = b_ptr + rn[None, :] * s_bn + rk[:, None] * s_bk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        ka = tl.load(a, mask=(rm[:, None] < M) & (rk[None, :] < K - k * BLOCK_K), other=0.0)
        kb = tl.load(b, mask=(rn[None, :] < N) & (rk[:, None] < K - k * BLOCK_K), other=0.0)
        acc += tl.dot(ka, kb)
        a += BLOCK_K * s_ak
        b += BLOCK_K * s_bk
    c = c_ptr + rm[:, None] * s_cm + rn[None, :] * s_cn
    tl.store(c, acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = triton.autotune(configs=configs(_arch()), key=["M", "N", "K"],
                                  prune_configs_by={"early_config_prune": _prune},
                                  do_bench=_bench)(_body)
    return _KERNEL


def matmul_abt(a, b, out=None):
    """a @ b.T, b as (N, K), autotuned per (M, N, K) on first call."""
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty(m, n, device=a.device, dtype=a.dtype) if out is None else out
    kern = _kernel()
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
    kern[grid](a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
               out.stride(0), out.stride(1))
    return out


def picked(m: int, n: int, k: int):
    """The config the autotuner picked for a shape, as a string, or None."""
    kern = _kernel()
    for key, cfg in kern.cache.items():
        if tuple(key[:3]) == (m, n, k):
            return str(cfg)
    return None
