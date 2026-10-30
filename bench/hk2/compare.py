"""torch vs Triton (autotuned) vs hk2 on the same GPU, same process, interleaved rounds.

    python bench/hk2/compare.py [--only gemm|attn] [--rounds 10]
"""
import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import triton_ref  # noqa: E402
from gemm import matmul as hk2_matmul  # noqa: E402

import hk2 as hk  # noqa: E402
from hk2.kernels.attention import attention as hk2_attention  # noqa: E402


def timeit(fn, reps=5):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def race(arms, rounds):
    for f in arms.values():
        f()
    ts = {n: [] for n in arms}
    for _ in range(rounds):
        for n, f in arms.items():
            ts[n].append(timeit(f))
    return {n: statistics.median(t) for n, t in ts.items()}


def gemm_cfgs(arch):
    if arch.startswith("gfx1"):
        return [(128, 256, 64, 8), (128, 128, 64, 4)]
    return [(256, 256, 64, 8), (256, 128, 64, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"# {arch}: TF/s, median of {a.rounds} interleaved rounds")
    if a.only in ("", "gemm"):
        for m, n, k in ((4096, 4096, 4096), (8192, 8192, 8192), (2048, 8192, 4096)):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            arms = {"torch": lambda: x @ w.T, "triton": lambda: triton_ref.matmul_nt(x, w)}
            for bm, bn, bk, nw in gemm_cfgs(arch):
                c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                arms[f"hk2 {bm}x{bn}x{bk}x{nw}"] = (lambda c=c, bm=bm, bn=bn, bk=bk, nw=nw: hk2_matmul[
                    (hk.cdiv(m, bm), hk.cdiv(n, bn))](x, w, c, BM=bm, BN=bn, BK=bk, TRANS_B=True, num_warps=nw))
            ref = (x @ w.T).float()
            assert (triton_ref.matmul_nt(x, w).float() - ref).abs().max() < 1e-2 * ref.abs().max()
            t = race(arms, a.rounds)
            fl = 2 * m * n * k
            print(f"gemm {m}x{n}x{k}: " + " | ".join(f"{k_} {fl / v / 1e9:.0f}" for k_, v in t.items()))
    if a.only in ("", "attn"):
        for causal in (False, True):
            for B, H, S, D in ((1, 32, 4096, 128), (4, 32, 2048, 128), (1, 32, 8192, 64)):
                q, k_, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
                ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k_.float(), v.float(),
                                                                       is_causal=causal)
                for name, f in (("triton", triton_ref.attention), ("hk2", hk2_attention)):
                    err = (f(q, k_, v, causal=causal).float() - ref).abs().max().item()
                    assert err < 5e-2, (name, err)
                arms = {"sdpa": lambda: torch.nn.functional.scaled_dot_product_attention(q, k_, v, is_causal=causal),
                        "triton": lambda: triton_ref.attention(q, k_, v, causal=causal),
                        "hk2": lambda: hk2_attention(q, k_, v, causal=causal)}
                t = race(arms, a.rounds)
                fl = 4 * B * H * S * S * D * (0.5 if causal else 1)
                print(f"attn {B}x{H}x{S}x{D} causal={causal}: " +
                      " | ".join(f"{k2} {fl / v2 / 1e9:.1f}" for k2, v2 in t.items()))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
