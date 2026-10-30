"""hk2 GEMM vs torch (hipBLASLt): C = A @ B^T (the nn.Linear layout) or A @ B, bf16, f32 accumulation.

    python bench/hk2/gemm.py [--shapes 4096x4096x4096,...] [--configs 128x128x64x4,...]

Timing: device events, interleaved rounds (hk2 config, torch, ...), median per arm.
"""
import argparse
import statistics
import sys

import torch

import hk2 as hk


@hk.jit
def matmul(a, b, c, BM: hk.constexpr = 128, BN: hk.constexpr = 128, BK: hk.constexpr = 64,
           TRANS_B: hk.constexpr = True):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN), hk.f32)
    for k in range(0, a.shape[1], BK):
        x = hk.load(a, (m0, k), (BM, BK))
        if TRANS_B:
            y = hk.load(b, (n0, k), (BN, BK)).T
        else:
            y = hk.load(b, (k, n0), (BK, BN))
        acc = hk.dot(x, y, acc)
    hk.store(c, (m0, n0), acc.to(c.dtype))


def timeit(fn, rounds, reps=10):
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rounds)]
    for i in range(rounds):
        st[i].record()
        for _ in range(reps):
            fn()
        en[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) / reps for s, e in zip(st, en)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="4096x4096x4096,8192x8192x8192,2048x8192x4096")
    ap.add_argument("--configs", default="128x128x64x4,256x128x64x8,256x256x64x8,128x128x32x4")
    ap.add_argument("--nn", action="store_true", help="B is KxN (default: B is NxK, as nn.Linear)")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"# {arch}  {'A@B' if args.nn else 'A@B^T'}  bf16")
    for shp in args.shapes.split(","):
        m, n, k = map(int, shp.split("x"))
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) if args.nn else \
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = a @ (b if args.nn else b.T)
        arms = {"torch": lambda: torch.matmul(a, b if args.nn else b.T)}
        info = {}
        for cfg in args.configs.split(","):
            bm, bn, bk, nw = map(int, cfg.split("x"))
            c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            grid = (hk.cdiv(m, bm), hk.cdiv(n, bn))
            try:
                kern = matmul.warmup(a, b, c, BM=bm, BN=bn, BK=bk, TRANS_B=not args.nn, num_warps=nw)
                matmul[grid](a, b, c, BM=bm, BN=bn, BK=bk, TRANS_B=not args.nn, num_warps=nw)
            except Exception as e:  # noqa: BLE001
                print(f"  {cfg}: {type(e).__name__}: {str(e)[:200]}")
                continue
            err = (c.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
            info[cfg] = (kern.stats, err)
            arms[cfg] = (lambda c=c, grid=grid, bm=bm, bn=bn, bk=bk, nw=nw:
                         matmul[grid](a, b, c, BM=bm, BN=bn, BK=bk, TRANS_B=not args.nn, num_warps=nw))
        for f in arms.values():
            f()
        times = {name: [] for name in arms}
        for _ in range(args.rounds):
            for name, f in arms.items():
                times[name] += timeit(f, 1)
        flop = 2 * m * n * k
        base = statistics.median(times["torch"])
        print(f"{m}x{n}x{k}: torch {flop / base / 1e9:.1f} TF/s ({base * 1e3:.0f} us)")
        for name, ts in times.items():
            if name == "torch":
                continue
            t = statistics.median(ts)
            st, err = info[name]
            print(f"  {name:>14}: {flop / t / 1e9:7.1f} TF/s  {base / t:5.3f}x torch  relerr {err:.1e}  "
                  f"vgpr {st.get('vgprs')} agpr {st.get('agprs')} spill {st.get('scratch')} "
                  f"occ {st.get('occupancy')} lds {st.get('lds')} bar {st.get('barriers')}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
