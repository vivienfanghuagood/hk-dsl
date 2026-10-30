"""hk2 flash attention forward vs torch SDPA (AOTriton / CK flash on ROCm). bf16, (B, H, S, D).

    python bench/hk2/attention.py [--cases 1x32x4096x128,...] [--causal] [--cfg BMxBNxW,...]
"""
import argparse
import statistics

import torch

from hk2.kernels.attention import attention, attn_fwd


def timeit(fn, reps=5):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1x32x4096x128,4x32x2048x128,1x32x8192x64")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--cfg", default="")
    ap.add_argument("--rounds", type=int, default=10)
    a = ap.parse_args()
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    print(f"# {arch} attention fwd bf16 causal={a.causal}")
    for case in a.cases.split(","):
        B, H, S, D = map(int, case.split("x"))
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=a.causal)
        ref = sdpa().float()
        arms = {"sdpa": sdpa}
        cfgs = [c for c in a.cfg.split(",") if c] or [""]
        for c in cfgs:
            kw = {}
            if c:
                bm, bn, w = map(int, c.split("x"))
                kw = dict(BM=bm, BN=bn, num_warps=w)
            name = "hk2" + (f" {c}" if c else "")
            try:
                out = attention(q, k, v, causal=a.causal, **kw)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: {type(e).__name__}: {str(e)[:160]}")
                continue
            err = (out.float() - ref).abs().max().item()
            print(f"  {name}: max err {err:.3e}")
            arms[name] = (lambda kw=kw: attention(q, k, v, causal=a.causal, **kw))
        for f in arms.values():
            f()
        ts = {n: [] for n in arms}
        for _ in range(a.rounds):
            for n, f in arms.items():
                ts[n].append(timeit(f))
        flop = 4 * B * H * S * S * D * (0.5 if a.causal else 1)
        base = statistics.median(ts["sdpa"])
        line = f"{case}: sdpa {flop / base / 1e9:.1f} TF/s"
        for n, t in ts.items():
            if n != "sdpa":
                t = statistics.median(t)
                line += f" | {n} {flop / t / 1e9:.1f} TF/s ({base / t:.3f}x)"
        print(line)
    for key, kern in attn_fwd.cache.items():
        print("  stats", {k: key[i] for i, k in ((0, "arch"), (1, "warps"))}, kern.stats)


if __name__ == "__main__":
    main()
