"""Training attention on gfx1201: fwd+bwd of hk.ops.attn_bwd.attention against torch SDPA (AOTriton
on ROCm), one process, interleaved round by round, inputs BSHD as a QKV projection writes them.

    HK_ARCH=gfx1201 python tools/attn-bwd-gfx1201/bench.py [--rounds 5]

Each shape is checked first (rel-L2 of o, dq, dk, dv against SDPA's own outputs, bound 2e-2), then
timed with CUDA events: the forward alone, and forward + backward (torch.autograd.grad), median of
20 per round; the table is the median over rounds. TF counts 4 B H S^2 D for the forward and 2.5x
that for the backward, halved for causal.
"""
import argparse
import statistics

import torch
import torch.nn.functional as F

from hk.ops import attn_bwd

SHAPES = [  # name, B, S, H, H_kv, D, causal
    ("P1", 2, 2048, 16, 16, 128, True),     # a 1.28B trainer: hidden 2048, 16 heads, mbs 2
    ("P2", 1, 2048, 16, 16, 128, True),
    ("NC", 2, 2048, 16, 16, 128, False),
    ("G4", 2, 2048, 16, 4, 128, True),      # GQA 4
    ("L1B", 1, 4096, 32, 8, 64, True),      # head_dim 64, GQA 4
    ("L8B", 1, 4096, 32, 8, 128, True),
]


def med(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal,
        enable_gqa=k.shape[2] != q.shape[2]).transpose(1, 2)


def hk_attn(q, k, v, causal):
    return attn_bwd.attention(q, k, v, causal=causal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    rounds = ap.parse_args().rounds
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print("| shape | B S H/H_kv D mask | SDPA fwd+bwd ms | hk fwd+bwd ms | x "
          "| hk fwd ms | hk bwd ms (TF) |")
    print("|---|---|---|---|---|---|---|")
    for name, b, s, h, hkv, d, causal in SHAPES:
        # q, k and v as views of one fused projection output, read in place by hk
        qkv = torch.randn(b, s, h + 2 * hkv, d, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
        q, k, v = qkv[:, :, :h], qkv[:, :, h:h + hkv], qkv[:, :, h + hkv:]
        do = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
        want = [sdpa(q, k, v, causal)]
        want += torch.autograd.grad(want[0], (q, k, v), do)
        got = [hk_attn(q, k, v, causal)]
        got += torch.autograd.grad(got[0], (q, k, v), do)
        for n, g, w in zip(("o", "dq", "dk", "dv"), got, want):
            r = ((g.float() - w.float()).norm() / w.float().norm()).item()
            assert r < 2e-2, f"{name} {n}: rel-L2 {r:.2e} against SDPA"
        fns = {}
        for tag, f in (("sdpa", sdpa), ("hk", hk_attn)):
            fns[tag + "_f"] = lambda f=f: f(q, k, v, causal)
            fns[tag + "_fb"] = lambda f=f: torch.autograd.grad(f(q, k, v, causal), (q, k, v), do)
        t = {key: [] for key in fns}
        for _ in range(rounds):
            for key, fn in fns.items():
                t[key].append(med(fn))
        m = {key: statistics.median(x) for key, x in t.items()}
        fl = 4 * b * h * s * s * d * (0.5 if causal else 1.0)
        bwd = m["hk_fb"] - m["hk_f"]
        print(f"| {name} | {b} {s} {h}/{hkv} {d} {'causal' if causal else 'full'} "
              f"| {m['sdpa_fb']:.3f} | {m['hk_fb']:.3f} | {m['sdpa_fb'] / m['hk_fb']:.2f} "
              f"| {m['hk_f']:.3f} | {bwd:.3f} ({2.5 * fl / bwd / 1e9:.0f}) |", flush=True)


if __name__ == "__main__":
    main()
