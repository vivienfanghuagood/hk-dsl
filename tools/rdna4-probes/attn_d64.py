"""hk attention at d64 on gfx1201, GQA 4 (32 q / 8 kv heads) and MHA 16: tilings vs gfx1100's, causal and full,
n 1024-8192; one process, interleaved (CUDA events)."""
import sys, statistics
import torch, torch.nn.functional as F
sys.path.insert(0, (sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])) + "/python")
from hk.ops import attn as A
def bench_all(fns, rounds=7, it=10):
    t = {a: [] for a in fns}; names = list(fns)
    for r in range(rounds):
        for a in names[r % len(names):] + names[:r % len(names)]:
            f = fns[a]; f(); f(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(it): f()
            e1.record(); torch.cuda.synchronize(); t[a].append(e0.elapsed_time(e1) / it)
    return {a: statistics.median(v) for a, v in t.items()}
d = 64
for h, hkv in ((32, 8),):
    for n in (1024, 2048, 4096, 8192):
        for causal in (True, False):
            q = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, hkv, n, d, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True)
            fns = {}
            cfgs = [dict(), dict(warps=16, heads_first=True), dict(warps=16), dict(warps=12, heads_first=True)]
            if causal:
                cfgs += [dict(warps=16, heads_first=True, warp_skip=True), dict(warps=16, kv_block=64, heads_first=True, warp_skip=True),
                         dict(warps=12, heads_first=True, warp_skip=True)]
            for c in cfgs:
                name = "d64_" + ("c" if causal else "f") + "_" + ("_".join(f"{a}{int(b)}" for a, b in c.items()) or "gfx1100")
                kern = A.attn_kernel(name, head_dim=d, causal=causal, **c)
                o = torch.empty_like(q); kern(q, k, v, o); torch.cuda.synchronize()
                if ((o.float() - ref).abs().max() / ref.abs().max()).item() > 2e-2:
                    print(name, "WRONG"); continue
                fns[name[6:]] = (lambda kern=kern, o=o: kern(q, k, v, o))
            t = bench_all(fns)
            base = t["gfx1100"]
            print(f"n{n} {'causal' if causal else 'full'}: gfx1100={base:.3f} ms | " + "  ".join(f"{a}={v:.3f} ({base/v:.2f}x)" for a, v in sorted(t.items(), key=lambda kv: kv[1])[:4]), flush=True)
