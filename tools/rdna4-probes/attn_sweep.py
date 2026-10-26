"""hk attention tilings on gfx1201 (hk.ops.attn._TILING) against AITER's FlyDSL kernel (aiter on PYTHONPATH):
warps, kv_block, heads_first, warp_skip; 32 query heads over 8 KV heads, d128, one process, interleaved."""
import sys, statistics
import torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
import hk
from hk.ops import attn as A
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_func

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

h, hkv, d = 32, 8, 128
for n, causal in ((1024, True), (2048, True), (4096, True), (8192, True), (4096, False)):
    torch.manual_seed(0)
    q = torch.randn(1, h, n, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, hkv, n, d, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
    qb = q.transpose(1, 2).contiguous()
    kb, vb = (x.repeat_interleave(h // hkv, 1).transpose(1, 2).contiguous() for x in (k, v))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal, enable_gqa=True)
    fns = {}
    for hf, ws, w, kvb in ((0, 0, 12, 32), (1, 0, 16, 64), (1, 1, 16, 64), (1, 0, 16, 32), (1, 1, 16, 32), (1, 1, 12, 32), (1, 1, 8, 32), (1, 1, 8, 64)):
        if True:
            if True:
                name = f"sw3_{'c' if causal else 'f'}_hf{int(hf)}_ws{ws}_w{w}_kv{kvb}"
                kern = A.attn_kernel(name, head_dim=d, causal=causal, warps=w, kv_block=kvb, heads_first=bool(hf), warp_skip=bool(ws))
                o = torch.empty_like(q); kern(q, k, v, o); torch.cuda.synchronize()
                err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
                if err > 2e-2:
                    print(name, "WRONG", err); continue
                fns[name[6:]] = (lambda kern=kern, o=o: kern(q, k, v, o))
    out = {}
    fns["aiter"] = lambda: out.__setitem__("a", flydsl_flash_attn_func(qb, kb, vb, causal=causal))
    t = bench_all(fns)
    fl = 4 * h * n * n * d // (2 if causal else 1)
    print(f"== n{n} {'causal' if causal else 'full'}")
    for a, ms in sorted(t.items(), key=lambda x: x[1]):
        print(f"  {ms:.4f} ms {fl / ms / 1e9:6.1f} TF  {a}", flush=True)
