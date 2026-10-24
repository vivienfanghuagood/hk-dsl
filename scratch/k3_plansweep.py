"""Tail-split plans swept per grid size: (n_full, parts, n_split) for nblk Q blocks, all units on one
interleaved clock. K3.plan is monkeypatched; _PLANS cache cleared per plan."""
import torch
from hk.ops import attn_k3 as K3
_RULE = K3.plan


def t_ms(f, iters=20, trials=7):
    f(); f(); torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            f()
        e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters)
    return sorted(ts)[len(ts) // 2]


for h, nq, nkv in [(8, 4096, 4141), (24, 4307, 4307), (16, 4096, 4141), (32, 4096, 4141), (56, 14720, 14720)]:
    nb = h * -(-nq // 128)
    q = torch.randn((1, nq, h, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k, v = (torch.randn((1, nkv, h, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(2))
    cands = {"none": (nb, 1, 0)}
    cands["rule"] = _RULE(nb, q.device)
    for slots in (144, 192):
        rem = nb % slots
        for p in (2, 3, 4):
            if rem:
                cands[f"s{slots}p{p}"] = (nb - rem, p, rem)
    res = {}
    for nm, pl in cands.items():
        K3.plan = lambda nblk, device, pl=pl: pl
        K3._PLANS.clear()
        res[nm] = t_ms(lambda: K3.attention(q, k, v), iters=20 if nb < 2000 else 3)
    best = min(res, key=res.get)
    print(f"H{h} q{nq} kv{nkv} nblk {nb}: " + "  ".join(f"{n} {t:.3f}" for n, t in res.items()) + f"  -> {best}", flush=True)
