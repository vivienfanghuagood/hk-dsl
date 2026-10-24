"""K3 variants A/B: each checked (max-rel, V=1, 5 repeats bitwise) then timed interleaved in one process,
with c1 and the current hk.ops.sdpa in the same rotation.

    python scratch/k3_ab.py [--shapes qwen h3] [--only NAME ...]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k3_attn  # noqa: E402

SHAPES = {"odd": (1, 3, 1408, 128), "qwen": (1, 32, 4096, 128), "h3": (1, 56, 14720, 128)}
VARIANTS = {
    "lazy8": dict(lazy=8.0),
    "abl_kread": dict(lazy=8.0, _ablate="kread"),
    "abl_vread": dict(lazy=8.0, _ablate="vread"),
    "abl_kvread": dict(lazy=8.0, _ablate="kread vread"),
    "abl_noexp": dict(lazy=8.0, _ablate="noexp"),
}
ABLATIONS = ("abl_",)   # timed without the checks (deliberately wrong)


def build(only):
    out = {"c1": k3_attn.ATTN}
    for name, kw in VARIANTS.items():
        if only and name not in only:
            continue
        out[name] = k3_attn.attn_kernel(f"k3v_{name}", **kw)
    return out


def check(name, kern, q, k, v, r):
    try:
        o = k3_attn.attention(q, k, v, attn=kern).clone()
        reps = [k3_attn.attention(q, k, v, attn=kern).clone() for _ in range(5)]
        ones = k3_attn.attention(q, k, torch.ones_like(v), attn=kern)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"  {name:14s} error {type(e).__name__}: {str(e)[:300]}", flush=True)
        return False
    err = ((o.float() - r).abs().max() / r.abs().max()).item()
    det = all(torch.equal(o, x) for x in reps)
    ones_ok = bool((ones == 1).all().item())
    ok = err <= 1e-2 and det and ones_ok
    if not ok:
        print(f"  {name:14s} FAIL max-rel {err:.2e} det {det} ones {ones_ok}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=["odd", "qwen", "h3"])
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    import hk.ops.sdpa as S

    kerns = build(a.only)
    for sname in a.shapes:
        shape = SHAPES[sname]
        b, h, n, d = shape
        torch.manual_seed(0)
        q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        good = {nm: kn for nm, kn in kerns.items()
                if nm.startswith(ABLATIONS) or check(nm, kn, q, k, v, r)}
        del r
        fns = {nm: (lambda kn=kn: k3_attn.attention(q, k, v, attn=kn)) for nm, kn in good.items()}
        fns["hk"] = lambda: S.attention(q, k, v)
        flops = 4 * b * h * n * n * d
        res = {nm: [] for nm in fns}
        iters = 10 if n <= 4096 else 3
        for _ in range(7):
            for nm, f in fns.items():
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(iters):
                    f()
                e1.record()
                torch.cuda.synchronize()
                res[nm].append(e0.elapsed_time(e1) / iters)
        print(sname, shape, flush=True)
        for nm, ts in res.items():
            t = sorted(ts)[len(ts) // 2]
            print(f"  {nm:14s} {t:9.3f} ms {flops / t / 1e9:6.1f} TF/s", flush=True)


if __name__ == "__main__":
    main()
