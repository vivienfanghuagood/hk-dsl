"""gfx950 attention: hk.ops.attn_cdna's workgroup numberings against each other (and AITER's).

    python3 tools/hk-bench/attn_mapping.py [--only BxHxHKVxNxD,...] [--causal 0,1] [--out f.json]

Arms, in one process, interleaved (every other round reversed), from the same values:

  block_first   attn_cdna as shipped: HipKittens' numbering, a query block of every head
                before the next (grid (H, N/256, B))
  spatial       attn_cdna(mapping="spatial"): a (batch, KV head) cluster an XCD at a time
                (arXiv 2511.02132; AITER's remap_workgroup_spatial)
  aiter         AITER's Triton MHA (aiter.ops.triton.attention.mha.flash_attn_func), its
  aiter_spatial   SWIZZLE "default" and "spatial" -- a reference, if the image's AITER has it

The two hk arms compute every tile the same way, so their outputs must be bitwise equal
(checked on every shape, before timing). Each arm is also checked against fp32 on three
32-row query blocks of the first and last batch, three heads. TFLOP/s from the best round,
4 B N^2 H D (halved under causal), as tools/hk-bench/attn_cdna.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

#: MHA at the head counts and lengths where the numbering matters (arXiv 2511.02132, AITER
#: #3936), then GQA controls whose KV heads already sit one an XCD, and a TP-sharded one.
CONFIGS = tuple((1, h, h, n, 128) for h in (16, 40, 64, 128) for n in (8192, 32768, 65536, 131072)) + (
    (1, 64, 8, 8192, 128), (1, 32, 8, 32768, 128), (8, 8, 1, 32768, 128))


def reference(q, k, v, causal, batches, heads):
    """fp32 o for three 32-row query blocks of `batches` x `heads`; q, k, v (B, H, N, D)."""
    import torch

    _, h, n, d = q.shape
    g = h // k.shape[1]
    rows = torch.cat([torch.arange(s, s + 32, device=q.device)
                      for s in (0, (n // 2) // 32 * 32, n - 32)])
    want = {}
    for b in batches:
        for hd in heads:
            s = (q[b, hd, rows].float() @ k[b, hd // g].float().T) / math.sqrt(d)
            if causal:
                s = s.masked_fill(torch.arange(n, device=q.device) > rows[:, None], float("-inf"))
            want[b, hd] = torch.softmax(s, -1) @ v[b, hd // g].float()
    return rows, want


def interleaved(arms, rounds, calls):
    import torch

    out = {n: [] for n in arms}
    order = list(arms)
    for fn in arms.values():
        fn()
    torch.cuda.synchronize()
    for r in range(rounds):
        for name in (order if r % 2 == 0 else order[::-1]):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(calls):
                arms[name]()
            e.record()
            torch.cuda.synchronize()
            out[name].append(s.elapsed_time(e) / calls)
    return out


def _aiter():
    """(flash_attn_func, mha_set_swizzle) from the image's AITER, or the reason it has none."""
    try:
        from aiter.ops.triton.attention.mha import flash_attn_func, mha_set_swizzle
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return (flash_attn_func, mha_set_swizzle), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated BxHxHKVxNxD")
    ap.add_argument("--causal", default="0,1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--arms", default="block_first,spatial,aiter,aiter_spatial")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    from hk.ops import attn_cdna

    want_arms = args.arms.split(",")
    aiter, why = (None, "not asked") if not any(a.startswith("aiter") for a in want_arms) \
        else _aiter()
    meta = dict(device=torch.cuda.get_device_name(),
                arch=torch.cuda.get_device_properties(0).gcnArchName,
                torch=torch.__version__, hip=torch.version.hip, date=time.strftime("%Y-%m-%d"),
                rounds=args.rounds, aiter=why or "available")
    print(json.dumps(meta), flush=True)

    configs = CONFIGS
    if args.only:
        configs = [tuple(int(x) for x in s.split("x")) for s in args.only.split(",")]
    results = []
    dev = torch.device("cuda")
    for (b, h, hkv, n, d) in configs:
        for causal in (bool(int(c)) for c in args.causal.split(",")):
            flop = 4.0 * b * n * n * h * d / (2 if causal else 1)
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16, generator=g)
            k = torch.randn(b, hkv, n, d, device=dev, dtype=torch.bfloat16, generator=g)
            v = torch.randn(b, hkv, n, d, device=dev, dtype=torch.bfloat16, generator=g)
            outs = {}
            arms = {}
            for m in want_arms:
                if m in ("block_first", "spatial"):
                    kern = attn_cdna.kernel_for(d, causal, **({} if m == "block_first" else {"mapping": m}))
                    o = outs[m] = torch.empty_like(q)
                    arms[m] = (lambda kern=kern, o=o: kern(q, k, v, o) or o)
            if aiter is not None:
                fa, set_swizzle = aiter
                qn, kn, vn = (x.transpose(1, 2).contiguous() for x in (q, k, v))
                for a, sw in (("aiter", "default"), ("aiter_spatial", "spatial")):
                    if a in want_arms:
                        def run(sw=sw):
                            set_swizzle(sw)
                            return fa(qn, kn, vn, causal=causal).transpose(1, 2)
                        arms[a] = run
            rows, want = reference(q, k, v, causal, sorted({0, b - 1}), sorted({0, h // 2, h - 1}))
            errs, ok, bitwise = {}, {}, None
            for name, fn in arms.items():
                try:
                    o = fn()
                    torch.cuda.synchronize()
                    errs[name] = max(((o[bi, hd, rows].float() - w).abs().max()
                                      / w.abs().max().clamp_min(1e-30)).item()
                                     for (bi, hd), w in want.items())
                except Exception as e:  # noqa: BLE001
                    errs[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: {errs[name]}", flush=True)
                    continue
                if errs[name] < 2e-2:
                    ok[name] = fn
                else:
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: FAILED the fp32 check "
                          f"(rel err {errs[name]:.3g})", flush=True)
            if "block_first" in outs and len(outs) > 1:
                bitwise = all(bool(torch.equal(outs["block_first"], o)) for o in outs.values())
                if not bitwise:
                    print(f"  {(b, h, hkv, n, d)} causal={causal}: a spatial arm is NOT bitwise "
                          f"block_first's", flush=True)
            t0 = time.time()
            for fn in ok.values():
                fn()
            torch.cuda.synchronize()
            calls = max(2, min(20, int(0.1 / max((time.time() - t0) / max(len(ok), 1), 1e-4))))
            times = interleaved(ok, args.rounds, calls)
            summ = {nm: {"best_ms": min(t), "median_ms": statistics.median(t),
                         "tflops": flop / (min(t) * 1e-3) / 1e12,
                         "spread": (max(t) - min(t)) / min(t),
                         "rounds": [round(x, 6) for x in t]} for nm, t in times.items()}
            rec = dict(b=b, h=h, h_kv=hkv, n=n, d=d, causal=causal, calls=calls, err=errs,
                       bitwise=bitwise, eager=summ)
            pairs = [(a, "block_first") for a in summ if a.startswith("spatial")]
            for x, y in pairs + [("aiter_spatial", "aiter")]:
                if x in summ and y in summ:
                    rec[f"{x}_over_{y}"] = summ[x]["tflops"] / summ[y]["tflops"]
            results.append(rec)
            line = "  ".join(f"{a} {s['tflops']:.0f} ({100 * s['spread']:.1f}%)"
                             for a, s in summ.items())
            ratios = "  ".join(f"{k_} {v_:.3f}" for k_, v_ in rec.items() if "_over_" in k_)
            print(f"B{b} H{h} Hkv{hkv} N{n} D{d} causal={int(causal)} bitwise={bitwise} "
                  f"TFLOP/s: {line}  {ratios}", flush=True)
            del q, k, v, outs, arms, ok, want
            torch.cuda.empty_cache()
    out = dict(meta=meta, results=results)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
