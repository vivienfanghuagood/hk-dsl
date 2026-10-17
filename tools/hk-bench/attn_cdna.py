"""hk's gfx950 flash attention forward against HipKittens', FlyDSL's and torch's.

    python3 tools/hk-bench/attn_cdna.py [--only BxHxHKVxNxD,...] [--rounds 7] [--out f.json]

Arms, each in its own operand layout, all from the same values:

  hk          hk.ops.attn_cdna, (B, H, N, D) -- SDPA's
  hipkittens  hk.ops.hipkittens.gqa_fwd, HipKittens' CDNA4 GQA forward, (B, N, H, D)
  flydsl      FlyDSL's flydsl_flash_attn_func, (B, N, H, D)
  sdpa        F.scaled_dot_product_attention(enable_gqa=True), (B, H, N, D)

HipKittens' attention builds only with ROCm 7.1's hipcc and hk's with the
image's: set HK_HIPCC_71 to 7.1's hipcc and the harness builds that arm with
it (hk.runtime reads HK_HIPCC per build). FlyDSL needs its wheel on
PYTHONPATH and FLYDSL_ROOT at its checkout. An arm that cannot run is left
out with the reason.

Every arm is checked against fp32 first (three 32-row query blocks of the
first and last batch, every head); one that is off by more than 2e-2 is not
timed. Then rounds of a few calls each, interleaved in one process, every
other round reversed; TFLOP/s from the best round, 4 B N^2 H D (halved under
causal).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

#: The shapes docs/design/cdna-hipkittens-ops.md measured HipKittens' GQA at.
CONFIGS = tuple((16, 64, 8, n, 128) for n in (1024, 2048, 4096, 8192)) + (
    (1, 32, 8, 4096, 128), (16, 64, 8, 4096, 64))


def rel_err(got, want) -> float:
    return ((got.float() - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


def reference(q, k, v, causal, batches):
    """fp32 o, (rows, H, D), for three 32-row query blocks of each batch in
    `batches`; q, k, v in (B, N, H, D)."""
    import torch

    _, n, h, d = q.shape
    g = h // k.shape[2]
    rows = torch.cat([torch.arange(s, s + 32, device=q.device)
                      for s in (0, (n // 2) // 32 * 32, n - 32)])
    want = {}
    for b in batches:
        kf = k[b].float().repeat_interleave(g, dim=1)
        vf = v[b].float().repeat_interleave(g, dim=1)
        s = torch.einsum("rhd,nhd->hrn", q[b, rows].float(), kf) / math.sqrt(d)
        if causal:
            s = s.masked_fill(torch.arange(n, device=q.device) > rows[:, None], float("-inf"))
        want[b] = torch.einsum("hrn,nhd->rhd", torch.softmax(s, -1), vf)
    return rows, want


def interleaved(arms, rounds, calls):
    """{name: [ms per call, one per round]}."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated BxHxHKVxNxD")
    ap.add_argument("--causal", default="0,1", help="which causal settings, 0,1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--arms", default="hk,hipkittens,flydsl,sdpa")
    ap.add_argument("--hk-opts", default="", help="attn_cdna options, k=v,...")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-check", action="store_true",
                    help="time arms that fail the fp32 check too (diagnostics only)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    from hk.ops import attn_cdna
    from hk.ops import hipkittens as hkk

    want_arms = set(args.arms.split(","))
    hk_opts = {}
    for kv in filter(None, args.hk_opts.split(",")):
        key, val = kv.split("=")
        hk_opts[key] = {"True": True, "False": False}.get(val, val)
        try:
            hk_opts[key] = float(val) if "." in val else int(val)
        except ValueError:
            pass
    fly = None
    if "flydsl" in want_arms:
        try:
            root = os.environ.get("FLYDSL_ROOT", "/home/hfang/hk-local/FlyDSL")
            sys.path.append(root)
            from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as fly
        except Exception as e:  # noqa: BLE001
            print(f"flydsl: not available ({type(e).__name__}: {e})", flush=True)
    hip71 = os.environ.get("HK_HIPCC_71", "")
    meta = dict(device=torch.cuda.get_device_name(),
                arch=torch.cuda.get_device_properties(0).gcnArchName,
                torch=torch.__version__, hip=torch.version.hip,
                hipcc=hkk.hipcc_version(), hipcc_hipkittens=hip71 or "(default)",
                date=time.strftime("%Y-%m-%d"), rounds=args.rounds, hk_opts=hk_opts)
    print(json.dumps(meta), flush=True)

    def hipkittens_build(fn):
        """Run `fn` with HipKittens' hipcc, so its first call builds with it."""
        if not hip71:
            return fn()
        old = os.environ.get("HK_HIPCC")
        os.environ["HK_HIPCC"] = hip71
        try:
            return fn()
        finally:
            if old is None:
                del os.environ["HK_HIPCC"]
            else:
                os.environ["HK_HIPCC"] = old

    configs = CONFIGS
    if args.only:
        configs = [tuple(int(x) for x in s.split("x")) for s in args.only.split(",")]
    results = []
    dev = torch.device("cuda")
    for (b, h, hkv, n, d) in configs:
        for causal in (bool(int(c)) for c in args.causal.split(",")):
            flop = 4.0 * b * n * n * h * d / (2 if causal else 1)
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.randn(b, n, h, d, device=dev, dtype=torch.bfloat16, generator=g)
            k = torch.randn(b, n, hkv, d, device=dev, dtype=torch.bfloat16, generator=g)
            v = torch.randn(b, n, hkv, d, device=dev, dtype=torch.bfloat16, generator=g)
            qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            ot = torch.empty_like(qt)
            rows, want = reference(q, k, v, causal, sorted({0, b - 1}))
            kern = attn_cdna.kernel_for(d, causal, **hk_opts)
            arms = {}
            if "hk" in want_arms:
                arms["hk"] = (lambda: kern(qt, kt, vt, ot) or ot, True)
            if "hipkittens" in want_arms:
                arms["hipkittens"] = (lambda: hkk.gqa_fwd(q, k, v, causal=causal), False)
            if fly is not None:
                arms["flydsl"] = (lambda: fly(q, k, v, causal=causal, num_kv_heads=hkv), False)
            if "sdpa" in want_arms:
                arms["sdpa"] = (lambda: F.scaled_dot_product_attention(
                    qt, kt, vt, is_causal=causal, enable_gqa=True), True)
            errs, ok = {}, {}
            for name, (fn, bhnd) in arms.items():
                try:
                    o = hipkittens_build(fn) if name == "hipkittens" else fn()
                    torch.cuda.synchronize()
                    if bhnd:
                        o = o.transpose(1, 2)
                    errs[name] = max(rel_err(o[bi, rows], w) for bi, w in want.items())
                except Exception as e:  # noqa: BLE001
                    errs[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: {errs[name]}", flush=True)
                    continue
                if errs[name] < 2e-2 or args.no_check:
                    ok[name] = fn
                if errs[name] >= 2e-2:
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: FAILED the fp32 "
                          f"check (rel err {errs[name]:.3g})", flush=True)
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
            rec = dict(b=b, h=h, h_kv=hkv, n=n, d=d, causal=causal, calls=calls,
                       err=errs, eager=summ)
            others = [summ[a]["tflops"] for a in ("hipkittens", "flydsl") if a in summ]
            if "hk" in summ and others:
                rec["hk_over_best"] = summ["hk"]["tflops"] / max(others)
            results.append(rec)
            line = "  ".join(f"{a} {s['tflops']:.0f} ({100 * s['spread']:.1f}%)"
                             for a, s in summ.items())
            ratio = f"  hk/best {rec['hk_over_best']:.3f}" if "hk_over_best" in rec else ""
            print(f"B{b} H{h} Hkv{hkv} N{n} D{d} causal={int(causal)} TFLOP/s: {line}{ratio}",
                  flush=True)
            del q, k, v, qt, kt, vt, ot, want
            torch.cuda.empty_cache()
    out = dict(meta=meta, results=results)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
