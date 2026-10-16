#!/usr/bin/env python3
"""HipKittens' CDNA4 GEMM and GQA forward (hk.ops.hipkittens) against what torch
runs for the same call, on an MI355X.

GEMM is C = A @ B.T in F.linear's layout, A (M, K) and B (N, K). The arms:

  hk           hk.ops.gemm.matmul, hk's own gfx950 GEMM (hk.ops.gemm_cdna), with --hk
  hipkittens   hk.ops.hipkittens.gemm_bf16 (M, N multiples of 256, K of 128)
  hipblaslt    F.linear, torch's default BLAS on gfx950
  rocblas      F.linear with preferred_blas_library("cublas")
  matmul_nn    torch.matmul(A, B) with B stored (K, N), the other common layout
  flydsl       FlyDSL's gemm_a16w16 ("nt", 256x256x64 HTI), with --flydsl

GQA forward is softmax(q k^T / sqrt(D)) v, with H / H_KV query heads per KV head:

  hipkittens   hk.ops.hipkittens.gqa_fwd, q (B, N, H, D) and k, v (B, N, H_KV, D)
  sdpa         F.scaled_dot_product_attention(enable_gqa=True) on (B, H, N, D)
               contiguous tensors, the backend torch picks
  sdpa_flash, sdpa_efficient
               the same with that backend forced (aotriton's flash and
               memory-efficient kernels on ROCm)
  sdpa_views   sdpa on transposed views of the (B, N, H, D) tensors, which is
               what a caller holding hk's layout pays without a copy
  flydsl       FlyDSL's flydsl_flash_attn_func (BSHD), with --flydsl

Every arm is checked against fp32 before it is timed: the full product for
GEMM, and three 32-row query blocks of the first and last batch for attention.
An arm that fails is reported and not timed. The arms then interleave in one
process: every round times each arm once, under one pair of events, and
alternate rounds run the arms in reverse order. For the GEMM, a round is one
pass over enough distinct weight copies to overflow the MI355X's 256 MB
Infinity Cache three times. The tables give the best round and the median;
TFLOP/s and ratios come from the best. `--graphs` also replays each GEMM pass
as a CUDA graph, which takes the host out of the small shapes.

    python3 tools/hk-bench/hipkittens_cdna.py gemm --flydsl --graphs --out gemm.json
    HK_HIPCC=/opt/rocm-7.1.0/bin/hipcc python3 tools/hk-bench/hipkittens_cdna.py gqa --flydsl --out gqa.json

HK_HIPCC matters here (see hk.ops.hipkittens). On the MI355X the GEMM is right
built by ROCm 7.2.3 and wrong built by 7.1. The attention kernels build only
under 7.1. FlyDSL comes from $FLYDSL_ROOT (its kernels/) plus an installed
flydsl wheel; aiter is not installable here.
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

#: Three times the MI355X's Infinity Cache (MALL): what one GEMM pass streams.
ROTATE_BYTES = 3 * 256 * 2**20

SQUARES = (1024, 2048, 3072, 4096, 6144, 8192)
#: (N, K) of Llama-3-8B and Qwen3-8B's linears, merged and unmerged, at TP=1.
LLM_NK = ((4096, 4096), (6144, 4096), (12288, 4096), (14336, 4096), (24576, 4096),
          (28672, 4096), (4096, 12288), (4096, 14336))
#: lm_head: Llama-3-8B's and Qwen3-8B's vocabularies, with --lm-head.
LM_HEAD_NK = ((128256, 4096), (151936, 4096))
LLM_M = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
GQA_CONFIGS = tuple((16, 64, 8, n, 128) for n in (1024, 2048, 4096, 8192)) + (
    (1, 32, 8, 4096, 128), (16, 64, 8, 4096, 64))


def rel_err(got, want) -> float:
    return ((got.float() - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


def interleaved(arms, rounds, calls, graphs=False):
    """{name: [ms per call, one per round]}. `arms[name](i)` runs call i of a pass."""
    import torch

    if graphs:
        passes = {}
        for name, fn in arms.items():
            try:
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for i in range(calls):
                        fn(i)
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    for i in range(calls):
                        fn(i)
                passes[name] = g.replay
            except Exception as e:  # noqa: BLE001
                print(f"    {name}: graph capture failed ({type(e).__name__}: "
                      f"{str(e).splitlines()[0][:120]}); not timed in graph mode", flush=True)
    else:
        passes = {n: (lambda fn=fn: [fn(i) for i in range(calls)]) for n, fn in arms.items()}
    out = {n: [] for n in passes}
    order = list(passes)
    for p in passes.values():
        p()
    torch.cuda.synchronize()
    for r in range(rounds):
        for name in (order if r % 2 == 0 else order[::-1]):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            passes[name]()
            e.record()
            torch.cuda.synchronize()
            out[name].append(s.elapsed_time(e) / calls)
    return out


def summarize(times, flop):
    return {n: {"best_ms": min(t), "median_ms": statistics.median(t),
                "tflops": flop / (min(t) * 1e-3) / 1e12, "rounds": [round(x, 6) for x in t]}
            for n, t in times.items()}


def flydsl_import(what):
    root = os.environ.get("FLYDSL_ROOT", "/home/hfang/hk-local/FlyDSL")
    if root not in sys.path:
        sys.path.append(root)
    if what == "gemm":
        from kernels.gemm.gemm_a16w16_gfx950 import gemm_a16w16
        return gemm_a16w16
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func
    return flydsl_flash_attn_func


# ---------------------------------------------------------------- GEMM


def gemm_points(args):
    ms = tuple(int(x) for x in args.m.split(",")) if args.m else LLM_M
    pts = [] if args.no_squares else [(m, m, m, "square") for m in SQUARES]
    for n, k in LLM_NK + (LM_HEAD_NK if args.lm_head else ()):
        pts += [(m, n, k, "llm") for m in ms if (m, n, k) != (m, m, m)]
    if args.only:
        keep = {tuple(int(x) for x in s.split("x")) for s in args.only.split(",")}
        pts = [p for p in pts if p[:3] in keep]
    return pts


def run_gemm(args):
    import torch
    import torch.nn.functional as F

    from hk.ops import gemm as hk_gemm
    from hk.ops import hipkittens as hkk

    dev = torch.device("cuda")
    default_blas = torch.backends.cuda.preferred_blas_library()
    fly = flydsl_import("gemm") if args.flydsl else None
    results = []
    for m, n, k, kind in gemm_points(args):
        flop = 2.0 * m * n * k
        copies = max(2, -(-ROTATE_BYTES // (n * k * 2)))
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * k ** -0.5
              for _ in range(copies)]
        wts = [w.t().contiguous() for w in ws]
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().t()

        def blas(lib):
            def call(i):
                torch.backends.cuda.preferred_blas_library(lib)
                y = F.linear(x, ws[i])
                torch.backends.cuda.preferred_blas_library(default_blas)
                return y
            return call

        arms = {}
        if args.hk:
            arms["hk"] = lambda i: hk_gemm.matmul(x, ws[i], out=out)
        if hkk.gemm_fits(m, n, k):
            arms["hipkittens"] = lambda i: hkk.gemm_bf16(x, ws[i], out=out)
        arms["hipblaslt"] = lambda i: F.linear(x, ws[i])
        arms["rocblas"] = blas("cublas")
        arms["matmul_nn"] = lambda i: torch.matmul(x, wts[i])
        if fly is not None:
            arms["flydsl"] = lambda i: fly(x, ws[i].t(), out=out, layout="nt")
        errs, ok = {}, {}
        for name, fn in arms.items():
            try:
                y = fn(0)
                torch.cuda.synchronize()
                errs[name] = rel_err(y, ref)
            except Exception as e:  # noqa: BLE001
                errs[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                print(f"  {m}x{n}x{k} {name}: {errs[name]}", flush=True)
                continue
            if errs[name] < 2e-2:
                ok[name] = fn
            else:
                print(f"  {m}x{n}x{k} {name}: FAILED the fp32 check (rel err {errs[name]:.3g})", flush=True)
        rec = dict(m=m, n=n, k=k, kind=kind, copies=copies, err=errs,
                   eager=summarize(interleaved(ok, args.rounds, copies), flop))
        if args.graphs:
            rec["graph"] = summarize(interleaved(ok, args.rounds, copies, graphs=True), flop)
        results.append(rec)
        line = "  ".join(f"{a} {v['tflops']:.0f}" for a, v in rec["eager"].items())
        gl = ("  | graph " + "  ".join(f"{a} {v['tflops']:.0f}" for a, v in rec["graph"].items())
              if args.graphs else "")
        print(f"{m:>5}x{n:>5}x{k:>5}  TFLOP/s eager: {line}{gl}", flush=True)
        del x, ws, wts, out, ref
        torch.cuda.empty_cache()
    return dict(kind="gemm", default_blas=str(default_blas), results=results)


# ---------------------------------------------------------------- GQA


def gqa_reference(q, k, v, causal, batches):
    """fp32 o for three 32-row query blocks of each batch in `batches`, all heads."""
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


def run_gqa(args):
    import warnings

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from hk.ops import hipkittens as hkk

    # Why a forced backend declined is in the error each arm records.
    warnings.filterwarnings("ignore", category=UserWarning, module=__name__)
    dev = torch.device("cuda")
    fly = flydsl_import("gqa") if args.flydsl else None
    configs = GQA_CONFIGS
    if args.only:
        configs = [tuple(int(x) for x in s.split("x")) for s in args.only.split(",")]
    results = []
    for (b, h, hkv, n, d) in configs:
        for causal in (False, True):
            flop = 4.0 * b * n * n * h * d / (2 if causal else 1)
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.randn(b, n, h, d, device=dev, dtype=torch.bfloat16, generator=g)
            k = torch.randn(b, n, hkv, d, device=dev, dtype=torch.bfloat16, generator=g)
            v = torch.randn(b, n, hkv, d, device=dev, dtype=torch.bfloat16, generator=g)
            qt, kt, vt = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            rows, want = gqa_reference(q, k, v, causal, sorted({0, b - 1}))

            def sdpa(backend=None, views=False):
                a = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) if views else (qt, kt, vt)

                def call():
                    if backend is None:
                        return F.scaled_dot_product_attention(*a, is_causal=causal, enable_gqa=True)
                    with sdpa_kernel([backend]):
                        return F.scaled_dot_product_attention(*a, is_causal=causal, enable_gqa=True)
                return call

            arms = {"hipkittens": lambda: hkk.gqa_fwd(q, k, v, causal=causal),
                    "sdpa": sdpa(),
                    "sdpa_flash": sdpa(SDPBackend.FLASH_ATTENTION),
                    "sdpa_efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION),
                    "sdpa_views": sdpa(views=True)}
            if fly is not None:
                arms["flydsl"] = lambda: fly(q, k, v, causal=causal, num_kv_heads=hkv)
            bhnd = {"sdpa", "sdpa_flash", "sdpa_efficient", "sdpa_views"}
            try:
                choice = int(torch._fused_sdp_choice(qt, kt, vt, is_causal=causal, enable_gqa=True))
                chosen = SDPBackend(choice).name
            except Exception as e:  # noqa: BLE001
                chosen = f"? ({type(e).__name__})"
            errs, ok = {}, {}
            for name, fn in arms.items():
                try:
                    o = fn()
                    torch.cuda.synchronize()
                    if name in bhnd:
                        o = o.transpose(1, 2)
                    errs[name] = max(rel_err(o[bi, rows], w) for bi, w in want.items())
                except Exception as e:  # noqa: BLE001
                    errs[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: {errs[name]}", flush=True)
                    continue
                if errs[name] < 2e-2:
                    ok[name] = fn
                else:
                    print(f"  {(b, h, hkv, n, d)} causal={causal} {name}: FAILED the fp32 check "
                          f"(rel err {errs[name]:.3g})", flush=True)
            # A round of a few calls per arm, ~100 ms of the slowest.
            t0 = time.time()
            for fn in ok.values():
                fn()
            torch.cuda.synchronize()
            calls = max(2, min(20, int(0.1 / max((time.time() - t0) / max(len(ok), 1), 1e-4))))
            times = interleaved({nm: (lambda i, fn=fn: fn()) for nm, fn in ok.items()},
                                args.rounds, calls)
            rec = dict(b=b, h=h, h_kv=hkv, n=n, d=d, causal=causal, sdpa_choice=chosen,
                       calls=calls, err=errs, eager=summarize(times, flop))
            results.append(rec)
            line = "  ".join(f"{a} {v['tflops']:.0f}" for a, v in rec["eager"].items())
            print(f"B{b} H{h} Hkv{hkv} N{n} D{d} causal={int(causal)} (sdpa picks {chosen}) "
                  f"TFLOP/s: {line}", flush=True)
            del q, k, v, qt, kt, vt, want
            torch.cuda.empty_cache()
    return dict(kind="gqa", hipcc=hkk.hipcc_version(), results=results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("gemm", "gqa"))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--only", default="", help="comma-separated MxNxK (gemm) or BxHxHKVxNxD (gqa)")
    ap.add_argument("--flydsl", action="store_true", help="add FlyDSL's gfx950 kernel")
    ap.add_argument("--graphs", action="store_true", help="gemm: also time CUDA-graph replays")
    ap.add_argument("--hk", action="store_true", help="gemm: add hk's own GEMM (hk.ops.gemm.matmul)")
    ap.add_argument("--m", default="", help="gemm: comma-separated M for the LLM shapes")
    ap.add_argument("--lm-head", action="store_true", help="gemm: add the two lm_head shapes")
    ap.add_argument("--no-squares", action="store_true", help="gemm: skip the square shapes")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    from hk.ops import hipkittens as hkk

    meta = dict(device=torch.cuda.get_device_name(), arch=torch.cuda.get_device_properties(0).gcnArchName,
                torch=torch.__version__, hip=torch.version.hip, hipcc=hkk.hipcc_version(),
                date=time.strftime("%Y-%m-%d"), rounds=args.rounds)
    print(json.dumps(meta), flush=True)
    res = run_gemm(args) if args.what == "gemm" else run_gqa(args)
    res["meta"] = meta
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
