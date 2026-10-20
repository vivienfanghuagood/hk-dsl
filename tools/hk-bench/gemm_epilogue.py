"""hk.templates.gemm's fused epilogues against what they replace.

For each linear layer of Qwen3-8B and Llama-3-8B (qkv, o_proj, gate_up,
down) at M in {16, 128, 1024, 4096}, interleaved in one process:

    hk          hk.ops.matmul(x, w)                         the plain GEMM
    tmpl        hk.templates.gemm(x, w, epilogue=E, ...)    E fused
    unfused     hk.ops.matmul then E in torch               two+ launches
    compile     torch.compile(F.linear + E)                 default mode
    blas        F.linear (hipBLASLt) + E in torch, eager
    blas_plain  F.linear alone, for reference

for E = bias+GELU(tanh) and E = residual add. Times are medians over rounds
of a burst of calls between two events (so the GPU is never waiting on a
launch, and a launch path slower than the kernel shows), in microseconds.

    python3 tools/hk-bench/gemm_epilogue.py [--m 16,128] [--layers qkv,o]
        [--models qwen3-8b] [--epilogues bias_gelu,residual] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hk  # noqa: E402

# Every case compiles its own graph; past dynamo's default limit of 8 it
# would quietly run the rest eagerly.
torch._dynamo.config.recompile_limit = 1 << 20
torch._dynamo.config.cache_size_limit = 1 << 20
from _timing import burn_in  # noqa: E402

#: (N, K) of each linear layer, unsharded.
MODELS = {
    "qwen3-8b": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096),
                 "down": (4096, 12288)},
    "llama3-8b": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096),
                  "down": (4096, 14336)},
}


def timeit(variants, rounds=7, burst=None, warmup=3):
    """{name: thunk} -> {name: median us per call}: rounds of bursts of
    `burst` calls between two events, the variants rotated within a round."""
    for _ in range(warmup):
        for fn in variants.values():
            fn()
    torch.cuda.synchronize()
    times = {k: [] for k in variants}
    names = list(variants)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            fn = variants[name]
            n = burst[name] if isinstance(burst, dict) else (burst or 10)
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(n):
                fn()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1e3 / n)
    return {k: statistics.median(v) for k, v in times.items()}


def case(model, layer, m, epilogue, use_compile=True):
    n, k = MODELS[model][layer]
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    res = torch.randn(m, n, device="cuda", dtype=torch.bfloat16, generator=g)
    T = hk.templates.gemm
    mm = hk.ops.matmul

    if epilogue == "bias_gelu":
        def tmpl():
            return T(x, w, epilogue=T.bias_gelu, bias=bias)

        def unfused():
            return F.gelu(mm(x, w) + bias, approximate="tanh")

        def eager(x, w):
            return F.gelu(F.linear(x, w, bias), approximate="tanh")

        ref = F.gelu(x.float() @ w.float().t() + bias.float(), approximate="tanh")
    else:
        def tmpl():
            return T(x, w, epilogue=T.residual, residual=res)

        def unfused():
            return mm(x, w) + res

        def eager(x, w):
            return F.linear(x, w) + res

        ref = x.float() @ w.float().t() + res.float()

    got = tmpl()
    err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
    variants = {
        "hk": lambda: mm(x, w),
        "tmpl": tmpl,
        "unfused": unfused,
        "blas": lambda: eager(x, w),
        "blas_plain": lambda: F.linear(x, w),
    }
    if use_compile:
        torch._dynamo.reset()
        comp = torch.compile(eager, dynamic=False)
        comp(x, w)
        variants["compile"] = lambda: comp(x, w)
    t = timeit(variants)
    flops = 2 * m * n * k
    return dict(model=model, layer=layer, m=m, n=n, k=k, epilogue=epilogue, err=err,
                us=t, tflops={k_: flops / v / 1e6 for k_, v in t.items()},
                plan=[str(s) for s in T.plan(m, n, k)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-8b,llama3-8b")
    ap.add_argument("--layers", default="qkv,o,gate_up,down")
    ap.add_argument("--m", default="16,128,1024,4096")
    ap.add_argument("--epilogues", default="bias_gelu,residual")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    meta = dict(arch=hk.target.current_arch(), device=torch.cuda.get_device_name(),
                torch=torch.__version__, hip=torch.version.hip,
                date=time.strftime("%Y-%m-%d"), stat="median us, bursts between events")
    print(json.dumps(meta), flush=True)
    burn_in(2.0)
    rows = []
    seen = set()
    for model in args.models.split(","):
        for layer in args.layers.split(","):
            nk = MODELS[model][layer]
            for m in (int(v) for v in args.m.split(",")):
                for ep in args.epilogues.split(","):
                    if (nk, m, ep) in seen:     # qkv and o are the same in both
                        continue
                    seen.add((nk, m, ep))
                    r = case(model, layer, m, ep, not args.no_compile)
                    rows.append(r)
                    u = r["us"]
                    line = (f"{model:9s} {layer:7s} M={m:5d} {ep:9s} "
                            f"hk {u['hk']:8.1f}  tmpl {u['tmpl']:8.1f} "
                            f"({u['tmpl'] / u['hk']:.3f}x hk)  unfused {u['unfused']:8.1f}  "
                            + (f"compile {u['compile']:8.1f}  " if "compile" in u else "")
                            + f"blas+ep {u['blas']:8.1f}  blas {u['blas_plain']:8.1f}  "
                            f"err {r['err']:.1e}")
                    print(line, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


if __name__ == "__main__":
    main()
