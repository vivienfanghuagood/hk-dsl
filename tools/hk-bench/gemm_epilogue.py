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

sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.argv[1:2] != ["--table"]:            # the table needs neither
    import torch
    import torch.nn.functional as F

    import hk  # noqa: E402
    from _timing import burn_in  # noqa: E402

    # Every case compiles its own graph; past dynamo's default limit of 8 it
    # would quietly run the rest eagerly.
    torch._dynamo.config.recompile_limit = 1 << 20
    torch._dynamo.config.cache_size_limit = 1 << 20

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


def graphed(variants, burst=10):
    """{name: thunk} -> {name: replay of `burst` calls captured in a CUDA
    graph}, for the ones that capture: the GPU's time, without the host's."""
    out = {}
    for name, fn in variants.items():
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(burst):
                    fn()
            out[name] = (lambda g=g: g.replay(), burst)
        except Exception as e:  # noqa: BLE001 -- recorded, not fatal
            print(f"  ({name} does not capture: {type(e).__name__}: {str(e)[:80]})")
    torch.cuda.synchronize()
    return out


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
    from hk.ops import decode_gemm  # noqa: PLC0415

    variants = {
        "hk": lambda: mm(x, w),
        "tmpl": tmpl,
        "unfused": unfused,
        "blas": lambda: eager(x, w),
        "blas_plain": lambda: F.linear(x, w),
    }
    if any(st.kernel == "decode" for st in T.plan(m, n, k)):
        # The plain kernel the template runs here, where hk.ops.matmul does
        # not (gfx1100 at M <= 32).
        variants["hk_decode"] = lambda: decode_gemm.matmul(x, w)
    if use_compile:
        torch._dynamo.reset()
        comp = torch.compile(eager, dynamic=False)
        comp(x, w)
        variants["compile"] = lambda: comp(x, w)
    t = timeit(variants)
    dev = {}
    if GRAPHS:
        gs = graphed(variants)
        d = timeit({k_: f for k_, (f, _) in gs.items()}, burst=1)
        dev = {k_: v / gs[k_][1] for k_, v in d.items()}
    flops = 2 * m * n * k
    return dict(model=model, layer=layer, m=m, n=n, k=k, epilogue=epilogue, err=err,
                us=t, dev_us=dev, tflops={k_: flops / v / 1e6 for k_, v in t.items()},
                plan=[str(s) for s in T.plan(m, n, k)])


#: Also time each variant replayed from a CUDA graph (--graphs).
GRAPHS = False


DEVICE = False


def table(paths):
    """The markdown table docs/templates/gemm.md shows, from result files
    (--device: the CUDA-graph times where a variant has one)."""
    global DEVICE
    if paths and paths[0] == "--device":
        DEVICE, paths = True, paths[1:]
    rows, meta = [], None
    for p in paths:
        doc = json.loads(Path(p).read_text())
        meta = meta or doc["meta"]
        rows += doc["rows"]
    print(f"{meta['device']} ({meta['arch']}), torch {meta['torch']}, HIP {meta['hip']}, "
          f"{meta['date']}; {meta['stat']}.")
    print()
    print("| layer (N x K) | M | epilogue | hk.ops.matmul | template | vs hk | "
          "unfused hk + torch | torch.compile | hipBLASLt + torch | template vs "
          "compile | vs hipBLASLt+ep |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    names = {"qkv": "qkv", "o": "o_proj", "gate_up": "gate_up", "down": "down"}
    for r in sorted(rows, key=lambda r: (r["epilogue"], r["n"], r["k"], r["m"])):
        u = dict(r["us"])
        if DEVICE and r.get("dev_us"):
            u.update(r["dev_us"])
        comp = u.get("compile")
        # Where the template runs the decode GEMV and hk.ops.matmul does not
        # (gfx1100, M <= 32), its plain kernel is decode_gemm's: marked *.
        base, mark = (u["hk_decode"], "*") if "hk_decode" in u else (u["hk"], "")
        print(f"| {r['model']} {names[r['layer']]} ({r['n']} x {r['k']}) | {r['m']} | "
              f"{r['epilogue']} | {base:.1f}{mark} | {u['tmpl']:.1f} | "
              f"{u['tmpl'] / base:.3f} | {u['unfused']:.1f} | "
              + (f"{comp:.1f}" if comp else "-") + f" | {u['blas']:.1f} | "
              + (f"{comp / u['tmpl']:.2f}x" if comp else "-")
              + f" | {u['blas'] / u['tmpl']:.2f}x |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        return table(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-8b,llama3-8b")
    ap.add_argument("--layers", default="qkv,o,gate_up,down")
    ap.add_argument("--m", default="16,128,1024,4096")
    ap.add_argument("--epilogues", default="bias_gelu,residual")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--graphs", action="store_true",
                    help="also the GPU's time: each variant replayed from a CUDA graph")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    global GRAPHS
    GRAPHS = args.graphs

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
                    d = r["dev_us"]
                    if d:
                        line += "\n    graph: " + "  ".join(
                            f"{k_} {v:.1f}" for k_, v in d.items())
                        if "hk" in d and "tmpl" in d:
                            line += f"  (tmpl {d['tmpl'] / d['hk']:.3f}x hk)"
                    print(line, flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


if __name__ == "__main__":
    main()
