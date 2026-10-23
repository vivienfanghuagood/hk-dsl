#!/usr/bin/env python3
"""Paged decode attention against the size of the allocation the cache lives in.

vLLM 0.31 keeps every layer's KV cache in one allocation (26.75 GiB on a W7900D with Qwen3-8B at
gpu_memory_utilization 0.9); the other bench tools allocate just the pages a batch reads. This
times Triton's kernel (as hk's backend calls it, `vllm_backend.triton_decode`) and hk's walk at its
planned split on the same pages, the same block table (sequential, as a fresh engine hands them
out) and one layer's view, with that layer carved out of allocations of growing size, at their
start and, where there is room, at their end. Every output is checked against the first.

    python3 tools/hk-bench/paged_alloc.py --batch 16,32 --context 1024,4096 --gib 0,2,26
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

from hk.integration import vllm_backend as vb  # noqa: E402

HQ, HKV, D, BLK = 32, 8, 128, 32
BF16 = torch.bfloat16


def t_ms(fn, reps, calls=10):
    """fn captured in a CUDA graph, `calls` times; the median over `reps` replays, per call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    xs = []
    for _ in range(reps):
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        xs.append(a.elapsed_time(b) / calls)
    return sorted(xs)[len(xs) // 2]


def one(store, lo, need, src, q, table, lens, qsl, unit, splits, scratch, reps):
    """(Triton ms, hk ms, Triton's output, hk's output) on the pages at store[lo:lo + need]."""
    n = q.shape[0]
    g = HQ // HKV
    cache = store[lo:lo + need]
    cache.copy_(src)
    k, v = cache[..., :D], cache[..., D:]
    out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)
    o2 = torch.empty_like(out)
    ctx = int(lens[0].item())
    tri = t_ms(lambda: vb.triton_decode(q, k, v, table, lens, qsl, out, D ** -0.5, ctx,
                                        unit, unit), reps)
    hk = t_ms(lambda: vb.decode(q.view(n, HKV, g, D), k, v, table, lens,
                                o2.view(n, HKV, g, D), scratch, splits), reps)
    return tri, hk, out, o2


def case(n, ctx, gibs, width, reps, ends):
    pages = (ctx + 1 + BLK - 1) // BLK
    need = n * pages + 1                              # block 0 unused, as in vLLM
    page_bytes = HKV * BLK * 2 * D * 2
    gen = torch.Generator(device="cuda").manual_seed(n * 7919 + ctx)
    q = torch.randn(n, HQ, D, device="cuda", dtype=BF16, generator=gen)
    src = torch.randn(need, HKV, BLK, 2 * D, device="cuda", dtype=BF16, generator=gen)
    w = max(width, pages)
    table = torch.zeros(n, w, device="cuda", dtype=torch.int32)
    table[:, :pages] = torch.arange(1, need, device="cuda", dtype=torch.int32).view(n, pages)
    lens = torch.full((n,), ctx + 1, device="cuda", dtype=torch.int32)
    qsl = torch.arange(n + 1, device="cuda", dtype=torch.int32)
    unit = torch.ones((), device="cuda", dtype=torch.float32)
    splits = vb.decode_splits(n, HKV, w)
    scratch = vb._scratch(q.device, HKV, D, max(256, n), HQ // HKV)
    ref = None
    for gib in gibs:
        nb = max(need, int(gib * 2**30 / page_bytes))
        store = torch.empty(nb, HKV, BLK, 2 * D, device="cuda", dtype=BF16)
        for where in (("start", "end") if nb > need and ends else ("start",)):
            lo = 0 if where == "start" else nb - need
            tri, hk, out, o2 = one(store, lo, need, src, q, table, lens, qsl, unit, splits,
                                   scratch, reps)
            if ref is None:
                ref = out.clone()
            diff = max((out - ref).abs().max().item(), (o2 - ref).abs().max().item())
            print(f"{ctx:>6} {n:>5} {nb * page_bytes / 2**30:10.2f} {where:>6} {tri:10.4f} "
                  f"{hk:8.4f} {tri / hk:7.2f}x  s{splits} {diff:10.3e}", flush=True)
        del store
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", default="32")
    ap.add_argument("--context", default="4096")
    ap.add_argument("--gib", default="0,2,8,16,26",
                    help="allocation sizes, GiB; 0 is just the pages read")
    ap.add_argument("--width", type=int, default=136,
                    help="block table row width (vLLM: max_model_len / 32)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-ends", action="store_true", help="only the start of each allocation")
    args = ap.parse_args()
    gibs = [float(x) for x in args.gib.split(",")]
    print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}  triton "
          f"{__import__('triton').__version__}; ms a call, median of {args.reps}")
    print(f"{'ctx':>6} {'batch':>5} {'alloc GiB':>10} {'where':>6} {'triton':>10} {'hk':>8} "
          f"{'tri/hk':>8}  split {'max|diff|':>10}")
    for ctx in (int(x) for x in args.context.split(",")):
        for n in (int(x) for x in args.batch.split(",")):
            case(n, ctx, gibs, args.width, args.reps, not args.no_ends)


if __name__ == "__main__":
    main()
