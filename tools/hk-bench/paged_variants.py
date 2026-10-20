#!/usr/bin/env python3
"""hk's paged decode variants against Triton's, kernel for kernel, on vLLM
0.31's cache.

The cache is one tensor a layer, `[blocks, kv_heads, block, 2 * head_dim]`,
K and V side by side in the last axis -- the layout vLLM 0.31 hands hk's
backend (LBHNC), so K and V are each a slice with a row stride of 2 x
head_dim. paged_ab.py still builds 0.16's two planes. Every arm walks
`--layers` caches per call, so K/V is read cold as a decode step reads it,
and is timed as a CUDA graph's replay (the GPU's time, the way a captured
decode step pays it), all arms interleaved in one process.

  triton      unified_attention as TRITON_ATTN calls it on 0.31: its 16-
              segment 3-D kernel and reduction up to 16 requests
              (MIN_LAUNCH_GRID_SIZE_2D // kv_heads), its 2-D kernel above
  tri3d       the 3-D kernel at every batch (segment buffers sized to fit)
  tri2d       the 2-D kernel at every batch: what hk's backend ran at 16
              requests when chapter 20 routed them to Triton
              (seq_threshold_3D = its TRITON_3D_MAX, 8)
  trinative   `triton` on the cache TRITON_ATTN itself keeps on 0.31: 16-token
              blocks, [blocks, block, kv_heads, 2 x head_dim] (LBNHC), so a
              token's heads sit side by side -- the stock backend's kernel on
              its own layout, the same K/V values
  <arm>@2p    the arm on vLLM 0.16's two-plane cache, (2, blocks, heads,
              block, head_dim), the same K/V: what paged_ab.py measures
  hk:fused:<splits>[:opt...]   the shipped walk + merge at a split count:
              `plan` for the backend's, `plan1024` for its plan before
              docs/history/38 (TARGET_WORKGROUPS 1024); options xw0 (the
              generic register transpose) and hs (the half-split loads)

Correctness first: every arm against Triton on the same cache, and with V
all ones, where every output element is exactly 1.0.

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/paged_variants.py \\
        --ctx 1024,4096 --batch 16,32,64 --arms triton,hk:fused:plan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

from _timing import burn_in                        # noqa: E402

HQ, HKV, D = 32, 8, 128
G = HQ // HKV
SCALE = D ** -0.5
PAGE = 32
BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
#: TritonAttentionMetadataBuilder on 0.31: MIN_LAUNCH_GRID_SIZE_2D // kv heads,
#: snapped to the nearest capture size (16), and its segment count.
TRITON_3D_MAX = 128 // HKV
SEGMENTS = 16


def make_layers(n, ctx, layers, gen, ragged=False, sequential=False):
    pages = -(-ctx // PAGE)
    nb = n * pages
    kvs = []
    for _ in range(layers):
        kv = torch.randn(nb, HKV, PAGE, 2 * D, device="cuda", dtype=BF16,
                         generator=gen)
        kvs.append(kv)
    # Shuffled, as an allocator that has run a while leaves it; sequential,
    # as a fresh engine hands out a batch's blocks.
    table = (torch.arange(nb, device="cuda") if sequential else
             torch.randperm(nb, device="cuda", generator=gen)).view(n, pages)
    table = table.to(I32).contiguous()
    if ragged:
        lens = ctx - torch.randint(0, PAGE * 2, (n,), device="cuda",
                                   generator=gen)
        lens = lens.to(I32)
    else:
        lens = torch.full((n,), ctx, device="cuda", dtype=I32)
    q = torch.randn(n, HQ, D, device="cuda", dtype=BF16, generator=gen)
    return kvs, table, lens, q


def kv_of(kv):
    if kv.dim() == 5:                    # two planes: (2, blocks, heads, block, d)
        return kv[0], kv[1]
    return kv[..., :D], kv[..., D:]


def two_planes(kvs):
    return [torch.stack((kv[..., :D], kv[..., D:])).contiguous() for kv in kvs]


def triton_arm(n, ctx, kvs, table, lens, q, out, force_3d=False,
               force_2d=False):
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )
    rows = max(n, TRITON_3D_MAX) if force_3d else TRITON_3D_MAX
    thr = rows if force_3d else (0 if force_2d else TRITON_3D_MAX)
    segm = (torch.empty(rows, HQ, SEGMENTS, D, device="cuda", dtype=F32),
            torch.empty(rows, HQ, SEGMENTS, device="cuda", dtype=F32),
            torch.empty(rows, HQ, SEGMENTS, device="cuda", dtype=F32))
    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    one = torch.ones((), device="cuda", dtype=F32)

    def one_layer(kv):
        kc, vc = kv_of(kv)
        unified_attention(
            q=q, k=kc.transpose(1, 2), v=vc.transpose(1, 2), out=out,
            cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens,
            max_seqlen_k=ctx, softmax_scale=SCALE, causal=True,
            alibi_slopes=None, use_alibi_sqrt=False, window_size=(-1, -1),
            block_table=table, softcap=0, q_descale=None,
            k_descale=one.expand((n, HKV)), v_descale=one.expand((n, HKV)),
            seq_threshold_3D=thr, num_par_softmax_segments=SEGMENTS,
            softmax_segm_output=segm[0], softmax_segm_max=segm[1],
            softmax_segm_expsum=segm[2], sinks=None, output_scale=None,
            mm_prefix_range=None)

    def run():
        for kv in kvs:
            one_layer(kv)
    return run, one_layer


def native_layers(kvs, table, n, ctx):
    """The same K/V in TRITON_ATTN's own 0.31 cache: 16-token blocks,
    LBNHC, a fresh shuffled table."""
    tb = 16
    pages = -(-ctx // tb)
    nb = n * pages
    gen = torch.Generator(device="cuda").manual_seed(n + ctx + 1)
    t16 = torch.randperm(nb, device="cuda", generator=gen).view(n, pages)
    t16 = t16.to(I32).contiguous()
    out = []
    for kv in kvs:
        # hk's page p of request r holds tokens [32 i, 32 i + 32) of its
        # table column i, as [heads, tokens, 2D].
        tok = kv[table.long()]                       # (n, pages32, H, 32, 2D)
        tok = tok.permute(0, 1, 3, 2, 4).reshape(n, -1, HKV, 2 * D)[:, :ctx]
        nat = torch.empty(nb, tb, HKV, 2 * D, device="cuda", dtype=BF16)
        nat[t16.long().view(-1)] = tok.reshape(n * pages, tb, HKV, 2 * D)
        out.append(nat)
        del tok
    return out, t16


def native_arm(n, ctx, nats, t16, lens, q, out):
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )
    segm = (torch.empty(TRITON_3D_MAX, HQ, SEGMENTS, D, device="cuda", dtype=F32),
            torch.empty(TRITON_3D_MAX, HQ, SEGMENTS, device="cuda", dtype=F32),
            torch.empty(TRITON_3D_MAX, HQ, SEGMENTS, device="cuda", dtype=F32))
    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    one = torch.ones((), device="cuda", dtype=F32)

    def one_layer(nat):
        unified_attention(
            q=q, k=nat[..., :D], v=nat[..., D:], out=out,
            cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens,
            max_seqlen_k=ctx, softmax_scale=SCALE, causal=True,
            alibi_slopes=None, use_alibi_sqrt=False, window_size=(-1, -1),
            block_table=t16, softcap=0, q_descale=None,
            k_descale=one.expand((n, HKV)), v_descale=one.expand((n, HKV)),
            seq_threshold_3D=TRITON_3D_MAX, num_par_softmax_segments=SEGMENTS,
            softmax_segm_output=segm[0], softmax_segm_max=segm[1],
            softmax_segm_expsum=segm[2], sinks=None, output_scale=None,
            mm_prefix_range=None)

    def run():
        for nat in nats:
            one_layer(nat)
    return run, one_layer


def plan(n, ctx, target=None):
    """The backend's split count; `target` replays it with another
    TARGET_WORKGROUPS (`plan1024`: the one before docs/history/38)."""
    from hk.integration import vllm_backend
    from hk.ops import paged
    if target is None:
        return vllm_backend.decode_splits(n, HKV, -(-40960 // PAGE))
    saved = paged.TARGET_WORKGROUPS
    paged.TARGET_WORKGROUPS = target
    try:
        return vllm_backend.decode_splits(n, HKV, -(-40960 // PAGE))
    finally:
        paged.TARGET_WORKGROUPS = saved


def hk_arm(spec, n, ctx, kvs, table, lens, q, out):
    from hk.ops import paged
    from hk.runtime.torch_ext import torch_op

    parts = spec.split(":")
    kind = parts[1]
    if parts[2] == "plan":
        splits = plan(n, ctx)
    elif parts[2].startswith("plan"):
        splits = plan(n, ctx, int(parts[2][4:]))
    else:
        splits = int(parts[2])
    extra = parts[3:]
    if kind == "fused":
        # options: xw0 = the generic register transpose (HK_RDNA_XPOSE_WORDS=0),
        # hs = the half-split loads (HK_RT_HALF_SPLIT=1)
        defines = {}
        if "xw0" in extra:
            defines["HK_RDNA_XPOSE_WORDS"] = 0
        if "hs" in extra:          # each 32-byte run loaded once, halves swapped
            defines["HK_RT_HALF_SPLIT"] = 1
        defines = defines or None
        walk = paged.fused_kernel(D, splits, defines=defines)
    else:
        raise ValueError(f"unknown hk decode {kind!r}")
    walk = torch_op(walk)
    merge = torch_op(paged.merge_kernel(D, splits)) if splits > 1 else None
    scratch = paged.fused_scratch(n * splits, HKV, D, q.device, group=G)
    q4, o4 = q.view(n, HKV, G, D), out.view(n, HKV, G, D)

    def one_layer(kv):
        kc, vc = kv_of(kv)
        paged.decode(walk, merge, q4, kc, vc, table, lens, o4, scratch, splits)

    def run():
        for kv in kvs:
            one_layer(kv)
    return run, one_layer, splits


def graphed(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def interleave_graphs(replays, rounds, reps):
    """{name: replay} -> {name: [ms per replay, one per round]}, rounds
    rotating the order, each a burst of `reps` replays between events."""
    for fn in replays.values():
        fn()
    torch.cuda.synchronize()
    out = {k: [] for k in replays}
    names = list(replays)
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            fn = replays[name]
            fn()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(reps):
                fn()
            e.record()
            e.synchronize()
            out[name].append(s.elapsed_time(e) / reps)
    return out


def check(name, got, want, rel=2e-2):
    got, want = got.float(), want.float()
    ref = want.abs().max().item()
    err = (got - want).abs().max().item()
    ok = ref > 0 and err == err and err <= rel * ref
    print(f"  check {name:<22} max|err| {err:.2e} of {ref:.2e}  "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    return ok


def run_case(n, ctx, arms, layers, rounds, reps, ragged, sequential=False):
    gen = torch.Generator(device="cuda").manual_seed(n * 7919 + ctx)
    kvs, table, lens, q = make_layers(n, ctx, layers, gen, ragged, sequential)
    fns, ones, outs, splits = {}, {}, {}, {}
    nats = t16 = None
    if "trinative" in arms:
        nats, t16 = native_layers(kvs, table, n, ctx)
    kv2 = two_planes(kvs) if any(a.endswith("@2p") for a in arms) else None
    layers_of = {a: (kv2 if a.endswith("@2p") else kvs) for a in arms}
    for a0 in arms:
        a = a0[:-3] if a0.endswith("@2p") else a0
        out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)
        if a == "trinative":
            fns[a0], ones[a0] = native_arm(n, ctx, nats, t16, lens, q, out)
            splits[a0] = "-"
        elif a in ("triton", "tri3d", "tri2d"):
            fns[a0], ones[a0] = triton_arm(n, ctx, layers_of[a0], table, lens,
                                           q, out, force_3d=(a == "tri3d"),
                                           force_2d=(a == "tri2d"))
            splits[a0] = "-"
        else:
            fns[a0], ones[a0], splits[a0] = hk_arm(a, n, ctx, layers_of[a0],
                                                   table, lens, q, out)
        outs[a0] = out
    ok = True
    # Against Triton on the first layer, then V = 1 exactly on a copy.
    ref_name = "triton" if "triton" in arms else arms[0]
    for a in arms:
        ones[a](nats[0] if a == "trinative" else layers_of[a][0])
    torch.cuda.synchronize()
    for a in arms:
        if a != ref_name:
            ok &= check(f"{a} n{n} c{ctx}", outs[a], outs[ref_name])
    v1 = kvs[0].clone()
    v1[..., D:] = 1.0
    n1 = p1 = None
    if nats is not None:
        n1 = nats[0].clone()
        n1[..., D:] = 1.0
    if kv2 is not None:
        p1 = kv2[0].clone()
        p1[1] = 1.0
    for a in arms:
        outs[a].zero_()
        ones[a](n1 if a == "trinative" else p1 if a.endswith("@2p") else v1)
        torch.cuda.synchronize()
        exact = bool((outs[a] == 1.0).all())
        if not exact:
            bad = (outs[a] != 1.0).sum().item()
            print(f"  exact {a}: {bad} elements != 1.0  FAIL", flush=True)
        ok &= exact
    del v1, n1, p1
    reps_ = {a: graphed(fns[a]) for a in arms}
    times = interleave_graphs(reps_, rounds, reps)
    gb = sum(int(x) for x in lens.tolist()) * HKV * D * 2 * 2 / 1e9
    res = {}
    for a in arms:
        ts = sorted(times[a])
        med = ts[len(ts) // 2] / layers
        best = ts[0] / layers
        res[a] = dict(ms=med, best=best, gbs=gb / (med * 1e-3),
                      splits=splits[a], rounds=[t / layers for t in times[a]])
    del kvs, nats, kv2, layers_of
    torch.cuda.empty_cache()
    return ok, res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", default="1024,4096")
    ap.add_argument("--batch", default="16,32,64")
    ap.add_argument("--arms", default="triton,hk:fused:plan")
    ap.add_argument("--layers", type=int, default=0,
                    help="caches per call; 0: enough for 1.5 GB of K/V")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ragged", action="store_true",
                    help="lengths ctx - [0, 64) instead of ctx")
    ap.add_argument("--sequential", action="store_true",
                    help="each request's pages in order, as a fresh engine "
                         "allocates them, instead of shuffled")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    arms = args.arms.split(",")

    from hk.ops import paged  # noqa: F401 -- build everything up front
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}",
          flush=True)
    burn_in()
    allres, failed = [], False
    print(f"\nms per layer (median of {args.rounds} rounds), GB/s of K+V; "
          f"ratio = triton / arm")
    for ctx in (int(c) for c in args.ctx.split(",")):
        for n in (int(b) for b in args.batch.split(",")):
            per_layer = n * ctx * HKV * D * 4
            layers = args.layers or max(4, min(36, int(1.5e9 // per_layer)))
            ok, res = run_case(n, ctx, arms, layers, args.rounds, args.reps,
                               args.ragged, args.sequential)
            failed |= not ok
            base = res.get("triton", {}).get("ms")
            for a in arms:
                r = res[a]
                ratio = f"{base / r['ms']:.3f}x" if base else ""
                print(f"  ctx {ctx:>5} n {n:>3} L{layers:<3}{a:<24}"
                      f"s{r['splits']!s:<4}{r['ms']:8.4f} ms (best "
                      f"{r['best']:.4f}){r['gbs']:7.0f} GB/s  {ratio}",
                      flush=True)
            allres.append(dict(ctx=ctx, n=n, layers=layers, ok=ok, arms=res))
    if args.json:
        Path(args.json).write_text(json.dumps(allres, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
