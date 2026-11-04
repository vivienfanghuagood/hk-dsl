#!/usr/bin/env python3
"""Paged attention on RDNA: hk against vLLM's own kernels, at real model shapes, in one process.

    python3 tools/hk-bench/attention_bench.py decode  --model minicpm5
    python3 tools/hk-bench/attention_bench.py decode  --model qwen3.8 --batch 1,4,16 --ctx 8192
    python3 tools/hk-bench/attention_bench.py prefill --model minicpm5 --reqs 0:8192,16384:2048

README.md in this directory has the details. In short:

* **The kernels vLLM runs**, on the cache vLLM keeps. `triton` is vLLM's `unified_attention`
  (TRITON_ATTN), called as its backend calls it (for a decode batch, the 3-D split-softmax kernel
  below its batch threshold and the 2-D one above); `rocm_attn` is ROCM_ATTN's
  `chunked_prefill_paged_decode` on ROCM_ATTN's own cache (prefix-prefill for prefill, its paged
  decode for decode); `hk` is hk's route: for decode hk.ops.paged_decode, for prefill
  hk.integration.vllm_prefill's stand-in for `unified_attention` (hk.ops.paged_prefill), both on
  TRITON_ATTN's cache in place. `hk_rocm` (prefill) is the route inside ROCM_ATTN: ROCM_ATTN's
  `chunked_prefill_paged_decode` on its own cache with its prefix-prefill rebound to hk's
  stand-in, its decode rows still ROCM_ATTN's.
* **Cold KV.** Every call reads its own layer's cache, and there are enough layers that the
  cycle is at least COLD_BYTES -- several times the last-level cache (a W7900D's Infinity Cache
  is 96 MB, an R9700's 64 MB) -- so no arm is served from it.
* **Same process, interleaved.** Arms alternate round by round, best of `--rounds`. Decode
  bursts (one call a layer) are captured as CUDA graphs and replayed, so host cost is not
  counted; prefill calls are milliseconds and run eagerly.
* **Correct first.** Each arm's output on layer 0 is compared with a float32 reference before
  it is timed; an arm that is wrong (max |error| over `--tol`) is reported as wrong, not timed.

Decode bandwidth is the KV bytes a call reads (every request's K and V, every kv head) over its
time. Prefill rate is the causal attention FLOPs (4 * d * heads * the query-key pairs at or
below the diagonal) over its time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

#: name -> query heads, kv heads, head_dim, page (tokens), TRITON_ATTN cache order -- one TP
#: rank's share. The page and order are what vLLM gives these models on Radeon (hybrid models'
#: attention pages are sized to their mamba state).
MODELS = {
    "minicpm5": dict(hq=16, hkv=2, d=128, page=16, order="planes"),        # MiniCPM5-2B
    "qwen3-8b": dict(hq=32, hkv=8, d=128, page=16, order="planes"),        # Qwen3-8B
    "qwen3.8": dict(hq=24, hkv=4, d=256, page=784, order="heads"),         # Qwen3.8-27B, TP1
    "qwen3.8-tp2": dict(hq=12, hkv=2, d=256, page=784, order="heads"),     # its TP2 rank
    "minicpm-v-tp2": dict(hq=8, hkv=1, d=256, page=544, order="tokens"),   # MiniCPM-V-4.7-35B
    "qwen3.6": dict(hq=16, hkv=2, d=256, page=2064, order="tokens"),       # Qwen3.6-35B-A3B
}

#: the cycle of layer caches is at least this many bytes
COLD_BYTES = 1 << 30


def cache_views(nb, page, hkv, d, order, device):
    """(storage, k, v, kv): vLLM's K and V views of one layer's TRITON_ATTN cache, and hk's 4-D
    view of the same memory."""
    if order == "planes":
        st = torch.empty(nb, 2, page, hkv, d, device=device, dtype=torch.bfloat16)
        k, v = st[:, 0], st[:, 1]
        kv = st.view(nb, 2, page, hkv * d)
    elif order == "tokens":
        st = torch.empty(nb, page, hkv, 2 * d, device=device, dtype=torch.bfloat16)
        k, v = st[..., :d], st[..., d:]
        kv = st.view(nb, 1, page, hkv * 2 * d)
    else:  # heads
        st = torch.empty(nb, hkv, page, 2 * d, device=device, dtype=torch.bfloat16)
        k, v = st[..., :d].transpose(1, 2), st[..., d:].transpose(1, 2)
        kv = st
    return st, k, v, kv


class Case:
    """A batch: `reqs` = [(sequence length, query length)] (a decode's query length is 1),
    `layers` caches, the batch's metadata, the queries, and (lazily) ROCM_ATTN's caches."""

    def __init__(self, m, reqs, device="cuda", seed=0):
        self.m, self.reqs = m, reqs
        hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
        g = torch.Generator(device=device).manual_seed(seed)
        lens = torch.tensor([s for s, _ in reqs], dtype=torch.int32)
        qlens = [n for _, n in reqs]
        self.lens_host = lens
        pages = [-(-int(n) // page) for n in lens]
        nb = sum(pages) + 1
        layer_bytes = nb * page * hkv * d * 2 * 2
        self.layers = max(2, min(64, math.ceil(COLD_BYTES / layer_bytes)))
        self.kv_bytes = int(lens.sum()) * hkv * d * 2 * 2
        self.flops = sum(4 * d * hq * (n * (s - n) + n * (n + 1) // 2) for s, n in reqs)
        self.caches = []
        for _ in range(self.layers):
            st, k, v, kv = cache_views(nb, page, hkv, d, m["order"], device)
            st.normal_(generator=g)
            self.caches.append((k, v, kv))
        self._rocm = None
        # pages scattered over the cache, as an engine's allocator leaves them
        perm = torch.randperm(nb - 1, generator=torch.Generator().manual_seed(seed + 1)) + 1
        bt = torch.zeros(len(reqs), max(pages), dtype=torch.int32)
        at = 0
        for r, n in enumerate(pages):
            bt[r, :n] = perm[at:at + n]
            at += n
        self.block_table = bt.to(device)
        self.seq_lens = lens.to(device)
        starts = [0]
        for n in qlens:
            starts.append(starts[-1] + n)
        self.q_start = torch.tensor(starts, dtype=torch.int32, device=device)
        self.max_q = max(qlens)
        self.q = torch.randn(starts[-1], hq, d, device=device, dtype=torch.bfloat16, generator=g)
        self.scale = d ** -0.5

    def rocm(self, layer):
        """ROCM_ATTN's caches for layer `layer`, the same values -- K x-split (blocks, kv_heads,
        d / 8, page, 8), V (blocks, kv_heads, d, page) -- and the batch's new tokens' K and V."""
        if self._rocm is None:
            self._rocm = []
            for k, v, _ in self.caches:
                nb, page, hkv, d = k.shape
                k5 = k.permute(0, 2, 1, 3).reshape(nb, hkv, page, d // 8, 8).permute(0, 1, 3, 2, 4)
                # the new tokens' K and V (ROCM_ATTN's prefill reads them from the call, not
                # from the cache): each request's last query-length rows
                news = [[], []]
                for r, (s, n) in enumerate(self.reqs):
                    idx = self.block_table[r, :-(-s // page)].long()
                    for i, t in enumerate((k, v)):
                        news[i].append(t[idx].reshape(-1, hkv, d)[s - n:s])
                self._rocm.append((k5.contiguous(), v.permute(0, 2, 3, 1).contiguous(),
                                   torch.cat(news[0]).contiguous(),
                                   torch.cat(news[1]).contiguous()))
        return self._rocm[layer]

    def reference(self, layer=0):
        """float32 causal attention of the batch's queries over layer `layer`'s cache."""
        m = self.m
        hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
        k, v, _ = self.caches[layer]
        out = torch.empty(self.q.shape, device=self.q.device, dtype=torch.float32)
        for r, (s, n) in enumerate(self.reqs):
            idx = self.block_table[r, :-(-s // page)].long()
            kk = k[idx].reshape(-1, hkv, d)[:s].float()
            vv = v[idx].reshape(-1, hkv, d)[:s].float()
            q0 = int(self.q_start[r])
            for a in range(0, n, 512):         # query chunks: the scores of a 16k prefill are GBs
                b = min(n, a + 512)
                qq = self.q[q0 + a:q0 + b].float().view(b - a, hkv, hq // hkv, d)
                sc = torch.einsum("nhgd,shd->hgns", qq, kk) * self.scale
                pos = (s - n) + torch.arange(a, b, device=sc.device)
                sc = sc.masked_fill(torch.arange(s, device=sc.device)[None, :] > pos[:, None],
                                    float("-inf"))
                out[q0 + a:q0 + b] = torch.einsum("hgns,shd->nhgd", torch.softmax(sc, -1),
                                                  vv).reshape(b - a, hq, d)
        return out


def _triton_scratch(case):
    m = case.m
    hq, hkv, d = m["hq"], m["hkv"], m["d"]
    thr = 128 // hkv            # vLLM's seq_threshold_3D (MIN_LAUNCH_GRID_SIZE_2D // kv heads)
    segs = 16                   # NUM_PAR_SOFTMAX_SEGMENTS
    dp = 1 << (d - 1).bit_length()
    dev = case.q.device
    return dict(seq_threshold_3D=thr, num_par_softmax_segments=segs,
                softmax_segm_output=torch.empty(thr, hq, segs, dp, dtype=torch.float32, device=dev),
                softmax_segm_max=torch.empty(thr, hq, segs, dtype=torch.float32, device=dev),
                softmax_segm_expsum=torch.empty(thr, hq, segs, dtype=torch.float32, device=dev))


def _u_args(case, layer, out):
    k, v, _ = case.caches[layer]
    desc = torch.ones(1, device=case.q.device, dtype=torch.float32).expand(len(case.reqs),
                                                                          case.m["hkv"])
    return dict(q=case.q, k=k, v=v, out=out, cu_seqlens_q=case.q_start, max_seqlen_q=case.max_q,
                seqused_k=case.seq_lens, max_seqlen_k=int(case.lens_host.max()),
                softmax_scale=case.scale, causal=True, window_size=(-1, -1),
                block_table=case.block_table, softcap=0, q_descale=None, k_descale=desc,
                v_descale=desc)


def triton_arm(case):
    """vLLM's unified_attention as TRITON_ATTN's forward calls it."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: PLC0415

    out = torch.empty_like(case.q)
    extra = _triton_scratch(case)

    def call(layer):
        unified_attention(**_u_args(case, layer, out), **extra)
    return call, out


def rocm_arm(case):
    """vLLM's ROCM_ATTN: chunked_prefill_paged_decode on its own (x-split) cache."""
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: PLC0415
        chunked_prefill_paged_decode)

    out = torch.empty_like(case.q)
    one = torch.ones(1, device=case.q.device, dtype=torch.float32)
    case.rocm(0)

    def call(layer):
        k5, v4, k_new, v_new = case.rocm(layer)
        chunked_prefill_paged_decode(
            query=case.q, key=k_new, value=v_new, output=out,
            kv_cache_dtype="auto", key_cache=k5, value_cache=v4, block_table=case.block_table,
            query_start_loc=case.q_start, seq_lens=case.seq_lens,
            max_seq_len=int(case.lens_host.max()), max_query_len=case.max_q, k_scale=one,
            v_scale=one, sm_scale=case.scale)
    return call, out


def hk_decode_arm(case, splits=None, warps=None, step=8, v_both=None):
    """hk.ops.paged_decode on TRITON_ATTN's cache: the split walk, and the merge if split (as
    hk.integration.vllm_prefill runs it with HK_VLLM_DECODE=1)."""
    from hk.ops import paged_decode as pd  # noqa: PLC0415
    from hk.runtime.torch_ext import launcher  # noqa: PLC0415

    m = case.m
    hq, hkv, d, page = m["hq"], m["hkv"], m["d"], m["page"]
    grp, n = hq // hkv, len(case.reqs)
    ps, pw, pv = pd.plan(n, hkv, d, page, wgps=pd.device_wgps(case.q.device))
    s, warps = splits or ps, warps or pw
    v_both = pv if v_both is None else v_both
    try:
        walk = launcher(pd.walk_for(d, grp, page, m["order"], s, warps=warps, v_both=v_both))
    except Exception:  # noqa: BLE001 -- both V bands in flight spill under some compilers
        if not v_both:
            raise
        walk = launcher(pd.walk_for(d, grp, page, m["order"], s, warps=warps, v_both=False))
    merge = launcher(pd.merge_for(d, grp, s, step)) if s > 1 else None
    out = torch.empty_like(case.q)
    o_part, ml = pd.scratch(n * s, hkv, grp, d, case.q.device)
    qh = case.q.unsqueeze(0)
    o4 = out.view(n, hkv, grp, d)

    def call(layer):
        _, _, kv = case.caches[layer]
        walk(qh, kv, case.block_table, case.seq_lens, case.q_start, o4, o_part, ml)
        if merge is not None:
            merge(case.seq_lens, case.q_start, o4, o_part, ml)
    call.splits = s
    return call, out


def hk_prefill_arm(case):
    """hk's TRITON_ATTN route as vLLM calls it: hk.integration.vllm_prefill's stand-in for
    unified_attention (hk.ops.paged_prefill, with the V^T pre-pack where it applies). A call it
    does not take goes to vLLM's kernel, as in serving -- `hk_calls` in the row says how many
    it took. The step's exact sizes are given to the guard as vLLM's TRITON_ATTN metadata
    builder gives them in serving (vllm_prefill._STATS), so mixed batches are judged -- and split
    -- as served."""
    import inspect  # noqa: PLC0415

    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: PLC0415

    from hk.integration import vllm_prefill as vp  # noqa: PLC0415

    vp._ORIG_U = unified_attention
    vp._U_SIG = inspect.signature(unified_attention)
    starts = [0]
    for _, n in case.reqs:
        starts.append(starts[-1] + n)
    vp._STATS[id(case.q_start)] = vp.batch_stats(starts, case.lens_host.tolist(), len(case.reqs))
    out = torch.empty_like(case.q)
    extra = _triton_scratch(case)

    def call(layer):
        vp.unified_attention(**_u_args(case, layer, out), **extra)
    call.stats = vp.PATCH.stats
    return call, out


def hk_rocm_arm(case):
    """hk's ROCM_ATTN route as vLLM runs it: ROCM_ATTN's chunked_prefill_paged_decode on its own
    cache, with its prefix-prefill (context_attention_fwd, requests of more than one query)
    rebound to hk.integration.vllm_prefill's stand-in (hk.ops.paged_prefill, layout="rocm");
    decode rows stay ROCM_ATTN's paged decode. The rebinding is swapped in around each call, so
    the rocm_attn arm keeps vLLM's own kernel."""
    import inspect  # noqa: PLC0415

    from vllm.v1.attention.ops import chunked_prefill_paged_decode as cppd  # noqa: PLC0415
    from vllm.v1.attention.ops import prefix_prefill  # noqa: PLC0415

    from hk.integration import vllm_prefill as vp  # noqa: PLC0415

    orig = prefix_prefill.context_attention_fwd
    vp._ORIG = orig
    vp._P_SIG = inspect.signature(orig)
    out = torch.empty_like(case.q)
    one = torch.ones(1, device=case.q.device, dtype=torch.float32)
    case.rocm(0)

    def call(layer):
        k5, v4, k_new, v_new = case.rocm(layer)
        cppd.context_attention_fwd = vp.context_attention_fwd
        try:
            cppd.chunked_prefill_paged_decode(
                query=case.q, key=k_new, value=v_new, output=out,
                kv_cache_dtype="auto", key_cache=k5, value_cache=v4,
                block_table=case.block_table, query_start_loc=case.q_start,
                seq_lens=case.seq_lens, max_seq_len=int(case.lens_host.max()),
                max_query_len=case.max_q, k_scale=one, v_scale=one, sm_scale=case.scale)
        finally:
            cppd.context_attention_fwd = orig
    call.stats = vp.PATCH.stats
    return call, out


def graphed(call, layers):
    """A CUDA graph of one call a layer; returns its replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(layers):
            call(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(layers):
            call(i)
    return g.replay


def eager(call, layers):
    def burst():
        for i in range(layers):
            call(i)
    return burst


def time_arms(bursts, layers, rounds, reps=3):
    """{arm: best us a call}: arms alternating, `reps` bursts each a round."""
    best = {k: float("inf") for k in bursts}
    for fn in bursts.values():
        fn()
    torch.cuda.synchronize()
    for _ in range(rounds):
        for name, fn in bursts.items():
            for _ in range(reps):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                best[name] = min(best[name], a.elapsed_time(b) * 1000 / layers)
    return best


def kernel_times(bursts, layers, calls=3):
    """{arm: {kernel: us a layer}}, from the torch profiler: where an arm's time goes."""
    from collections import defaultdict  # noqa: PLC0415

    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    got = {}
    for name, fn in bursts.items():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(calls):
                fn()
            torch.cuda.synchronize()
        tot = defaultdict(float)
        for e in p.events():
            if e.device_type == torch.autograd.DeviceType.CUDA and e.device_time_total > 0:
                tot[e.name.split("(")[0][:48]] += e.device_time_total
        got[name] = {k: round(v / (calls * layers), 2) for k, v in tot.items()}
    return got


def check(arms, case):
    """{arm: max |error| against float32} on layer 0 (inf where not finite)."""
    ref = case.reference(0)
    errs = {}
    for name, (call, out) in arms.items():
        out.fill_(float("nan"))
        call(0)
        torch.cuda.synchronize()
        e = (out.float() - ref).abs().max().item()
        errs[name] = e if math.isfinite(e) else float("inf")
    return errs


def _arms(args, case, kind):
    want = [a for a in args.arms.split(",") if a]
    arms = {}
    if "triton" in want:
        arms["triton"] = triton_arm(case)
    if "rocm_attn" in want:
        arms["rocm_attn"] = rocm_arm(case)
    if "hk" in want:
        arms["hk"] = hk_decode_arm(case) if kind == "decode" else hk_prefill_arm(case)
    if "hk_rocm" in want:
        arms["hk_rocm"] = hk_rocm_arm(case)
    for v in (args.variants.split(",") if kind == "decode" and args.variants else ()):
        sp, rest = v.split("x")
        vb = rest.endswith("v")              # ...v: both V bands in flight (v_both)
        w, st = (rest.rstrip("v").split("m") + ["8"])[:2]
        arms[f"hk_{v}"] = hk_decode_arm(case, int(sp), int(w), int(st), v_both=vb)
    return arms


def _one(args, m, kind, label, reqs, seed):
    """One allocation of a shape (seed: its values, its page order, and where its caches land):
    {arm: best us}, plus the row's other fields."""
    case = Case(m, reqs, seed=seed)
    arms = _arms(args, case, kind)
    hkn = "hk" if "hk" in arms else ("hk_rocm" if "hk_rocm" in arms else None)
    before = dict(arms[hkn][0].stats) if kind == "prefill" and hkn else None
    errs = check(arms, case)
    ok = {n: a for n, a in arms.items() if errs[n] <= args.tol}
    bursts = {n: (graphed if kind == "decode" else eager)(c, case.layers)
              for n, (c, _) in ok.items()}
    t = time_arms(bursts, case.layers, args.rounds)
    extra = dict(layers=case.layers, err={n: round(e, 5) for n, e in errs.items()},
                 kv_bytes=case.kv_bytes, flops=case.flops,
                 wrong=[n for n in arms if n not in ok])
    if kind == "decode" and "hk" in arms:
        extra["splits"] = arms["hk"][0].splits
    if before is not None:
        after = arms[hkn][0].stats
        extra["hk_calls"] = {p: after[p] - before.get(p, 0) for p in after}
    if args.profile:
        extra["kernels_us"] = kernel_times(bursts, case.layers)
    del case, arms, bursts
    torch.cuda.empty_cache()
    return t, extra


def burn_in(seconds):
    """Sustained memory traffic until the clocks settle: an idle W7900D starts low and takes about
    a second to ramp, which a microsecond-scale decode would otherwise measure."""
    import time  # noqa: PLC0415

    a = torch.empty(1 << 28, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
    del a, b


def run(args, kind, shapes):
    """Each shape `--seeds` times, a fresh allocation each (on a W7900D where a small decode's
    caches land moves it by up to ~10% from one allocation to the next): each arm's median over
    the allocations of its best-of-rounds time."""
    m = MODELS[args.model]
    rows = []
    if args.burn_in > 0:
        burn_in(args.burn_in)
    for label, reqs in shapes:
        runs = [_one(args, m, kind, label, reqs, seed) for seed in range(args.seeds)]
        names = set.intersection(*(set(t) for t, _ in runs))
        t = {n: sorted(r[0][n] for r in runs)[len(runs) // 2] for n in names}
        extra = runs[0][1]
        row = dict(kind=kind, model=args.model, shape=label, layers=extra["layers"],
                   err=extra["err"], us={n: round(v, 2) for n, v in t.items()})
        if args.seeds > 1:
            row["us_by_seed"] = {n: [round(r[0][n], 2) for r in runs] for n in names}
        if kind == "decode":
            row["gbs"] = {n: round(extra["kv_bytes"] / v / 1e3, 1) for n, v in t.items()}
            if "splits" in extra:
                row["splits"] = extra["splits"]
        else:
            row["tflops"] = {n: round(extra["flops"] / v / 1e6, 2) for n, v in t.items()}
            if "hk_calls" in extra:
                row["hk_calls"] = extra["hk_calls"]
        rivals = {n: v for n, v in t.items() if not n.startswith("hk")}
        if "hk" in t and rivals:
            best = min(rivals, key=rivals.get)
            row["best_rival"], row["hk_speedup"] = best, round(rivals[best] / t["hk"], 3)
        if "kernels_us" in extra:
            row["kernels_us"] = extra["kernels_us"]
        if extra["wrong"]:
            row["wrong"] = extra["wrong"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def table(rows):
    """The rows as a markdown table."""
    kind = rows[0]["kind"]
    unit, rate = ("GB/s", "gbs") if kind == "decode" else ("TF/s", "tflops")
    arms = sorted({a for r in rows for a in list(r["us"]) + r.get("wrong", [])},
                  key=lambda a: (a != "hk", a))
    out = ["| shape | " + " | ".join(f"{a}: us ({unit})" for a in arms)
           + " | hk vs the faster of triton, rocm_attn |",
           "|" + "---|" * (len(arms) + 2)]
    for r in rows:
        cells = [f"{r['us'][a]} ({r[rate][a]})" if a in r["us"] else
                 ("wrong" if a in r.get("wrong", ()) else "-") for a in arms]
        sp = f"{r['hk_speedup']}x vs {r['best_rival']}" if "hk_speedup" in r else "-"
        out.append(f"| {r['shape']} | " + " | ".join(cells) + f" | {sp} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    for kind in ("decode", "prefill"):
        p = sub.add_parser(kind)
        p.add_argument("--model", default="minicpm5", choices=sorted(MODELS))
        p.add_argument("--arms", default="hk,triton,rocm_attn",
                       help="hk, triton (TRITON_ATTN), rocm_attn (ROCM_ATTN: its own kernel on "
                            "pages of a power of two, its Triton fallback on others), hk_rocm "
                            "(prefill: hk's route inside ROCM_ATTN)")
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--burn-in", type=float, default=3.0,
                       help="seconds of memory traffic before the first shape (clock ramp)")
        p.add_argument("--seeds", type=int, default=3,
                       help="allocations of each shape; the row is each arm's median")
        p.add_argument("--tol", type=float, default=2e-2, help="max |error| against float32")
        p.add_argument("--json", help="also write the rows here")
        p.add_argument("--profile", action="store_true",
                       help="also each arm's kernels' device time (torch profiler)")
        if kind == "decode":
            p.add_argument("--batch", default="1,2,4,8,16,32")
            p.add_argument("--ctx", default="8192,16384,32768")
            p.add_argument("--variants", default="",
                           help="extra hk arms SPLITSxWARPS[mSTEP],... (tuning the split plan; "
                                "STEP: splits a merge warp reads)")
        else:
            p.add_argument("--reqs", default="0:8192,0:16384,16384:2048",
                           help="requests CTX:QLEN (a cached prefix, then QLEN new tokens); "
                                "'+' joins requests into one batch")
    args = ap.parse_args()
    if args.kind == "decode":
        shapes = [(f"B{b} ctx{c}", [(c, 1)] * b) for c in map(int, args.ctx.split(","))
                  for b in map(int, args.batch.split(","))]
    else:
        shapes = []
        for spec in args.reqs.split(","):
            reqs = [(int(c) + int(n), int(n)) for c, n in (x.split(":") for x in spec.split("+"))]
            shapes.append((spec, reqs))
    rows = run(args, args.kind, shapes)
    print()
    print(table(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
