#!/usr/bin/env python3
"""gfx950's paged attention: hk against what an engine runs on an MI355X.

    python3 tools/hk-bench/paged_cdna.py decode [--batch 1,16,128] [--ctx 512,8192]
    python3 tools/hk-bench/paged_cdna.py prefill [--case 1x2048+0,...]

Decode arms, each at its own engine's cache layout, holding the same logical
K/V (Qwen3-8B's attention: 32 query heads, 8 KV heads, D = 128):

  hk          hk.ops.paged.decode: the walk and its merge, the split count the
              vLLM backend plans (vllm_backend.decode_splits); hk's head-major
              pages of 32, (blocks, kv heads, 32, D) each of K and V
  hk_vllm31   the same on vLLM 0.31's [B, H, N, 2C] block: K and V two
              row-strided views of one tensor
  triton16    TRITON_ATTN: vLLM's unified_attention (Triton), NHD pages of 16,
              its 3-D kernel at <= 16 requests, as its builder decides
  triton_hk   the same kernel reading hk's head-major cache (transposed view)
  rocm16/32   ROCM_ATTN, the default on gfx950: chunked_prefill_paged_decode,
              which runs ops.paged_attention_rocm (HIP) on its 5-D K / 4-D V
              pages of 16 or 32
  aiter16/32  ROCM_AITER_FA: torch.ops.aiter.paged_attention_v1, NHD
  aiter_ua    ROCM_AITER_UNIFIED_ATTN: aiter's unified_attention, its gfx950
              Gluon kernel by default, NHD pages of 16 (aiter_ua_hk: on hk's)
  sdpa        torch: the request's pages gathered, then SDPA (enable_gqa)

An arm whose engine is not installed is left out, with the reason. Every arm
is checked against fp32 on a few requests first; one off by more than 2e-2
is not timed. Each arm's calls rotate through enough layers' caches that a
burst reads past the 256 MB Infinity Cache, as a decode step does; the burst
is captured in a CUDA graph and replayed (what vLLM's full decode graphs
run), rounds interleaved across arms, the order reversed every other round;
the median round, us per layer.

Prefill arms (`prefill`): requests with a cached prefix and new tokens, all
in the cache, causal from the prefix on:

  hk          hk.ops.paged_prefill on hk's pages
  triton16    TRITON_ATTN's unified_attention, NHD pages of 16
  rocm16      ROCM_ATTN: context_attention_fwd (Triton) over the cache
  aiter_ua    aiter's unified_attention (its gfx950 Gluon kernel)
  sdpa        gathered, SDPA with the prefix-offset causal mask
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

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

HQ, HKV, D = 32, 8, 128
G = HQ // HKV
SCALE = D ** -0.5
PAGE = 32                       # hk's page
BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32
#: A burst reads at least this much, so back-to-back calls do not find their
#: K/V in the 256 MB Infinity Cache.
COLD_BYTES = 768 * 2**20
MAX_MODEL_LEN = 40960


def rel_err(got, want):
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


# ----------------------------------------------------------------- the cache

class Logical:
    """One layer's K/V as hk's pages, a scrambled table of 32-token pages,
    and every other layout derived from those two."""

    def __init__(self, n, ctx, gen):
        self.n, self.ctx = n, ctx
        pages = -(-ctx // PAGE)
        self.pages = pages
        nb = n * pages
        self.k = torch.randn(nb, HKV, PAGE, D, device="cuda", dtype=BF16, generator=gen)
        self.v = torch.randn_like(self.k)
        perm = torch.randperm(nb, device="cuda", generator=gen)
        self.table = perm.view(n, pages).to(I32).contiguous()
        self.lens = torch.full((n,), ctx, device="cuda", dtype=I32)

    def table_for(self, block):
        """The same pages cut into `block`-token ones: hk page p is pages
        p*r .. p*r + r - 1."""
        r = PAGE // block
        t = self.table.long()[:, :, None] * r + torch.arange(r, device="cuda")
        return t.reshape(self.n, -1).to(I32).contiguous()

    def nhd(self, block):
        """(blocks, block, kv heads, D) each, vLLM's NHD."""
        r = PAGE // block
        def f(x):
            nb = x.shape[0]
            return (x.view(nb, HKV, r, block, D).permute(0, 2, 3, 1, 4)
                    .reshape(nb * r, block, HKV, D).contiguous())
        return f(self.k), f(self.v)

    def rocm(self, block):
        """ROCM_ATTN's: K (blocks, kv heads, D/8, block, 8), V (blocks, kv
        heads, D, block)."""
        r = PAGE // block
        nb = self.k.shape[0]
        k = self.k.view(nb, HKV, r, block, D).permute(0, 2, 1, 3, 4).reshape(nb * r, HKV, block, D)
        v = self.v.view(nb, HKV, r, block, D).permute(0, 2, 1, 3, 4).reshape(nb * r, HKV, block, D)
        x = 8
        kk = k.view(nb * r, HKV, block, D // x, x).permute(0, 1, 3, 2, 4).contiguous()
        vv = v.permute(0, 1, 3, 2).contiguous()
        return kk, vv

    def vllm31(self):
        """vLLM 0.31's [B, H, N, 2C]: K and V side by side in the last axis."""
        blk = torch.cat([self.k, self.v], dim=-1).contiguous()
        return blk[..., :D], blk[..., D:]

    def gathered(self, i, n_tok):
        pages = self.table[i, : -(-n_tok // PAGE)].long()
        kk = self.k[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n_tok]
        vv = self.v[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n_tok]
        return kk, vv


def ref_decode(lg: Logical, q, rows):
    out = {}
    for i in rows:
        kk, vv = lg.gathered(i, int(lg.lens[i]))
        qq = q[i].float().view(HKV, G, D)
        s = torch.einsum("hgd,hnd->hgn", qq, kk.float()) * SCALE
        out[i] = torch.einsum("hgn,hnd->hgd", torch.softmax(s, -1), vv.float()).reshape(HQ, D)
    return out


# ------------------------------------------------------------------- decode

def parse_vars(spec):
    """"splits=4,stages=3;kreg=1" -> [{"splits": 4, "stages": 3}, {"kreg": 1}]."""
    out = []
    for one in filter(None, (spec or "").split(";")):
        out.append({k: int(v) for k, v in (kv.split("=") for kv in one.split(","))})
    return out


def decode_arms(lgs, q, want, hk_vars=(), only=None):
    """{name: call(layer)}: call(l) runs layer l's decode into its own
    output, which the check reads."""
    n = lgs[0].n
    arms, skipped = {}, {}
    rows = sorted(want)

    def check(name, fn, out_of):
        try:
            fn(0)
            torch.cuda.synchronize()
            o = out_of()
            err = max(rel_err(o[i].view(HQ, D), want[i]) for i in rows)
        except Exception as e:                       # noqa: BLE001
            skipped[name] = f"raised {type(e).__name__}: {str(e)[:200]}"
            return
        if not err <= 2e-2:
            skipped[name] = f"wrong: max rel err {err:.3e}"
            return
        arms[name] = fn
        print(f"  check {name:<10} {err:.2e}", flush=True)

    # hk
    from hk.integration import vllm_backend as vb
    from hk.ops import paged
    splits = vb.decode_splits(n, HKV, -(-MAX_MODEL_LEN // PAGE))
    qv = q.view(n, HKV, G, D)
    out_hk = torch.empty_like(qv)
    scratch = paged.fused_scratch(n * splits, HKV, D, q.device, group=G)
    walk, merge = vb._op("walk", D, splits), (vb._op("merge", D, splits) if splits > 1 else None)
    check("hk", lambda l: paged.decode(walk, merge, qv, lgs[l].k, lgs[l].v, lgs[l].table,
                                       lgs[l].lens, out_hk, scratch, splits),
          lambda: out_hk.view(n, HQ, D))
    for var in hk_vars:
        from hk.ops import paged_cdna
        opts = dict(var)
        s = opts.pop("splits", splits)
        wk = paged_cdna.kernel_for(D, s, "fused", **{k: v for k, v in opts.items()})
        mg = paged.merge_kernel(D, s) if s > 1 else None
        sc = paged.fused_scratch(n * s, HKV, D, q.device, group=G)
        o_v = torch.empty_like(qv)
        tag = "hk[" + ",".join(f"{k}={v}" for k, v in sorted(var.items())) + "]"
        check(tag, lambda l, wk=wk, mg=mg, sc=sc, o_v=o_v, s=s: paged.decode(
            wk, mg, qv, lgs[l].k, lgs[l].v, lgs[l].table, lgs[l].lens, o_v, sc, s),
            lambda o_v=o_v: o_v.view(n, HQ, D))
    kv31 = [lg.vllm31() for lg in lgs]
    out31 = torch.empty_like(qv)
    check("hk_vllm31", lambda l: paged.decode(walk, merge, qv, kv31[l][0], kv31[l][1],
                                              lgs[l].table, lgs[l].lens, out31, scratch, splits),
          lambda: out31.view(n, HQ, D))

    # vLLM's Triton unified_attention
    try:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    except Exception as e:                            # noqa: BLE001
        unified_attention = None
        skipped["triton16"] = skipped["triton_hk"] = f"no vLLM: {type(e).__name__}"
    one = torch.ones((), device="cuda", dtype=F32)
    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    if unified_attention is not None:
        thr, segs = 128 // HKV, 16
        so = torch.empty(thr, HQ, segs, D, device="cuda", dtype=F32)
        sm = torch.empty(thr, HQ, segs, device="cuda", dtype=F32)
        sl = torch.empty_like(sm)

        def ua(qq, kc, vc, table, lens, out, ctx):
            unified_attention(q=qq, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
                              seqused_k=lens, max_seqlen_k=ctx, softmax_scale=SCALE,
                              causal=True, window_size=(-1, -1), block_table=table,
                              softcap=0, q_descale=None,
                              k_descale=one.expand((n, HKV)), v_descale=one.expand((n, HKV)),
                              seq_threshold_3D=thr, num_par_softmax_segments=segs,
                              softmax_segm_output=so, softmax_segm_max=sm,
                              softmax_segm_expsum=sl)
        nhd16 = [lg.nhd(16) for lg in lgs]
        t16 = [lg.table_for(16) for lg in lgs]
        out_t = torch.empty_like(q)
        ctx = lgs[0].ctx
        check("triton16", lambda l: ua(q, nhd16[l][0], nhd16[l][1], t16[l], lgs[l].lens, out_t, ctx),
              lambda: out_t)
        out_th = torch.empty_like(q)
        check("triton_hk", lambda l: ua(q, lgs[l].k.transpose(1, 2), lgs[l].v.transpose(1, 2),
                                        lgs[l].table, lgs[l].lens, out_th, ctx),
              lambda: out_th)

    # ROCM_ATTN
    try:
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
            chunked_prefill_paged_decode)
    except Exception as e:                            # noqa: BLE001
        chunked_prefill_paged_decode = None
        skipped["rocm16"] = skipped["rocm32"] = f"no vLLM: {type(e).__name__}"
    if chunked_prefill_paged_decode is not None:
        for block in (16, 32):
            lay = [lg.rocm(block) for lg in lgs]
            tb = [lg.table_for(block) for lg in lgs]
            out_r = torch.empty_like(q)

            def rocm(l, lay=lay, tb=tb, out_r=out_r):
                chunked_prefill_paged_decode(
                    query=q, key=None, value=None, output=out_r, kv_cache_dtype="auto",
                    key_cache=lay[l][0], value_cache=lay[l][1], block_table=tb[l],
                    query_start_loc=cu, seq_lens=lgs[l].lens, max_seq_len=lgs[l].ctx,
                    max_query_len=1, k_scale=one, v_scale=one, sm_scale=SCALE)
            check(f"rocm{block}", rocm, lambda out_r=out_r: out_r)

    # ROCM_AITER_FA's decode, and aiter's unified_attention
    try:
        import aiter  # noqa: F401
        have_aiter = hasattr(torch.ops.aiter, "paged_attention_v1")
    except Exception as e:                            # noqa: BLE001
        have_aiter = False
        skipped["aiter16"] = skipped["aiter32"] = skipped["aiter_ua"] = f"no aiter: {type(e).__name__}"
    if have_aiter:
        part = 256
        for block in (16, 32):
            lay = [lg.nhd(block) for lg in lgs]
            tb = [lg.table_for(block) for lg in lgs]
            out_a = torch.empty_like(q)
            parts = -(-lgs[0].ctx // part)
            ws = torch.empty(n * HQ * parts * D * 2 + 2 * n * HQ * parts * 4,
                             dtype=torch.uint8, device="cuda")

            def ait(l, lay=lay, tb=tb, out_a=out_a, ws=ws):
                torch.ops.aiter.paged_attention_v1(
                    out_a, ws, q, lay[l][0], lay[l][1], SCALE, tb[l], cu[:n], lgs[l].lens,
                    lgs[l].ctx, None, "auto", "NHD", 0.0, one, one, None, part, 1, 0)
            check(f"aiter{block}", ait, lambda out_a=out_a: out_a)
        try:
            from aiter.ops.triton.unified_attention import unified_attention as aua
            lay = [lg.nhd(16) for lg in lgs]
            tb = [lg.table_for(16) for lg in lgs]
            out_u = torch.empty_like(q)

            def aiter_ua(l):
                aua(q=q, k=lay[l][0], v=lay[l][1], out=out_u, cu_seqlens_q=cu, max_seqlen_q=1,
                    seqused_k=lgs[l].lens, max_seqlen_k=lgs[l].ctx, softmax_scale=SCALE,
                    causal=True, alibi_slopes=None, window_size=(-1, -1), block_table=tb[l],
                    softcap=0, q_descale=None, k_descale=None, v_descale=None)
            check("aiter_ua", aiter_ua, lambda: out_u)
            out_uh = torch.empty_like(q)

            def aiter_ua_hk(l):
                aua(q=q, k=lgs[l].k.transpose(1, 2), v=lgs[l].v.transpose(1, 2), out=out_uh,
                    cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lgs[l].lens,
                    max_seqlen_k=lgs[l].ctx, softmax_scale=SCALE, causal=True,
                    alibi_slopes=None, window_size=(-1, -1), block_table=lgs[l].table,
                    softcap=0, q_descale=None, k_descale=None, v_descale=None)
            check("aiter_ua_hk", aiter_ua_hk, lambda: out_uh)
        except Exception as e:                        # noqa: BLE001
            skipped["aiter_ua"] = f"{type(e).__name__}: {str(e)[:120]}"

    # torch: gather, then SDPA
    out_s = torch.empty_like(q)

    def sdpa(l):
        lg = lgs[l]
        kk = lg.k[lg.table.long()].permute(0, 2, 1, 3, 4).reshape(n, HKV, -1, D)[:, :, :lg.ctx]
        vv = lg.v[lg.table.long()].permute(0, 2, 1, 3, 4).reshape(n, HKV, -1, D)[:, :, :lg.ctx]
        o = F.scaled_dot_product_attention(q.view(n, HQ, 1, D), kk, vv, scale=SCALE,
                                           enable_gqa=True)
        out_s.copy_(o.view(n, HQ, D))
    check("sdpa", sdpa, lambda: out_s)
    return arms, skipped, splits


# ----------------------------------------------------------------- timing

def graphed(fn, layers, reps):
    """A CUDA graph of `reps` passes over the layers, or None (eager)."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for l in range(layers):
                fn(l)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                for l in range(layers):
                    fn(l)
        torch.cuda.synchronize()
        return g
    except Exception as e:                            # noqa: BLE001
        print(f"    (not captured: {type(e).__name__}: {str(e)[:100]}; eager)")
        torch.cuda.synchronize()
        return None


def time_arms(arms, layers, rounds, reps):
    runs = {}
    for name, fn in arms.items():
        g = graphed(fn, layers, reps)
        if g is not None:
            runs[name] = (g.replay, True)
        else:
            def eager(fn=fn):
                for _ in range(reps):
                    for l in range(layers):
                        fn(l)
            runs[name] = (eager, False)
    for run, _ in runs.values():
        run()
    torch.cuda.synchronize()
    times = {k: [] for k in runs}
    names = list(runs)
    for r in range(rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            run = runs[name][0]
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e) * 1e3 / (reps * layers))
    return {k: dict(us=statistics.median(v), best=min(v),
                    spread=max(abs(x - statistics.median(v)) for x in v) / statistics.median(v),
                    graph=runs[k][1]) for k, v in times.items()}


def burn_in(seconds=3.0):
    a = torch.randn(8192, 8192, device="cuda", dtype=BF16)
    b = torch.empty_like(a)
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()


def run_decode(args):
    results = []
    for ctx in args.ctx:
        for n in args.batch:
            gen = torch.Generator(device="cuda").manual_seed(n * 100003 + ctx)
            layer_bytes = n * ctx * HKV * D * 2 * 2
            layers = max(2, -(-COLD_BYTES // layer_bytes))
            lgs = [Logical(n, ctx, gen) for _ in range(layers)]
            q = torch.randn(n, HQ, D, device="cuda", dtype=BF16, generator=gen)
            rows = sorted({0, n // 2, n - 1})
            want = ref_decode(lgs[0], q, rows)
            print(f"\ndecode n={n} ctx={ctx}: {layers} layers of {layer_bytes / 2**20:.1f} MB",
                  flush=True)
            arms, skipped, splits = decode_arms(lgs, q, want, parse_vars(args.hk_vars))
            if args.only:
                keep = set(args.only.split(","))
                arms = {k: v for k, v in arms.items() if k in keep or k.startswith("hk")}
            for k, why in skipped.items():
                print(f"  skip {k}: {why}")
            reps = max(1, min(20, int(2e4 // max(1, layers))))
            t = time_arms(arms, layers, args.rounds, max(1, reps // 10))
            kv = layer_bytes
            best_alt = min((v["us"] for k, v in t.items() if not k.startswith("hk")), default=None)
            for k, v in sorted(t.items(), key=lambda kv: kv[1]["us"]):
                print(f"  {k:<10} {v['us']:8.2f} us  {kv / v['us'] / 1e6:6.2f} TB/s  "
                      f"spread {v['spread']:.1%}{'' if v['graph'] else '  (eager)'}")
            if best_alt and "hk" in t:
                print(f"  hk / best alternative: {best_alt / t['hk']['us']:.3f}x  (splits {splits})")
            results.append(dict(n=n, ctx=ctx, layers=layers, splits=splits, times=t,
                                skipped=skipped))
            del lgs, arms
            torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------------ prefill

class Prefill:
    """Requests [(ctx, q_len)]: each sequence's ctx + q_len tokens in hk's
    pages under a scrambled table, its last q_len the queries."""

    def __init__(self, reqs, gen):
        self.reqs = reqs
        self.seq = [c + n for c, n in reqs]
        pages = [-(-s // PAGE) for s in self.seq]
        self.max_pages = max(pages)
        nb = sum(pages) + 1
        self.k = torch.randn(nb, HKV, PAGE, D, device="cuda", dtype=BF16, generator=gen)
        self.v = torch.randn_like(self.k)
        perm = torch.randperm(nb, device="cuda", generator=gen)
        table = torch.zeros(len(reqs), self.max_pages, dtype=I32, device="cuda")
        at = 0
        for i, p in enumerate(pages):
            table[i, :p] = perm[at:at + p].to(I32)
            at += p
        self.table = table.contiguous()
        self.lens = torch.tensor(self.seq, device="cuda", dtype=I32)
        starts = [0]
        for _, n in reqs:
            starts.append(starts[-1] + n)
        self.starts = starts
        self.q_start = torch.tensor(starts, device="cuda", dtype=I32)
        self.n_tok = starts[-1]
        self.q = torch.randn(self.n_tok, HQ, D, device="cuda", dtype=BF16, generator=gen)
        # The new tokens' K and V, as an engine passes them beside the cache.
        ks, vs = [], []
        for i, (c, n) in enumerate(reqs):
            kk, vv = self.gathered(i)
            ks.append(kk[:, c:c + n].permute(1, 0, 2))
            vs.append(vv[:, c:c + n].permute(1, 0, 2))
        self.k_new = torch.cat(ks).contiguous()
        self.v_new = torch.cat(vs).contiguous()

    def gathered(self, i):
        s = self.seq[i]
        pages = self.table[i, : -(-s // PAGE)].long()
        kk = self.k[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :s]
        vv = self.v[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :s]
        return kk, vv

    def table_for(self, block):
        r = PAGE // block
        t = self.table.long()[:, :, None] * r + torch.arange(r, device="cuda")
        return t.reshape(len(self.reqs), -1).to(I32).contiguous()

    def nhd(self, block):
        r = PAGE // block
        def f(x):
            nb = x.shape[0]
            return (x.view(nb, HKV, r, block, D).permute(0, 2, 3, 1, 4)
                    .reshape(nb * r, block, HKV, D).contiguous())
        return f(self.k), f(self.v)

    def rocm(self, block):
        r = PAGE // block
        nb = self.k.shape[0]
        k = self.k.view(nb, HKV, r, block, D).permute(0, 2, 1, 3, 4).reshape(nb * r, HKV, block, D)
        v = self.v.view(nb, HKV, r, block, D).permute(0, 2, 1, 3, 4).reshape(nb * r, HKV, block, D)
        kk = k.view(nb * r, HKV, block, D // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
        return kk, v.permute(0, 1, 3, 2).contiguous()

    def flops(self):
        return sum(4.0 * HQ * D * (n * c + n * n / 2) for c, n in self.reqs)

    def reference(self, picks):
        """fp32 rows of (request, query head) picks: {(i, h): (n, D)}."""
        out = {}
        for i, h in picks:
            c, n = self.reqs[i]
            kk, vv = self.gathered(i)
            q0 = self.starts[i]
            qq = self.q[q0:q0 + n, h].float()
            sc = (qq @ kk[h // G].float().T) * SCALE
            pos = c + torch.arange(n, device="cuda")
            keep = torch.arange(self.seq[i], device="cuda")[None, :] <= pos[:, None]
            sc = sc.masked_fill(~keep, float("-inf"))
            out[(i, h)] = torch.softmax(sc, -1) @ vv[h // G].float()
        return out


def prefill_arms(pf: Prefill, want):
    arms, skipped = {}, {}
    n_req = len(pf.reqs)
    max_q = max(n for _, n in pf.reqs)
    max_s = max(pf.seq)
    one = torch.ones((), device="cuda", dtype=F32)

    def check(name, fn, out):
        try:
            fn()
            torch.cuda.synchronize()
            err = 0.0
            for (i, h), w in want.items():
                q0 = pf.starts[i]
                err = max(err, rel_err(out[q0:q0 + pf.reqs[i][1], h], w))
        except Exception as e:                       # noqa: BLE001
            skipped[name] = f"raised {type(e).__name__}: {str(e)[:200]}"
            return
        if not err <= 2e-2:
            skipped[name] = f"wrong: max rel err {err:.3e}"
            return
        arms[name] = fn
        print(f"  check {name:<10} {err:.2e}", flush=True)

    from hk.ops import paged_prefill
    tiles = paged_prefill.work_list([(i, n) for i, (_, n) in enumerate(pf.reqs)], "cuda")
    kern = paged_prefill.kernel_for(D, G, SCALE)
    o_hk = torch.empty_like(pf.q)
    check("hk", lambda: kern(pf.q.view(pf.n_tok, HQ * D), pf.k, pf.v, pf.table, pf.lens,
                             pf.q_start, tiles, o_hk.view(pf.n_tok, HQ * D)), o_hk)

    try:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    except Exception as e:                            # noqa: BLE001
        unified_attention = None
        skipped["triton16"] = f"no vLLM: {type(e).__name__}"
    if unified_attention is not None:
        k16, v16 = pf.nhd(16)
        t16 = pf.table_for(16)
        o_t = torch.empty_like(pf.q)
        check("triton16", lambda: unified_attention(
            q=pf.q, k=k16, v=v16, out=o_t, cu_seqlens_q=pf.q_start, max_seqlen_q=max_q,
            seqused_k=pf.lens, max_seqlen_k=max_s, softmax_scale=SCALE, causal=True,
            window_size=(-1, -1), block_table=t16, softcap=0, q_descale=None,
            k_descale=one.expand((n_req, HKV)), v_descale=one.expand((n_req, HKV))), o_t)
        o_th = torch.empty_like(pf.q)
        check("triton_hk", lambda: unified_attention(
            q=pf.q, k=pf.k.transpose(1, 2), v=pf.v.transpose(1, 2), out=o_th,
            cu_seqlens_q=pf.q_start, max_seqlen_q=max_q, seqused_k=pf.lens,
            max_seqlen_k=max_s, softmax_scale=SCALE, causal=True, window_size=(-1, -1),
            block_table=pf.table, softcap=0, q_descale=None,
            k_descale=one.expand((n_req, HKV)), v_descale=one.expand((n_req, HKV))), o_th)
    try:
        from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
            chunked_prefill_paged_decode)
        kr, vr = pf.rocm(16)
        tr = pf.table_for(16)
        o_r = torch.empty_like(pf.q)
        check("rocm16", lambda: chunked_prefill_paged_decode(
            query=pf.q, key=pf.k_new, value=pf.v_new, output=o_r, kv_cache_dtype="auto",
            key_cache=kr, value_cache=vr, block_table=tr, query_start_loc=pf.q_start,
            seq_lens=pf.lens, max_seq_len=max_s, max_query_len=max_q, k_scale=one,
            v_scale=one, sm_scale=SCALE), o_r)
    except Exception as e:                            # noqa: BLE001
        skipped["rocm16"] = f"{type(e).__name__}: {str(e)[:150]}"
    try:
        from aiter.ops.triton.unified_attention import unified_attention as aua
        o_u = torch.empty_like(pf.q)
        check("aiter_ua_hk", lambda: aua(
            q=pf.q, k=pf.k.transpose(1, 2), v=pf.v.transpose(1, 2), out=o_u,
            cu_seqlens_q=pf.q_start, max_seqlen_q=max_q, seqused_k=pf.lens,
            max_seqlen_k=max_s, softmax_scale=SCALE, causal=True, alibi_slopes=None,
            window_size=(-1, -1), block_table=pf.table, softcap=0, q_descale=None,
            k_descale=None, v_descale=None), o_u)
    except Exception as e:                            # noqa: BLE001
        skipped["aiter_ua_hk"] = f"{type(e).__name__}: {str(e)[:150]}"
    o_s = torch.empty_like(pf.q)

    def sdpa():
        for i, (c, n) in enumerate(pf.reqs):
            kk, vv = pf.gathered(i)
            q0 = pf.starts[i]
            qq = pf.q[q0:q0 + n].transpose(0, 1)[None]
            keep = (torch.arange(pf.seq[i], device="cuda")[None, :]
                    <= (c + torch.arange(n, device="cuda"))[:, None])
            o = F.scaled_dot_product_attention(qq, kk[None], vv[None], attn_mask=keep,
                                               scale=SCALE, enable_gqa=True)
            o_s[q0:q0 + n].copy_(o[0].transpose(0, 1))
    check("sdpa", sdpa, o_s)
    return arms, skipped


def run_prefill(args):
    results = []
    for spec in args.cases.split(";"):
        reqs = [tuple(int(x) for x in r.split("+")) for r in spec.split(",")]
        gen = torch.Generator(device="cuda").manual_seed(len(reqs) * 7919 + sum(map(sum, reqs)))
        pf = Prefill(reqs, gen)
        picks = [(0, 0), (len(reqs) - 1, HQ - 1)]
        want = pf.reference(picks)
        print(f"\nprefill {spec} (ctx+q): {pf.n_tok} new tokens", flush=True)
        arms, skipped = prefill_arms(pf, want)
        for k, why in skipped.items():
            print(f"  skip {k}: {why}")
        runs = {k: (lambda fn=fn: (lambda l: fn())) () for k, fn in arms.items()}
        t = time_arms(runs, 1, args.rounds, 3)
        fl = pf.flops()
        best_alt = min((v["us"] for k, v in t.items() if k != "hk"), default=None)
        for k, v in sorted(t.items(), key=lambda kv: kv[1]["us"]):
            print(f"  {k:<12} {v['us']:9.1f} us  {fl / v['us'] / 1e6:7.1f} TFLOP/s  "
                  f"spread {v['spread']:.1%}{'' if v['graph'] else '  (eager)'}")
        if best_alt and "hk" in t:
            print(f"  hk / best alternative: {best_alt / t['hk']['us']:.3f}x")
        results.append(dict(reqs=reqs, times=t, skipped=skipped, flops=fl))
        del pf
        torch.cuda.empty_cache()
    return results


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["decode", "prefill"])
    ap.add_argument("--cases", default="0+2048;4096+512;2048+1024,2048+1024,2048+1024,2048+1024;"
                    "0+2048,0+2048,0+2048,0+2048,0+2048,0+2048,0+2048,0+2048;"
                    "0+300,1000+77,4000+1000,64+2048",
                    help="prefill: ';'-separated batches of ctx+q_len requests")
    ap.add_argument("--batch", default="1,4,16,32,64,128")
    ap.add_argument("--ctx", default="512,1024,2048,4096,8192")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--json")
    ap.add_argument("--hk-vars", help="extra hk arms: 'splits=4,stages=3;kreg=1'")
    ap.add_argument("--only", help="alternatives to time besides hk's arms")
    args = ap.parse_args()
    args.batch = [int(x) for x in args.batch.split(",")]
    args.ctx = [int(x) for x in args.ctx.split(",")]
    print(f"device {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    burn_in()
    res = run_decode(args) if args.what == "decode" else run_prefill(args)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
