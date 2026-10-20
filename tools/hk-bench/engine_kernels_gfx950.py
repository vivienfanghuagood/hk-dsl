"""hk's gfx950 kernels against exactly what vLLM calls on an MI355X, at its shapes.

`headroom` says which families a step spends its time in; this measures, for
each family hk has a gfx950 kernel for, hk's kernel against the engine's own
entry point -- the function the engine calls, so whatever aiter, CK, hipBLASLt
or Triton kernel it dispatches to is what is timed -- in one process,
interleaved, every arm checked against fp32 first.

    VLLM_ROCM_USE_AITER=1 python3 tools/hk-bench/engine_kernels_gfx950.py \\
        --model Qwen/Qwen3-8B --json k.json [--only linear,attn,rows,moe]

* linear: `torch.ops.vllm.rocm_unquantized_gemm` (aiter's tuned GEMM, the
  skinny wvSplitK kernels, hipBLASLt -- whichever it picks) against every hk
  gfx950 GEMM candidate that takes the shape (`hk.ops.gemm_cdna`, the
  four-warp kernel, the decode GEMV), and HipKittens' GEMM where it fits.
  Weights rotate through copies past the 256 MB Infinity Cache, as a
  model's do: a decode step reads every weight once.
* attn: prefill of equal-length causal prompts (the step `headroom` saw).
  ROCM_ATTN's `chunked_prefill_paged_decode` (Triton) and ROCM_AITER_FA's
  `flash_attn_varlen_func` (CK) against hk's `attn_cdna` -- with the
  (tokens, heads, D) <-> (B, H, N, D) copies a patch would pay, and without --
  and HipKittens' GQA, which reads the engine's layout as it is. Decode:
  ROCM_ATTN's paged kernel, timed for the record; hk has none on gfx950.
* rows: aiter's fused add + RMSNorm (`torch.ops.vllm_aiter.fused_add_rms_norm`,
  what vLLM's compiled graph calls with aiter on) against hk's; SiLU-and-mul
  as Inductor compiles it in the graph against hk's.
* moe: aiter's fused MoE (vLLM's AITER path) against hk's MoE GEMM on vLLM's
  sorted lists, both GEMMs and the activation, per layer.

Each arm's time is the median over rounds of a burst of calls between two
events (calls queued back to back, as a captured step queues them); the
spread is the largest distance of a round from that median.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

ROTATE_BYTES = 768 << 20


#: Arms whose host enqueue outran the spin in some round (their time may hold
#: host gaps).
EAGER = set()

_CLOCK_HZ = 2.4e9


def bench(arms, rounds=7, calls=20):
    """{name: f(i)} -> {name: (median ms per call, spread)}; f(i) runs one call
    on input set i. Each burst of `calls` calls is queued behind a GPU spin
    long enough to cover its host enqueue (docs/history/32), so the calls run
    back to back as a captured step runs them: a launch's host cost is not
    timed -- aiter's tuned GEMM spends ~60 us of Python a call at M = 16,
    which an unqueued burst would charge to its kernels. Arms rotate their
    order each round."""
    import time as _t

    names = list(arms)
    per = {n: [] for n in names}
    host = {}
    for n in names:
        for i in range(3):
            arms[n](i)
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for i in range(calls):
            arms[n](i)
        host[n] = _t.perf_counter() - t0
        torch.cuda.synchronize()
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            f = arms[n]
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            spin = 2 * host[n] + 1e-3
            torch.cuda._sleep(int(spin * _CLOCK_HZ))
            s.record()
            t0 = _t.perf_counter()
            for i in range(calls):
                f(i)
            e.record()
            enq = _t.perf_counter() - t0
            torch.cuda.synchronize()
            if enq > spin:   # the spin ended before the host had queued the burst
                EAGER.add(n)
            per[n].append(s.elapsed_time(e) / calls)
    out = {}
    for n, v in per.items():
        med = statistics.median(v)
        out[n] = (med, max(abs(x / med - 1) for x in v))
    return out


def kernels_of(fn):
    """The device kernels one call of fn launches, by name."""
    from torch.profiler import ProfilerActivity, profile

    from hk.integration import headroom

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"ek_{os.getpid()}.json")
    prof.export_chrome_trace(path)
    names = [n for n, _, _ in headroom.device_events(headroom.read_trace(path))]
    os.unlink(path)
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def rel_err(y, ref):
    return ((y.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


# ------------------------------------------------------------------ linear


def linear(cfg, ms, rounds, log):
    import vllm.model_executor.layers.utils  # noqa: F401 -- registers the op

    from hk.autotune import routes as _routes
    from hk.ops import gemm_cdna, hipkittens

    engine = torch.ops.vllm.rocm_unquantized_gemm
    rows = []
    for name, n, k, calls in _routes.linear_shapes(cfg):
        for m in ms:
            if name == "lm_head" and m > 64:
                continue
            g = torch.Generator(device="cuda").manual_seed(0)
            copies = max(2, -(-ROTATE_BYTES // (n * k * 2)))
            ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
                  * k ** -0.5 for _ in range(copies)]
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ref = x[:4096].float() @ ws[0].float().t()
            arms = {"engine": lambda i, ws=ws, x=x: engine(x, ws[i % len(ws)], None)}
            for c in gemm_cdna.CANDIDATES:
                if gemm_cdna.fits(c, m, n, k):
                    arms[c] = (lambda i, c=c, ws=ws, x=x, out=out:
                               gemm_cdna.run(c, x, ws[i % len(ws)], out))
            if hipkittens.gemm_fits(m, n, k):
                arms["hipkittens"] = (lambda i, ws=ws, x=x, out=out:
                                      hipkittens.gemm_bf16(x, ws[i % len(ws)], out=out))
            ok, errs = {}, {}
            for a, f in arms.items():
                try:
                    y = f(0)
                    y = out if y is None else y
                    errs[a] = rel_err(y[:4096], ref)
                except Exception as e:  # noqa: BLE001 -- a candidate that cannot run is not one
                    errs[a] = f"{type(e).__name__}: {str(e)[:120]}"
                    continue
                if a == "engine" or errs[a] < 2e-2:
                    ok[a] = f
            eng_k = kernels_of(lambda: engine(x, ws[0], None))
            t = bench(ok, rounds=rounds, calls=2 * copies if m <= 256 else copies)
            routed = gemm_cdna.route(m, n, k)
            hk_arms = {a: v for a, v in t.items() if a not in ("engine", "hipkittens")}
            best = min(hk_arms, key=lambda a: hk_arms[a][0]) if hk_arms else None
            row = {"name": name, "n": n, "k": k, "m": m, "calls_per_step": calls,
                   "engine_ms": t["engine"][0], "engine_spread": t["engine"][1],
                   "engine_kernels": eng_k,
                   "hk_routed": routed,
                   "hk_routed_ms": t.get(routed, (None,))[0],
                   "hk_best": best, "hk_best_ms": t[best][0] if best else None,
                   "hipkittens_ms": t.get("hipkittens", (None,))[0],
                   "all": {a: round(v[0], 5) for a, v in t.items()},
                   "errors": {a: e for a, e in errs.items() if not isinstance(e, float)
                              or e >= 2e-2}}
            row["ratio_routed"] = (row["engine_ms"] / row["hk_routed_ms"]
                                   if row["hk_routed_ms"] else None)
            row["ratio_best"] = row["engine_ms"] / row["hk_best_ms"] if best else None
            bw = (n * k + m * k + m * n) * 2 / (row["engine_ms"] * 1e-3) / 1e12
            log(f"linear {name:8s} {n}x{k} M={m:<6d} engine {row['engine_ms']*1e3:9.1f} us "
                f"({bw:.2f} TB/s eff) [{eng_k[0][:40] if eng_k else '?'}] | hk routed "
                f"{routed} {fmt_ratio(row['ratio_routed'])} | hk best {best} "
                f"{fmt_ratio(row['ratio_best'])}"
                + (f" | HipKittens {row['engine_ms'] / row['hipkittens_ms']:.3f}x"
                   if row["hipkittens_ms"] else ""))
            rows.append(row)
            del ws, x, out
            torch.cuda.empty_cache()
    return rows


def fmt_ratio(r):
    return "-" if r is None else f"{r:.3f}x"


# ------------------------------------------------------------------ attention


def attention(cfg, specs, rounds, log):
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode)
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    import hk
    from hk.ops import hipkittens

    text = cfg.get("text_config", cfg)
    h, hkv = text["num_attention_heads"], text.get("num_key_value_heads")
    d = text.get("head_dim") or text["hidden_size"] // h
    scale = d ** -0.5
    rows = []
    for b, n in specs:
        t_ = b * n
        g = torch.Generator(device="cuda").manual_seed(1)
        q = torch.randn(t_, h, d, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(t_, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(t_, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
        o = torch.empty_like(q)
        cu = torch.arange(0, t_ + 1, n, device="cuda", dtype=torch.int32)
        lens = torch.full((b,), n, device="cuda", dtype=torch.int32)
        bs = 16
        blocks = b * (n // bs)
        kv = torch.zeros(2, blocks, bs * hkv * d, device="cuda", dtype=torch.bfloat16)
        kc, vc = PagedAttention.split_kv_cache(kv, hkv, d)
        table = torch.arange(blocks, device="cuda", dtype=torch.int32).view(b, n // bs)
        one = torch.ones((), device="cuda", dtype=torch.float32)

        def rocm_attn(i):
            chunked_prefill_paged_decode(q, k, v, o, "auto", kc, vc, table, cu, lens, n, n,
                                         one, one, sm_scale=scale)
            return o

        def aiter_fa(i):
            from aiter import flash_attn_varlen_func
            flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                   max_seqlen_q=n, max_seqlen_k=n, min_seqlen_q=1,
                                   dropout_p=0.0, softmax_scale=scale, causal=True,
                                   window_size=(-1, -1), out=o)
            return o

        qb = q.view(b, n, h, d).transpose(1, 2).contiguous()
        kb = k.view(b, n, hkv, d).transpose(1, 2).contiguous()
        vb = v.view(b, n, hkv, d).transpose(1, 2).contiguous()
        ob = torch.empty_like(qb)

        def hk_kernel(i):
            hk.ops.attention(qb, kb, vb, causal=True, out=ob)
            return ob.transpose(1, 2).reshape(t_, h, d)

        def hk_with_copies(i):
            qq = q.view(b, n, h, d).transpose(1, 2).contiguous()
            kk = k.view(b, n, hkv, d).transpose(1, 2).contiguous()
            vv = v.view(b, n, hkv, d).transpose(1, 2).contiguous()
            oo = hk.ops.attention(qq, kk, vv, causal=True)
            o.view(b, n, h, d).copy_(oo.transpose(1, 2))
            return o

        def hipkittens_gqa(i):
            return hipkittens.gqa_fwd(q.view(b, n, h, d), k.view(b, n, hkv, d),
                                      v.view(b, n, hkv, d), causal=True).view(t_, h, d)

        # fp32 reference on the first prompt's first 512 queries
        qr = q[:n].float().view(n, hkv, h // hkv, d)
        kr, vr = k[:n].float(), v[:n].float()
        s_ = torch.einsum("qgrd,kgd->grqk", qr[:512], kr) * scale
        mask = torch.ones(512, n, device="cuda", dtype=torch.bool).tril()
        s_ = s_.masked_fill(~mask, float("-inf"))
        ref = torch.einsum("grqk,kgd->qgrd", s_.softmax(-1), vr).reshape(512, h, d)
        arms = {"rocm_attn (Triton)": rocm_attn, "aiter_fa (CK)": aiter_fa,
                "hk attn_cdna": hk_kernel, "hk attn_cdna + layout copies": hk_with_copies,
                "HipKittens GQA": hipkittens_gqa}
        ok, errs, kern = {}, {}, {}
        for a, f in arms.items():
            try:
                y = f(0)
                errs[a] = rel_err(y[:512], ref)
                kern[a] = kernels_of(lambda f=f: f(0))
            except Exception as e:  # noqa: BLE001
                errs[a] = f"{type(e).__name__}: {str(e)[:200]}"
                continue
            if errs[a] < 2e-2:
                ok[a] = f
        t = bench(ok, rounds=rounds, calls=5)
        flops = b * 2 * h * n * n * d     # causal: half of 4 h n^2 d
        row = {"phase": "prefill", "batch": b, "n": n, "h": h, "h_kv": hkv, "d": d,
               "ms": {a: round(v[0], 4) for a, v in t.items()},
               "spread": {a: round(v[1], 4) for a, v in t.items()},
               "tflops": {a: round(flops / (v[0] * 1e-3) / 1e12, 1) for a, v in t.items()},
               "kernels": kern, "errors": errs}
        log(f"attn prefill {b}x{n}: " + ", ".join(
            f"{a} {v[0]:.3f} ms ({row['tflops'][a]:.0f} TF)" for a, v in t.items())
            + "".join(f"; {a} FAILED: {e}" for a, e in errs.items()
                      if not isinstance(e, float) or e >= 2e-2))
        rows.append(row)
        del q, k, v, o, kv, qb, kb, vb, ob
        torch.cuda.empty_cache()
    return rows


def decode_attention(cfg, specs, rounds, log):
    """ROCM_ATTN's decode kernel at a context, for the record: hk has no
    paged decode on gfx950, so there is no hk arm."""
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        chunked_prefill_paged_decode)
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    text = cfg.get("text_config", cfg)
    h, hkv = text["num_attention_heads"], text.get("num_key_value_heads")
    d = text.get("head_dim") or text["hidden_size"] // h
    rows = []
    for b, ctx in specs:
        bs = 16
        blocks = b * (ctx // bs)
        kv = torch.randn(2, blocks, bs * hkv * d, device="cuda", dtype=torch.bfloat16)
        kc, vc = PagedAttention.split_kv_cache(kv, hkv, d)
        table = torch.randperm(blocks, device="cuda", dtype=torch.int32).view(b, ctx // bs)
        q = torch.randn(b, h, d, device="cuda", dtype=torch.bfloat16)
        o = torch.empty_like(q)
        cu = torch.arange(0, b + 1, device="cuda", dtype=torch.int32)
        lens = torch.full((b,), ctx, device="cuda", dtype=torch.int32)
        one = torch.ones((), device="cuda", dtype=torch.float32)
        f = (lambda i: chunked_prefill_paged_decode(q, None, None, o, "auto", kc, vc, table,
                                                    cu, lens, ctx, 1, one, one,
                                                    sm_scale=d ** -0.5))
        t = bench({"rocm_attn": f}, rounds=rounds, calls=20)
        kv_bytes = b * ctx * 2 * hkv * d * 2
        ms_ = t["rocm_attn"][0]
        row = {"phase": "decode", "batch": b, "context": ctx, "ms": ms_,
               "kv_gb_per_s": kv_bytes / (ms_ * 1e-3) / 1e9, "kernels": kernels_of(lambda: f(0))}
        log(f"attn decode {b} x {ctx}: rocm_attn {ms_ * 1e3:.1f} us a layer, "
            f"{row['kv_gb_per_s']:.0f} GB/s of KV; hk: no paged decode on gfx950")
        rows.append(row)
        del kv, kc, vc
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------ rows


def row_ops(cfg, ms, rounds, log):
    import vllm.kernels.aiter_ops  # noqa: F401 -- registers torch.ops.vllm_aiter

    import hk

    text = cfg.get("text_config", cfg)
    hid, inter = text["hidden_size"], text["intermediate_size"]
    eps = text.get("rms_norm_eps", 1e-6)
    rows = []
    aiter_add = getattr(torch.ops.vllm_aiter, "fused_add_rms_norm", None)
    aiter_rms = getattr(torch.ops.vllm_aiter, "rms_norm", None)
    silu_ref = torch.compile(lambda x: torch.nn.functional.silu(x[:, :inter]) * x[:, inter:],
                             dynamic=False)
    for m in ms:
        g = torch.Generator(device="cuda").manual_seed(2)
        sets = max(2, min(16, -(-(256 << 20) // (m * hid * 2 * 3))))
        xs = [torch.randn(m, hid, device="cuda", dtype=torch.bfloat16, generator=g)
              for _ in range(sets)]
        rs = [torch.randn(m, hid, device="cuda", dtype=torch.bfloat16, generator=g)
              for _ in range(sets)]
        w = torch.rand(hid, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
        x0, r0 = xs[0].clone(), rs[0].clone()
        s_ = x0.float() + r0.float()
        ref = s_ * torch.rsqrt(s_.pow(2).mean(-1, keepdim=True) + eps) * w.float()
        arms, errs = {}, {}
        if aiter_add is not None:
            y = aiter_add(x0.clone(), r0.clone(), w, eps)
            errs["aiter fused_add_rms_norm"] = rel_err(y[0], ref)
            arms["aiter fused_add_rms_norm"] = lambda i: aiter_add(xs[i % sets],
                                                                   rs[i % sets], w, eps)
        xa, ra = x0.clone(), r0.clone()
        hk.ops.fused_add_rms_norm(xa, ra, w, eps)
        errs["hk fused_add_rms_norm"] = rel_err(xa, ref)
        arms["hk fused_add_rms_norm"] = lambda i: hk.ops.fused_add_rms_norm(
            xs[i % sets], rs[i % sets], w, eps)
        t = bench(arms, rounds=rounds, calls=4 * sets)
        kern = {a: kernels_of(lambda f=f: f(0)) for a, f in arms.items()}
        row = {"op": "fused_add_rms_norm", "m": m, "cols": hid,
               "ms": {a: v[0] for a, v in t.items()}, "spread": {a: v[1] for a, v in t.items()},
               "errors": errs, "kernels": kern}
        e_, h_ = t.get("aiter fused_add_rms_norm"), t["hk fused_add_rms_norm"]
        log(f"fused_add_rms_norm {m}x{hid}: aiter {e_[0] * 1e3 if e_ else float('nan'):.2f} us"
            f", hk {h_[0] * 1e3:.2f} us -> {e_[0] / h_[0] if e_ else float('nan'):.3f}x")
        rows.append(row)
        if aiter_rms is not None:
            arms = {"aiter rms_norm": lambda i: aiter_rms(xs[i % sets], w, eps),
                    "hk rms_norm": lambda i: hk.ops.rms_norm(xs[i % sets], w, eps)}
            ref2 = xs[0].float() * torch.rsqrt(xs[0].float().pow(2).mean(-1, keepdim=True)
                                               + eps) * w.float()
            errs = {a: rel_err(f(0), ref2) for a, f in arms.items()}
            t = bench(arms, rounds=rounds, calls=4 * sets)
            rows.append({"op": "rms_norm", "m": m, "cols": hid,
                         "ms": {a: v[0] for a, v in t.items()}, "errors": errs})
            log(f"rms_norm {m}x{hid}: aiter {t['aiter rms_norm'][0] * 1e3:.2f} us, hk "
                f"{t['hk rms_norm'][0] * 1e3:.2f} us -> "
                f"{t['aiter rms_norm'][0] / t['hk rms_norm'][0]:.3f}x")
        del xs, rs
        gu = [torch.randn(m, 2 * inter, device="cuda", dtype=torch.bfloat16, generator=g)
              for _ in range(max(2, min(16, -(-(256 << 20) // (m * inter * 6)))))]
        outs = torch.empty(m, inter, device="cuda", dtype=torch.bfloat16)
        ref = (torch.nn.functional.silu(gu[0][:, :inter].float()) * gu[0][:, inter:].float())
        arms = {"inductor silu_mul": lambda i: silu_ref(gu[i % len(gu)]),
                "hk silu_mul": lambda i: hk.ops.silu_mul(gu[i % len(gu)][:, :inter],
                                                         gu[i % len(gu)][:, inter:], out=outs)}
        errs = {a: rel_err(f(0), ref) for a, f in arms.items()}
        t = bench(arms, rounds=rounds, calls=4 * len(gu))
        rows.append({"op": "silu_mul", "m": m, "cols": inter,
                     "ms": {a: v[0] for a, v in t.items()}, "errors": errs})
        log(f"silu_mul {m}x{inter}: inductor {t['inductor silu_mul'][0] * 1e3:.2f} us, hk "
            f"{t['hk silu_mul'][0] * 1e3:.2f} us -> "
            f"{t['inductor silu_mul'][0] / t['hk silu_mul'][0]:.3f}x")
        del gu
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------ moe


def moe(cfg, tokens, rounds, log):
    """One MoE layer: aiter's fused MoE as vLLM's AITER path calls it, against
    hk's MoE GEMM for both GEMMs on vLLM's sort (moe_align) with the
    activation between them and the weighted sum over the top-k."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)

    from hk.ops import moe as hk_moe

    import hk

    text = cfg.get("text_config", cfg)
    e, top_k = text["num_experts"], text["num_experts_per_tok"]
    hid, inter = text["hidden_size"], text["moe_intermediate_size"]
    g = torch.Generator(device="cuda").manual_seed(3)
    w1 = torch.randn(e, 2 * inter, hid, device="cuda", dtype=torch.bfloat16,
                     generator=g) * hid ** -0.5
    w2 = torch.randn(e, hid, inter, device="cuda", dtype=torch.bfloat16,
                     generator=g) * inter ** -0.5
    # vLLM's AITER path shuffles the experts' weights at load
    # (convert_to_unquantized_kernel_format); hk reads them as loaded.
    w1s, w2s = rocm_aiter_ops.shuffle_weights(w1.clone(), w2.clone())
    rows = []
    for t_ in tokens:
        # Routings rotate between calls, so a few tokens' experts are not
        # found in the cache from the call before.
        sets = 8 if t_ <= 64 else 2
        xs, wl, il = [], [], []
        for _ in range(sets):
            xs.append(torch.randn(t_, hid, device="cuda", dtype=torch.bfloat16, generator=g))
            logits = torch.randn(t_, e, device="cuda", dtype=torch.float32, generator=g)
            wts, ids = torch.topk(logits.softmax(-1), top_k, dim=-1)
            wl.append((wts / wts.sum(-1, keepdim=True)).float())
            il.append(ids.to(torch.int32))
        x, wts, ids = xs[0], wl[0], il[0]

        def aiter_moe(i):
            j = i % sets
            return rocm_aiter_ops.fused_moe(xs[j], w1s, w2s, wl[j], il[j])

        bm = 16 if t_ <= 256 else 64
        sched_ok = (hk_moe.schedule_for(bm, hid) is not None
                    and hk_moe.schedule_for(bm, inter) is not None)
        if not sched_ok:
            bm = 16
        h1 = torch.empty(t_ * top_k, 2 * inter, device="cuda", dtype=torch.bfloat16)
        h2 = torch.empty(t_ * top_k, hid, device="cuda", dtype=torch.bfloat16)
        act = torch.empty(t_ * top_k, inter, device="cuda", dtype=torch.bfloat16)

        def hk_layer(i):
            j = i % sets
            sorted_ids, expert_ids, ntp = moe_align_block_size(il[j], bm, e)
            hk_moe.moe_gemm(xs[j], w1, h1, sorted_ids, expert_ids, ntp, top_k=top_k,
                            block_m=bm)
            hk.ops.silu_mul(h1[:, :inter], h1[:, inter:], out=act)
            hk_moe.moe_gemm(act, w2, h2, sorted_ids, expert_ids, ntp, top_k=1, block_m=bm,
                            topk_weights=wl[j])
            return h2.view(t_, top_k, hid).sum(1)

        ref = torch.zeros(t_, hid, device="cuda", dtype=torch.float32)
        for j in range(top_k):
            for ex in ids[:, j].unique().tolist():
                sel = ids[:, j] == ex
                a = x[sel].float() @ w1[ex].float().t()
                a = torch.nn.functional.silu(a[:, :inter]) * a[:, inter:]
                ref[sel] += wts[sel, j:j + 1] * (a @ w2[ex].float().t())
        arms, errs, kern = {}, {}, {}
        for a, f in {"aiter fused_moe": aiter_moe, "hk moe_gemm": hk_layer}.items():
            try:
                errs[a] = rel_err(f(0), ref)
                kern[a] = kernels_of(lambda f=f: f(0))
            except Exception as ex_:  # noqa: BLE001
                errs[a] = f"{type(ex_).__name__}: {str(ex_)[:200]}"
                continue
            if errs[a] < 3e-2:
                arms[a] = f
        t = bench(arms, rounds=rounds, calls=10)
        row = {"tokens": t_, "experts": e, "top_k": top_k, "hidden": hid, "inter": inter,
               "ms": {a: v[0] for a, v in t.items()}, "spread": {a: v[1] for a, v in t.items()},
               "errors": errs, "kernels": kern}
        log(f"moe {t_} tokens: " + ", ".join(f"{a} {v[0] * 1e3:.1f} us" for a, v in t.items())
            + "".join(f"; {a} FAILED: {ee}" for a, ee in errs.items()
                      if not isinstance(ee, float) or ee >= 3e-2))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--only", default="linear,attn,rows")
    ap.add_argument("--ms", default="1,16,64,2048,16384")
    ap.add_argument("--attn", default="1x2048,8x2048")
    ap.add_argument("--decode", default="1x2048,16x2048,64x2048")
    ap.add_argument("--row-ms", default="1,16,64,16384")
    ap.add_argument("--moe-tokens", default="1,16,64,2048")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    from hk.autotune import routes as _routes

    cfg = _routes.model_config(args.model)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    import vllm

    meta = {"model": args.model, "device": torch.cuda.get_device_name(),
            "torch": torch.__version__, "hip": torch.version.hip, "vllm": vllm.__version__,
            "VLLM_ROCM_USE_AITER": os.environ.get("VLLM_ROCM_USE_AITER"),
            "date": time.strftime("%Y-%m-%d")}
    try:
        import importlib.metadata as md
        meta["aiter"] = md.version("amd-aiter")
    except Exception:  # noqa: BLE001
        pass
    res = {"meta": meta}
    only = set(args.only.split(","))
    ints = lambda s: [int(v) for v in s.split(",") if v]  # noqa: E731
    pairs = lambda s: [tuple(int(x) for x in p.split("x")) for p in s.split(",") if p]  # noqa: E731
    if "linear" in only:
        res["linear"] = linear(cfg, ints(args.ms), args.rounds, log)
    if "attn" in only:
        res["attn"] = attention(cfg, pairs(args.attn), args.rounds, log)
        res["attn_decode"] = decode_attention(cfg, pairs(args.decode), args.rounds, log)
    if "rows" in only:
        res["rows"] = row_ops(cfg, ints(args.row_ms), args.rounds, log)
    if "moe" in only:
        res["moe"] = moe(cfg, ints(args.moe_tokens), args.rounds, log)
    res["eager_arms"] = sorted(EAGER)
    if EAGER:
        log(f"timed eagerly (their bursts could not be captured): {sorted(EAGER)}")
    res["log"] = lines
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
