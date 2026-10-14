#!/usr/bin/env python3
"""Decode attention, layer for layer, against the Triton attention vLLM ships.

Both arms are what each backend does per layer per decode step, called the way
vLLM 0.16.1 calls it, so the wrapper around each kernel is in the number:

  stock_layer   triton_reshape_and_cache_flash + unified_attention, with the
                arguments TritonAttentionImpl.do_kv_cache_update and .forward
                build (2-D kernel above 16 requests, 16-segment 3-D kernel and
                its reduction at or below, as TritonAttentionMetadataBuilder
                decides)
  stock_attn    unified_attention alone
  hk_layer      HKAttentionImpl.forward (the cache write is inside it), as
                routed: Triton's kernel on hk's cache up to
                vllm_backend.TRITON_DECODE_MAX requests (every batch, on
                vLLM 0.31), hk's above
  hk_attn       HKAttentionImpl._decode_all (no cache write), as routed
  hk_kernel     hk's split decode alone, walk and merge, at every batch
  tri_on_hk     unified_attention reading hk's head-major cache

The first run of this file, against the backend before its decode was fused
(a torch scatter, a padded query copy, a pybind launch on stream 0, a torch
merge and an output copy per layer), is the baseline in
docs/history/17-the-decode-step-captured-whole.md.

Two clocks. `eager` times one call between CUDA events with nothing captured,
which is how attention runs inside vLLM's piecewise graphs: the host's launch
gaps count. `graph` captures the call into a CUDA graph and times replays, the
GPU time alone, which is what a backend captured whole costs. An arm that
cannot be captured prints its reason instead of a number.

Correctness before timing: every hk arm's output is checked against stock's on
the same K/V, each written through its own backend's cache write. And once, up
front, the cache write itself: vLLM's two reshape_and_cache_flash ops (C++ and
Triton) into hk's head-major layout through a transposed view, against the
torch scatter the hk backend does, bit for bit, with slot -1 entries that must
be skipped.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/paged_ab.py \\
        --ctx 1024,4096,16384 --batch 1,4,8,16,32 [--layers 36]

`--layers 36` is the one to believe for a decode step. With one cache, every
replay rereads K/V the previous one left in the 96 MB Infinity Cache, and a
one-layer graph's replay overhead (~13 us) sits on top of both arms; at
1024 tokens and one request that hid hk losing by 10 us a layer.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

from _timing import burn_in, interleave            # noqa: E402

#: Qwen3-8B's attention.
HQ, HKV, D = 32, 8, 128
G = HQ // HKV
SCALE = D ** -0.5
LAYERS = 36
HBM_GB_S = 864.0
#: The Triton backend's page: vLLM's default block size on ROCm.
T_BLOCK = 16
#: TritonAttentionMetadataBuilder: MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads, and
#: NUM_PAR_SOFTMAX_SEGMENTS. 16 is also a capture size, so graph mode keeps it.
SEQ_THRESHOLD_3D = 128 // HKV
SEGMENTS = 16

BF16, F32, I32 = torch.bfloat16, torch.float32, torch.int32


def check(name, got, want, rel=2e-2):
    """Max error relative to max |want|; a zero reference or a NaN fails."""
    got, want = got.float(), want.float()
    ref = want.abs().max().item()
    err = (got - want).abs().max().item()
    ok = ref > 0 and err == err and err <= rel * ref
    print(f"  check {name:<12} max|err| {err:.3e}  max|ref| {ref:.3e}  "
          f"{'ok' if ok else 'FAIL'}", flush=True)
    return ok


def pages_and_slots(n, ctx, block, gen):
    """A shuffled page table, as an allocator that has run a while leaves
    it, and the slot of every token position under it."""
    pages = ctx // block
    perm = torch.randperm(n * pages, device="cuda", generator=gen)
    table = perm.view(n, pages).to(I32)
    pos = torch.arange(ctx, device="cuda")
    slots = (table[:, pos // block].long() * block + pos % block)
    return table, slots                                # (n, pages), (n, ctx)


# ---------------------------------------------------------------- the cache

def cache_write_check():
    """vLLM's cache writes into the head-major layout, against torch's."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    gen = torch.Generator(device="cuda").manual_seed(1)
    nb, block, n = 64, 32, 300
    key = torch.randn(n, HKV, D, device="cuda", dtype=BF16, generator=gen)
    value = torch.randn_like(key)
    slots = torch.randperm(nb * block, device="cuda", generator=gen)[:n]
    slots[::7] = -1                                   # padding: must be skipped
    k_scale = torch.ones((), device="cuda", dtype=F32)

    def fresh():
        kv = torch.full((2, nb, HKV, block, D), 7.0, device="cuda", dtype=BF16)
        return kv[0], kv[1]

    k0, v0 = fresh()
    live = slots >= 0
    s = slots[live]
    k0[s // block, :, s % block] = key[live]
    v0[s // block, :, s % block] = value[live]

    ok = True
    for name, write in (
        ("c++", lambda kc, vc: ops.reshape_and_cache_flash(
            key, value, kc, vc, slots, "auto", k_scale, k_scale)),
        ("triton", lambda kc, vc: triton_reshape_and_cache_flash(
            key, value, kc, vc, slots, "auto", k_scale, k_scale)),
    ):
        k1, v1 = fresh()
        try:
            # (num_blocks, block, heads, d) over head-major storage.
            write(k1.transpose(1, 2), v1.transpose(1, 2))
            torch.cuda.synchronize()
            same = torch.equal(k1, k0) and torch.equal(v1, v0)
        except Exception as e:                        # noqa: BLE001
            same = False
            print(f"  cache {name}: raised {type(e).__name__}: {e}")
        print(f"  cache write {name:<7} into head-major, slot -1 skipped: "
              f"{'bit-identical' if same else 'DIFFERS'}", flush=True)
        ok = ok and same
    return ok


# ---------------------------------------------------------------- the arms

def stock_arms(n, ctx, key, value, q, k_new, v_new, gen, layers=1):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )
    table, slots = pages_and_slots(n, ctx, T_BLOCK, gen)
    kv = torch.zeros(n * ctx // T_BLOCK, 2, T_BLOCK, HKV, D, device="cuda",
                     dtype=BF16)
    key_cache, value_cache = kv.unbind(1)
    scale1 = torch.ones((), device="cuda", dtype=F32)
    triton_reshape_and_cache_flash(
        key.view(-1, HKV, D), value.view(-1, HKV, D), key_cache, value_cache,
        slots.reshape(-1), "auto", scale1, scale1)
    new_slots = slots[:, -1].contiguous()
    kvs = [kv] + [kv.clone() for _ in range(layers - 1)]

    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    seq_lens = torch.full((n,), ctx, device="cuda", dtype=I32)
    segm_o = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, D, device="cuda",
                         dtype=F32)
    segm_m = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, device="cuda",
                         dtype=F32)
    segm_l = torch.empty_like(segm_m)
    out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)

    def attn_one(kv):
        kc, vc = kv.unbind(1)
        unified_attention(
            q=q[:n], k=kc, v=vc, out=out[:n], cu_seqlens_q=cu,
            max_seqlen_q=1, seqused_k=seq_lens, max_seqlen_k=ctx,
            softmax_scale=SCALE, causal=True, alibi_slopes=None,
            use_alibi_sqrt=False, window_size=(-1, -1), block_table=table,
            softcap=0, q_descale=None,
            k_descale=scale1.expand((cu.shape[0] - 1, kc.shape[2])),
            v_descale=scale1.expand((cu.shape[0] - 1, kc.shape[2])),
            seq_threshold_3D=SEQ_THRESHOLD_3D,
            num_par_softmax_segments=SEGMENTS, softmax_segm_output=segm_o,
            softmax_segm_max=segm_m, softmax_segm_expsum=segm_l, sinks=None,
            output_scale=None, mm_prefix_range=None)

    def attn():
        for kv in kvs:
            attn_one(kv)

    def layer():
        for kv in kvs:
            kc, vc = kv.unbind(1)
            triton_reshape_and_cache_flash(k_new, v_new, kc, vc, new_slots,
                                           "auto", scale1, scale1)
            attn_one(kv)

    return {"stock_layer": layer, "stock_attn": attn}, out


def hk_arms(n, ctx, key, value, q, k_new, v_new, gen, layers=1):
    from hk.integration import vllm_backend
    from hk.ops import paged

    _, impl_cls = vllm_backend._build()
    impl = impl_cls(num_heads=HQ, head_size=D, scale=SCALE, num_kv_heads=HKV,
                    alibi_slopes=None, sliding_window=None,
                    kv_cache_dtype="auto")
    block = paged.KV_BLOCK
    table, slots = pages_and_slots(n, ctx, block, gen)
    kv = torch.zeros(2, n * ctx // block, HKV, block, D, device="cuda",
                     dtype=BF16)
    impl._update_cache(None, key.view(-1, HKV, D), value.view(-1, HKV, D),
                       kv[0], kv[1], slots.reshape(-1))
    kvs = [kv] + [kv.clone() for _ in range(layers - 1)]
    md = types.SimpleNamespace(
        num_actual_tokens=n, slot_mapping=slots[:, -1].contiguous(),
        max_query_len=1, max_seq_len=ctx, block_table=table,
        seq_lens=torch.full((n,), ctx, device="cuda", dtype=I32),
        query_start_loc=torch.arange(n + 1, device="cuda", dtype=I32))
    # (tokens, heads, head_dim): the view vLLM hands an impl's forward.
    out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)

    def layer():
        for kv in kvs:
            impl.forward(None, q, k_new, v_new, kv, md, out)

    def attn():
        for kv in kvs:
            impl._decode_all(q, out, kv[0], kv[1], md, n)

    # The launches as the backend plans them, nothing around them.
    splits = impl._plan(n)
    scratch = vllm_backend._scratch(q.device, HKV, D, impl._rows, G)
    q4, o4 = q.view(n, HKV, G, D), out.view(n, HKV, G, D)

    def kernel():
        for kv in kvs:
            vllm_backend.decode(q4, kv[0], kv[1], table, md.seq_lens, o4,
                                scratch, splits)

    # Triton's kernel reading this backend's cache: it addresses K/V by
    # stride, so the head-major pages are a (blocks, block, heads, d) view.
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )
    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    segm_o = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, D, device="cuda",
                         dtype=F32)
    segm_m = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, device="cuda",
                         dtype=F32)
    segm_l = torch.empty_like(segm_m)
    scale1 = torch.ones((), device="cuda", dtype=F32)
    tri_out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)

    def tri_on_hk():
        for kv in kvs:
            kc, vc = kv[0].transpose(1, 2), kv[1].transpose(1, 2)
            unified_attention(
                q=q[:n], k=kc, v=vc, out=tri_out[:n], cu_seqlens_q=cu,
                max_seqlen_q=1, seqused_k=md.seq_lens, max_seqlen_k=ctx,
                softmax_scale=SCALE, causal=True, alibi_slopes=None,
                use_alibi_sqrt=False, window_size=(-1, -1), block_table=table,
                softcap=0, q_descale=None,
                k_descale=scale1.expand((n, HKV)),
                v_descale=scale1.expand((n, HKV)),
                seq_threshold_3D=SEQ_THRESHOLD_3D,
                num_par_softmax_segments=SEGMENTS, softmax_segm_output=segm_o,
                softmax_segm_max=segm_m, softmax_segm_expsum=segm_l,
                sinks=None, output_scale=None, mm_prefix_range=None)

    arms = {"hk_layer": layer, "hk_attn": attn, "hk_kernel": kernel,
            "tri_on_hk": tri_on_hk}
    return arms, (out, tri_out), {}, splits


# ---------------------------------------------------------------- clocks

def graphed(fn):
    """Capture fn; return a replay thunk."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def run_case(n, ctx, rounds, layers=1):
    gen = torch.Generator(device="cuda").manual_seed(n * 100003 + ctx)
    key = torch.randn(n, ctx, HKV, D, device="cuda", dtype=BF16, generator=gen)
    value = torch.randn_like(key)
    q = torch.randn(n, HQ, D, device="cuda", dtype=BF16, generator=gen)
    # The step's own token is the last position, already in the cache: the
    # layer arms rewrite it with the same values, so every call is the same.
    k_new = key[:, -1].contiguous()
    v_new = value[:, -1].contiguous()

    stock, s_out = stock_arms(n, ctx, key, value, q, k_new, v_new, gen,
                              layers)
    hk, h_out, uncapturable, splits = hk_arms(n, ctx, key, value, q, k_new,
                                              v_new, gen, layers)
    del key, value

    stock["stock_attn"]()
    hk["hk_attn"]()
    hk["tri_on_hk"]()
    torch.cuda.synchronize()
    h_out, t_out = h_out
    if not check(f"n{n} c{ctx}", h_out, s_out):
        return None
    if not check("tri_on_hk", t_out, s_out):
        return None

    eager = interleave({**stock, **hk}, rounds=rounds)
    graph, refused = {}, {}
    for name, fn in {**stock, **hk}.items():
        if name in uncapturable:
            refused[name] = uncapturable[name]
            continue
        try:
            graph[name] = graphed(fn)
        except Exception as e:                        # noqa: BLE001
            refused[name] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    gbest = interleave(graph, rounds=rounds)
    for name, why in refused.items():
        print(f"  {name}: not captured ({why})")
        gbest[name] = None
    per = {k: v / layers for k, v in eager.items()}
    gper = {k: (v / layers if v is not None else None) for k, v in gbest.items()}
    return per, gper, splits


#: Batches with prefill in them, as vLLM's defaults make them: (name, [(ctx,
#: q_len)]), decodes first, the order vLLM's reorder gives the backend.
PREFILL_CASES = (
    ("prompt 2048", [(0, 2048)]),
    ("prompt 8192", [(0, 8192)]),
    ("chunk 512 @4096", [(4096, 512)]),
    ("4 x chunk 1024 @1024", [(1024, 1024)] * 4),
    ("8 x prefix 3000 + 200", [(3000, 200)] * 8),
    ("16 decodes + chunk 512", [(2047, 1)] * 16 + [(2048, 512)]),
)


def prefill_case(reqs, rounds):
    """One mixed batch: each side writes the new tokens' K/V through its own
    cache write and attends, eagerly -- a step with prefill in it runs
    between vLLM's graph pieces."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention,
    )
    from hk.integration import vllm_backend
    from hk.ops import paged, paged_prefill

    gen = torch.Generator(device="cuda").manual_seed(len(reqs))
    seq = [c + n for c, n in reqs]
    q_lens = [n for _, n in reqs]
    r = len(reqs)
    n_new = sum(q_lens)
    starts = [0]
    for n in q_lens:
        starts.append(starts[-1] + n)
    keys = [torch.randn(s, HKV, D, device="cuda", dtype=BF16, generator=gen)
            for s in seq]
    vals = [torch.randn_like(k) for k in keys]
    q = torch.randn(n_new, HQ, D, device="cuda", dtype=BF16, generator=gen)
    k_new = torch.cat([k[-n:] for k, n in zip(keys, q_lens)])
    v_new = torch.cat([v[-n:] for v, n in zip(vals, q_lens)])
    cu = torch.tensor(starts, device="cuda", dtype=I32)
    lens = torch.tensor(seq, device="cuda", dtype=I32)

    def layout(block):
        pages = max(-(-s // block) for s in seq)
        perm = torch.randperm(r * pages, device="cuda", generator=gen)
        table = perm.view(r, pages).to(I32)
        slots_all, slots_new = [], []
        for i, s in enumerate(seq):
            pos = torch.arange(s, device="cuda")
            sl = table[i, pos // block].long() * block + pos % block
            slots_all.append(sl)
            slots_new.append(sl[-q_lens[i]:])
        return table, r * pages, torch.cat(slots_all), torch.cat(slots_new)

    # stock
    t_table, t_blocks, t_all, t_new = layout(T_BLOCK)
    kv_t = torch.zeros(t_blocks, 2, T_BLOCK, HKV, D, device="cuda", dtype=BF16)
    one = torch.ones((), device="cuda", dtype=F32)
    kc, vc = kv_t.unbind(1)
    triton_reshape_and_cache_flash(torch.cat(keys), torch.cat(vals), kc, vc,
                                   t_all, "auto", one, one)
    segm = (torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, D, device="cuda",
                        dtype=F32),
            torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, device="cuda",
                        dtype=F32),
            torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, device="cuda",
                        dtype=F32))
    s_out = torch.empty(n_new, HQ, D, device="cuda", dtype=BF16)

    def stock():
        kc, vc = kv_t.unbind(1)
        triton_reshape_and_cache_flash(k_new, v_new, kc, vc, t_new, "auto",
                                       one, one)
        unified_attention(
            q=q, k=kc, v=vc, out=s_out, cu_seqlens_q=cu,
            max_seqlen_q=max(q_lens), seqused_k=lens, max_seqlen_k=max(seq),
            softmax_scale=SCALE, causal=True, alibi_slopes=None,
            use_alibi_sqrt=False, window_size=(-1, -1), block_table=t_table,
            softcap=0, q_descale=None,
            k_descale=one.expand((r, HKV)), v_descale=one.expand((r, HKV)),
            seq_threshold_3D=SEQ_THRESHOLD_3D,
            num_par_softmax_segments=SEGMENTS, softmax_segm_output=segm[0],
            softmax_segm_max=segm[1], softmax_segm_expsum=segm[2], sinks=None,
            output_scale=None, mm_prefix_range=None)

    # hk
    _, impl_cls = vllm_backend._build()
    impl = impl_cls(num_heads=HQ, head_size=D, scale=SCALE, num_kv_heads=HKV,
                    alibi_slopes=None, sliding_window=None,
                    kv_cache_dtype="auto")
    h_table, h_blocks, h_all, h_new = layout(paged.KV_BLOCK)
    kv_h = torch.zeros(2, h_blocks, HKV, paged.KV_BLOCK, D, device="cuda",
                       dtype=BF16)
    impl._update_cache(None, torch.cat(keys), torch.cat(vals), kv_h[0],
                       kv_h[1], h_all)
    nd = 0
    while nd < r and q_lens[nd] == 1:
        nd += 1
    md = types.SimpleNamespace(
        num_actual_tokens=n_new, slot_mapping=h_new, max_query_len=max(q_lens),
        max_seq_len=max(seq), block_table=h_table, seq_lens=lens,
        query_start_loc=cu, hk_num_decodes=nd,
        hk_prefill_tiles=paged_prefill.work_list(
            [(i, q_lens[i]) for i in range(nd, r)], "cuda") if nd < r else None)
    h_out = torch.empty(n_new, HQ, D, device="cuda", dtype=BF16)

    def hk_layer():
        impl.forward(None, q, k_new, v_new, kv_h, md, h_out)

    stock()
    hk_layer()
    torch.cuda.synchronize()
    ok = check(f"r{r} t{n_new}", h_out, s_out)
    best = interleave({"stock": stock, "hk": hk_layer}, rounds=rounds)
    # Causal attention FLOPs: 4 * d per (query, key) pair it does not mask.
    flops = sum(4 * D * HQ * (n * c + n * (n + 1) / 2) for c, n in reqs)
    return ok, best, flops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", default="1024,4096,16384")
    ap.add_argument("--batch", default="1,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--prefill", action="store_true",
                    help="batches with prefill in them instead of decode")
    ap.add_argument("--layers", type=int, default=1,
                    help="each arm walks this many caches per call, as a "
                         "decode step walks its layers': at 36, a layer's K/V "
                         "is no longer in the Infinity Cache when it is read "
                         "again. Times stay per layer.")
    args = ap.parse_args()

    if args.prefill:
        torch.manual_seed(0)
        print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}")
        burn_in()
        print(f"\n{'batch':<26}{'stock ms':>10}{'hk ms':>10}{'hk/stock':>10}"
              f"{'hk TFLOP/s':>12}")
        failed = False
        for name, reqs in PREFILL_CASES:
            ok, best, flops = prefill_case(reqs, args.rounds)
            failed |= not ok
            print(f"{name:<26}{best['stock']:>10.3f}{best['hk']:>10.3f}"
                  f"{best['stock'] / best['hk']:>9.2f}x"
                  f"{flops / (best['hk'] * 1e-3) / 1e12:>12.1f}", flush=True)
            torch.cuda.empty_cache()
        return 1 if failed else 0

    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}")
    print("cache write:")
    cache_ok = cache_write_check()
    burn_in()

    arms = ("stock_layer", "stock_attn", "hk_layer", "hk_attn", "hk_kernel",
            "tri_on_hk")
    head = f"{'ctx':>6}{'batch':>6}{'split':>6}  clock "
    print("\nms per layer call; GB/s is K+V read over the attention arm's time")
    print(head + "".join(f"{a:>12}" for a in arms) +
          f"{'stockGB/s':>11}{'hkGB/s':>9}")
    failed = False
    for ctx in (int(c) for c in args.ctx.split(",")):
        for n in (int(b) for b in args.batch.split(",")):
            res = run_case(n, ctx, args.rounds, args.layers)
            if res is None:
                failed = True
                continue
            eager, graph, splits = res
            gb = n * ctx * HKV * D * 2 * 2 / 1e9
            for clock, t in (("eager", eager), ("graph", graph)):
                cells = "".join(
                    f"{t[a]:>12.4f}" if isinstance(t[a], float) else f"{'-':>12}"
                    for a in arms)
                sg = gb / (t["stock_attn"] * 1e-3)
                hg = (f"{gb / (t['hk_attn'] * 1e-3):>9.0f}"
                      if isinstance(t["hk_attn"], float) else f"{'-':>9}")
                print(f"{ctx:>6}{n:>6}{splits:>6}  {clock:<6}{cells}"
                      f"{sg:>11.0f}{hg}", flush=True)
            torch.cuda.empty_cache()
    return 1 if (failed or not cache_ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
