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
  hk_layer      HKAttentionImpl.forward (the cache write is inside it)
  hk_attn       HKAttentionImpl._decode_all (no cache write)
  hk_kernel     the fused decode launch alone, no wrapper around it

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
        --ctx 1024,4096,16384 --batch 1,4,8,16,32
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

def stock_arms(n, ctx, key, value, q, k_new, v_new, gen):
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

    cu = torch.arange(n + 1, device="cuda", dtype=I32)
    seq_lens = torch.full((n,), ctx, device="cuda", dtype=I32)
    segm_o = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, D, device="cuda",
                         dtype=F32)
    segm_m = torch.empty(SEQ_THRESHOLD_3D, HQ, SEGMENTS, device="cuda",
                         dtype=F32)
    segm_l = torch.empty_like(segm_m)
    out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)

    def attn():
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

    def layer():
        kc, vc = kv.unbind(1)
        triton_reshape_and_cache_flash(k_new, v_new, kc, vc, new_slots,
                                       "auto", scale1, scale1)
        attn()

    return {"stock_layer": layer, "stock_attn": attn}, out


def hk_arms(n, ctx, key, value, q, k_new, v_new, gen):
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
    md = types.SimpleNamespace(
        num_actual_tokens=n, slot_mapping=slots[:, -1].contiguous(),
        max_query_len=1, max_seq_len=ctx, block_table=table,
        seq_lens=torch.full((n,), ctx, device="cuda", dtype=I32),
        query_start_loc=torch.arange(n + 1, device="cuda", dtype=I32))
    # (tokens, heads, head_dim): the view vLLM hands an impl's forward.
    out = torch.empty(n, HQ, D, device="cuda", dtype=BF16)

    def layer():
        impl.forward(None, q, k_new, v_new, kv, md, out)

    def attn():
        impl._decode_all(q, out, kv[0], kv[1], md, n)

    # The launches as the backend plans them, nothing around them.
    splits = impl._plan(n)
    scratch = vllm_backend._scratch(q.device, HKV, D, impl._rows, G)
    q4, o4 = q.view(n, HKV, G, D), out.view(n, HKV, G, D)

    def kernel():
        vllm_backend.decode(q4, kv[0], kv[1], table, md.seq_lens, o4,
                            scratch, splits)

    arms = {"hk_layer": layer, "hk_attn": attn, "hk_kernel": kernel}
    return arms, out, {}, splits


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


def run_case(n, ctx, rounds):
    gen = torch.Generator(device="cuda").manual_seed(n * 100003 + ctx)
    key = torch.randn(n, ctx, HKV, D, device="cuda", dtype=BF16, generator=gen)
    value = torch.randn_like(key)
    q = torch.randn(n, HQ, D, device="cuda", dtype=BF16, generator=gen)
    # The step's own token is the last position, already in the cache: the
    # layer arms rewrite it with the same values, so every call is the same.
    k_new = key[:, -1].contiguous()
    v_new = value[:, -1].contiguous()

    stock, s_out = stock_arms(n, ctx, key, value, q, k_new, v_new, gen)
    hk, h_out, uncapturable, splits = hk_arms(n, ctx, key, value, q, k_new,
                                              v_new, gen)
    del key, value

    stock["stock_attn"]()
    hk["hk_attn"]()
    torch.cuda.synchronize()
    if not check(f"n{n} c{ctx}", h_out, s_out):
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
    return eager, gbest, splits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", default="1024,4096,16384")
    ap.add_argument("--batch", default="1,4,8,16,32")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}")
    print("cache write:")
    cache_ok = cache_write_check()
    burn_in()

    arms = ("stock_layer", "stock_attn", "hk_layer", "hk_attn", "hk_kernel")
    head = f"{'ctx':>6}{'batch':>6}{'split':>6}  clock "
    print("\nms per layer call; GB/s is K+V read over the attention arm's time")
    print(head + "".join(f"{a:>12}" for a in arms) +
          f"{'stockGB/s':>11}{'hkGB/s':>9}")
    failed = False
    for ctx in (int(c) for c in args.ctx.split(",")):
        for n in (int(b) for b in args.batch.split(",")):
            res = run_case(n, ctx, args.rounds)
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
