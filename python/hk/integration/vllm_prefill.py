"""vLLM's ROCM_ATTN prefill on hk's paged prefill, in place on ROCM_ATTN's own cache.

ROCM_ATTN -- what vLLM's ROCm platform picks first for a decoder -- runs a step through
`chunked_prefill_paged_decode`: requests with more than one query go to the Triton
prefix-prefill kernel (`prefix_prefill.context_attention_fwd`, `skip_decode=True`), and the
one-query ones to the paged decode (ll4mi). This route replaces only the first. It takes the
same arguments and reads the same cache -- K in ROCM_ATTN's x-split layout through a strided
global, V as the per-page V^T it already is (hk.ops.paged_prefill, layout="rocm") -- so the
cache write, the block size, the decode and every other backend stay vLLM's. The new tokens'
K/V are in the cache before the call (ROCM_ATTN writes it in do_kv_cache_update), so `k`/`v`
are not read.

    HK_VLLM_PREFILL=1 vllm serve <model>

A call it does not take falls back to vLLM's kernel unchanged and is counted: another dtype
or an fp8 cache, ALiBi, a sliding window, sinks, fp8 output, non-causal, a head size without a
kernel, a page that is not 16 or 32 tokens, block-table pointers, or another arch. Nothing is
read on the host, so a call is graph-safe.

Kernel against vLLM's Triton prefix-prefill on the MiniCPM5-2B shape (a 16384-token prompt in
chunks of 2048, H16/HKV2/d128, one process, caches rotated past the Infinity Cache, a W7900D):
18.2 ms against 60.5, 60.6 TF/s against 18.2. The end-to-end numbers go in docs/performance.md
once the route's gate has run.
"""

from __future__ import annotations

from typing import Optional

from . import _common, engine

PREFIXES = ("vllm",)

PATCH = engine.Patch("hk_vllm::prefill", paths=("hk", "fallback"), graph="safe")

#: Head sizes and pages the kernel is instantiated for (256: dsplit, a query row across a pair
#: of warps).
HEAD_DIMS = (64, 128, 256)
PAGES = (16, 32)

_ORIG = None
_OPS: dict = {}


def _built(make):
    """(launch, Q tile rows) from `make()`, or None where the kernel does not build spill-free
    under this compiler (the build's spill gate refuses it): the call then falls back."""
    from ..runtime.resources import ResourceError  # noqa: PLC0415
    from ..runtime.torch_ext import launcher  # noqa: PLC0415

    k, tile = make()
    try:
        return launcher(k), tile
    except ResourceError as e:
        import sys  # noqa: PLC0415

        print(f"[hk] prefill: {k.name} does not build spill-free here; its calls stay vLLM's "
              f"({str(e).splitlines()[1].strip() if len(str(e).splitlines()) > 1 else e})",
              file=sys.stderr)
        return None


def _op(head_dim: int, group: int, scale: Optional[float], page: int):
    key = (head_dim, group, scale, page)
    if key not in _OPS:
        from ..ops import paged_prefill as pp  # noqa: PLC0415

        def make():
            if head_dim > 128:
                return pp.paged_prefill_kernel(
                    f"paged_prefill_rocm_d{head_dim}_g{group}_p{page}_ds"
                    f"{'' if scale is None else '_s%08x' % (hash(scale) & 0xffffffff)}",
                    head_dim=head_dim, group=group, scale=scale, layout="rocm", page=page,
                    dsplit=True, warps=12), 96
            return pp.rocm_kernel_for(head_dim, group, scale, page), pp.Q_TILE
        _OPS[key] = _built(make)
    return _OPS[key]


def why_not(q, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
            alibi_slopes, sliding_window, skip_decode, fp8_out_scale, sinks,
            is_block_table_ptr, causal) -> str:
    """Why hk does not take this call, or "" if it does."""
    import torch  # noqa: PLC0415

    from . import switches  # noqa: PLC0415

    if switches.arch() != "gfx1100":
        return f"arch {switches.arch()}"
    if q.dtype != torch.bfloat16 or o.dtype != torch.bfloat16:
        return f"dtype {q.dtype}"
    if not str(kv_cache_dtype).startswith("auto") or k_cache.dtype != torch.bfloat16:
        return f"kv cache {kv_cache_dtype}"
    if alibi_slopes is not None or sinks is not None or fp8_out_scale is not None:
        return "alibi / sinks / fp8 output"
    if sliding_window is not None and sliding_window > 0:
        return "sliding window"
    if not causal or not skip_decode or is_block_table_ptr:
        return "non-causal / decodes included / block-table pointers"
    if k_cache.dim() != 5 or v_cache.dim() != 4 or k_cache.shape[-1] * k_cache.element_size() != 16:
        return f"cache layout {tuple(k_cache.shape)}"
    if q.dim() != 3 or q.shape[-1] not in HEAD_DIMS:
        return f"head size {q.shape[-1]}"
    if v_cache.shape[3] not in PAGES or k_cache.shape[3] != v_cache.shape[3]:
        return f"page {v_cache.shape[3]}"
    if q.shape[1] % k_cache.shape[1]:
        return "query heads not a multiple of kv heads"
    for t in (b_loc, b_start_loc, b_seq_len):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    for t in (q, o):
        if t.stride(-1) != 1 or t.stride(-2) != t.shape[-1]:
            return "q / o heads not packed"
    return ""


def context_attention_fwd(q, k, v, o, kv_cache_dtype: str, k_cache, v_cache, b_loc,
                          b_start_loc, b_seq_len, max_seq_len, max_input_len, k_scale,
                          v_scale, alibi_slopes=None, sliding_window=None, sm_scale=None,
                          skip_decode=False, fp8_out_scale=None, sinks=None,
                          is_block_table_ptr: bool = False, causal: bool = True):
    """vLLM's prefix-prefill entry point (its parameters, in its order), hk's kernel where it
    takes the call."""
    if why_not(q, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
               alibi_slopes, sliding_window, skip_decode, fp8_out_scale, sinks,
               is_block_table_ptr, causal):
        PATCH.count("fallback")
        return _ORIG(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc,
                     b_seq_len, max_seq_len, max_input_len, k_scale, v_scale,
                     alibi_slopes, sliding_window, sm_scale, skip_decode, fp8_out_scale,
                     sinks, is_block_table_ptr, causal)
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    n, h, d = q.shape
    scale = None if sm_scale is None or sm_scale == d ** -0.5 else float(sm_scale)
    got = _op(d, h // k_cache.shape[1], scale, int(v_cache.shape[3]))
    if got is None:
        PATCH.count("fallback")
        return _ORIG(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc,
                     b_seq_len, max_seq_len, max_input_len, k_scale, v_scale,
                     alibi_slopes, sliding_window, sm_scale, skip_decode, fp8_out_scale,
                     sinks, is_block_table_ptr, causal)
    op, tile = got
    op(q.reshape(n, h * d), *pp.rocm_views(k_cache, v_cache), b_loc, b_seq_len, b_start_loc,
       pp.q_tiles(int(max_input_len), q.device, tile), o.reshape(n, h * d))
    PATCH.count("hk")
    if SHADOW:
        _shadow(o, (q, k, v, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
                    max_seq_len, max_input_len, k_scale, v_scale, alibi_slopes, sliding_window,
                    sm_scale, skip_decode, fp8_out_scale, sinks, is_block_table_ptr, causal))
    return None


#: HK_VLLM_PREFILL_SHADOW=1: after every hk call, vLLM's kernel on the same arguments into a
#: scratch output, and the two compared row by row (prefill rows only) in bf16 ulps of the
#: row's largest value -- the in-situ check of a drop-in. Reads the device each call: a
#: diagnostic, never a serving mode. SHADOW_STATS holds the calls, the worst row and its call.
SHADOW = __import__("os").environ.get("HK_VLLM_PREFILL_SHADOW", "") not in ("", "0")
SHADOW_STATS = {"calls": 0, "worst_ulps": 0.0, "worst_call": -1, "over_4": 0}


def _shadow(o, args) -> None:
    import torch  # noqa: PLC0415

    q, k, v, kvd, kc, vc, bl, bs, bsl, *rest = args
    ref = torch.empty_like(o)
    _ORIG(q, k, v, ref, kvd, kc, vc, bl, bs, bsl, *rest)
    qs = bs.tolist()
    rows = [r for i in range(len(qs) - 1) if qs[i + 1] - qs[i] > 1 for r in range(qs[i], qs[i + 1])]
    _compare(o, ref, rows)


def _compare(o, ref, rows) -> None:
    import sys  # noqa: PLC0415

    import torch  # noqa: PLC0415

    st = SHADOW_STATS
    st["calls"] += 1
    if not rows:
        return
    idx = torch.tensor(rows, device=o.device)
    a, b = o[idx].float(), ref[idx].float()
    top = b.abs().amax(dim=-1).clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(top)) - 7)        # a bf16 ulp at the row's max
    err = ((a - b).abs().amax(dim=-1) / ulp).amax().item()
    if err > st["worst_ulps"]:
        st["worst_ulps"], st["worst_call"] = err, st["calls"]
    if err > 4:
        st["over_4"] += 1
    if st["calls"] % 500 == 0 or err > 4:
        print(f"[hk] prefill shadow: {st}", file=sys.stderr, flush=True)


_GUARDED = PATCH.guard(context_attention_fwd)


# ---- TRITON_ATTN: one unified_attention launch takes the whole batch -------------------------

#: vLLM's unified_attention, while patched.
_ORIG_U = None

#: (head_dim, group, scale) -> (launch, Q tile rows) for layout="flash" with decodes on.
_FLASH: dict = {}


def _flash_op(head_dim: int, group: int, scale: Optional[float]):
    key = (head_dim, group, scale)
    if key not in _FLASH:
        from ..ops import paged_prefill as pp  # noqa: PLC0415

        def make():
            ds = head_dim > 128
            warps = 16 if ds else 12
            tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
            return pp.paged_prefill_kernel(
                f"paged_prefill_flash_d{head_dim}_g{group}{'_ds' if ds else ''}{tag}",
                head_dim=head_dim, group=group, scale=scale, layout="flash", dsplit=ds,
                warps=warps, decodes=True), 16 * (warps // 2 if ds else warps)
        _FLASH[key] = _built(make)
    return _FLASH[key]


#: unified_attention's arguments that must be off for hk to take a call: name -> the test.
_U_OFF = {
    "alibi_slopes": lambda v: v is None, "sinks": lambda v: v is None,
    "output_scale": lambda v: v is None, "qq_bias": lambda v: v is None,
    "mm_prefix_range": lambda v: v is None, "rswa_prefix_lens": lambda v: v is None,
    "softcap": lambda v: not v, "q_descale": lambda v: v is None,
    "window_size": lambda v: v is None or tuple(v) == (-1, -1),
    "chunk_lookback": lambda v: v is None or v < 0,
    "kv_quant_mode": lambda v: v is None or getattr(v, "name", str(v)) in ("NONE", "0"),
    "k_scale_cache": lambda v: v is None, "v_scale_cache": lambda v: v is None,
}


def _flash_kv(k, v):
    """The (blocks, 2, block, kv_heads * head_dim) tensor TRITON_ATTN's key and value caches
    are the halves of, or None if they are not that (another layout, padded strides)."""
    import torch  # noqa: PLC0415

    if k.dim() != 4 or v.shape != k.shape or k.stride() != v.stride():
        return None
    nb, bs, hkv, d = k.shape
    blk = bs * hkv * d
    if (k.stride() != (2 * blk, hkv * d, d, 1)
            or v.data_ptr() - k.data_ptr() != blk * k.element_size()):
        return None
    return torch.as_strided(k, (nb, 2, bs, hkv * d), (2 * blk, blk, hkv * d, 1))


def u_why_not(a: dict) -> str:
    """Why hk does not take this unified_attention call (`a`: its arguments by name), or ""."""
    import torch  # noqa: PLC0415

    from . import switches  # noqa: PLC0415

    if switches.arch() != "gfx1100":
        return f"arch {switches.arch()}"
    if engine.capturing():
        return "being captured (a mixed batch's grid follows its max query length)"
    if int(a.get("max_seqlen_q", 1)) <= 1:
        return "decode only"
    if not a.get("causal", True):
        return "non-causal"
    for name, ok in _U_OFF.items():
        if name in a and not ok(a[name]):
            return f"{name} set"
    q, k, v, out = a["q"], a["k"], a["v"], a["out"]
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return f"dtype {q.dtype} / cache {k.dtype}"
    if q.dim() != 3 or q.shape[-1] not in (64, 128, 256):
        return f"head size {q.shape[-1]}"
    if k.dim() != 4 or k.shape[1] != 16:
        return f"cache {tuple(k.shape)} (pages of 16)"
    if q.shape[1] % k.shape[2]:
        return "query heads not a multiple of kv heads"
    for t in (a["block_table"], a["seqused_k"], a["cu_seqlens_q"]):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    for t in (q, out):
        if t.stride(-1) != 1 or t.stride(-2) != t.shape[-1]:
            return "q / out heads not packed"
    if _flash_kv(k, v) is None:
        return "key/value caches are not the halves of one (blocks, 2, ...) tensor"
    return ""


def unified_attention(*args, **kwargs):
    """vLLM's unified_attention (its signature, set on apply), hk's flash-layout prefill --
    the whole batch, decodes included -- where it takes a call with prefill in it."""
    a = kwargs if not args else _U_SIG.bind(*args, **kwargs).arguments
    if u_why_not(a):
        PATCH.count("fallback")
        return _ORIG_U(*args, **kwargs)
    q, out = a["q"], a["out"]
    n, h, d = q.shape
    kv = _flash_kv(a["k"], a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    got = _flash_op(d, h // a["k"].shape[2], scale)
    if got is None:
        PATCH.count("fallback")
        return _ORIG_U(*args, **kwargs)
    op, tile = got
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    op(q.reshape(n, h * d), kv, kv, a["block_table"], a["seqused_k"], a["cu_seqlens_q"],
       pp.q_tiles(int(a["max_seqlen_q"]), q.device, tile), out.reshape(n, h * d))
    PATCH.count("hk")
    if SHADOW:
        import torch  # noqa: PLC0415

        ref = torch.empty_like(out)
        b = dict(a)
        b["out"] = ref
        _ORIG_U(**b)
        _compare(out, ref, list(range(n)))
    return None


_U_SIG = None
_GUARDED_U = PATCH.guard(unified_attention)


def apply() -> str:
    """Route ROCM_ATTN's prefill. Returns what happened, in one line."""
    global _ORIG
    if not _common.importable("vllm"):
        return "vllm is not importable; prefill not patched"
    if _ORIG is not None:
        return "prefill: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "prefill: not patched -- not a ROCm platform"
    from vllm.v1.attention.ops import prefix_prefill  # noqa: PLC0415

    _ORIG = prefix_prefill.context_attention_fwd
    names = _common.rebind(PREFIXES, _ORIG, _GUARDED)
    unames = _apply_unified()
    return (f"prefill -> hk where hk.ops.paged_prefill takes ROCM_ATTN's prefill "
            f"({len(names)} bindings rebound) and TRITON_ATTN's batches with prefill in them "
            f"({len(unames)} rebound)")


def _apply_unified() -> list:
    global _ORIG_U, _U_SIG
    import inspect  # noqa: PLC0415

    from vllm.v1.attention.ops import triton_unified_attention as tua  # noqa: PLC0415

    _ORIG_U = tua.unified_attention
    _U_SIG = inspect.signature(_ORIG_U)
    # vLLM probes signatures (vllm_backend checks keywords against them): the stand-in shows
    # the engine's own.
    unified_attention.__signature__ = _U_SIG
    return _common.rebind(PREFIXES, _ORIG_U, _GUARDED_U)


def revert() -> str:
    global _ORIG
    if _ORIG is None:
        return "prefill: not patched"
    global _ORIG_U
    names = _common.rebind(PREFIXES, _GUARDED, _ORIG)
    if _ORIG_U is not None:
        names += _common.rebind(PREFIXES, _GUARDED_U, _ORIG_U)
        _ORIG_U = None
    _ORIG = None
    return f"prefill -> vllm ({len(names)} rebound)"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary()
