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
kernel, a page other than 16 or 32 tokens, block-table pointers, or another arch. (On
TRITON_ATTN, below, a page may be any multiple of 16: hybrid models' pages, 544 tokens for
MiniCPM-V-4.7's, 784 for Qwen3.8's.) Nothing is
read on the host, so a call is graph-safe.
"""

from __future__ import annotations

from typing import Optional

from . import _common, engine

PREFIXES = ("vllm",)

PATCH = engine.Patch("hk_vllm::prefill", paths=("hk", "fallback"), graph="safe")

#: Head sizes the kernel is instantiated for (256: dsplit, a query row across a pair of warps).
HEAD_DIMS = (64, 128, 256)


def page_why_not(page: int) -> str:
    """Why hk does not take a cache of `page`-token pages, or "": any multiple of 16 tokens (a
    32-key block lies in one page, spans whole ones, or is two 16-key halves of two pages)."""
    if page > 0 and page % 16 == 0:
        return ""
    return f"page {page}: hk takes pages of a multiple of 16 tokens"

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
    if k_cache.shape[3] != v_cache.shape[3]:
        return f"K page {k_cache.shape[3]} and V page {v_cache.shape[3]}"
    if int(v_cache.shape[3]) not in (16, 32):
        # larger pages are hybrid models', whose rows past a sequence's end hold another
        # layer's bytes; layout="rocm" stages V^T pages unmasked (TRITON_ATTN's route masks)
        return (f"page {v_cache.shape[3]}: on ROCM_ATTN's cache hk takes pages of 16 or 32 "
                f"(it stages V^T pages as they are, past a sequence's end too)")
    if q.shape[1] % k_cache.shape[1]:
        return "query heads not a multiple of kv heads"
    for t in (b_loc, b_start_loc, b_seq_len):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    for t in (q, o):
        if t.stride(-1) != 1 or t.stride(-2) != t.shape[-1]:
            return "q / o heads not packed"
    return ""


#: context_attention_fwd's parameters the stand-in reads (the rest of vLLM's must be at their
#: defaults, or the call falls back: _p_unknown).
_P_READ = frozenset((
    "q", "k", "v", "o", "kv_cache_dtype", "k_cache", "v_cache", "b_loc", "b_start_loc",
    "b_seq_len", "max_seq_len", "max_input_len", "k_scale", "v_scale", "alibi_slopes",
    "sliding_window", "sm_scale", "skip_decode", "fp8_out_scale", "sinks", "is_block_table_ptr",
    "causal"))

#: of those, the ones vLLM's context_attention_fwd must still have (the contract's check)
_P_NEED = frozenset((
    "q", "k", "v", "o", "kv_cache_dtype", "k_cache", "v_cache", "b_loc", "b_start_loc",
    "b_seq_len", "max_seq_len", "max_input_len", "k_scale", "v_scale", "sm_scale",
    "skip_decode"))

#: vLLM's context_attention_fwd's signature, set on apply (the stand-in shows it as its own).
_P_SIG = None


def _p_unknown(a: dict) -> str:
    """A parameter of this vLLM's context_attention_fwd that hk neither reads nor tests and that
    is not at its default, as a refusal reason, or "". interleaved_v_pack_factor (> 0: ROCM_ATTN's
    V cache interleaved for its fused writer, not the per-page V^T layout="rocm" reads) is one."""
    for name, val in a.items():
        if name in _P_READ:
            continue
        p = _P_SIG.parameters.get(name) if _P_SIG is not None else None
        default = p.default if p is not None else None
        try:
            if val is default or (not hasattr(val, "shape") and val == default):
                continue
        except Exception:  # noqa: BLE001 -- an uncomparable value is a set one
            pass
        return f"{name}={val!r} (a parameter hk does not take)"
    return ""


def context_attention_fwd(*args, **kwargs):
    """vLLM's prefix-prefill entry point (its signature, set on apply: newer vLLMs add
    parameters), hk's kernel where it takes the call."""
    a = dict(kwargs) if not args else _P_SIG.bind(*args, **kwargs).arguments
    g = {n: (a[n] if n in a else (_P_SIG.parameters[n].default if _P_SIG is not None and
                                  n in _P_SIG.parameters else d))
         for n, d in (("alibi_slopes", None), ("sliding_window", None), ("sm_scale", None),
                      ("skip_decode", False), ("fp8_out_scale", None), ("sinks", None),
                      ("is_block_table_ptr", False), ("causal", True))}
    q, o, k_cache, v_cache = a["q"], a["o"], a["k_cache"], a["v_cache"]
    unknown = _p_unknown(a)
    if unknown:
        _P_REFUSED[unknown] = _P_REFUSED.get(unknown, 0) + 1
    if unknown or why_not(q, o, a["kv_cache_dtype"], k_cache, v_cache, a["b_loc"],
                          a["b_start_loc"], a["b_seq_len"], g["alibi_slopes"],
                          g["sliding_window"], g["skip_decode"], g["fp8_out_scale"], g["sinks"],
                          g["is_block_table_ptr"], g["causal"]):
        PATCH.count("fallback")
        return _ORIG(*args, **kwargs)
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    n, h, d = q.shape
    sm_scale = g["sm_scale"]
    scale = None if sm_scale is None or sm_scale == d ** -0.5 else float(sm_scale)
    got = _op(d, h // k_cache.shape[1], scale, int(v_cache.shape[3]))
    if got is None:
        PATCH.count("fallback")
        return _ORIG(*args, **kwargs)
    op, tile = got
    op(q.reshape(n, h * d), *pp.rocm_views(k_cache, v_cache), a["b_loc"], a["b_seq_len"],
       a["b_start_loc"], pp.q_tiles(int(a["max_input_len"]), q.device, tile), o.reshape(n, h * d))
    PATCH.count("hk")
    if SHADOW:
        _shadow(o, args, kwargs, a)
    return None


#: refusals by _p_unknown, by reason (the per-call refusal of a parameter hk does not take)
_P_REFUSED: dict = {}


#: HK_VLLM_PREFILL_SHADOW=1: after every hk call, vLLM's kernel on the same arguments into a
#: scratch output, and the two compared row by row (prefill rows only) in bf16 ulps of the
#: row's largest value -- the in-situ check of a drop-in. Reads the device each call: a
#: diagnostic, never a serving mode. SHADOW_STATS holds the calls, the worst row and its call.
SHADOW = __import__("os").environ.get("HK_VLLM_PREFILL_SHADOW", "") not in ("", "0")
SHADOW_STATS = {"calls": 0, "worst_ulps": 0.0, "worst_call": -1, "over_4": 0}


def _shadow(o, args, kwargs, a) -> None:
    import torch  # noqa: PLC0415

    ref = torch.empty_like(o)
    b = dict(a)
    b["o"] = ref
    _ORIG(**b)
    qs = a["b_start_loc"].tolist()
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

#: (head_dim, group, scale, page) -> (launch, Q tile rows) for layout="flash" with decodes on.
_FLASH: dict = {}


def _ptag(page: int) -> str:
    return "" if page == 16 else f"_p{page}"


def _otag(order: str) -> str:
    return "" if order == "planes" else f"_{order}"


def _flash_op(head_dim: int, group: int, scale: Optional[float], page: int = 16,
              order: str = "planes"):
    key = (head_dim, group, scale, page, order)
    if key not in _FLASH:
        from ..ops import paged_prefill as pp  # noqa: PLC0415

        def make():
            ds = head_dim > 128
            warps = 16 if ds else 12
            tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
            return pp.paged_prefill_kernel(
                f"paged_prefill_flash_d{head_dim}_g{group}{'_ds' if ds else ''}{_ptag(page)}"
                f"{_otag(order)}{tag}", head_dim=head_dim, group=group, scale=scale,
                layout="flash", dsplit=ds, warps=warps, decodes=True, page=page,
                kv_order=order), 16 * (warps // 2 if ds else warps)
        _FLASH[key] = _built(make)
    return _FLASH[key]


#: V^T pre-pack for TRITON_ATTN (paged_prefill vt_pages): the batch's V pages transposed once a
#: call into a scratch, instead of once per (query head, tile) workgroup inside the kernel. Taken
#: when the scratch -- requests x the longest request's pages -- stays under VT_CAP_BYTES;
#: HK_VLLM_PREFILL_VT=0 turns it off.
VT = __import__("os").environ.get("HK_VLLM_PREFILL_VT", "1") not in ("0", "off", "no", "false")
VT_CAP_BYTES = 256 << 20
_VT_OPS: dict = {}
_VT_SCRATCH: dict = {}


def _vt_ops(head_dim: int, group: int, scale: Optional[float], page: int = 16,
            order: str = "planes"):
    """(pack launch, vt_pages launch, Q tile rows), or None (not built here, or head_dim 256)."""
    key = (head_dim, group, scale, page, order)
    if key not in _VT_OPS:
        _VT_OPS[key] = None
        if head_dim <= 128:
            from ..ops import paged_prefill as pp  # noqa: PLC0415

            tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
            main = _built(lambda: (pp.paged_prefill_kernel(
                f"paged_prefill_flashvt_d{head_dim}_g{group}{_ptag(page)}{_otag(order)}{tag}",
                head_dim=head_dim, group=group, scale=scale, layout="flash", warps=12,
                decodes=True, vt_pages=True, page=page, kv_order=order), 192))
            pack = _built(lambda: (pp.vt_pack_kernel(
                f"paged_vt_pack_d{head_dim}{_ptag(page)}{_otag(order)}", head_dim=head_dim,
                page=page, kv_order=order), 0))
            if main is not None and pack is not None:
                _VT_OPS[key] = (pack[0], main[0], main[1])
    return _VT_OPS[key]


def _vt_scratch(device, slots: int, hkv: int, d: int, page: int = 16):
    import torch  # noqa: PLC0415

    key = (str(device), hkv, d, page)
    got = _VT_SCRATCH.get(key)
    if got is None or got.shape[0] < slots:
        got = _VT_SCRATCH[key] = torch.empty(1 << (slots - 1).bit_length(), hkv, d, page,
                                             device=device, dtype=torch.bfloat16)
    return got


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
    "rswa_window": lambda v: v is None,
}

#: unified_attention's arguments hk reads, or that change nothing it computes: its split-softmax
#: scratch for the 3-D decode kernel, Intel's tensor-descriptor switch, and two modifiers of
#: features _U_OFF already requires off (ALiBi's sqrt, a multimodal prefix's window clamp).
#: Any other argument -- one a later vLLM adds -- must be at its default, or the call falls back.
_U_READ = frozenset((
    "q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k", "max_seqlen_k",
    "softmax_scale", "causal", "block_table", "k_descale", "v_descale", "seq_threshold_3D",
    "num_par_softmax_segments", "softmax_segm_output", "softmax_segm_max",
    "softmax_segm_expsum", "use_td", "use_alibi_sqrt", "mm_prefix_clamp_sliding_window"))


def _unknown_set(a: dict) -> str:
    """The first argument hk neither reads nor tests that is not at its default, or ""."""
    for name, val in a.items():
        if name in _U_READ or name in _U_OFF:
            continue
        p = _U_SIG.parameters.get(name) if _U_SIG is not None else None
        default = p.default if p is not None else None
        if val is default:
            continue
        try:
            if not hasattr(val, "shape") and val == default:
                continue
        except Exception:  # noqa: BLE001 -- an uncomparable value is a set one
            pass
        return name
    return ""


def _flash_kv(k, v):
    """(kv, kv_order): the 4-D tensor TRITON_ATTN's key and value caches -- (blocks, page,
    kv_heads, head_dim) views -- are both views of, as paged_prefill's layout="flash" takes it,
    or None if they are not one of its orders (another layout, padded strides):

    * "planes": (blocks, 2, page, kv_heads * head_dim), K then V a block (vLLM until its
      per-layer layouts);
    * "tokens": a head's K and V side by side, token-major, (blocks, 1, page, kv_heads * 2 *
      head_dim) (vLLM's (B, H, N, 2 * head_size) cache laid out [B, N, H, C]);
    * "heads": the same head-major, (blocks, kv_heads, page, 2 * head_dim) ([B, H, N, C]).
    """
    import torch  # noqa: PLC0415

    if k.dim() != 4 or v.shape != k.shape or k.stride() != v.stride() or k.stride(-1) != 1:
        return None
    nb, bs, hkv, d = k.shape
    blk = bs * hkv * d
    off = (v.data_ptr() - k.data_ptr()) // k.element_size()
    st = k.stride()
    if st == (2 * blk, hkv * d, d, 1) and off == blk:
        return torch.as_strided(k, (nb, 2, bs, hkv * d), (2 * blk, blk, hkv * d, 1)), "planes"
    if off != d:
        return None
    if st[0] == 2 * blk and st[1] == hkv * 2 * d and (st[2] == 2 * d or hkv == 1):
        return (torch.as_strided(k, (nb, 1, bs, hkv * 2 * d), (2 * blk, 2 * blk, hkv * 2 * d, 1)),
                "tokens")
    if st == (2 * blk, 2 * d, bs * 2 * d, 1):
        return (torch.as_strided(k, (nb, hkv, bs, 2 * d), (2 * blk, bs * 2 * d, 2 * d, 1)),
                "heads")
    return None


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
    unknown = _unknown_set(a)
    if unknown:
        return f"{unknown} set (an argument hk does not know)"
    q, k, v, out = a["q"], a["k"], a["v"], a["out"]
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return f"dtype {q.dtype} / cache {k.dtype}"
    if q.dim() != 3 or q.shape[-1] not in (64, 128, 256):
        return f"head size {q.shape[-1]}"
    if k.dim() != 4:
        return f"cache {tuple(k.shape)}"
    why = page_why_not(int(k.shape[1]))
    if why:
        return why
    if q.shape[1] % k.shape[2]:
        return "query heads not a multiple of kv heads"
    for t in (a["block_table"], a["seqused_k"], a["cu_seqlens_q"]):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    for t in (q, out):
        if t.stride(-1) != 1 or t.stride(-2) != t.shape[-1]:
            return "q / out heads not packed"
    if not out.is_contiguous():
        return "output not contiguous"
    if _flash_kv(k, v) is None:
        return (f"key/value caches {tuple(k.shape)} strides {tuple(k.stride())} (V at +"
                f"{(v.data_ptr() - k.data_ptr()) // k.element_size()}): not a layout hk reads")
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
    kv, order = _flash_kv(a["k"], a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    page = int(a["k"].shape[1])
    got = _flash_op(d, h // a["k"].shape[2], scale, page, order)
    if got is None:
        PATCH.count("fallback")
        return _ORIG_U(*args, **kwargs)
    op, tile = got
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    v_src = kv
    if VT and d <= 128:
        hkv = a["k"].shape[2]
        max_k = int(a.get("max_seqlen_k") or 0)
        slots = len(a["seqused_k"]) * -(-max_k // page)
        vt_ops = (_vt_ops(d, h // hkv, scale, page, order)
                  if 0 < slots * hkv * d * page * 2 <= VT_CAP_BYTES else None)
        if vt_ops is not None:
            pack, op, tile = vt_ops
            v_src = _vt_scratch(q.device, slots, hkv, d, page)
            pack(kv, a["block_table"], a["seqused_k"], pp.vt_pack_span(max_k, q.device, page),
                 v_src)
    args = [q.reshape(n, h * d), kv, v_src, a["block_table"], a["seqused_k"], a["cu_seqlens_q"],
            pp.q_tiles(int(a["max_seqlen_q"]), q.device, tile), out.reshape(n, h * d)]
    if d <= 128:
        # the GQA-packed decodes' views: q and the output as (1, tokens, heads, head_dim)
        args += [q.unsqueeze(0), out.unsqueeze(0)]
    op(*args)
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

    global _P_SIG
    import inspect  # noqa: PLC0415

    _ORIG = prefix_prefill.context_attention_fwd
    _P_SIG = inspect.signature(_ORIG)
    # vLLM's callers pass keywords, and probes may read the signature: the stand-in shows vLLM's
    context_attention_fwd.__signature__ = _P_SIG
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
