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

    vllm serve <model>                     # on by default on gfx1100
    HK_VLLM_PREFILL=0 vllm serve <model>   # off: vLLM's prefill kernels

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

PATCH = engine.Patch("hk_vllm::prefill", paths=("hk", "fallback", "decode", "declined"),
                     graph="safe")

#: The GPUs the kernels are built and tested for: RDNA3 and RDNA4 (wave32, 64 KiB of LDS a
#: workgroup; the WMMA operand layouts are HipKittens' per arch, behind the same DSL ops).
ARCHES = ("gfx1100", "gfx1201")

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


def _built(make, quiet: bool = False):
    """(launch, Q tile rows) from `make()`, or None where the kernel does not build spill-free
    under this compiler (the build's spill gate refuses it): the call then falls back."""
    from ..runtime.resources import ResourceError  # noqa: PLC0415
    from ..runtime.torch_ext import launcher  # noqa: PLC0415

    k, tile = make()
    try:
        return launcher(k), tile
    except ResourceError as e:
        import sys  # noqa: PLC0415

        if quiet:
            return None
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

    if switches.arch() not in ARCHES:
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

    if switches.arch() not in ARCHES:
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


#: The default-on rule's per-shape clause on TRITON_ATTN: hk takes a batch with prefill in it
#: only where its kernel measured faster than vLLM's (tools/hk-bench/attention_bench.py, one
#: process, cold KV, against the faster of TRITON_ATTN's and ROCM_ATTN's kernels, MiniCPM5-2B,
#: Qwen3.8-27B and MiniCPM-V-4.7 at TP2). Two kinds of batch it is not faster on:
#: - decode rows beside a small prefill: hk's kernel walks each decode's whole context as one
#:   16-row fragment in one workgroup, where vLLM's splits it;
#: - a short chunk: too few query tiles (query rows x kv heads) to fill the card, as there is
#:   no KV split.
#: The route reads nothing on the host, so it judges from the sizes it is given: the requests,
#: the longest query, the longest context and the kv heads. It bounds the decode work from
#: above, (requests - 1) x the longest context, and the prefill's from below, the longest
#: query's square over 2, and declines a batch where the first is more than MIX_RATIO times
#: the second; it declines a short chunk -- its longest query times its kv heads under
#: MIN_ROWS -- over a context longer than SHORT_CTX tokens; and any chunk whose context is more
#: than CTX_PER_Q times its length. Measured per arch and
#: head size class (up to 128; 256), set with margin: where unsure it declines. It is checked
#: first, on sizes alone, so a declined batch costs nothing more than vLLM's own call: it goes
#: to vLLM's kernel unchanged and is counted as "declined".
#: HK_VLLM_PREFILL_GUARD=0 turns the guard off (to measure it).
GUARD = __import__("os").environ.get("HK_VLLM_PREFILL_GUARD", "1") != "0"
#: (arch, head size class) -> (MIX_RATIO, MIN_ROWS, SHORT_CTX, CTX_PER_Q)
GUARDS = {
    # gfx1100 (W7900D). Mixed: wins at a ratio <= 0.0039, losses from 0.0078. Up to 128 (2 kv
    # heads): a short prompt wins (1.33x at 16-256 tokens; 512:64 too), a chunk under 512 rows
    # over a context of 1k+ loses (1024:64 0.85x, 2048:128 0.91x), 4096:256 (512 rows) wins.
    # 256: no chunk measured loses (32 tokens x 1 kv head over 32k: a win).
    ("gfx1100", 128): (0.003, 512, 600, 64),
    ("gfx1100", 256): (0.003, 16, 0, 1024),
    # gfx1201 (R9700). Mixed: as gfx1100. Up to 128: under 1024 rows only thin wins (1.03-1.14x)
    # or losses (32-token chunks over 2k+: 0.77x); 0:512 1.09x. 256: a short prompt wins
    # (1.7x), a chunk under 512 rows over 1k+ loses or ties (1024:64 0.95-1.00x, 1024:128 0.91x).
    ("gfx1201", 128): (0.003, 1024, 0, 128),
    ("gfx1201", 256): (0.003, 512, 600, 128),
}
_DECLINED_SAID: list = []


def guard_why_not(reqs: int, max_q: int, max_k: int, head_dim: int, kv_heads: int,
                  arch: str) -> str:
    """Why the guard declines a TRITON_ATTN batch with prefill in it, from its host-known sizes
    (`reqs` requests, longest query `max_q`, longest context `max_k`, `kv_heads`), or ""."""
    if not GUARD:
        return ""
    key = (arch, 256 if head_dim > 128 else 128)
    g = GUARDS.get(key)
    if g is None:
        return f"no measured guard for {key}"
    mix, min_rows, short_ctx, ctx_per_q = g
    if max_k <= 0:
        return "no context length given"
    if reqs > 1 and (reqs - 1) * max_k > mix * max_q * max_q / 2:
        return (f"up to {reqs - 1} decodes over up to {max_k} tokens beside a {max_q}-token "
                f"prefill (vLLM's kernel splits a decode's context, hk's does not)")
    if max_q * kv_heads < min_rows and max_k > short_ctx:
        return f"a {max_q}-token chunk over a {max_k}-token context, {kv_heads} kv head(s)"
    if max_k > ctx_per_q * max_q:
        return f"a {max_q}-token chunk over a {max_k}-token context"
    return ""


def _guard(a: dict) -> str:
    """guard_why_not for a unified_attention call (`a`: its arguments by name), read before any
    other check: sizes only. "" for a call whose shapes it does not know (the checks after it
    refuse those)."""
    from . import switches  # noqa: PLC0415

    q, k, used = a.get("q"), a.get("k"), a.get("seqused_k")
    arch = switches.arch()
    if (not GUARD or arch not in ARCHES or getattr(q, "dim", None) is None or q.dim() != 3
            or getattr(k, "dim", None) is None or k.dim() != 4 or used is None):
        return ""
    return guard_why_not(int(used.shape[0]), int(a.get("max_seqlen_q", 0)),
                         int(a.get("max_seqlen_k") or 0), int(q.shape[-1]), int(k.shape[2]), arch)


def _note_declined(why: str) -> None:
    """The first declined batch of this process says so, once, with where the rest are counted."""
    if not _DECLINED_SAID:
        _DECLINED_SAID.append(True)
        import sys  # noqa: PLC0415

        print(f"[hk] prefill: a batch went to vLLM's kernel, where hk's is not faster ({why}); "
              f"such batches are counted as 'declined' in {PATCH.name} "
              f"(python3 -m hk.integration --status; HK_VLLM_PREFILL_GUARD=0 turns this off)",
              file=sys.stderr, flush=True)


#: A call with only decodes in it (one query a request) goes to hk's split decode
#: (hk.ops.paged_decode) on the same cache, where `decode_takes` says it does. HK_VLLM_DECODE:
#: unset, head_dim 256 only -- there hk beats every stock kernel that can read those pages;
#: "1", head_dim 128 too (opt-in: at one request and 8k tokens ROCM_ATTN's paged decode is as
#: fast); "0", none.
def decode_switches(env: str):
    """(decode on at all, head_dim 128 too) for an HK_VLLM_DECODE value."""
    v = (env or "").strip().lower()
    return v not in ("0", "off", "no", "false"), v in ("1", "on", "yes", "true")


DECODE, DECODE_D128 = decode_switches(__import__("os").environ.get("HK_VLLM_DECODE", ""))

#: Decode batches hk takes: at head_dim 128, up to this many (request, kv head) pairs -- past
#: it vLLM's Triton kernel has the machine full and the two read the cache at the same rate
#: (tools/hk-bench/attention_bench.py: MiniCPM5-2B, B8 at 8k-32k: 0.98-1.0x); at 256 every
#: batch (Qwen3.8-27B: 1.06-2.2x at B1-B16).
DECODE_MAX_PAIRS_D128 = 8
#: The GPUs the decode is measured and tested on.
DECODE_ARCHES = ("gfx1100",)
_DEC: dict = {}
_DEC_SCRATCH: dict = {}


def decode_takes(reqs: int, kv_heads: int, head_dim: int) -> bool:
    """Whether hk's split decode takes a decode batch of `reqs` requests: every batch at head_dim
    256; at 128 (with HK_VLLM_DECODE=1) up to DECODE_MAX_PAIRS_D128 (request, kv head) pairs."""
    if head_dim > 128:
        return True
    return DECODE_D128 and reqs * kv_heads <= DECODE_MAX_PAIRS_D128


def d_why_not(a: dict) -> str:
    """Why hk's decode does not take this decode-only unified_attention call, or ""."""
    import torch  # noqa: PLC0415

    from . import switches  # noqa: PLC0415

    if switches.arch() not in DECODE_ARCHES:
        return f"arch {switches.arch()}"
    if int(a.get("max_seqlen_q", 1)) != 1:
        return "not decode-only"
    for name, ok in _U_OFF.items():
        if name in a and not ok(a[name]):
            return f"{name} set"
    unknown = _unknown_set(a)
    if unknown:
        return f"{unknown} set (an argument hk does not know)"
    q, k, out = a["q"], a["k"], a["out"]
    if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return f"dtype {q.dtype} / cache {k.dtype}"
    if q.dim() != 3 or q.shape[-1] not in (128, 256):
        return f"head size {q.shape[-1]}"
    if k.dim() != 4:
        return f"cache {tuple(k.shape)}"
    why = page_why_not(int(k.shape[1]))
    if why:
        return why
    if q.shape[1] % k.shape[2] or q.shape[1] // k.shape[2] > 16:
        return "query heads not a multiple of kv heads, or more than 16 a kv head"
    for t in (a["block_table"], a["seqused_k"], a["cu_seqlens_q"]):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    if q.stride(-1) != 1 or q.stride(-2) != q.shape[-1] or not out.is_contiguous():
        return "q heads not packed / output not contiguous"
    if not decode_takes(len(a["seqused_k"]), int(k.shape[2]), int(q.shape[-1])):
        return ("head_dim 128 decode (HK_VLLM_DECODE=1 takes it)" if not DECODE_D128 else
                "decode batch past hk's (vLLM's kernel reads as fast)")
    if _flash_kv(k, a["v"]) is None:
        return "key/value caches: not a layout hk reads"
    return ""


def _decode_ops(head_dim: int, group: int, page: int, order: str, splits: int,
                scale: Optional[float], warps: int = 4, v_both: bool = False):
    key = (head_dim, group, page, order, splits, scale, warps, v_both)
    if key not in _DEC:
        from ..ops import paged_decode as pd  # noqa: PLC0415

        walk = _built(lambda: (pd.walk_for(head_dim, group, page, order, splits, scale,
                                           warps=warps, v_both=v_both), 0), quiet=v_both)
        if walk is None and v_both:
            # both V bands in flight spills under some compilers (clang 22): the one-band walk
            walk = _built(lambda: (pd.walk_for(head_dim, group, page, order, splits, scale,
                                               warps=warps, v_both=False), 0))
        merge = (_built(lambda: (pd.merge_for(head_dim, group, splits), 0)) if splits > 1
                 else (None, 0))
        _DEC[key] = None if walk is None or merge is None else (walk[0], merge[0])
    return _DEC[key]


def _decode_scratch(device, rows: int, hkv: int, group: int, d: int):
    """The split partials, kept across calls (a captured graph holds them); grown outside a
    capture only -- None if a capture needs more than there is."""
    from ..ops import paged_decode as pd  # noqa: PLC0415

    key = (str(device), hkv, group, d)
    got = _DEC_SCRATCH.get(key)
    if got is None or got[0].shape[0] < rows:
        if engine.capturing():
            return None
        got = _DEC_SCRATCH[key] = pd.scratch(max(rows, 256), hkv, group, d, device)
    return got


def _decode(a: dict) -> bool:
    """hk's split decode on a decode-only batch; False if it cannot (falls back)."""
    from ..ops import paged_decode as pd  # noqa: PLC0415

    q, out, k = a["q"], a["out"], a["k"]
    n, h, d = q.shape
    hkv, page = int(k.shape[2]), int(k.shape[1])
    group = h // hkv
    reqs = len(a["seqused_k"])
    kv, order = _flash_kv(k, a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    splits, warps, v_both = pd.plan(reqs, hkv, d, page, wgps=pd.device_wgps(q.device))
    ops = _decode_ops(d, group, page, order, splits, scale, warps, v_both)
    scratch = _decode_scratch(q.device, reqs * splits, hkv, group, d) if ops else None
    if ops is None or scratch is None:
        return False
    walk, merge = ops
    o4 = out.view(n, hkv, group, d)
    walk(q.unsqueeze(0), kv, a["block_table"], a["seqused_k"], a["cu_seqlens_q"], o4, *scratch)
    if merge is not None:
        merge(a["seqused_k"], a["cu_seqlens_q"], o4, *scratch)
    return True


def unified_attention(*args, **kwargs):
    """vLLM's unified_attention (its signature, set on apply), hk's flash-layout prefill --
    the whole batch, decodes included -- where it takes a call with prefill in it; with
    HK_VLLM_DECODE=1, hk's split decode where it takes a decode-only call."""
    a = kwargs if not args else _U_SIG.bind(*args, **kwargs).arguments
    if DECODE and int(a.get("max_seqlen_q", 1)) <= 1:
        if not d_why_not(a) and _decode(a):
            PATCH.count("decode")
            if SHADOW and not engine.capturing():
                import torch  # noqa: PLC0415

                ref = torch.empty_like(a["out"])
                b = dict(a)
                b["out"] = ref
                _ORIG_U(**b)
                live = (a["seqused_k"] > 0).nonzero().flatten().tolist()
                _compare(a["out"], ref, [r for r in live if r < a["out"].shape[0]])
            return None
        PATCH.count("fallback")
        return _ORIG_U(*args, **kwargs)
    why = _guard(a)
    if why:
        PATCH.count("declined")
        _note_declined(why)
        return _ORIG_U(*args, **kwargs)
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
