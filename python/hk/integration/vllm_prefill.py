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

    vllm serve <model>                     # on by default on gfx1100 and gfx1201
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

PATCH = engine.Patch("hk_vllm::prefill",
                     paths=("hk", "fallback", "decode", "declined", "split", "kvsplit"),
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


#: hk's kernels read a tensor through a buffer resource -- 32-bit byte offsets from its base -- so a
#: cache view spanning 4 GiB or more wraps: the wrong page read, or a fault. Measured on gfx1100 with
#: Qwen3.8's cache (4 kv heads, d256): 2^20 tokens a layer (4.3 GB) faults the prefill and the decode,
#: 983k (4.0 GB) is read right. A call over such a cache goes to vLLM's kernel.
SPAN_LIMIT = 1 << 32


def span_why_not(*ts) -> str:
    """Why hk cannot address these cache views (each one's bytes from its first element to its
    last, as its strides lay them out, at least SPAN_LIMIT), or ""."""
    for t in ts:
        span = (sum((n - 1) * st for n, st in zip(t.shape, t.stride()) if n > 0) + 1) \
            * t.element_size()
        if span >= SPAN_LIMIT:
            return (f"cache view spans {span / 2 ** 30:.2f} GiB; hk's buffer loads address "
                    f"{SPAN_LIMIT / 2 ** 30:.0f} GiB")
    return ""


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
    why = span_why_not(k_cache, v_cache)
    if why:
        return why
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
              order: str = "planes", decodes: bool = True):
    key = (head_dim, group, scale, page, order, decodes)
    if key not in _FLASH:
        from ..ops import paged_prefill as pp  # noqa: PLC0415

        def make():
            ds = head_dim > 128
            warps = 16 if ds else 12
            tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
            return pp.paged_prefill_kernel(
                f"paged_prefill_flash_d{head_dim}_g{group}{'_ds' if ds else ''}{_ptag(page)}"
                f"{_otag(order)}{'' if decodes else '_nd'}{tag}", head_dim=head_dim, group=group,
                scale=scale, layout="flash", dsplit=ds, warps=warps, decodes=decodes, page=page,
                kv_order=order), 16 * (warps // 2 if ds else warps)
        _FLASH[key] = _built(make)
    return _FLASH[key]


#: KV-split prefill (paged_prefill's kv_splits) for a short chunk over a long context: its few Q
#: tiles a head walk the context in runs, a workgroup each, and merge_kernel combines them.
#: HK_VLLM_PREFILL_KVSPLIT=0 turns it off; =N (N > 1) takes every prefill call with N splits,
#: the guard aside (to measure them); HK_VLLM_PREFILL_KVS_WARPS sets its tile (warps x 16 rows).
_KVS_ENV = __import__("os").environ.get("HK_VLLM_PREFILL_KVSPLIT", "1").strip().lower()
KVSPLIT = _KVS_ENV not in ("0", "off", "no", "false")
KVSPLIT_FORCE = int(_KVS_ENV) if _KVS_ENV.isdigit() and int(_KVS_ENV) > 1 else 0
KVS_WARPS = int(__import__("os").environ.get("HK_VLLM_PREFILL_KVS_WARPS", "0") or 0)
#: HK_VLLM_PREFILL_KVS_PACK=1 / 0 forces the GQA-packed tile (paged_prefill's pack) on or off for
#: the KV split's every call, the plan aside (to measure them)
_KVS_PACK_ENV = __import__("os").environ.get("HK_VLLM_PREFILL_KVS_PACK", "").strip()
KVS_PACK = None if _KVS_PACK_ENV == "" else _KVS_PACK_ENV not in ("0", "off", "no", "false")
_KVS_OPS: dict = {}


def _kvs_ops(head_dim: int, group: int, scale: Optional[float], page: int, order: str,
             splits: int, warps: int, pack: bool = False):
    """(walk launch, merge launch, Q tile tokens) for kv_splits = `splits`, decodes=False, GQA-packed
    tiles with `pack`, or None (not built spill-free here, or `group` does not divide a packed
    tile's warps)."""
    key = (head_dim, group, scale, page, order, splits, warps, pack)
    if key not in _KVS_OPS:
        from ..ops import paged_prefill as pp  # noqa: PLC0415

        ds = head_dim > 128
        rows_warps = warps // 2 if ds else warps
        if pack and rows_warps % group:
            _KVS_OPS[key] = None
            return None
        tile = 16 * rows_warps // (group if pack else 1)
        tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
        walk = _built(lambda: (pp.paged_prefill_kernel(
            f"paged_prefill_flash_d{head_dim}_g{group}{'_ds' if ds else ''}{_ptag(page)}"
            f"{_otag(order)}_nd_w{warps}_ks{splits}{'_pk' if pack else ''}{tag}", head_dim=head_dim,
            group=group, scale=scale, layout="flash", dsplit=ds, warps=warps, decodes=False,
            page=page, kv_order=order, kv_splits=splits, pack=pack), tile))
        merge = _built(lambda: (pp.merge_kernel(
            f"paged_prefill_merge_d{head_dim}_ks{splits}_t{tile}", head_dim=head_dim,
            kv_splits=splits, q_tile=tile), tile))
        _KVS_OPS[key] = (walk[0], merge[0], tile) if walk and merge else None
    return _KVS_OPS[key]


def kvs_warps(head_dim: int) -> int:
    return KVS_WARPS or (8 if head_dim > 128 else 4)


#: (arch, head size class) -> (the longest query below this, the longest sequence at least,
#: (warps, splits) up to SHORT_Q queries, (warps, splits) past): where the KV split takes a
#: prefill call, in place of both the guard's verdict and the unsplit kernel. Fitted on the
#: 36-shape sweep of three models (contexts 1k-32k x chunks 16-2048, batches of short chunks,
#: chunks beside decodes) with every (warps, splits) of {4, 8} x {4, 8, 16} that builds
#: spill-free (d256: 8 warps), against TRITON_ATTN; the route's own sweep against the faster of
#: TRITON_ATTN and ROCM_ATTN is tools/hk-bench/results/<date>/<arch>/prefill-kvsplit-sweep.json.
#: gfx1100: d128 1.27-5.2x up to 512-token chunks (longer ones tie the unsplit kernel); d256
#: 2.5-8.5x up to 128 (the unsplit kernel is as fast past). gfx1201: d128 up to 256 (the unsplit
#: kernel faster from 512), 1.4-3.3x over 4k+ but 1.06x over 1k; d256 1.5-5.1x up to 256 over 4k+,
#: ~1.15x over 1k: both from 2048 tokens.
KVSPLITS = {("gfx1100", 128): (1024, 0, (4, 8), (8, 8)),
            ("gfx1100", 256): (256, 0, (8, 4), (8, 4)),
            ("gfx1201", 128): (512, 2048, (4, 8), (8, 8)),
            ("gfx1201", 256): (512, 2048, (8, 8), (8, 4))}
SHORT_Q = 64
#: (arch, head size class, group) -> rows (longest query at most, longest sequence at least,
#: splits): where the KV split takes GQA-packed tiles (paged_prefill's pack: a workgroup a kv
#: head's query heads of 16 tokens, its K and V read once a tile, not once a query head), the first
#: row that fits, ahead of KVSPLITS' bounds; KVSPLITS' plan elsewhere. A packed tile is group
#: warps (twice that for d256), at most KVS_PACK_WARPS; its call has few workgroups, so the walk is
#: cut finer. The more query heads a kv head has, the more a packed tile saves: group 8 takes it
#: further than group 6. Fitted on tools/hk-bench's prefill sweep with every splits of {4, 8, 16,
#: 32} against TRITON_ATTN and the unpacked plan; gfx1100:
#: MiniCPM5 (d128, group 8) 1.2-2.8x the unpacked plan up to 64-token chunks, Qwen3.8 (d256, group
#: 6) 1.5-2.8x up to 32, MiniCPM-V (d256, group 8) 1.3-4.2x up to 512 (where the unsplit kernel
#: took it). 64 splits (paged_prefill.merge_kernel's loop form) over 16k-64k, against the rows of
#: 16 / 32 (tools/hk-bench/results/2026-11-05/gfx1100/prefill-kvsplit-64.json): MiniCPM5 16-token
#: chunks 1.14-1.25x, MiniCPM-V 32-64-token chunks 1.11-1.21x (16k:32 level); Qwen3.8 at most
#: 1.04x, not taken. gfx1201, from 16k (the unpacked plan as fast over 4k): MiniCPM5 1.4-3.4x up to 64,
#: Qwen3.8 1.1-4.2x up to 512 from 2k, MiniCPM-V 1.1-3.3x up to 256.
KVS_PACKED = {("gfx1100", 128, 8): ((16, 16384, 64), (32, 4096, 32), (64, 4096, 16)),
              ("gfx1100", 256, 6): ((16, 16384, 32), (32, 4096, 16)),
              ("gfx1100", 256, 8): ((16, 16384, 32), (64, 16384, 64), (512, 4096, 16)),
              ("gfx1201", 128, 8): ((16, 16384, 32), (64, 16384, 16)),
              ("gfx1201", 256, 6): ((16, 16384, 16), (512, 2048, 8)),
              ("gfx1201", 256, 8): ((16, 16384, 32), (32, 16384, 16), (256, 16384, 8))}
KVS_PACK_WARPS = 16


def kvs_pack_warps(head_dim: int, group: int) -> int:
    """A 16-token packed tile's warps: a q_block-row block a query head (a pair for d256)."""
    return group * (2 if head_dim > 128 else 1)
#: Beside decodes, the KV split is taken only while their contexts, summed, are at most this times
#: the prefill's (query, key) pairs: the decodes then go to vLLM's kernel in a second call, and
#: where they are most of the work the split gains nothing (gfx1201, MiniCPM5: a 32-token chunk
#: beside 7 decodes over 8k, a ratio of 0.22, 0.97-1.01x; a 64-token chunk over 16k beside 3, 0.023,
#: 1.98x; the sweep's mixed batches, 0.0024-0.016, 1.6-6.5x).
KVS_DEC_RATIO = 0.05
#: (arch, head size class) -> the longest sequence at least for the unpacked split beside decodes:
#: gfx1201 d128 over 8k it gains nothing (MiniCPM5, a 32-64-token chunk beside 1-3 decodes over 8k:
#: 0.89-1.09x), over 16k-32k it does (a 128-256-token chunk beside 2-4: 2.4-3.1x).
KVS_DEC_MIN_CTX = {("gfx1201", 128): 16384}
#: (arch, head size class, group) -> the most decode tokens beside the unpacked split, in place of
#: KVS_DEC_RATIO: gfx1100 Qwen3.8 (d256, group 6), a 64-128-token chunk over 8k-32k beside 1-31
#: decodes over 8k-32k -- 2.1-10.3x the faster stock kernel up to 15 x 32k or 31 x 8k decode tokens
#: (ratios to 0.93, which KVS_DEC_RATIO declined: vLLM's kernel there was 0.37-1.0x ROCM_ATTN's),
#: level (1.0-1.05x) at 31 x 32k.
KVS_DEC_MAX_TOKENS = {("gfx1100", 256, 6): 1 << 19}
#: KVS_DEC_RATIO for a packed plan (KVS_PACKED), and the most decodes beside it per (arch, head size
#: class, group): its prefill part is cheap enough that splitting off the decodes -- vLLM's decode
#: call splits their contexts, its mixed-batch kernel does not -- wins even where they are most of
#: the work. A 16-64-token chunk over 8k-32k beside 1-31 decodes over 8k-32k, against the faster
#: stock kernel: gfx1100 1.35-20x up to 15 decodes (ratios up to 3.75), where the unpacked route
#: declined them, but beside 31 short ones the second call costs more than it saves (MiniCPM5,
#: 0.80-1.06x); gfx1201 MiniCPM-V 1.4-5.7x up to 15, Qwen3.8 1.46-5.65x up to 3 but from 7 down to
#: 0.72x (over 8k), MiniCPM5 1.64-3.41x up to 2^18 decode tokens (7 x 32k, 15 x 8k) but 1.06-1.16x
#: at 15 x 32k; Qwen3.8 over 16k-32k 1.28-2.93x up to 7 decodes and ~170k decode tokens (two seeds,
#: runs of a 16-64-token chunk beside 5-15 decodes over 8k-32k), but 0.83-1.16x from 11 over 16k and
#: 1.11x at 7 x 32k. Qwen3.8 over 32k-64k with a chunk of 64+ tokens: 1.21-4.41x beside 7-15
#: decodes over 8k-32k at any count of their tokens (to 15 x 32k; 64-512-token chunks), where a
#: 32-token chunk over 32k fell to 0.93x beside 11 x 32k and a 64-token one over 16k to 0.92-1.04x
#: beside 11-15 x 32k (tools/hk-bench/results/2026-11-05/gfx1201/prefill-kvsplit-qwen3.8-dec.json).
#: Each entry is tiers (longest sequence at least, longest query at least, most decodes, most decode
#: tokens or None), the last the call reaches in both.
KVS_PACK_DEC_RATIO = 4.0
KVS_PACK_MAX_DECODES = {("gfx1201", 128, 8): ((0, 0, 15, 1 << 18),),
                        ("gfx1201", 256, 6): ((0, 0, 3, None), (16384, 0, 7, 170000),
                                              (32768, 64, 15, None))}
KVS_PACK_MAX_DECODES_ELSE = ((0, 0, 15, None),)


def kvsplit_plan(a: dict, pack: bool = True):
    """(splits, warps, pack) where the KV split takes this prefill call, or None: exact sizes known;
    beside decodes (which go to vLLM's kernel, as in SPLIT) no request without queries, SPLIT on and
    the decodes' work under KVS_DEC_RATIO of the prefill's (or KVS_DEC_MAX_TOKENS) and its context
    KVS_DEC_MIN_CTX (packed:
    KVS_PACK_DEC_RATIO, and at most KVS_PACK_MAX_DECODES of them); the longest query short enough and the longest sequence long enough (KVSPLITS).
    Packed tiles where a KVS_PACKED row fits, first (not with `pack` False: KVSPLITS' plan alone)."""
    from . import switches  # noqa: PLC0415

    if not KVSPLIT:
        return None
    st = _step_stats(a)
    if st is None or not st["max_q"]:
        return None
    if st["decodes"] and (st["empty"] or not SPLIT):
        return None

    def decodes_fit(ratio):
        return not st["decodes"] or st["dec_work"] <= ratio * st["pre_work"]

    d = int(a["q"].shape[-1])
    group = int(a["q"].shape[1]) // int(a["k"].shape[2])
    pw = kvs_pack_warps(d, group)
    arch, cls = switches.arch(), 256 if d > 128 else 128
    max_dec, max_dec_tok = [(n, t) for c, mq, n, t in KVS_PACK_MAX_DECODES.get(
        (arch, cls, group), KVS_PACK_MAX_DECODES_ELSE)
        if st["pre_ctx"] >= c and st["max_q"] >= mq][-1]
    for max_q, min_ctx, sp in KVS_PACKED.get((arch, cls, group), ()) if pack else ():
        if (st["max_q"] <= max_q and st["pre_ctx"] >= min_ctx and pw <= KVS_PACK_WARPS
                and decodes_fit(KVS_PACK_DEC_RATIO) and st["decodes"] <= max_dec
                and (max_dec_tok is None or st["dec_work"] <= max_dec_tok)):
            return sp, pw, True
    max_tok = KVS_DEC_MAX_TOKENS.get((arch, cls, group))
    fits = (decodes_fit(KVS_DEC_RATIO) if max_tok is None
            else not st["decodes"] or st["dec_work"] <= max_tok)
    if not fits or (st["decodes"] and st["pre_ctx"] < KVS_DEC_MIN_CTX.get((arch, cls), 0)):
        return None
    plan = KVSPLITS.get((arch, cls))
    if plan is None or st["max_q"] >= plan[0] or st["pre_ctx"] < plan[1]:
        return None
    w, sp = plan[2] if st["max_q"] <= SHORT_Q else plan[3]
    return sp, w, False


#: V^T pre-pack for TRITON_ATTN (paged_prefill vt_pages): the batch's V pages transposed once a
#: call into a scratch, instead of once per (query head, tile) workgroup inside the kernel. Taken
#: when the scratch -- requests x the longest request's pages -- stays under VT_CAP_BYTES;
#: HK_VLLM_PREFILL_VT=0 turns it off.
VT = __import__("os").environ.get("HK_VLLM_PREFILL_VT", "1") not in ("0", "off", "no", "false")
VT_CAP_BYTES = 256 << 20
_VT_OPS: dict = {}
_VT_SCRATCH: dict = {}


def _vt_ops(head_dim: int, group: int, scale: Optional[float], page: int = 16,
            order: str = "planes", decodes: bool = True):
    """(pack launch, vt_pages launch, Q tile rows), or None (not built here, or head_dim 256)."""
    key = (head_dim, group, scale, page, order, decodes)
    if key not in _VT_OPS:
        _VT_OPS[key] = None
        if head_dim <= 128:
            from ..ops import paged_prefill as pp  # noqa: PLC0415

            tag = "" if scale is None else "_s%08x" % (hash(scale) & 0xffffffff)
            main = _built(lambda: (pp.paged_prefill_kernel(
                f"paged_prefill_flashvt_d{head_dim}_g{group}{_ptag(page)}{_otag(order)}"
                f"{'' if decodes else '_nd'}{tag}",
                head_dim=head_dim, group=group, scale=scale, layout="flash", warps=12,
                decodes=decodes, vt_pages=True, page=page, kv_order=order), 192))
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
    kv = _flash_kv(k, v)
    if kv is None:
        return (f"key/value caches {tuple(k.shape)} strides {tuple(k.stride())} (V at +"
                f"{(v.data_ptr() - k.data_ptr()) // k.element_size()}): not a layout hk reads")
    return span_why_not(kv[0])


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
    # gfx1100 (W7900D). Mixed: up to 128, wins to a ratio of 0.0068 (>= 1.55x); 256, Qwen3.8
    # loses from 0.0049 and MiniCPM-V from 0.0059, wins at 0.0039. Up to 128 (2 kv
    # heads): a short prompt wins (1.33x at 16-256 tokens; 512:64 too), a chunk under 512 rows
    # over a context of 1k+ loses (1024:64 0.85x, 2048:128 0.91x), 4096:256 (512 rows) wins.
    # 256: no chunk measured loses (32 tokens x 1 kv head over 32k: a win).
    ("gfx1100", 128): (0.006, 512, 600, 64),
    ("gfx1100", 256): (0.003, 16, 0, 1024),
    # gfx1201 (R9700). Mixed: up to 128, wins to 0.0068 (>= 1.64x); 256, MiniCPM-V >= 1.19x to
    # 0.0049 and thin from 0.0059, losses from 0.0078. Up to 128: under 1024 rows only thin
    # wins (1.03-1.14x) or losses (32-token chunks over 2k+: 0.77x); 0:512 1.09x. 256: a short
    # prompt wins (1.7x), a chunk under 512 rows over 1k+ loses or ties (1024:64 0.95-1.00x,
    # 1024:128 0.91x).
    ("gfx1201", 128): (0.006, 1024, 0, 128),
    ("gfx1201", 256): (0.004, 512, 600, 128),
}
_DECLINED_SAID: list = []

#: With exact sizes (EXACT below): a batch is declined when its decodes' contexts, summed, are
#: more than EXACT_RATIO times its prefill's (query, key) pairs. Fitted on the same measured rows
#: with the batches' true decode contexts, under the first loss: gfx1100 up to 128 clean to
#: 0.0068, a loss at 0.0078; 256 clean to 0.0039, a loss at 0.0049; gfx1201 up to 128 a loss at
#: 0.0078; 256 a thin 1.02x at 0.0068.
EXACT_RATIO = {("gfx1100", 128): 0.006, ("gfx1100", 256): 0.0035,
               ("gfx1201", 128): 0.006, ("gfx1201", 256): 0.005}

#: The exact sizes of the steps vLLM's TRITON_ATTN metadata builder built, from host arrays it
#: already has (no sync): {id(the step's query_start_loc): batch_stats(...)}. The stand-in finds
#: its call's by the identity of `cu_seqlens_q`, which TRITON_ATTN passes as that tensor. A few
#: entries, the latest kept: one a KV-cache group a step.
_STATS: dict = {}
_STATS_KEEP = 8
#: TritonAttentionMetadataBuilder.build as vLLM had it, while the hook is on (apply())
_ORIG_BUILD = None
#: why the hook is not on, if it is not ("" when it is)
STATS_WHY_NOT = "not applied"


def batch_stats(query_start_loc, seq_lens, num_reqs: int) -> dict:
    """A step's sizes from host arrays (`query_start_loc` num_reqs + 1 long, `seq_lens`): the
    decode rows (one query), their contexts summed, the prefill's (query, key) pairs under the
    causal mask, its longest query and that query's context."""
    import numpy as np  # noqa: PLC0415

    qsl = np.asarray(query_start_loc, dtype=np.int64)[: num_reqs + 1]
    s = np.asarray(seq_lens, dtype=np.int64)[:num_reqs]
    q = np.diff(qsl)
    dec, pre = q == 1, q > 1
    qp, sp = q[pre], s[pre]
    i = int(np.argmax(qp)) if qp.size else 0
    return dict(reqs=int(num_reqs), decodes=int(dec.sum()), empty=int((q == 0).sum()),
                dec_work=int(s[dec].sum()),
                pre_work=float((qp * (sp - qp) + qp * (qp + 1) / 2).sum()),
                max_q=int(qp[i]) if qp.size else 0, pre_ctx=int(sp[i]) if qp.size else 0)


def _stats_hook(orig):
    """vLLM's TritonAttentionMetadataBuilder.build, unchanged, recording the step's sizes from the
    common metadata's host arrays: `query_start_loc_cpu`, and the cached `_seq_lens_cpu` or
    `seq_lens_cpu_upper_bound` -- never the `seq_lens_cpu` property, which copies from the device
    when nothing is cached (a sync)."""
    import functools  # noqa: PLC0415

    @functools.wraps(orig)
    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        md = orig(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        try:
            cam = common_attn_metadata
            seq = getattr(cam, "_seq_lens_cpu", None)
            if seq is None:
                seq = getattr(cam, "seq_lens_cpu_upper_bound", None)
            qsl = getattr(cam, "query_start_loc_cpu", None)
            key = getattr(md, "query_start_loc", None)
            if seq is not None and qsl is not None and key is not None:
                if len(_STATS) >= _STATS_KEEP:
                    _STATS.pop(next(iter(_STATS)))
                _STATS[id(key)] = batch_stats(qsl, seq, int(cam.num_reqs))
        except Exception:  # noqa: BLE001 -- the hook must never break vLLM's build
            pass
        return md
    return build


def exact_why_not(st: dict, head_dim: int, kv_heads: int, arch: str, split: bool = False) -> str:
    """guard_why_not with a step's exact sizes (`st`: batch_stats): the decodes' real contexts
    against the prefill's pairs, and the longest prefill's own query and context. `split`: the
    decodes go to vLLM's kernel apart (SPLIT), so only the prefill's own rules apply."""
    if not GUARD:
        return ""
    key = (arch, 256 if head_dim > 128 else 128)
    g, exact = GUARDS.get(key), EXACT_RATIO.get(key)
    if g is None or exact is None:
        return f"no measured guard for {key}"
    _mix, min_rows, short_ctx, ctx_per_q = g
    if st["decodes"] and st["dec_work"] > exact * st["pre_work"] and not split:
        return (f"{st['decodes']} decodes over {st['dec_work']} tokens beside a prefill of "
                f"{st['pre_work']:.0f} pairs (vLLM's kernel splits a decode's context, hk's does "
                f"not)")
    max_q, ctx = st["max_q"], st["pre_ctx"]
    if max_q * kv_heads < min_rows and ctx > short_ctx:
        return f"a {max_q}-token chunk over a {ctx}-token context, {kv_heads} kv head(s)"
    if ctx > ctx_per_q * max_q:
        return f"a {max_q}-token chunk over a {ctx}-token context"
    return ""


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
    st = _step_stats(a)
    if st is not None:
        return exact_why_not(st, int(q.shape[-1]), int(k.shape[2]), arch)
    return guard_why_not(int(used.shape[0]), int(a.get("max_seqlen_q", 0)),
                         int(a.get("max_seqlen_k") or 0), int(q.shape[-1]), int(k.shape[2]), arch)


def _step_stats(a: dict):
    """This call's step sizes (batch_stats) from the TRITON_ATTN builder hook, or None."""
    used = a.get("seqused_k")
    st = _STATS.get(id(a.get("cu_seqlens_q")))
    return st if st is not None and used is not None and st["reqs"] == int(used.shape[0]) else None


#: Mixed batches the guard declines for their decodes: with the step's exact sizes, hk takes the
#: prefill rows (the kernel built with decodes=False skips one-query requests) and vLLM's own
#: kernel the decode rows, as a batch of one query a request (`_split_decodes`) -- vLLM splits a
#: decode's context, hk's prefill kernel does not. HK_VLLM_PREFILL_SPLIT=0 turns it off (the
#: guard then declines the whole batch, as before); "all" splits every such batch, SPLITS aside
#: (to measure them).
_SPLIT_ENV = __import__("os").environ.get("HK_VLLM_PREFILL_SPLIT", "1").strip().lower()
SPLIT = _SPLIT_ENV not in ("0", "off", "no", "false")
SPLIT_ALL = _SPLIT_ENV == "all"
#: (arch, head size class) -> (the longest query at least, decode work at most this times the
#: prefill's): the split costs launches and gives the decode rows to TRITON_ATTN's decode, which
#: ROCM_ATTN's beats, so it pays off on a sizable prefill. Measured against the faster of
#: TRITON_ATTN and ROCM_ATTN (tools/hk-bench/results/2026-11-04/<arch>/prefill-split-sweep.json:
#: the 68-shape set, and the chunks vLLM's default 2048-token budget makes -- 1024-2040 tokens,
#: first or after a prefix, beside 4-16 decodes). From 1024 tokens, gfx1100: d128 1.08-2.59x to
#: a ratio of 0.125, d256 1.40-5.8x to 0.25; gfx1201: d128 1.21-2.58x to 0.028 (a first chunk
#: beside heavier decodes 0.70-1.03x), d256 1.16-2.8x to 0.111 (0.97x at 0.125). Prompts of
#: 128-512 tokens beside decodes: some lose (0.26-1.03x), so they stay declined.
SPLITS = {("gfx1100", 128): (1024, 0.13), ("gfx1100", 256): (1024, 0.25),
          ("gfx1201", 128): (1024, 0.03), ("gfx1201", 256): (1024, 0.10)}


def split_takes(a: dict) -> bool:
    """Whether a call the guard declines is taken split: exact sizes known, decodes in it, no
    request without queries (each request's first row is its own), the prefill sizable enough
    against the decodes (SPLITS), and the prefill rows alone passing the guard's other rules."""
    from . import switches  # noqa: PLC0415

    st = _step_stats(a) if SPLIT else None
    if st is None or not st["decodes"] or st["empty"]:
        return False
    q, k = a["q"], a["k"]
    d, arch = int(q.shape[-1]), switches.arch()
    sp = SPLITS.get((arch, 256 if d > 128 else 128))
    if not SPLIT_ALL and (sp is None or st["max_q"] < sp[0]
                          or st["dec_work"] > sp[1] * st["pre_work"]):
        return False
    return not exact_why_not(st, d, int(k.shape[2]), arch, split=True)


_ARANGE: dict = {}


def _split_decodes(a: dict) -> None:
    """The decode rows of a call hk's decodes=False kernel left alone: vLLM's kernel on the batch
    as one query a request -- each request's first row, a prefill request as a one-key row whose
    answer is dropped -- and its answers copied back to the decodes' rows. All on the device: no
    sync, so it adds launches, not waits."""
    import torch  # noqa: PLC0415

    q, out, cu, used = a["q"], a["out"], a["cu_seqlens_q"], a["seqused_k"]
    r = int(used.shape[0])
    first = cu[:-1].long()
    is_dec = (cu[1:] - cu[:-1]) == 1
    key = (str(q.device), r + 1)
    if key not in _ARANGE:
        _ARANGE[key] = torch.arange(r + 1, device=q.device, dtype=cu.dtype)
    q1 = q.index_select(0, first)
    out1 = torch.empty_like(q1)
    b = dict(a)
    b.update(q=q1, out=out1, cu_seqlens_q=_ARANGE[key], max_seqlen_q=1,
             seqused_k=torch.where(is_dec, used, torch.ones_like(used)))
    _ORIG_U(**b)
    out.index_copy_(0, first, torch.where(is_dec[:, None, None], out1, out.index_select(0, first)))


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

#: Decode batches hk takes: at head_dim 128, up to this many (request, kv head) pairs; at 256
#: every batch (Qwen3.8-27B: 1.06-2.2x at B1-B16). Serving MiniCPM5-2B on a W7900D (TRITON_ATTN +
#: TunableOp, 3 alternating starts; HK_VLLM_DECODE=1 against vLLM's decode): one request, TPOT
#: 1.03 / 1.08 / 1.20 / 1.40x at 4k / 8k / 16k / 32k tokens; four requests (8 pairs) 0.98x at
#: 4k-8k -- so 4 pairs, one or two requests. Decided on the batch alone, not on its lengths: a
#: decode-only step runs as a CUDA graph captured once a batch size, so what the route takes at
#: capture is what every replay runs, whatever the lengths.
DECODE_MAX_PAIRS_D128 = 4
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
    kv = _flash_kv(k, a["v"])
    if kv is None:
        return "key/value caches: not a layout hk reads"
    return span_why_not(kv[0])


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
    if int(a.get("max_seqlen_q", 1)) > 1 and not KVSPLIT_FORCE:
        plan = kvsplit_plan(a)
        if plan is not None and plan[2]:
            if _kvsplit(a, *plan):
                return None
            plan = kvsplit_plan(a, pack=False)        # packed tiles not built here
        if plan is not None and _kvsplit(a, *plan):
            return None
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
    split = bool(why) and split_takes(a)
    if why and not split:
        # a batch with prefill in it is never captured (piecewise graphs leave attention out, and
        # full graphs hold decode-only steps), so the count skips the capture query
        PATCH.stats["declined"] += 1
        if not _DECLINED_SAID:
            _note_declined(why)
        return _ORIG_U(*args, **kwargs)
    if u_why_not(a):
        PATCH.count("fallback")
        return _ORIG_U(*args, **kwargs)
    if KVSPLIT_FORCE and _kvsplit(a, KVSPLIT_FORCE):
        return None
    decodes = not split
    q, out = a["q"], a["out"]
    n, h, d = q.shape
    kv, order = _flash_kv(a["k"], a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    page = int(a["k"].shape[1])
    got = _flash_op(d, h // a["k"].shape[2], scale, page, order, decodes)
    if decodes and SPLIT:
        # the split's decodes=False kernels, built with the first ones (vLLM's warm-up run) so
        # the first mixed batch of a fresh cache does not compile them on the serving path
        _flash_op(d, h // a["k"].shape[2], scale, page, order, False)
        if VT and d <= 128:
            _vt_ops(d, h // a["k"].shape[2], scale, page, order, False)
    if decodes:
        _kvs_warm(a)                        # the KV split's, likewise
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
        vt_ops = (_vt_ops(d, h // hkv, scale, page, order, decodes)
                  if 0 < slots * hkv * d * page * 2 <= VT_CAP_BYTES else None)
        if vt_ops is not None:
            pack, op, tile = vt_ops
            v_src = _vt_scratch(q.device, slots, hkv, d, page)
            pack(kv, a["block_table"], a["seqused_k"], pp.vt_pack_span(max_k, q.device, page),
                 v_src)
    args = [q.reshape(n, h * d), kv, v_src, a["block_table"], a["seqused_k"], a["cu_seqlens_q"],
            pp.q_tiles(int(a["max_seqlen_q"]), q.device, tile), out.reshape(n, h * d)]
    if d <= 128 and decodes:
        # the GQA-packed decodes' views: q and the output as (1, tokens, heads, head_dim)
        args += [q.unsqueeze(0), out.unsqueeze(0)]
    op(*args)
    if split:
        _split_decodes(a)
        PATCH.count("split")
    PATCH.count("hk")
    if SHADOW:
        import torch  # noqa: PLC0415

        ref = torch.empty_like(out)
        b = dict(a)
        b["out"] = ref
        _ORIG_U(**b)
        _compare(out, ref, list(range(n)))
    return None


def _kvs_warm(a: dict) -> None:
    """The KV split's kernels, both plans, built with the whole-batch ones (vLLM's warm-up run is
    such a batch), so the first short chunk of a fresh cache does not compile them on the serving
    path."""
    from . import switches  # noqa: PLC0415

    q, k = a["q"], a["k"]
    d = int(q.shape[-1])
    if not KVSPLIT:
        return
    kv = _flash_kv(k, a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    group = int(q.shape[1]) // int(k.shape[2])
    pw = kvs_pack_warps(d, group)
    plan = KVSPLITS.get((switches.arch(), 256 if d > 128 else 128))
    packed = KVS_PACKED.get((switches.arch(), 256 if d > 128 else 128, group), ())
    builds = {(*plan[2], False), (*plan[3], False)} if plan else set()
    if pw <= KVS_PACK_WARPS:
        builds |= {(pw, sp, True) for *_, sp in packed}
    for w, sp, pk in builds:
        _kvs_ops(d, group, scale, int(k.shape[1]), kv[1], sp, w, pk)


def _kvsplit(a: dict, splits: int, warps: int = 0, pack: bool = False) -> bool:
    """A prefill call with paged_prefill's kv_splits = `splits` and its merge; vLLM's kernel
    takes the decode rows, if any (`_split_decodes`). False where it does not build here."""
    if u_why_not(a):
        return False
    q, out = a["q"], a["out"]
    n, h, d = q.shape
    kv, order = _flash_kv(a["k"], a["v"])
    sc = a["softmax_scale"]
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    page = int(a["k"].shape[1])
    if KVS_PACK is not None:
        pack = KVS_PACK
    got = _kvs_ops(d, h // a["k"].shape[2], scale, page, order, splits, warps or kvs_warps(d),
                   pack)
    if got is None:
        return False
    walk, merge, tile = got
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    used = a["seqused_k"]
    tiles = pp.q_tiles(int(a["max_seqlen_q"]), q.device, tile)
    o_part, ml_part = pp.split_parts(splits, len(used) * tiles.shape[0] * tile, h, d, q.device)
    o2 = out.reshape(n, h * d)
    walk(q.reshape(n, h * d), kv, kv, a["block_table"], used, a["cu_seqlens_q"], tiles, o2,
         o_part, ml_part)
    merge(used, a["cu_seqlens_q"], tiles, o_part, ml_part, o2)
    st = _step_stats(a)
    if st is not None and st["decodes"]:
        _split_decodes(a)
        PATCH.count("split")
    PATCH.count("kvsplit")
    PATCH.count("hk")
    if SHADOW:
        import torch  # noqa: PLC0415

        ref = torch.empty_like(out)
        b = dict(a)
        b["out"] = ref
        _ORIG_U(**b)
        _compare(out, ref, list(range(n)))
    return True


_U_SIG = None
_GUARDED_U = PATCH.guard(unified_attention)


# ---- ROCM_ATTN's decode: hk's split decode on ROCM_ATTN's own cache --------------------------

#: A ROCM_ATTN call with only decodes in it (`chunked_prefill_paged_decode`, max_query_len 1)
#: goes to hk.ops.paged_decode over ROCM_ATTN's own cache (kv_order="rocm": the x-split K
#: staged, the V^T pages read as they are), one-warp workgroups, 4 a WGP (paged_decode.plan_rocm).
#: Prefill and mixed calls keep vLLM's function (whose prefix-prefill is hk's, above).
#: Opt-in, HK_VLLM_ROCM_DECODE=1. W7900D, MiniCPM5-2B, one process (attention_bench, hk_rocm arm,
#: tools/hk-bench/results/2026-11-06/gfx1100/decode-rocm-cache.json): against ROCM_ATTN's paged
#: decode (ll4mi) 0.95-1.29x, B1 1.19-1.29x from 16k (0.95x at 4k), B2-B8 1.01-1.25x; weighted
#: by a measured real serving mix 1.10-1.14x against the faster stock kernel, under the 1.15x the
#: default-on rule asks for. Serving (same process, graphs): TPOT 1.01x (2k-8k) to 1.07x (64k) at
#: one request, 1.02-1.03x at 2-8, TTFT unchanged.
ROCM_DECODE = __import__("os").environ.get("HK_VLLM_ROCM_DECODE", "").strip().lower() in (
    "1", "on", "yes", "true")
#: Decode batches it takes: up to this many (request, kv head) pairs (B8 at 2 kv heads); B16 was
#: as fast as vLLM's.
ROCM_DEC_MAX_PAIRS = 16
#: Where it is measured and tested.
ROCM_DEC_ARCHES = ("gfx1100",)
#: vLLM's chunked_prefill_paged_decode while patched, and its signature
_ORIG_C = None
_C_SIG = None
#: its parameters the stand-in reads; any other one not at its default falls back
_C_READ = frozenset((
    "query", "key", "value", "output", "kv_cache_dtype", "key_cache", "value_cache",
    "block_table", "query_start_loc", "seq_lens", "max_seq_len", "max_query_len", "k_scale",
    "v_scale", "alibi_slopes", "sliding_window", "sm_scale", "output_scale", "sinks",
    "is_block_table_ptr", "causal"))
_C_DEFAULTS = (("alibi_slopes", None), ("sliding_window", None), ("sm_scale", None),
               ("output_scale", None), ("sinks", None), ("is_block_table_ptr", False),
               ("causal", True))


def _c_unknown(a: dict) -> str:
    for name, val in a.items():
        if name in _C_READ:
            continue
        p = _C_SIG.parameters.get(name) if _C_SIG is not None else None
        default = p.default if p is not None else None
        try:
            if val is default or (not hasattr(val, "shape") and val == default):
                continue
        except Exception:  # noqa: BLE001 -- an uncomparable value is a set one
            pass
        return f"{name}={val!r} (a parameter hk does not take)"
    return ""


def rd_why_not(a: dict) -> str:
    """Why hk's decode does not take this ROCM_ATTN chunked_prefill_paged_decode call, or ""."""
    import torch  # noqa: PLC0415

    from . import switches  # noqa: PLC0415

    if not ROCM_DECODE:
        return "HK_VLLM_ROCM_DECODE=1 takes it"
    if switches.arch() not in ROCM_DEC_ARCHES:
        return f"arch {switches.arch()}"
    if int(a.get("max_query_len", 2)) != 1:
        return "not decode-only"
    unknown = _c_unknown(a)
    if unknown:
        return unknown
    g = {n: a.get(n, d) for n, d in _C_DEFAULTS}
    if g["alibi_slopes"] is not None or g["sinks"] is not None or g["output_scale"] is not None:
        return "alibi / sinks / fp8 output"
    if g["sliding_window"] is not None and g["sliding_window"] > 0:
        return "sliding window"
    if not g["causal"] or g["is_block_table_ptr"]:
        return "non-causal / block-table pointers"
    q, out, kc, vc = a["query"], a["output"], a["key_cache"], a["value_cache"]
    if (q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16 or kc.dtype != torch.bfloat16
            or not str(a["kv_cache_dtype"]).startswith("auto")):
        return f"dtype {q.dtype} / cache {a['kv_cache_dtype']}"
    if kc.dim() != 5 or vc.dim() != 4 or kc.shape[-1] * kc.element_size() != 16:
        return f"cache layout {tuple(kc.shape)}"
    if q.dim() != 3 or q.shape[-1] != 128:
        return f"head size {q.shape[-1]}"
    if int(vc.shape[3]) != 16 or int(kc.shape[3]) != 16:
        return f"page {vc.shape[3]}: hk's decode on ROCM_ATTN's cache takes pages of 16"
    hkv = int(kc.shape[1])
    if q.shape[1] % hkv or q.shape[1] // hkv > 16:
        return "query heads not a multiple of kv heads, or more than 16 a kv head"
    if len(a["seq_lens"]) * hkv > ROCM_DEC_MAX_PAIRS:
        return "decode batch past hk's (vLLM's kernel reads about as fast)"
    for t in (a["block_table"], a["seq_lens"], a["query_start_loc"]):
        if t.dtype != torch.int32 or not t.is_contiguous():
            return "metadata not contiguous int32"
    if q.stride(-1) != 1 or q.stride(-2) != q.shape[-1] or not out.is_contiguous():
        return "q heads not packed / output not contiguous"
    return span_why_not(kc, vc)


def _rocm_decode(a: dict) -> bool:
    """hk's split decode on ROCM_ATTN's cache; False if it cannot (falls back)."""
    from ..ops import paged_decode as pd  # noqa: PLC0415
    from ..ops import paged_prefill as pp  # noqa: PLC0415

    q, out, kc, vc = a["query"], a["output"], a["key_cache"], a["value_cache"]
    n, h, d = q.shape
    hkv = int(kc.shape[1])
    group = h // hkv
    reqs = len(a["seq_lens"])
    sc = a.get("sm_scale")
    scale = None if sc is None or sc == d ** -0.5 else float(sc)
    splits = pd.plan_rocm(reqs, hkv, wgps=pd.device_wgps(q.device))
    ops = _decode_ops(d, group, 16, "rocm", splits, scale, 1, False)
    scratch = _decode_scratch(q.device, reqs * splits, hkv, group, d) if ops else None
    if ops is None or scratch is None:
        return False
    walk, merge = ops
    kv, vt = pp.rocm_views(kc, vc)
    o4 = out.view(n, hkv, group, d)
    walk(q.unsqueeze(0), kv, a["block_table"], a["seq_lens"], a["query_start_loc"], o4,
         *scratch, vt)
    if merge is not None:
        merge(a["seq_lens"], a["query_start_loc"], o4, *scratch)
    return True


def chunked_prefill_paged_decode(*args, **kwargs):
    """vLLM ROCM_ATTN's chunked_prefill_paged_decode (its signature, set on apply): hk's split
    decode where it takes a decode-only call; everything else is vLLM's function."""
    a = dict(kwargs) if not args else _C_SIG.bind(*args, **kwargs).arguments
    if int(a.get("max_query_len", 2)) == 1:
        if not rd_why_not(a) and _rocm_decode(a):
            PATCH.count("decode")
            if SHADOW and not engine.capturing():
                import torch  # noqa: PLC0415

                b = dict(a)
                b["output"] = ref = torch.empty_like(a["output"])
                _ORIG_C(**b)
                live = (a["seq_lens"] > 0).nonzero().flatten().tolist()
                _compare(a["output"], ref, [r for r in live if r < ref.shape[0]])
            return None
        PATCH.count("fallback")
    return _ORIG_C(*args, **kwargs)


_GUARDED_C = PATCH.guard(chunked_prefill_paged_decode)


def _apply_rocm_decode() -> list:
    global _ORIG_C, _C_SIG
    import inspect  # noqa: PLC0415

    from vllm.v1.attention.ops import chunked_prefill_paged_decode as cppd  # noqa: PLC0415

    _ORIG_C = cppd.chunked_prefill_paged_decode
    _C_SIG = inspect.signature(_ORIG_C)
    chunked_prefill_paged_decode.__signature__ = _C_SIG
    return _common.rebind(PREFIXES, _ORIG_C, _GUARDED_C)


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
    try:
        cnames = _apply_rocm_decode()
    except Exception as e:  # noqa: BLE001 -- an older vLLM: the prefill routes stand alone
        cnames, cwhy = [], f"; ROCM_ATTN decode not patched ({type(e).__name__})"
    else:
        cwhy = ""
    _apply_stats()
    return (f"prefill -> hk where hk.ops.paged_prefill takes ROCM_ATTN's prefill "
            f"({len(names)} bindings rebound) and TRITON_ATTN's batches with prefill in them "
            f"({len(unames)} rebound); ROCM_ATTN's decode-only calls -> hk.ops.paged_decode "
            f"({len(cnames)} rebound){cwhy}")


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


def stats_why_not(builder_cls) -> str:
    """Why the batch-size hook cannot go on this vLLM's TRITON_ATTN metadata builder, or "": it
    wraps `build(common_prefix_len, common_attn_metadata, ...)` and reads the common metadata's
    host arrays. Without them the guard keeps its size-only bound, which declines when unsure."""
    import dataclasses  # noqa: PLC0415
    import inspect  # noqa: PLC0415

    build = getattr(builder_cls, "build", None)
    if build is None:
        return "no TritonAttentionMetadataBuilder.build"
    params = list(inspect.signature(build).parameters)
    if params[1:3] != ["common_prefix_len", "common_attn_metadata"]:
        return f"build{tuple(params)}: not (self, common_prefix_len, common_attn_metadata, ...)"
    try:
        from vllm.v1.attention.backends.utils import CommonAttentionMetadata  # noqa: PLC0415

        fields = {f.name for f in dataclasses.fields(CommonAttentionMetadata)}
    except Exception as e:  # noqa: BLE001
        return f"no CommonAttentionMetadata ({type(e).__name__})"
    if "query_start_loc_cpu" not in fields or "num_reqs" not in fields:
        return "CommonAttentionMetadata has no query_start_loc_cpu / num_reqs"
    if not fields & {"_seq_lens_cpu", "seq_lens_cpu_upper_bound"}:
        return "CommonAttentionMetadata has no host sequence lengths"
    return ""


def _apply_stats() -> None:
    global _ORIG_BUILD, STATS_WHY_NOT
    try:
        from vllm.v1.attention.backends import triton_attn  # noqa: PLC0415

        cls = triton_attn.TritonAttentionMetadataBuilder
    except Exception as e:  # noqa: BLE001
        STATS_WHY_NOT = f"no TRITON_ATTN metadata builder ({type(e).__name__})"
        return
    STATS_WHY_NOT = stats_why_not(cls)
    if STATS_WHY_NOT or _ORIG_BUILD is not None:
        return
    _ORIG_BUILD = cls.build
    cls.build = _stats_hook(cls.build)


def _revert_stats() -> None:
    global _ORIG_BUILD, STATS_WHY_NOT
    if _ORIG_BUILD is None:
        return
    from vllm.v1.attention.backends import triton_attn  # noqa: PLC0415

    triton_attn.TritonAttentionMetadataBuilder.build = _ORIG_BUILD
    _ORIG_BUILD = None
    _STATS.clear()
    STATS_WHY_NOT = "not applied"


def revert() -> str:
    global _ORIG
    if _ORIG is None:
        return "prefill: not patched"
    _revert_stats()
    global _ORIG_U
    names = _common.rebind(PREFIXES, _GUARDED, _ORIG)
    if _ORIG_U is not None:
        names += _common.rebind(PREFIXES, _GUARDED_U, _ORIG_U)
        _ORIG_U = None
    global _ORIG_C
    if _ORIG_C is not None:
        names += _common.rebind(PREFIXES, _GUARDED_C, _ORIG_C)
        _ORIG_C = None
    _ORIG = None
    return f"prefill -> vllm ({len(names)} rebound)"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary()
