"""A vLLM attention backend that runs on the generated kernels.

This is the part an SDPA drop-in cannot reach. vLLM V1 keeps KV in a paged
cache and dispatches through `AttentionImpl.forward(..., kv_cache,
attn_metadata)`; there is no dense tensor on that path, so the only way into a
decoder is to *be* a backend. vLLM's backend enum is closed and ROCm will not
select `CUSTOM`, so it is registered *over* a member the platform accepts --
`TRITON_ATTN`, unless `HK_VLLM_BACKEND` names another (see `SLOT` below) --
through `vllm.v1.attention.backends.registry.register_backend`, vLLM's own
hook for third-party backends. In diagnostics it calls itself `HK_ATTN`.

    HK_PATCH_VLLM=1 HK_VLLM_BACKEND=1 vllm serve <model> \\
        --attention-backend TRITON_ATTN

with vLLM's defaults: CUDA graphs, prefix caching and chunked prefill on.

The backend declares its own KV cache layout -- `(2, num_blocks,
num_kv_heads, block_size, head_size)` -- because `gl` derives strides from
dims and vLLM's default `(2, num_blocks, block_size, num_kv_heads,
head_size)` cannot yield a contiguous `(block_size, head_dim)` tile for one
head. Having its own layout means every attention it serves is its own:

* decode: vLLM's own Triton kernel, on this backend's cache, captured whole
  in vLLM's full decode graphs. On vLLM 0.31 it beats hk's paged walk and
  merge (`hk.ops.paged`) at every batch measured (TRITON_DECODE_MAX);
  hk's walk takes decode where this vLLM has no compatible kernel;
* prefill (`hk.ops.paged_prefill`): one launch for every prefill request in
  the batch, cached prefixes and chunks included -- the new tokens' K/V are
  in the cache before either runs.

docs/integration.md is the reference; docs/history/17-the-decode-step-
captured-whole.md how it was measured against vLLM's Triton attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

from . import engine

#: What this backend calls itself in diagnostics.
NAME = "HK_ATTN"

#: The paths a forward takes, counted, and the CUDA-graph contract. A decode
#: step is one cache write and its launches, all on torch's stream, planned on
#: shapes alone: hk's or Triton's by the batch size, the split count from the
#: batch size and max_model_len, the scratch allocated once at its largest,
#: and the metadata check that reads the device only outside capture. "decode"
#: counts hk's decode, "decode_triton" Triton's on hk's cache. The builder claims
#: UNIFORM_SINGLE_TOKEN_DECODE, so vLLM captures decode steps whole and runs
#: steps with prefill in them eagerly, between its pieces; `forward` refuses
#: to have one of those captured.
PATCH = engine.Patch(NAME, paths=("decode", "decode_triton", "prefill"),
                     graph="safe")

#: What `get_name()` answers, which is not the same thing.
#:
#: `vllm/model_executor/layers/attention/attention.py` does
#: `AttentionBackendEnum[self.attn_backend.get_name()]` -- it round-trips the
#: name back through the enum -- so a backend registered as an override has to
#: answer with the *member it overrides*, not with what it is. Returning
#: "HK_ATTN" here fails at model construction with "Unknown attention
#: backend", which is a clear message for a confusing rule.
_SLOT_NAME = "TRITON_ATTN"

#: The enum member to register under.
#:
#: vLLM's `AttentionBackendEnum` is closed -- `register_backend` records an
#: override for an existing member rather than adding one -- and it does
#: reserve `CUSTOM` for third parties. But `RocmPlatform.get_attn_backend_cls`
#: has an allowlist that `CUSTOM` is not on, so on ROCm selecting it fails
#: with "Attention backend CUSTOM is not supported on ROCm" before any of this
#: code runs. What does work is overriding a member the platform already
#: accepts: `get_path()` consults the override table, so a backend registered
#: under `TRITON_ATTN` is what `--attention-backend TRITON_ATTN` resolves to.
#:
#: That is a loaded gun -- `TRITON_ATTN` is also what ROCm picks by default --
#: so registration is its own opt-in (`HK_VLLM_BACKEND=1`) rather than
#: something `HK_PATCH_VLLM=1` does on the way past. Replacing the decoder
#: attention of a serving engine should take saying so.
SLOT = "TRITON_ATTN"

#: The metadata builder class, made once: vLLM asks for it per KV-cache group.
_BUILDER: list = []

#: Decode scratch, one set per (device, kv heads, head_dim) for every layer:
#: layers run one after another on one stream, and a set per layer would be
#: 36 copies of the largest one. Allocated at its largest on first use, so a
#: captured graph's pointers stay valid.
_SCRATCH: dict = {}

#: (kind, head_dim, splits) -> a decode launch as a torch op, on torch's
#: stream: "walk" is the paged walk (which with one split writes the answer
#: itself), "merge" the launch that folds the splits' partials.
_OPS: dict = {}

_UNIT: dict = {}


def _engine_limits():
    """(max_model_len, max_num_seqs) of the engine being built, or vLLM's
    defaults outside one."""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        return (int(cfg.model_config.max_model_len),
                int(cfg.scheduler_config.max_num_seqs))
    except Exception:            # noqa: BLE001 -- no engine: a bench, a test
        return 40960, 256


def _unit_scale(device):
    import torch

    t = _UNIT.get(device)
    if t is None:
        t = _UNIT[device] = torch.ones((), device=device, dtype=torch.float32)
    return t


def _scratch(device, n_kv_heads: int, head_dim: int, rows: int, group: int):
    from ..ops import paged as hk_paged

    key = (device, n_kv_heads, head_dim, group)
    got = _SCRATCH.get(key)
    if got is None or got[0].shape[0] < rows:
        got = _SCRATCH[key] = hk_paged.fused_scratch(
            rows, n_kv_heads, head_dim, device, group=group)
    return got


def decode_splits(n: int, n_kv_heads: int, pages_cap: int) -> int:
    """The decode's split count for `n` requests: the plan
    (hk.ops.paged.plan_splits) rounded to a power of two.

    A function of the batch size and of max_model_len (through `pages_cap`),
    never of the step's max_seq_len: a graph captured on a dummy batch of
    length-1 sequences would keep whatever it planned there for every replay.
    Rounded because each count is its own compile, and unrounded the counts
    a batch passes through on its way from 1 to 256 requests are nineteen;
    every best split measured (hk_paged.TARGET_WORKGROUPS) was a power of two
    anyway."""
    from ..ops import paged as hk_paged

    want = hk_paged.plan_splits(n, n_kv_heads, pages_cap)
    return 1 << round(math.log2(want))


def _op(kind: str, head_dim: int, splits: int):
    op = _OPS.get((kind, head_dim, splits))
    if op is None:
        from ..ops import paged as hk_paged
        from ..runtime.torch_ext import torch_op

        k = (hk_paged.fused_kernel(head_dim, splits) if kind == "walk"
             else hk_paged.merge_kernel(head_dim, splits))
        op = _OPS[(kind, head_dim, splits)] = torch_op(k)
    return op


def _prefill_op(head_dim: int, group: int, scale: float):
    op = _OPS.get(("prefill", head_dim, group, scale))
    if op is None:
        from ..ops import paged_prefill as hk_prefill
        from ..runtime.torch_ext import torch_op

        op = _OPS[("prefill", head_dim, group, scale)] = torch_op(
            hk_prefill.kernel_for(head_dim, group, scale))
    return op


def decode(q, k_cache, v_cache, block_table, seq_lens, out, scratch,
           splits: int) -> None:
    """One decode, as torch ops: the paged walk, and with more than one split
    the merge (hk.ops.paged.decode)."""
    from ..ops import paged as hk_paged

    d = q.shape[-1]
    merge = _op("merge", d, splits) if splits > 1 else None
    hk_paged.decode(_op("walk", d, splits), merge, q, k_cache, v_cache,
                    block_table, seq_lens, out, scratch, splits)


#: Pure-decode batches up to this many requests are run by vLLM's own Triton
#: kernel, reading this backend's cache through a transposed view (it
#: addresses K/V by stride; the output is bit-identical to the stock
#: backend's). That is hk losing a kernel, recorded as such. Measured with
#: every layer's cache cold, as a decode step reads them
#: (tools/hk-bench/paged_ab.py --layers 36, graph clock, ms per layer,
#: Triton on hk's cache / hk's walk and merge). On vLLM 0.16.1, where hk
#: still took 16 requests up:
#:
#:     ctx 1024   n1 0.0174/0.0278   n4 0.0388/0.0509   n8 0.0685/0.0845   n16 0.1293/0.1228
#:     ctx 4096   n1 0.0410/0.0527   n4 0.1109/0.1233   n8 0.2078/0.2219   n16 0.4018/0.3939
#:
#: On vLLM 0.31 (triton-rocm 3.7.1) Triton's 2-D kernel wins at every batch
#: measured, so every decode batch is Triton's:
#:
#:     ctx 1024   n16 0.1201/0.1264   n32 0.1804/0.2166   n64 0.3505/0.3972
#:     ctx 4096   n16 0.3738/0.4001   n32 0.6907/0.7587
#:
#: 0 turns it off, and hk's walk takes every batch.
TRITON_DECODE_MAX = 1 << 16

#: Up to this many requests Triton runs its 3-D kernel (16 KV segments and a
#: reduction); above, its 2-D one. The 3-D kernel's per-segment buffers are
#: sized for this many rows.
TRITON_3D_MAX = 8

#: (device, q heads, head_dim) -> the 3-D kernel's per-segment buffers.
_SEGM: dict = {}

_TRITON: list = []


def _triton_decode():
    """vLLM's unified_attention and the keyword arguments this vLLM's
    version of it takes, or None where it is missing. Checked against the
    signature rather than assumed: the arguments moved between releases."""
    if not _TRITON:
        try:
            import inspect

            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )

            params = set(inspect.signature(unified_attention).parameters)
            _TRITON.append((unified_attention, params))
        except Exception:        # noqa: BLE001 -- no vLLM, or it moved
            _TRITON.append(None)
    return _TRITON[0]


def _segm(device, n_heads: int, head_dim: int):
    import torch

    key = (device, n_heads, head_dim)
    got = _SEGM.get(key)
    if got is None:
        rows, segs = TRITON_3D_MAX, 16
        d = 1 << (head_dim - 1).bit_length()
        got = _SEGM[key] = (
            torch.empty(rows, n_heads, segs, d, device=device,
                        dtype=torch.float32),
            torch.empty(rows, n_heads, segs, device=device,
                        dtype=torch.float32),
            torch.empty(rows, n_heads, segs, device=device,
                        dtype=torch.float32))
    return got


def _kv_scales(layer, device):
    ks = getattr(layer, "_k_scale", None)
    vs = getattr(layer, "_v_scale", None)
    if ks is None or vs is None:
        ks = vs = _unit_scale(device)
    return ks, vs


def triton_decode(q, k_cache, v_cache, block_table, seq_lens, query_start,
                  out, scale: float, max_seq_len: int, k_scale,
                  v_scale) -> bool:
    """One decode of `q.shape[0]` <= TRITON_DECODE_MAX requests by Triton's
    3-D kernel on hk's head-major cache. False if this vLLM has no kernel
    with the arguments it needs, and nothing was launched."""
    got = _triton_decode()
    if got is None:
        return False
    fn, params = got
    n, h, d = q.shape
    o, m, l = _segm(q.device, h, d)
    hkv = k_cache.shape[1]
    kw = dict(q=q, k=k_cache.transpose(1, 2), v=v_cache.transpose(1, 2),
              out=out, cu_seqlens_q=query_start, max_seqlen_q=1,
              seqused_k=seq_lens, max_seqlen_k=max_seq_len,
              softmax_scale=scale,
              causal=True, alibi_slopes=None, window_size=(-1, -1),
              block_table=block_table, softcap=0, q_descale=None,
              k_descale=k_scale.expand((n, hkv)),
              v_descale=v_scale.expand((n, hkv)),
              seq_threshold_3D=TRITON_3D_MAX, num_par_softmax_segments=16,
              softmax_segm_output=o, softmax_segm_max=m,
              softmax_segm_expsum=l)
    optional = dict(use_alibi_sqrt=False, sinks=None, output_scale=None,
                    mm_prefix_range=None)
    if not set(kw) <= params:
        return False
    kw.update({k: v for k, v in optional.items() if k in params})
    fn(**kw)
    return True


def _build():
    """Construct the backend classes against the installed vLLM.

    Done inside a function, and imported lazily, because this module is
    imported by `hk.integration` on a machine that may have no vLLM at all --
    and because vLLM moves these symbols between releases often enough that an
    import error at the top of the file would take the whole package with it.
    """
    import torch
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionType,
        MultipleOf,
    )

    from ..ops import attn as hk_attn
    from ..ops import paged as hk_paged
    from ..ops import paged_prefill as hk_prefill

    class HKAttentionBackend(AttentionBackend):
        accept_output_buffer: bool = True
        supported_dtypes: ClassVar[list] = [torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            # The overridden member's name; see _SLOT_NAME.
            return _SLOT_NAME

        @staticmethod
        def get_impl_cls():
            return HKAttentionImpl

        @staticmethod
        def get_builder_cls():
            # The metadata this needs -- block_table, seq_lens, slot_mapping,
            # query_start_loc -- is exactly what the ROCm backend's builder
            # already assembles, and reimplementing it would be copying a file
            # to change nothing in it. Everything but its CUDA-graph claim.
            if _BUILDER:
                return _BUILDER[0]
            from vllm.v1.attention.backends.rocm_attn import (
                RocmAttentionMetadataBuilder,
            )

            support = type(RocmAttentionMetadataBuilder._cudagraph_support)

            from vllm.v1.attention.backends.utils import (
                split_decodes_and_prefills,
            )

            class HKAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
                # Decode steps are captured whole; a step with prefill in it
                # runs eagerly between vLLM's pieces. See PATCH.
                _cudagraph_support = support.UNIFORM_SINGLE_TOKEN_DECODE

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # Decodes first, so `forward` takes them as one run.
                    init = getattr(self, "_init_reorder_batch_threshold", None)
                    if init is not None:
                        init(1)
                    else:
                        self.reorder_batch_threshold = 1

                def build(self, common_prefix_len, common_attn_metadata,
                          fast_build=False):
                    md = super().build(common_prefix_len, common_attn_metadata,
                                       fast_build)
                    # On the host, where vLLM already has the lengths: how
                    # many leading decodes, and the prefill's work list.
                    nd, n_pf, _, _ = split_decodes_and_prefills(
                        common_attn_metadata, decode_threshold=1)
                    md.hk_num_decodes = nd
                    md.hk_prefill_tiles = None
                    if n_pf:
                        qs = common_attn_metadata.query_start_loc_cpu.tolist()
                        md.hk_prefill_tiles = hk_prefill.work_list(
                            [(r, qs[r + 1] - qs[r]) for r in range(nd, nd + n_pf)],
                            self.device)
                    return md

            _BUILDER.append(HKAttentionMetadataBuilder)
            return HKAttentionMetadataBuilder

        @staticmethod
        def get_supported_kernel_block_sizes():
            # The page size *is* the kernel's KV block, so there is exactly one.
            return [hk_paged.KV_BLOCK]

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return list(hk_attn.HEAD_DIMS)

        @staticmethod
        def get_kv_cache_shape(num_blocks: int, block_size: int,
                               num_kv_heads: int, head_size: int,
                               cache_dtype_str: str = "auto") -> tuple:
            if block_size != hk_paged.KV_BLOCK:
                raise ValueError(
                    f"{NAME} needs block_size {hk_paged.KV_BLOCK}, got "
                    f"{block_size}: the page size is the kernel's KV block, "
                    f"not a tiling choice."
                )
            # Head-major, which is the whole reason this backend exists. See
            # the module docstring.
            return (2, num_blocks, num_kv_heads, block_size, head_size)

        @classmethod
        def supported_kv_cache_layouts(cls):
            """vLLM 0.31 and later lay the cache out themselves -- logically
            [layers, blocks, heads, tokens, K|V], in the physical order a
            backend declares here -- and ignore get_kv_cache_shape. LBHNC is
            this backend's head-major page, one layer's [B, H, N, 2C] block
            contiguous, so K and V are each a slice of the last axis (a row
            stride of 2C) that a gl takes as it is. vLLM's default, LBNHC,
            puts heads inside tokens, which no gl can address."""
            from vllm.v1.kv_cache_layout import KVCacheLayout

            return [KVCacheLayout.LBHNC]

        @classmethod
        def validate_head_size(cls, head_size: int) -> None:
            # Defined here rather than inherited: the base class grew this in
            # a later release than the one this was first run against, and a
            # backend that silently accepts a head_dim it has no kernel for
            # fails at launch time with a shape error instead of here with a
            # sentence.
            if head_size not in cls.get_supported_head_sizes():
                raise ValueError(
                    f"{NAME} has kernels for head_size "
                    f"{cls.get_supported_head_sizes()}, not {head_size}. "
                    f"Instantiate one in hk.ops.attn and hk.ops.paged, or run "
                    f"this model on TRITON_ATTN."
                )

        @staticmethod
        def use_cascade_attention(*args, **kwargs) -> bool:
            return False

    class HKAttentionImpl(AttentionImpl):
        def __init__(self, num_heads: int, head_size: int, scale: float,
                     num_kv_heads: int, alibi_slopes, sliding_window,
                     kv_cache_dtype: str, logits_soft_cap=None,
                     attn_type=None, kv_sharing_target_layer_name=None,
                     sinks=None, **kw) -> None:
            unsupported = {
                "alibi_slopes": alibi_slopes,
                "sliding_window": sliding_window,
                "sinks": sinks,
            }
            bad = [k for k, v in unsupported.items() if v]
            if bad:
                raise NotImplementedError(
                    f"{NAME} does not implement {', '.join(bad)}. The kernel "
                    f"is plain causal attention; anything that changes the "
                    f"mask needs the mask written, not a flag threaded."
                )
            if logits_soft_cap:
                raise NotImplementedError(f"{NAME} has no logits_soft_cap")
            if not kv_cache_dtype.startswith("auto"):
                raise NotImplementedError(
                    f"{NAME} keeps the cache in the model dtype, not "
                    f"{kv_cache_dtype}"
                )
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = float(scale)
            self.num_kv_heads = num_kv_heads
            self.group = num_heads // num_kv_heads
            self._checked = False
            HKAttentionBackend.validate_head_size(head_size)
            # What the decode plan is a function of, read once (see
            # decode_splits), and the scratch sized for its largest batch.
            max_len, max_seqs = _engine_limits()
            self._pages_cap = -(-max_len // hk_paged.KV_BLOCK)
            self._splits = {}
            self._rows = max(
                max_seqs,
                max(m * self._plan(m) for m in range(1, max_seqs + 1)))

        def _plan(self, n: int) -> int:
            s = self._splits.get(n)
            if s is None:
                s = self._splits[n] = decode_splits(n, self.num_kv_heads,
                                                    self._pages_cap)
            return s

        # -- the cache -----------------------------------------------------

        def _update_cache(self, layer, key, value, k_cache, v_cache,
                          slot_mapping):
            """vLLM's own cache write, into this backend's head-major pages.

            reshape_and_cache_flash takes its block, page and head strides
            from the tensor, so a `(blocks, block, heads, d)` view of
            head-major storage is a layout it writes correctly --
            tools/hk-bench/paged_ab.py checks it bit for bit against a torch
            scatter. One launch, and it skips slot -1, which is what a CUDA
            graph's padding tokens carry: the scatter this replaced was six
            launches and sent them to the last slot of the last block.
            """
            from vllm import _custom_ops as ops

            ks, vs = _kv_scales(layer, key.device)
            ops.reshape_and_cache_flash(
                key, value, k_cache.transpose(1, 2), v_cache.transpose(1, 2),
                slot_mapping, "auto", ks, vs)

        # -- the launch ----------------------------------------------------

        def _kv(self, kv_cache):
            """(k_cache, v_cache), each (blocks, kv_heads, block, head_dim).
            Up to vLLM 0.30 the layer's cache is get_kv_cache_shape's
            (2, ...); from 0.31 it is [B, H, N, 2C], K and V side by side
            in the last axis -- two strided views of one tensor."""
            if kv_cache.dim() == 5:
                return kv_cache[0], kv_cache[1]
            d = self.head_size
            return kv_cache[..., :d], kv_cache[..., d:2 * d]

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            if attn_metadata is None:                 # profiling run
                return output.fill_(0)
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(f"{NAME} has no fused output quant")

            n = attn_metadata.num_actual_tokens
            k_cache, v_cache = self._kv(kv_cache)
            self._update_cache(layer, key[:n], value[:n], k_cache, v_cache,
                               attn_metadata.slot_mapping[:n])

            # The common case, and the one that must not touch the host.
            # `max_query_len` is already a Python int in the metadata, so a
            # pure-decode step can be recognised without reading a device
            # tensor -- and with one token per request, `query` is already in
            # request order, so there is no gather either. The first version
            # of this called .tolist() on two tensors unconditionally, which
            # is a device sync per layer per step: 28 of them on this model,
            # in front of a kernel that takes microseconds.
            if attn_metadata.max_query_len == 1:
                self._decode_all(query, output, k_cache, v_cache,
                                 attn_metadata, n, layer)
                return output
            if engine.capturing():
                raise engine.GraphCaptureError(
                    f"{NAME} is being captured on a batch with prefill in it. "
                    f"It claims UNIFORM_SINGLE_TOKEN_DECODE: a prefill step "
                    f"reads its query lengths on the host, which a replay "
                    f"would not repeat.")

            # A step with prefill in it. The builder ordered the batch decodes
            # first and counted them on the host, so the decodes are the
            # first `nd` tokens and go to the decode; the rest are one paged
            # prefill launch over the builder's work list -- chunks and cached
            # prefixes included, since every new token's K/V is in the cache
            # by now. A batch vLLM did not reorder is still right: whatever is
            # not a leading decode is prefilled, a lone token as a short chunk.
            nd = getattr(attn_metadata, "hk_num_decodes", 0)
            if nd:
                self._decode_all(query, output, k_cache, v_cache,
                                 attn_metadata, nd, layer)
            tiles = getattr(attn_metadata, "hk_prefill_tiles", None)
            if tiles is not None:
                self._prefill(query, output, k_cache, v_cache, attn_metadata,
                              n, tiles)
                PATCH.count("prefill")
            return output

        def _prefill(self, query, output, k_cache, v_cache, attn_metadata, n,
                     tiles):
            """Every prefill request in the batch, one launch: the engine's
            rows as (tokens, heads * head_dim), which a q split off a fused
            qkv projection is too, strided by the projection's width."""
            h, d = self.num_heads, self.head_size
            if not self._checked:
                self._check(attn_metadata.block_table, attn_metadata.seq_lens)
            _prefill_op(d, self.group, self.scale)(
                query[:n].view(n, h * d), k_cache, v_cache,
                attn_metadata.block_table, attn_metadata.seq_lens,
                attn_metadata.query_start_loc, tiles,
                output[:n].view(n, h * d))

        def _decode_all(self, query, output, k_cache, v_cache, attn_metadata,
                        n, layer=None):
            """Every request has exactly one token: the walk and its merge,
            which read the query and write the answer where vLLM keeps them
            -- or, for up to TRITON_DECODE_MAX requests, Triton's kernel on
            the same cache. No copy, no sync, nothing allocated past the
            first call."""
            if n <= TRITON_DECODE_MAX and triton_decode(
                    query[:n], k_cache, v_cache,
                    attn_metadata.block_table[:n], attn_metadata.seq_lens[:n],
                    attn_metadata.query_start_loc[:n + 1], output[:n],
                    self.scale, attn_metadata.max_seq_len,
                    *_kv_scales(layer, query.device)):
                PATCH.count("decode_triton")
                return
            g, hkv, d = self.group, self.num_kv_heads, self.head_size
            q = query[:n].view(n, hkv, g, d)
            if not q.is_contiguous():
                # A q split off a fused qkv projection is strided by the
                # projection's width, a second stride a gl does not carry.
                q = q.contiguous()
            out = output[:n].view(n, hkv, g, d)
            table = attn_metadata.block_table[:n]
            lens = attn_metadata.seq_lens[:n]
            if not self._checked and not engine.capturing():
                self._check(table, lens)
            decode(q, k_cache, v_cache, table, lens, out,
                   _scratch(query.device, hkv, d, self._rows, g),
                   self._plan(n))
            PATCH.count("decode")

        def _check(self, table, lens):
            """Once per layer object, never under capture: it reads the device.

            The kernel reads these two as int32 and indexes memory with the
            result, so a dtype or a stride that is not what it assumes is not
            a wrong answer, it is a page fault several kernels later."""
            import torch

            self._checked = True
            for name, t in (("block_table", table), ("seq_lens", lens)):
                if t.dtype != torch.int32 or not t.is_contiguous():
                    raise TypeError(
                        f"{NAME}: {name} is {t.dtype} "
                        f"{'contiguous' if t.is_contiguous() else 'strided'}"
                        f" {tuple(t.shape)}; the kernel reads it as "
                        f"contiguous int32.")
            mx = int(lens.max())
            cols = table.shape[-1]
            if -(-mx // hk_paged.KV_BLOCK) > cols:
                raise ValueError(
                    f"{NAME}: seq_len {mx} needs "
                    f"{-(-mx // hk_paged.KV_BLOCK)} pages but block_table "
                    f"has {cols} columns.")

    return HKAttentionBackend, HKAttentionImpl


def register(slot: Optional[str] = None) -> str:
    """Register the backend with vLLM. Idempotent."""
    import os

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    want = slot or os.environ.get("HK_VLLM_BACKEND_SLOT", SLOT)
    globals()["_SLOT_NAME"] = want
    backend, impl = _build()
    globals()["HKAttentionBackend"] = backend
    globals()["HKAttentionImpl"] = impl

    member = getattr(AttentionBackendEnum, want, None)
    if member is None:
        raise RuntimeError(
            f"this vLLM has no AttentionBackendEnum.{want}; the members it "
            f"has are {sorted(m.name for m in AttentionBackendEnum)}. Set "
            f"HK_VLLM_BACKEND_SLOT to one of them."
        )
    register_backend(member, f"{__name__}.HKAttentionBackend")
    return f"{NAME} registered over {want} (--attention-backend {want})"
