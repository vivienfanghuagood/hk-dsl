"""vLLM's fused-MoE GEMMs on hk's kernel (`HK_VLLM_MOE=1`).

vLLM's unquantized MoE runs `TritonExperts`: it sorts the batch's
(token, choice) pairs by expert, then makes each of the layer's two GEMMs one
call of `invoke_fused_moe_triton_kernel` on those lists. Rebinding that one
name -- by identity, in every vLLM module that imported it -- routes every
bf16 MoE layer without touching a model file, and keeps vLLM's sort, its
activation and its sum over the top-k as they are.

The stand-in takes what `hk.ops.moe.takes` takes: bf16, no scales, no bias, a
BLOCK_SIZE_M with a schedule. Everything else -- fp8, int8, int4, a biased
expert, a 128-row block -- goes back to vLLM's kernel, and the counters say
how many did. The MoE layer is a custom op the compiler does not trace into,
so the stand-in runs on every call, captured or not, and the compile cache
has nothing of it to keep.

    HK_PATCH_VLLM=1 HK_VLLM_MOE=1 vllm serve allenai/OLMoE-1B-7B-0125-Instruct
"""

from __future__ import annotations

from typing import Callable, Optional

from . import _common, engine

PREFIXES = ("vllm",)

#: Up to this many tokens a step, the patch asks vLLM to sort into 16-row
#: blocks, the size hk's GEMM streams weights fastest at; above it vLLM's own
#: config stands. Measured by tools/hk-bench/moe_ab.py.
BLOCK16_MAX_TOKENS = 256

#: An arch's own limit, past which the call stays vLLM's. gfx1201's is 2 tokens,
#: where vLLM skips the sort (tokens * top_k * 4 <= experts) and gives each
#: (token, choice) pair a block of its own: there hk's GEMM streams the few chosen
#: experts faster than Triton's (OLMoE decode at one request 1.12x, Gate C in the
#: R9700 vLLM image). With the sort, a model's real routing puts several tokens
#: on each chosen expert and hk's 16-row blocks lose (decode 0.96x at 4 requests,
#: 0.83x at 8), though tools/hk-bench/moe_ab.py's uniform routing has hk ahead to
#: 8 tokens; from 16 both kernels stream at ~96% of GDDR6.
BLOCK16_MAX_TOKENS_AT = {"gfx1201": 2}


def block16_max() -> int:
    """This arch's token limit for hk's MoE GEMM (BLOCK16_MAX_TOKENS_AT)."""
    from ..target import current_arch  # noqa: PLC0415

    return BLOCK16_MAX_TOKENS_AT.get(current_arch(), BLOCK16_MAX_TOKENS)

_ORIG: Optional[Callable] = None
_ORIG_CONFIG: Optional[Callable] = None

PATCH = engine.Patch("hk_vllm::moe", paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def _plain(A_scale, B_scale, B_bias, block_shape, *quant_flags) -> bool:
    return not any(quant_flags) and A_scale is None and B_scale is None \
        and B_bias is None and block_shape is None


def invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights,
                                   sorted_token_ids, expert_ids,
                                   num_tokens_post_padded, mul_routed_weight,
                                   top_k, config, compute_type, use_fp8_w8a8,
                                   use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                                   per_channel_quant, block_shape=None,
                                   B_bias=None):
    """Stands in for vLLM's: hk's GEMM where it takes the call, vLLM's
    otherwise. Same arguments, same in-place write of C."""
    from ..ops import moe  # noqa: PLC0415

    # 16-row blocks only: the batches `try_get_optimal_moe_config` below
    # sent to 16 rows. At vLLM's 64- and 128-row blocks the GEMM loses.
    bm = config.get("BLOCK_SIZE_M", 0)
    # C is (tokens, top_k, N) for both GEMMs of the layer.
    tokens = C.shape[0] if C.dim() == 3 else A.shape[0]
    if (bm == 16 and A.is_cuda and tokens <= block16_max()
            and _plain(A_scale, B_scale, B_bias, block_shape, use_fp8_w8a8,
                       use_int8_w8a8, use_int8_w8a16, use_int4_w4a16)
            and moe.takes(A, B, bm)):
        moe._launch(A, B, C, sorted_token_ids, expert_ids, num_tokens_post_padded,
                    top_k=top_k, block_m=bm,
                    topk_weights=topk_weights if mul_routed_weight else None)
        PATCH.count("hk")
        return None
    PATCH.count("fallback")
    return _ORIG(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids,
                 expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,
                 config, compute_type, use_fp8_w8a8, use_int8_w8a8,
                 use_int8_w8a16, use_int4_w4a16, per_channel_quant,
                 block_shape, B_bias)


_GUARDED = PATCH.guard(invoke_fused_moe_triton_kernel)


def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M,
                               block_shape=None):
    """vLLM's config, with BLOCK_SIZE_M 16 for an unquantized layer at a
    batch hk's GEMM takes: the sort and the GEMM both read it, so changing it
    here changes both together."""
    config = _ORIG_CONFIG(w1_shape, w2_shape, top_k, dtype, M, block_shape)
    from ..ops import moe  # noqa: PLC0415

    if (dtype is None and block_shape is None and M <= block16_max()
            and moe.schedule_for(16, w1_shape[2]) is not None
            and moe.schedule_for(16, w2_shape[2]) is not None):
        config = dict(config, BLOCK_SIZE_M=16)
    return config


def apply() -> str:
    """Route vLLM's MoE GEMMs. Returns what happened, in one line."""
    global _ORIG, _ORIG_CONFIG
    if not _common.importable("vllm"):
        return "vllm is not importable; MoE not patched"
    if _ORIG is not None:
        return "moe: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "moe: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.fused_moe import fused_moe as fm  # noqa: PLC0415

    _ORIG = fm.invoke_fused_moe_triton_kernel
    _ORIG_CONFIG = fm.try_get_optimal_moe_config
    names = _common.rebind(PREFIXES, _ORIG, _GUARDED)
    cfg = _common.rebind(PREFIXES, _ORIG_CONFIG, try_get_optimal_moe_config)
    return (f"moe -> hk where hk.ops.moe takes the GEMM ({len(names)} kernel and "
            f"{len(cfg)} config bindings rebound; 16-row blocks up to "
            f"{block16_max()} tokens)")


def revert() -> str:
    global _ORIG, _ORIG_CONFIG
    if _ORIG is None:
        return "moe: not patched"
    names = _common.rebind(PREFIXES, _GUARDED, _ORIG)
    _common.rebind(PREFIXES, try_get_optimal_moe_config, _ORIG_CONFIG)
    _ORIG = _ORIG_CONFIG = None
    return f"moe -> vllm ({len(names)} rebound)"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary()
