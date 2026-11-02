"""vLLM's 4-bit MoE GEMMs (AWQ through moe_wna16) on hk's kernel (`HK_VLLM_MOE_WNA16=1`).

On ROCm an AWQ checkpoint's routed experts run `TritonWNA16Experts` (vLLM's moe_wna16): the batch's
(token, choice) pairs sorted by expert, then each of the layer's two GEMMs one call of
`invoke_fused_moe_wna16_triton_kernel` on 4-bit codes uint8 [E, N, K/2], scales [E, N, K/group] and
zero codes uint8 [E, N/2, K/group]. Rebinding that one name -- by identity, in every vLLM module that
imported it -- puts hk's GEMM (`hk.ops.moe.moe_gemm_w4`: the MoE GEMM with `hk.load_w4(zp=,
rows=True)`) under every such layer, on vLLM's own tensors: no repack, no second copy of the experts,
and vLLM's sort, activation and top-k sum as they are.

The stand-in takes a call at up to `MAX_TOKENS` tokens a step, with zero codes (AWQ, or GPTQ with
zero points), a 16-bit activation of the scales' type, a K the schedule divides and an (experts, N,
K, group) in `VALIDATED` (any, with HK_VLLM_MOE_WNA16=force); at those batches the config patch
below asks vLLM to sort into 16-row blocks. Everything else -- prefill chunks,
symmetric GPTQ without zero points, int8 weights -- goes back to vLLM's kernel, counted by reason
in `REFUSED`. The MoE layer is a custom op the compiler does not trace into, so the stand-in runs on
every call, captured or not.

    HK_VLLM_MOE_WNA16=1 vllm serve Qwen3.6-35B-A3B-AWQ ... --dtype float16
    HK_VLLM_MOE_WNA16=force vllm serve <another AWQ MoE checkpoint> --dtype float16   # unvalidated
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Optional, Tuple

from . import _common, engine

PREFIXES = ("vllm",)

#: Up to this many tokens a step, vLLM is asked to sort into 16-row blocks and hk's GEMM takes the
#: call. Measured on Qwen3.6-35B-A3B-AWQ's shapes (E 256, top 8, w13 1024 x 2048, w2 2048 x 512,
#: group 32) on a W7900D, real captured routing, cold weights, same process as vLLM's kernel with
#: the tuned config: 2.6x at 1 token, 3.2x at 4, 2.9x at 8, 2.6x at 16.
MAX_TOKENS = 32

#: The (experts, N, K, group) the route was validated on -- Qwen3.6-35B-A3B-AWQ's two expert GEMMs
#: (w13 and w2), its e2e gate. Any other is refused (counted "not validated") unless
#: HK_VLLM_MOE_WNA16=force, so turning the route on cannot reach a model it was not measured on.
VALIDATED = frozenset({(256, 1024, 2048, 32), (256, 2048, 512, 32)})

_ORIG: Optional[Callable] = None
_FORCE = False
_ORIG_CONFIG: Optional[Callable] = None

PATCH = engine.Patch("hk_vllm::moe_wna16", paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats

#: Calls left on vLLM's kernel, by reason (Python-side, so eager and capture only).
REFUSED: Counter = Counter()


def schedule(k: int) -> Optional[Tuple[int, int]]:
    """(slices, warps) for depth K: eight 16-deep slices a load and two loads a warp -- the
    measured best at both of Qwen3.6's depths (w13 K 2048: 8 warps; w2 K 512: 2), against 4 / 16
    slices and 2 / 4 / 16 warps -- or None where K does not split so."""
    if k < 256:
        return None
    warps = 1 << (min(16, k // 256).bit_length() - 1)      # a power of two, at most 16
    while warps > 1 and k % (128 * warps):
        warps //= 2
    return (8, warps) if warps >= 1 and k % (128 * warps) == 0 else None


def _why_not(A, B, C, B_scale, B_zp, tokens, config, use_int8_w8a16, use_int4_w4a16,
             block_shape) -> Optional[str]:
    import torch  # noqa: PLC0415

    if not use_int4_w4a16 or use_int8_w8a16:
        return "not int4"
    if B_zp is None:
        return "no zero points"
    if not block_shape or block_shape[0] != 0 or block_shape[1] % 16:
        return "block shape"
    if tokens > MAX_TOKENS:
        return "tokens"
    if config.get("BLOCK_SIZE_M") != 16:
        return "block_m not 16"
    if not A.is_cuda or A.dtype not in (torch.float16, torch.bfloat16) or B_scale.dtype != A.dtype:
        return "dtype"
    if B.dtype != torch.uint8 or B_zp.dtype != torch.uint8 or B.dim() != 3:
        return "weight layout"
    if not (B.is_contiguous() and B_scale.is_contiguous() and B_zp.is_contiguous()
            and C.is_contiguous()):
        return "not contiguous"
    if schedule(A.shape[1]) is None:
        return "depth"
    if not _FORCE and (B.shape[0], B.shape[1], A.shape[1], block_shape[1]) not in VALIDATED:
        return "not validated"
    return None


def invoke_fused_moe_wna16_triton_kernel(A, B, C, B_scale, B_zp, topk_weights,
                                         sorted_token_ids, expert_ids,
                                         num_tokens_post_padded, mul_routed_weight, top_k,
                                         config, compute_type, use_int8_w8a16, use_int4_w4a16,
                                         block_shape):
    """Stands in for vLLM's: hk's GEMM where it takes the call, vLLM's otherwise. Same arguments,
    same in-place write of C."""
    from ..ops import moe  # noqa: PLC0415

    # C is (tokens, top_k, N) for both GEMMs of the layer; A is (tokens, K) for the first and
    # (tokens * top_k, K) for the second, which vLLM calls with top_k 1.
    tokens = C.shape[0] if C.dim() == 3 else A.shape[0]
    why = _why_not(A, B, C, B_scale, B_zp, tokens, config, use_int8_w8a16, use_int4_w4a16,
                   block_shape)
    if why is None:
        e, n, kb = B.shape
        group = block_shape[1]
        k = A.shape[1]
        sl, wp = schedule(k)
        qw = B.view(_int32()).reshape(e * n, kb // 4)
        s = B_scale.reshape(e * n, k // group)
        zp = B_zp.view(_int8()).reshape(e * n // 2, k // group)
        if moe.takes_w4(A, qw, s, zp, n, 16, group, wp, sl):
            moe.moe_gemm_w4(A, qw, s, zp, C, sorted_token_ids, expert_ids, num_tokens_post_padded,
                            top_k=top_k, block_m=16, group=group, warps=wp, slices=sl,
                            topk_weights=topk_weights if mul_routed_weight else None)
            PATCH.count("hk")
            return None
        why = "shape"
    if not engine.capturing():
        REFUSED[why] += 1
    PATCH.count("fallback")
    return _ORIG(A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids, expert_ids,
                 num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
                 use_int8_w8a16, use_int4_w4a16, block_shape)


def _int32():
    import torch  # noqa: PLC0415

    return torch.int32


def _int8():
    import torch  # noqa: PLC0415

    return torch.int8


_GUARDED = PATCH.guard(invoke_fused_moe_wna16_triton_kernel)


def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
    """vLLM's config, with BLOCK_SIZE_M 16 for a 4-bit layer at a batch hk's GEMM takes: the sort
    and the GEMM both read it, so changing it here changes both together."""
    config = _ORIG_CONFIG(w1_shape, w2_shape, top_k, dtype, M, block_shape)
    if (dtype == "int4_w4a16" and block_shape and block_shape[0] == 0 and M <= MAX_TOKENS
            and schedule(w1_shape[2] * 2) is not None and schedule(w2_shape[2] * 2) is not None):
        config = dict(config, BLOCK_SIZE_M=16)
    return config


def apply() -> str:
    """Route vLLM's 4-bit MoE GEMMs. Returns what happened, in one line."""
    global _ORIG, _ORIG_CONFIG, _FORCE
    if not _common.importable("vllm"):
        return "vllm is not importable; MoE WNA16 not patched"
    import os  # noqa: PLC0415

    _FORCE = os.environ.get("HK_VLLM_MOE_WNA16", "").strip().lower() == "force"
    if _ORIG is not None:
        return "moe_wna16: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "moe_wna16: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.fused_moe import fused_moe as fm  # noqa: PLC0415

    _ORIG = fm.invoke_fused_moe_wna16_triton_kernel
    _ORIG_CONFIG = fm.try_get_optimal_moe_config
    names = _common.rebind(PREFIXES, _ORIG, _GUARDED)
    cfg = _common.rebind(PREFIXES, _ORIG_CONFIG, try_get_optimal_moe_config)
    return (f"moe_wna16 -> hk where hk.ops.moe.moe_gemm_w4 takes the GEMM ({len(names)} kernel "
            f"and {len(cfg)} config bindings rebound; 16-row blocks up to {MAX_TOKENS} tokens; "
            + ("any shape (force)" if _FORCE else f"{len(VALIDATED)} validated shapes") + ")")


def revert() -> str:
    global _ORIG, _ORIG_CONFIG
    if _ORIG is None:
        return "moe_wna16: not patched"
    names = _common.rebind(PREFIXES, _GUARDED, _ORIG)
    _common.rebind(PREFIXES, try_get_optimal_moe_config, _ORIG_CONFIG)
    _ORIG = _ORIG_CONFIG = None
    return f"moe_wna16 -> vllm ({len(names)} rebound)"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary() + (
        f"; refused {dict(REFUSED)}" if REFUSED else "")


def stats() -> Dict[str, int]:
    return dict(STATS)
