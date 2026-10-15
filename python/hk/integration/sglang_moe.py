"""SGLang's fused-MoE layers on gfx1100: routing that runs, and hk's GEMM.

Two rebindings, each by identity in every SGLang module that imported the
name, so no model file is touched:

* **`topk_softmax`, the routing.** On gfx1100 sgl_kernel's kernel fails on
  its own, in a process that ran nothing else, with `unspecified launch
  failure`. That is what stops every softmax-routed MoE model in SGLang on
  RDNA3, before any expert GEMM runs. The stand-in is the same softmax and
  top-k in torch, which is what SGLang's own `fused_topk_torch_native`
  computes. `apply_topk()` makes this change alone.
* **`invoke_fused_moe_kernel`, the expert GEMMs.** SGLang sorts the batch's
  (token, choice) pairs by expert as vLLM does, with the same sentinel, so
  `hk.ops.moe.moe_gemm` takes the call where `takes` does: bf16, no scales,
  zero points or bias, none of SGLang's fused epilogues, and 16-row blocks.
  `try_get_optimal_moe_config` asks for 16-row blocks up to
  `BLOCK16_MAX_TOKENS` tokens a step, for the up and the down GEMM together,
  since they share one sort. Every other call goes to SGLang's Triton
  kernel, and the counters say how many did.

    HK_SGLANG_MOE=1 python -m sglang.launch_server \\
        --model-path allenai/OLMoE-1B-7B-0125-Instruct \\
        --attention-backend triton --disable-cuda-graph

`HK_SGLANG_MOE=topk` makes only the routing change.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import _common, engine

PREFIXES = ("sglang",)

#: Up to this many tokens a step, the patch asks SGLang to sort into 16-row
#: blocks, the size hk's GEMM streams weights fastest at; above it SGLang's
#: own config stands. Measured by tools/hk-bench/sglang_moe_ab.py.
BLOCK16_MAX_TOKENS = 256

_ORIG: Optional[Callable] = None
_ORIG_CONFIG: Optional[Callable] = None
_ORIG_TOPK: Optional[Callable] = None

PATCH = engine.Patch("hk_sglang::moe", paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False,
                 moe_softcapping=0.0, correction_bias=None):
    """Stands in for sgl_kernel's: the same outputs, written in place."""
    import torch  # noqa: PLC0415

    logits = gating_output.float()
    if moe_softcapping:
        logits = moe_softcapping * torch.tanh(logits / moe_softcapping)
    scores = torch.softmax(logits, dim=-1)
    choice = scores if correction_bias is None else scores + correction_bias
    ids = choice.topk(topk_weights.shape[-1], dim=-1).indices
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    topk_weights.copy_(weights)
    topk_ids.copy_(ids)


def _plain(bias, A_scale, B_scale, B_zp, block_shape, add_output_mask, *flags) -> bool:
    return (bias is None and A_scale is None and B_scale is None and B_zp is None
            and block_shape is None and add_output_mask is None and not any(flags))


def invoke_fused_moe_kernel(A, B, bias, C, A_scale, B_scale, B_zp, topk_weights,
                            topk_ids, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, mul_routed_weight, top_k, config,
                            compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
                            use_int4_w4a16, per_channel_quant, block_shape=None,
                            no_combine=False, a_use_tma=False, b_use_tma=False,
                            c_sorted=False, filter_expert=True, fuse_sum_all_reduce=False,
                            router_topk=1, fuse_add_to_output=False, add_output_mask=None,
                            mask_output=False, lora_preserve_base=False, fuse_swiglu=False):
    """Stands in for SGLang's: hk's GEMM where it takes the call, SGLang's
    otherwise. Same arguments, same in-place write of C."""
    from ..ops import moe  # noqa: PLC0415

    bm = config.get("BLOCK_SIZE_M", 0)
    if (bm == 16 and A.is_cuda
            and _plain(bias, A_scale, B_scale, B_zp, block_shape, add_output_mask,
                       use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
                       no_combine, a_use_tma, b_use_tma, c_sorted, fuse_sum_all_reduce,
                       fuse_add_to_output, mask_output, lora_preserve_base, fuse_swiglu)
            and moe.takes(A, B, bm)):
        moe._launch(A, B, C, sorted_token_ids, expert_ids, num_tokens_post_padded,
                    top_k=top_k, block_m=bm,
                    topk_weights=topk_weights if mul_routed_weight else None)
        PATCH.count("hk")
        return None
    PATCH.count("fallback")
    return _ORIG(A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids,
                 sorted_token_ids, expert_ids, num_tokens_post_padded,
                 mul_routed_weight, top_k, config, compute_type, use_fp8_w8a8,
                 use_int8_w8a8, use_int8_w8a16, use_int4_w4a16, per_channel_quant,
                 block_shape=block_shape, no_combine=no_combine, a_use_tma=a_use_tma,
                 b_use_tma=b_use_tma, c_sorted=c_sorted, filter_expert=filter_expert,
                 fuse_sum_all_reduce=fuse_sum_all_reduce, router_topk=router_topk,
                 fuse_add_to_output=fuse_add_to_output, add_output_mask=add_output_mask,
                 mask_output=mask_output, lora_preserve_base=lora_preserve_base,
                 fuse_swiglu=fuse_swiglu)


_GUARDED = PATCH.guard(invoke_fused_moe_kernel)


def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, is_marlin=False,
                               block_shape=None, per_channel_quant=False,
                               return_down_config=False):
    """SGLang's config, with BLOCK_SIZE_M 16 for an unquantized layer at a
    batch hk's GEMM takes, in the up and the down config alike: both GEMMs
    read the one sort."""
    out = _ORIG_CONFIG(w1_shape, w2_shape, top_k, dtype, M, is_marlin=is_marlin,
                       block_shape=block_shape, per_channel_quant=per_channel_quant,
                       return_down_config=return_down_config)
    from ..ops import moe  # noqa: PLC0415

    if not (dtype is None and block_shape is None and not is_marlin
            and M <= BLOCK16_MAX_TOKENS
            and moe.schedule_for(16, w1_shape[2]) is not None
            and moe.schedule_for(16, w2_shape[2]) is not None):
        return out
    if return_down_config:
        config, (down_config, max_block_m) = out
        if down_config is not None:
            down_config = dict(down_config, BLOCK_SIZE_M=16)
        return dict(config, BLOCK_SIZE_M=16), (down_config, max_block_m)
    return dict(out, BLOCK_SIZE_M=16)


def _homes():
    """The defining modules, after importing every module that binds the
    names: a rebind reaches only modules already imported, and the plugin
    runs before the model -- and so the MoE runner -- is built."""
    from sglang.kernels.ops.moe import fused_moe_triton_kernels as kernels  # noqa: PLC0415
    from sglang.srt.layers.moe import topk  # noqa: PLC0415
    from sglang.srt.layers.moe.moe_runner import triton as _runner  # noqa: F401,PLC0415
    from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: PLC0415
        fused_moe as _fused_moe,  # noqa: F401
        fused_moe_triton_config as config,
    )

    return topk, kernels, config


def apply_topk() -> str:
    """Route SGLang's softmax top-k through torch. Returns what happened."""
    global _ORIG_TOPK
    if not _common.importable("sglang"):
        return "sglang is not importable; routing not patched"
    if _ORIG_TOPK is not None:
        return "topk: already patched"
    topk, _, _ = _homes()
    _ORIG_TOPK = topk.topk_softmax
    names = _common.rebind(PREFIXES, _ORIG_TOPK, topk_softmax)
    return f"topk_softmax -> torch ({len(names)} bindings rebound)"


def apply() -> str:
    """The routing, and SGLang's MoE GEMMs on hk where it takes them."""
    global _ORIG, _ORIG_CONFIG
    if not _common.importable("sglang"):
        return "sglang is not importable; MoE not patched"
    notes = [apply_topk()]
    if _ORIG is not None:
        return "; ".join(notes + ["moe: already patched"])
    _, kernels, config = _homes()
    _ORIG = kernels.invoke_fused_moe_kernel
    _ORIG_CONFIG = config.try_get_optimal_moe_config
    names = _common.rebind(PREFIXES, _ORIG, _GUARDED)
    cfg = _common.rebind(PREFIXES, _ORIG_CONFIG, try_get_optimal_moe_config)
    notes.append(f"moe -> hk where hk.ops.moe takes the GEMM ({len(names)} kernel and "
                 f"{len(cfg)} config bindings rebound; 16-row blocks up to "
                 f"{BLOCK16_MAX_TOKENS} tokens)")
    return "; ".join(notes)


def revert() -> str:
    global _ORIG, _ORIG_CONFIG, _ORIG_TOPK
    out = []
    if _ORIG is not None:
        names = _common.rebind(PREFIXES, _GUARDED, _ORIG)
        _common.rebind(PREFIXES, try_get_optimal_moe_config, _ORIG_CONFIG)
        _ORIG = _ORIG_CONFIG = None
        out.append(f"moe -> sglang ({len(names)} rebound)")
    if _ORIG_TOPK is not None:
        _common.rebind(PREFIXES, topk_softmax, _ORIG_TOPK)
        _ORIG_TOPK = None
        out.append("topk_softmax -> sgl_kernel")
    return "; ".join(out) or "moe: not patched"


def status() -> str:
    return "not patched" if _ORIG is None else PATCH.summary()
