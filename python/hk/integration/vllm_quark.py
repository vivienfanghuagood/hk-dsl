"""vLLM's AMD Quark W4A16 linear layers on hk's 4-bit forms (`HK_VLLM_QUARK`).

A Quark int4 weight-only checkpoint (`quant_method: quark`, weight int4 per group, no activation
quantization; e.g. amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16) loads through vLLM's `QuarkW4A16Int4`
scheme. Quark stores AutoAWQ's tensor layout -- `weight` int32 [K, N/8], `weight_scale` [K/group,
N], `weight_zero_point` int32 [K/group, N/8] -- with its own nibble order (`pack_method`) and, for
a symmetric checkpoint, signed codes. The scheme's `process_weights_after_loading` canonicalizes
that to AWQ's codes, converts them to the standard (GPTQ) format and hands the layer to the
MPLinearKernel vLLM chose for a uint4b8 (symmetric) or uint4 (zero points) weight. On gfx1201
that is `RDNAHybridW4A16LinearKernel`, the kernel vLLM's GPTQ layers get there.

The patch wraps the scheme's two entry points:

  * `process_weights_after_loading` runs vLLM's, then checks hk can read what it left in place
    (`vllm_gptq.hybrid_why`: int8 [N, K/2] ExLlama-shuffled codes, [N, K/group] scales, no zero
    points). No second copy of the weights is held.
  * `apply_weights` makes one call of the opaque custom op `hk_vllm::quark_w4a16_hybrid`: the
    fp16 batches vLLM would run on its Triton GEMM go to hk's forms (`hk.ops.w4a16.exllama_matmul`
    up to `vllm_gptq.HYBRID_MAX_M` tokens, `exllama_large_matmul` above: prefill), the batches
    its HIP skinny GEMM takes and every shape hk's forms do not stay on vLLM's op.

What it refuses, counted in `REFUSED` by reason (each such layer keeps vLLM's path): asymmetric
checkpoints (zero points), layers vLLM gave another kernel, and anything `hybrid_why` names. Off
by default until measured against the >= 1.15x rule; `HK_VLLM_QUARK=1` turns it on.

    HK_PATCH_VLLM=1 HK_VLLM_QUARK=1 vllm serve amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 --dtype float16
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from . import _common, engine, vllm_gptq

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None

#: Layers left on vLLM's path, by reason.
REFUSED: Counter = Counter()


def _fake(x, w_q, w_s, group, cu):
    return x.new_empty((x.shape[0], w_q.shape[0]))


PATCH = engine.Op(
    "hk_vllm::quark_w4a16_hybrid",
    "quark_w4a16_hybrid(Tensor x, Tensor w_q, Tensor w_s, int group, int cu) -> Tensor",
    _fake, paths=("hk", "hk_large", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def quark_w4a16_hybrid(x, w_q, w_s, group: int, cu: int):
    """y = x @ W.T on vLLM's own hybrid-kernel tensors: as `vllm_gptq.gptq_w4a16_hybrid`, counted
    here."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    n = w_q.shape[0]
    fp16 = x.dtype == torch.float16 and w_s.dtype == torch.float16
    if fp16 and m <= vllm_gptq.HYBRID_MAX_M and not vllm_gptq._vllm_skinny(m, k):
        PATCH.count("hk")
        return hk_w4.exllama_matmul(x, w_q.view(torch.int32), w_s, group)
    if fp16 and m > vllm_gptq.HYBRID_MAX_M and hk_w4.exllama_large_takes(n, k, group):
        PATCH.count("hk_large")
        return hk_w4.exllama_large_matmul(x, w_q.view(torch.int32), w_s, group)
    PATCH.count("fallback")
    return torch.ops.vllm.rdna_hybrid_w4a16_apply(x, w_q, w_s, None, None, cu, group)


def _why_not(self) -> str:
    """'' if the scheme's layer can be hk's after vLLM's repack, else why not (before looking at
    the tensors)."""
    from ..target import current_arch  # noqa: PLC0415

    kern = type(getattr(self, "kernel", None)).__name__
    if kern != vllm_gptq._HYBRID:
        return f"kernel {kern}"
    if current_arch() not in vllm_gptq.HYBRID_ARCHES:
        return f"kernel {kern} on {current_arch()}"
    if not getattr(self, "is_symmetric", False):
        return "zero points (asymmetric)"
    if not vllm_gptq._SKINNY:
        return "this vLLM's rdna_hybrid_w4a16 has no MAX_SKINNY_BATCH_SIZE / MEDIUM_SKINNY_LIMIT"
    if self.group_size <= 0 or self.group_size % 32:
        return f"group {self.group_size}"
    return ""


def process_weights_after_loading(self, layer) -> None:
    """vLLM's, then hk checks it can read the result in place."""
    _ORIG_PWAL(self, layer)
    why = _why_not(self) or vllm_gptq.hybrid_why(self.kernel, layer, self.group_size)
    if why:
        REFUSED[why] += 1
        return
    layer.hk_quark = True


def apply_weights(self, layer, x, bias=None):
    if not getattr(layer, "hk_quark", False):
        return _ORIG_APPLY(self, layer, x, bias)
    from vllm.utils.platform_utils import num_compute_units  # noqa: PLC0415

    w_q, w_s, _, _ = vllm_gptq._weights(self.kernel, layer)
    out = PATCH(x.reshape(-1, x.shape[-1]), w_q, w_s, self.group_size, num_compute_units())
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], w_q.shape[0])


def apply_patch() -> str:
    """Route vLLM's Quark W4A16 linear layers. Returns what happened, in one line."""
    global _ORIG_PWAL, _ORIG_APPLY
    if not _common.importable("vllm"):
        return "vllm is not importable; quark not patched"
    if _ORIG_APPLY is not None:
        return "quark: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "quark: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.quantization.quark.schemes import (  # noqa: PLC0415
        quark_w4a16_int4 as qs)

    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    vllm_gptq.read_skinny()                 # without it every layer is refused, counted
    PATCH.define(quark_w4a16_hybrid)
    cls = qs.QuarkW4A16Int4
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply_weights
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    return (f"quark -> hk for symmetric Quark W4A16 layers on RDNAHybridW4A16LinearKernel "
            f"({', '.join(vllm_gptq.HYBRID_ARCHES)}): the fp16 batches vLLM runs on Triton "
            f"(decode forms to {vllm_gptq.HYBRID_MAX_M} tokens, the large-batch form above), in "
            f"place (compile cache {_own_compile_cache('hk-quark')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is None:
        return "quark: not patched"
    from vllm.model_executor.layers.quantization.quark.schemes import (  # noqa: PLC0415
        quark_w4a16_int4 as qs)

    qs.QuarkW4A16Int4.process_weights_after_loading = _ORIG_PWAL
    qs.QuarkW4A16Int4.apply_weights = _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "quark -> vllm"


def status() -> str:
    if _ORIG_APPLY is None:
        return "not patched"
    refused = ", ".join(f"{k}: {v}" for k, v in REFUSED.items())
    return PATCH.summary() + (f"; layers left on vLLM's path: {refused}" if refused else "")
