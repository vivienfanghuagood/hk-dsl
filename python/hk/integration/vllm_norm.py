"""vLLM's GemmaRMSNorm on hk's fused (add+)RMSNorm (`HK_VLLM_NORM`).

`vllm.model_executor.layers.layernorm.GemmaRMSNorm` -- y = x_hat * (1 + w), the norm of Gemma and of
the Qwen3.5 family (Qwen3.5 / Qwen3.8, and the text models of MiniCPM-V-4.7) -- has no fused kernel
on CUDA or ROCm: its `forward_cuda` is its `forward_native`, vLLM IR ops that run as PyTorch ops
unless the model is compiled. Under torch.compile inductor fuses them; in eager mode (a deployment
that runs `enforce_eager`, as MiniCPM-V-4.7's do) every norm is ~8 kernels, 61 norms a decode token
on MiniCPM-V-4.7-1B, and the step is bound by their launches.

Here `forward_cuda` runs hk's weighted norm kernels at weight offset 1 -- the stored w, the one
added in fp32 inside the kernel, as vLLM's own math does -- with the residual add fused, out of
place as vLLM's native path is ((y, x + residual) or y), on torch's current stream
(`sglang.gemma_rms_norm`, the kernels SGLang's norm route uses). Against vLLM's native path on the
real calls of a MiniCPM-V-4.7-1B request: outputs within 1 bf16 ulp (0.10% of elements at most),
residuals bit-equal.

It leaves to vLLM, counted in `STATS` by reason: calls while torch.compile traces (inductor fuses
them there), and inputs hk's kernels do not take (`sglang._norm_refusal`: not a contiguous 16-bit
GPU tensor, a weight of another dtype or shape, a residual unlike x).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Tuple

_HOME = "vllm.model_executor.layers.layernorm"
_ORIG: Optional[Tuple[type, Callable]] = None

#: Which way each call went: "hk", or the reason it stayed vLLM's.
STATS: Counter = Counter()


def _compiling() -> bool:
    import torch  # noqa: PLC0415

    return bool(getattr(torch.compiler, "is_compiling", lambda: False)())


def apply() -> str:
    """Route vLLM's GemmaRMSNorm.forward_cuda through hk's weighted norm kernels. Returns what
    happened, in one line."""
    global _ORIG
    import importlib  # noqa: PLC0415

    from . import _common, sglang  # noqa: PLC0415

    if not _common.importable("vllm"):
        return "vllm is not importable; norm not patched"
    if _ORIG is not None:
        return "norm: already patched"
    try:
        L = importlib.import_module(_HOME)
    except ImportError as e:
        return f"norm: not patched -- {_HOME}: {e}"
    cls = getattr(L, "GemmaRMSNorm", None)
    if cls is None:
        return f"norm: not patched -- no GemmaRMSNorm in {_HOME}"
    orig = cls.forward_cuda

    def forward_cuda(self, x, residual=None):
        if _compiling():
            STATS["torch.compile (inductor fuses it)"] += 1
            return orig(self, x, residual)
        why = sglang._norm_refusal(self, x, residual)
        if why:
            STATS[why] += 1
            return orig(self, x, residual)
        STATS["hk"] += 1
        return sglang.gemma_rms_norm(x, residual, self.weight, self.variance_epsilon)

    forward_cuda.__wrapped__ = orig
    cls.forward_cuda = forward_cuda
    _ORIG = (cls, orig)
    return f"norm -> hk ({_HOME}.GemmaRMSNorm.forward_cuda, weight offset 1 in fp32; eager calls)"


def revert() -> str:
    global _ORIG
    if _ORIG is None:
        return "norm: not patched"
    cls, orig = _ORIG
    cls.forward_cuda = orig
    _ORIG = None
    return "norm -> vllm"


def stats() -> str:
    return f"vllm_norm {dict(STATS)}"


__all__ = ["apply", "revert", "stats", "STATS"]
