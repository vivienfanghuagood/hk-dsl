"""vLLM's GemmaRMSNorm on hk's fused (add+)RMSNorm (`HK_VLLM_NORM`).

`vllm.model_executor.layers.layernorm.GemmaRMSNorm` -- y = x_hat * (1 + w), the norm of Gemma and of
the Qwen3.5 family (Qwen3.5 / Qwen3.8, and the text models of MiniCPM-V-4.7) -- has no fused kernel
on CUDA or ROCm: its `forward_cuda` is its `forward_native`, vLLM IR ops that run as PyTorch ops
unless the model is compiled, where inductor generates a kernel for them. In eager mode (a
deployment that runs `enforce_eager`, as MiniCPM-V-4.7's do) every norm is ~8 kernels, 61 norms a
decode token on MiniCPM-V-4.7-1B, and the step is bound by their launches.

Here the layer runs hk's weighted norm kernels at weight offset 1 -- the stored w, the one
added in fp32 inside the kernel, as vLLM's own math does -- with the residual add fused, out of
place as vLLM's native path is ((y, x + residual) or y), on torch's current stream
(`sglang.gemma_rms_norm`, the kernels SGLang's norm route uses). Against vLLM's native path on the
real calls of a MiniCPM-V-4.7-1B request: outputs within 1 bf16 ulp (0.10% of elements at most),
residuals bit-equal.

In eager mode vLLM calls `forward_cuda`, which the route takes by default. Under torch.compile
vLLM's custom ops are off by default and it traces `forward_native`; with `HK_VLLM_NORM=compile` the
route takes that too (opt-in: under CUDA graphs it saves ~5 us a norm against inductor's kernel
in a decoder layer, a few percent of TPOT, not yet measured end to end). Both paths call
the kernels through two opaque torch custom ops, `torch.ops.hk_vllm.gemma_rms_norm` and
`gemma_add_rms_norm` (`engine.Op`), so torch.compile puts one node in its graph instead of
tracing hk's Python (which picks a plan from the row count, symbolic there), and a CUDA graph
captures one kernel. The ops launch on torch's current stream with no host sync (`engine.Op`'s
capture guard raises if one appears). The compiled graph depends on the patch, so the route gives
the engine a compile cache of its own (`vllm_gemm._own_compile_cache`, "hk-norm").

It leaves to vLLM, counted in `STATS` by reason, inputs hk's kernels do not take
(`sglang._norm_refusal`: not a contiguous 16-bit GPU tensor, a weight of another dtype or shape,
a residual unlike x). The counters are Python: a call compiled into vLLM's graph is counted when
the op runs (`NORM.stats` / `ADD.stats`, and `.captured` for calls made while capturing; a
CUDA-graph replay runs no Python), and nothing is written while torch.compile traces.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional, Tuple

from . import engine

_HOME = "vllm.model_executor.layers.layernorm"
_ORIG: Optional[Tuple[type, Callable, Callable]] = None

#: HK_VLLM_NORM values that also take the calls torch.compile traces (forward_native).
COMPILED = ("compile",)

#: Which way each eager call went: "hk", or the reason it stayed vLLM's.
STATS: Counter = Counter()


def _fake_norm(x, weight, eps):
    import torch  # noqa: PLC0415

    return torch.empty_like(x)


def _fake_add(x, residual, weight, eps):
    import torch  # noqa: PLC0415

    return torch.empty_like(x), torch.empty_like(x)


NORM = engine.Op("hk_vllm::gemma_rms_norm",
                 "gemma_rms_norm(Tensor x, Tensor weight, float eps) -> Tensor",
                 _fake_norm, paths=("hk",), graph="safe")
ADD = engine.Op("hk_vllm::gemma_add_rms_norm",
                "gemma_add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)",
                _fake_add, paths=("hk",), graph="safe")


def _compiling() -> bool:
    import torch  # noqa: PLC0415

    return bool(getattr(torch.compiler, "is_compiling", lambda: False)())


def _define() -> None:
    from . import sglang  # noqa: PLC0415

    def norm(x, weight, eps):
        NORM.count("hk")
        return sglang.gemma_rms_norm(x, None, weight, eps)

    def add(x, residual, weight, eps):
        ADD.count("hk")
        return sglang.gemma_rms_norm(x, residual, weight, eps)

    NORM.define(norm)
    ADD.define(add)


def _route(orig):
    """The patched forward: hk's ops for a call they take, `orig` for the rest."""
    from . import sglang  # noqa: PLC0415

    def forward(self, x, residual=None):
        why = sglang._norm_refusal(self, x, residual)
        compiling = _compiling()
        if why:
            if not compiling:
                STATS[why] += 1
            return orig(self, x, residual)
        if not compiling:
            STATS["hk"] += 1
        if residual is None:
            return NORM(x, self.weight, self.variance_epsilon)
        return ADD(x, residual, self.weight, self.variance_epsilon)

    forward.__wrapped__ = orig
    return forward


def apply() -> str:
    """Route vLLM's GemmaRMSNorm (forward_cuda, eager; forward_native, what torch.compile traces)
    through hk's weighted norm kernels. Returns what happened, in one line."""
    global _ORIG
    import importlib  # noqa: PLC0415

    from . import _common  # noqa: PLC0415

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
    from . import switches  # noqa: PLC0415

    _define()
    cuda, native = cls.forward_cuda, cls.forward_native
    cls.forward_cuda = _route(cuda)
    compiled = switches.value("vllm_norm").lower() in COMPILED
    if compiled:
        from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

        cls.forward_native = _route(native)
    _ORIG = (cls, cuda, native)
    if not compiled:
        return (f"norm -> hk ({_HOME}.GemmaRMSNorm.forward_cuda, weight offset 1 in fp32; eager "
                f"calls -- HK_VLLM_NORM=compile takes compiled ones too)")
    return (f"norm -> hk ({_HOME}.GemmaRMSNorm forward_cuda and forward_native, weight offset 1 "
            f"in fp32, torch.ops.hk_vllm.gemma_(add_)rms_norm, eager and compiled calls; compile "
            f"cache {_own_compile_cache('hk-norm')})")


def revert() -> str:
    global _ORIG
    if _ORIG is None:
        return "norm: not patched"
    cls, cuda, native = _ORIG
    cls.forward_cuda, cls.forward_native = cuda, native
    _ORIG = None
    return "norm -> vllm"


def stats() -> str:
    return f"vllm_norm {dict(STATS)}; {NORM.summary()}; {ADD.summary()}"


__all__ = ["apply", "revert", "stats", "STATS", "NORM", "ADD"]
