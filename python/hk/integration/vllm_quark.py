"""vLLM's AMD Quark W4A16 linear layers on hk's 4-bit forms (`HK_VLLM_QUARK`).

A Quark int4 weight-only checkpoint (`quant_method: quark`, weight int4 per group, no activation
quantization; e.g. amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16) loads through vLLM's `QuarkW4A16Int4`
scheme. Quark stores AutoAWQ's tensor layout -- `weight` int32 [K, N/8], `weight_scale` [K/group,
N], `weight_zero_point` int32 [K/group, N/8] -- with its own nibble order (`pack_method`) and, for
a symmetric checkpoint, signed codes. The scheme's `process_weights_after_loading` canonicalizes
that to AWQ's codes, converts them to the standard (GPTQ) format and hands the layer to the
MPLinearKernel vLLM chose for a uint4b8 (symmetric) weight -- the kernels vLLM's GPTQ layers get:

  * gfx1201, `RDNAHybridW4A16LinearKernel`. After vLLM's repack hk reads its tensors in place
    (`vllm_gptq.hybrid_why` checks them: int8 [N, K/2] ExLlama-shuffled codes, [N, K/group]
    scales), so no second copy is held. The opaque op `hk_vllm::quark_w4a16_hybrid` runs the fp16
    batches vLLM would send to its Triton GEMM on hk's forms (`hk.ops.w4a16.exllama_matmul` up to
    `vllm_gptq.HYBRID_MAX_M` tokens, `exllama_large_matmul` above: prefill); vLLM's HIP skinny
    GEMM keeps the batches it takes.
  * gfx1100, `RDNA3W4A16LinearKernel`, which shuffles the codes in place. Before vLLM's repack hk
    writes its symmetric layout of Quark's own codes beside them (`hk.ops.w4a16.repack_quark`), a
    second copy the size of the first, and the opaque op `hk_vllm::quark_w4a16` runs 16-bit
    batches of `vllm_gptq.MIN_M` to 64 tokens on hk's decode GEMM, the rest on vLLM's
    `gptq_gemm_rdna3` on vLLM's tensors.

What it refuses, counted in `REFUSED` by reason (each such layer keeps vLLM's path): asymmetric
checkpoints (zero points), layers vLLM gave another kernel, and shapes hk's forms do not take.
Off by default until measured against the >= 1.15x rule; `HK_VLLM_QUARK=1` turns it on.

    HK_PATCH_VLLM=1 HK_VLLM_QUARK=1 vllm serve amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 --dtype float16
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Callable, Optional

from . import _common, engine, vllm_gptq

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None

#: Layers left on vLLM's path, by reason.
REFUSED: Counter = Counter()


def _fake_hybrid(x, w_q, w_s, group, cu):
    return x.new_empty((x.shape[0], w_q.shape[0]))


HYBRID = engine.Op(
    "hk_vllm::quark_w4a16_hybrid",
    "quark_w4a16_hybrid(Tensor x, Tensor w_q, Tensor w_s, int group, int cu) -> Tensor",
    _fake_hybrid, paths=("hk", "hk_large", "fallback"), graph="safe")


def _fake(x, w_q, w_zp, w_s, hk_qw, hk_s, group):
    return x.new_empty((x.shape[0], hk_qw.shape[0]))


PATCH = engine.Op(
    "hk_vllm::quark_w4a16",
    "quark_w4a16(Tensor x, Tensor w_q, Tensor w_zp, Tensor w_s, Tensor hk_qw, Tensor hk_s, "
    "int group) -> Tensor",
    _fake, paths=("hk", "fallback"), graph="safe")

#: Which way each call went, by op; see `engine.Patch` for what a replay counts as.
STATS = {"hybrid": HYBRID.stats, "rdna3": PATCH.stats}


def quark_w4a16_hybrid(x, w_q, w_s, group: int, cu: int):
    """y = x @ W.T on vLLM's own hybrid-kernel tensors: as `vllm_gptq.gptq_w4a16_hybrid`, counted
    here."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    n = w_q.shape[0]
    fp16 = x.dtype == torch.float16 and w_s.dtype == torch.float16
    if fp16 and m <= vllm_gptq.HYBRID_MAX_M and not vllm_gptq._vllm_skinny(m, k):
        HYBRID.count("hk")
        return hk_w4.exllama_matmul(x, w_q.view(torch.int32), w_s, group)
    if fp16 and m > vllm_gptq.HYBRID_MAX_M and hk_w4.exllama_large_takes(n, k, group):
        HYBRID.count("hk_large")
        return hk_w4.exllama_large_matmul(x, w_q.view(torch.int32), w_s, group)
    HYBRID.count("fallback")
    return torch.ops.vllm.rdna_hybrid_w4a16_apply(x, w_q, w_s, None, None, cu, group)


def quark_w4a16(x, w_q, w_zp, w_s, hk_qw, hk_s, group: int):
    """y = x @ W.T for a Quark W held both ways: hk's symmetric decode GEMM where it takes the
    batch, vLLM's RDNA3 kernel on vLLM's tensors otherwise."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    run = (hk_w4.plan(m, hk_qw.shape[0], k, group, x.dtype, True)
           if x.dtype == hk_s.dtype and vllm_gptq.MIN_M <= m <= hk_w4.MAX_TOKENS * 16 else None)
    if run is not None:
        PATCH.count("hk")
        out = torch.empty((m, hk_qw.shape[0]), device=x.device, dtype=hk_s.dtype)
        run(x if x.is_contiguous() else x.contiguous(), hk_qw, hk_s, out)
        return out
    PATCH.count("fallback")
    return vllm_gptq._vllm_gptq(x, w_q, w_zp, w_s, None)


def _why_not(self) -> str:
    """'' if the scheme's layer can be hk's, else why not (before looking at the tensors)."""
    from ..target import current_arch  # noqa: PLC0415

    kern = type(getattr(self, "kernel", None)).__name__
    if kern == vllm_gptq._HYBRID:
        if current_arch() not in vllm_gptq.HYBRID_ARCHES:
            return f"kernel {kern} on {current_arch()}"
    elif kern != vllm_gptq._RDNA3:
        return f"kernel {kern}"
    if not getattr(self, "is_symmetric", False):
        return "zero points (asymmetric)"
    if kern == vllm_gptq._HYBRID:
        if not vllm_gptq._SKINNY:
            return "this vLLM's rdna_hybrid_w4a16 has no MAX_SKINNY_BATCH_SIZE / MEDIUM_SKINNY_LIMIT"
        if self.group_size <= 0 or self.group_size % 32:
            return f"group {self.group_size}"
        return ""
    if self.group_size <= 0 or self.group_size % 16:
        return f"group {self.group_size}"
    if any(n == "b_g_idx" for n, _ in vllm_gptq._gemm_plan()[0]):
        return "this vLLM's gptq_gemm_rdna3 takes an act-order index the Quark layers do not hold"
    return vllm_gptq._gemm_plan()[1]


def process_weights_after_loading(self, layer) -> None:
    """On the hybrid kernel vLLM's repack, then hk checks it can read the result in place; on the
    RDNA3 kernel hk's layout of Quark's codes first (vLLM's repack shuffles them in place)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    why = _why_not(self)
    if not why and type(self.kernel).__name__ == vllm_gptq._HYBRID:
        _ORIG_PWAL(self, layer)
        why = vllm_gptq.hybrid_why(self.kernel, layer, self.group_size)
        if why:
            REFUSED[why] += 1
        else:
            layer.hk_hybrid = True
        return
    hq = None
    if not why:
        w, s = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
        if w is None or s is None or s.dtype not in (torch.float16, torch.bfloat16):
            why = f"scales {getattr(s, 'dtype', None)}"
        elif w.shape[1] * 8 != s.shape[1] or s.shape[1] % 16 or w.shape[0] % 32:
            why = f"shape {s.shape[1]}x{w.shape[0]}"
        else:
            hq, hs, _ = hk_w4.repack_quark(w.data, s.data, self.group_size,
                                           pack_reorder=getattr(self, "pack_reorder", True),
                                           dtype=s.dtype)
    if why:
        REFUSED[why] += 1
        _ORIG_PWAL(self, layer)
        return
    _ORIG_PWAL(self, layer)
    w_q, w_s, w_zp, _ = vllm_gptq._weights(self.kernel, layer)
    if w_q is None or w_s is None or w_zp is None:
        REFUSED["vLLM's kernel holds no zero points here"] += 1
        return                              # vLLM's layer, untouched: its own apply runs
    layer.hk_qw = torch.nn.Parameter(hq, requires_grad=False)
    layer.hk_s = torch.nn.Parameter(hs, requires_grad=False)


def apply_weights(self, layer, x, bias=None):
    if getattr(layer, "hk_hybrid", False):
        from vllm.utils.platform_utils import num_compute_units  # noqa: PLC0415

        w_q, w_s, _, _ = vllm_gptq._weights(self.kernel, layer)
        out = HYBRID(x.reshape(-1, x.shape[-1]), w_q, w_s, self.group_size, num_compute_units())
        n = w_q.shape[0]
    elif hasattr(layer, "hk_qw"):
        w_q, w_s, w_zp, _ = vllm_gptq._weights(self.kernel, layer)
        out = PATCH(x.reshape(-1, x.shape[-1]), w_q, w_zp, w_s, layer.hk_qw, layer.hk_s,
                    self.group_size)
        n = layer.hk_qw.shape[0]
    else:
        return _ORIG_APPLY(self, layer, x, bias)
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], n)


#: vLLM's Quark W4A16 scheme. Never imported by hk itself: the package it lives in
#: (quark.schemes) also imports the OCP MX scheme, which imports vllm._aiter_ops and with it aiter,
#: and aiter JIT-builds kernels at import -- on RDNA with an image's ROCm clang that cannot (the
#: owner's README failure, 2026-11-13). vLLM imports it only for a Quark model; the route patches
#: it then (_Finder), and its contract is read from the source file (contract.gather_vllm_quark).
MODULE = "vllm.model_executor.layers.quantization.quark.schemes.quark_w4a16_int4"
_FINDER = None


def source_path() -> Optional[str]:
    """quark_w4a16_int4.py of the installed vLLM, found without importing anything; None if absent."""
    import importlib.util  # noqa: PLC0415

    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = os.path.join(list(spec.submodule_search_locations)[0], *MODULE.split(".")[1:]) + ".py"
    return path if os.path.isfile(path) else None


class _Finder:
    """Patches MODULE when vLLM imports it (a Quark model): importing it here would import aiter."""

    def find_spec(self, name, path, target=None):
        if name != MODULE:
            return None
        import importlib.machinery  # noqa: PLC0415

        _unhook()
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _patch_module(module)
        spec.loader.exec_module = exec_module
        return spec


def _unhook() -> None:
    global _FINDER
    if _FINDER is not None:
        if _FINDER in sys.meta_path:
            sys.meta_path.remove(_FINDER)
        _FINDER = None


def apply_patch() -> str:
    """Route vLLM's Quark W4A16 linear layers. Returns what happened, in one line."""
    global _FINDER
    if not _common.importable("vllm"):
        return "vllm is not importable; quark not patched"
    if _ORIG_APPLY is not None or _FINDER is not None:
        return "quark: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "quark: not patched -- not a ROCm platform"
    if source_path() is None:
        return "quark: not patched -- this vLLM has no Quark W4A16 scheme"
    if MODULE in sys.modules:
        return _patch_module(sys.modules[MODULE])
    _FINDER = _Finder()
    sys.meta_path.insert(0, _FINDER)
    return _what() + " (when vLLM loads a Quark model)"


def _patch_module(qs) -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is not None:
        return "quark: already patched"
    from . import contract  # noqa: PLC0415

    problems = contract.judge("vllm_quark", contract.quark_facts_of(qs.QuarkW4A16Int4))
    if problems:                            # read from the file before; the loaded class decides
        return contract.refuse_route("vllm", "vllm_quark", problems)
    PATCH.define(quark_w4a16)
    if vllm_gptq.read_skinny():             # else its hybrid layers are refused, counted
        HYBRID.define(quark_w4a16_hybrid)
    cls = qs.QuarkW4A16Int4
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply_weights
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    return _what()


def _what() -> str:
    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    return (f"quark -> hk for symmetric Quark W4A16 layers: on RDNAHybridW4A16LinearKernel "
            f"({', '.join(vllm_gptq.HYBRID_ARCHES)}) the fp16 batches vLLM runs on Triton, in "
            f"place (decode forms to {vllm_gptq.HYBRID_MAX_M} tokens, the large-batch form "
            f"above); on RDNA3W4A16LinearKernel 16-bit batches of {vllm_gptq.MIN_M} to 64 tokens "
            f"(compile cache {_own_compile_cache('hk-quark')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    _unhook()
    if _ORIG_APPLY is None:
        return "quark: not patched"
    qs = sys.modules[MODULE]
    qs.QuarkW4A16Int4.process_weights_after_loading = _ORIG_PWAL
    qs.QuarkW4A16Int4.apply_weights = _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "quark -> vllm"


def status() -> str:
    if _ORIG_APPLY is None:
        return "not patched"
    refused = ", ".join(f"{k}: {v}" for k, v in REFUSED.items())
    hyb = f"; {HYBRID.summary()}" if HYBRID.defined else ""
    return PATCH.summary() + hyb + (f"; layers left on vLLM's path: {refused}" if refused else "")
