"""vLLM's GPTQ linear layers on hk's 4-bit decode GEMM (`HK_VLLM_GPTQ`, on by default on gfx1100).

On RDNA3, vLLM runs a 4-bit symmetric GPTQ checkpoint through
`AutoGPTQLinearMethod` with `RDNA3W4A16LinearKernel`, its own HIP W4A16 GEMM
(`ops.gptq_gemm_rdna3`) on GPTQ's shuffled codes. The patch wraps the method's
two entry points:

  * `process_weights_after_loading` writes hk's symmetric layout of the codes
    (`hk.ops.w4a16.repack_gptq(..., symmetric=True)`: codes and scales in
    16-row blocks, no zero-times-scale) *before* vLLM's runs, since vLLM's
    shuffles the codes in place. It is a second copy of the codes, the size of
    the first, kept so that every batch hk does not take still runs vLLM's
    kernel on vLLM's own tensors. The zero point is 8, as
    `RDNA3W4A16LinearKernel` takes it for uint4b8; a layer whose stored zero
    points are not all 8 is refused (counted), not assumed.
  * `apply` makes one call of the opaque custom op `hk_vllm::gptq_w4a16`.
    Inside it the token count is a plain number, not a symbolic size. 16-bit
    batches of `MIN_M` to 64 tokens go to hk's kernel, and the rest to vLLM's.

What it refuses, counted in `REFUSED` by reason: act-order checkpoints
(`desc_act`, a column permutation hk's layout does not carry), anything but
4-bit symmetric with groups, and layers vLLM gave another kernel than
RDNA3W4A16LinearKernel. Each of those layers keeps vLLM's path.

Like the other linear patches it changes what the compiled graph calls, so it
gives the engine its own compile-cache root, `hk-gptq` under the current one.

    HK_PATCH_VLLM=1 vllm serve <a GPTQ-Int4 checkpoint> --dtype float16
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

from . import _common, engine

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None

#: The fewest tokens hk's kernel takes. Its symmetric form (packed-fp16 decode,
#: scales in 16-row blocks) is 1.04-1.12x vLLM's kernel at one token on every
#: Qwen3.8-27B TP1 shape and 1.07-1.31x at 2-4 (tools/hk-bench/gptq_w4a16.py on a
#: W7900D); the asymmetric form was 0.87-0.99x at one token, hence 4 before.
MIN_M = 1

#: Layers left on vLLM's path, by reason.
REFUSED: Counter = Counter()


def _fake(x, w_q, w_zp, w_s, g_idx, hk_qw, hk_s, hk_zs, group):
    return x.new_empty((x.shape[0], hk_qw.shape[0]))


PATCH = engine.Op(
    "hk_vllm::gptq_w4a16",
    "gptq_w4a16(Tensor x, Tensor w_q, Tensor w_zp, Tensor w_s, Tensor? g_idx, Tensor hk_qw, "
    "Tensor hk_s, Tensor hk_zs, int group) -> Tensor",
    _fake, paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def gptq_w4a16(x, w_q, w_zp, w_s, g_idx, hk_qw, hk_s, hk_zs, group: int):
    """y = x @ W.T for a 4-bit GPTQ W held both ways: hk's kernel where it
    takes the batch, vLLM's RDNA3 kernel on vLLM's tensors otherwise."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    m, k = x.shape
    run = (hk_w4.plan(m, hk_qw.shape[0], k, group, x.dtype, True)
           if x.dtype == hk_s.dtype and MIN_M <= m <= hk_w4.MAX_TOKENS * 16 else None)
    if run is not None:
        PATCH.count("hk")
        out = torch.empty((m, hk_qw.shape[0]), device=x.device, dtype=hk_s.dtype)
        run(x if x.is_contiguous() else x.contiguous(), hk_qw, hk_s, out)
        return out
    PATCH.count("fallback")
    return _vllm_gptq(x, w_q, w_zp, w_s, g_idx)


#: What this stand-in can pass to vLLM's `gptq_gemm_rdna3`, by its parameter names. vLLM builds
#: differ: some take the act-order index `b_g_idx` (an empty tensor for a checkpoint without act
#: order), some do not. The call is bound by name to whatever this build's signature is.
_GEMM_ARGS = {"a": "x", "b_q_weight": "w_q", "b_qzeros": "w_zp", "b_scales": "w_s",
              "b_g_idx": "g_idx", "use_v2_format": False}
_GEMM_PLAN: list = []


def plan_call(fn):
    """([(name, source)], problem) for a gptq_gemm_rdna3 signature: each parameter's source is
    one of the stand-in's tensors by name, or a constant. problem is "" when every parameter
    without a default has a source, else what is missing -- the layers then keep vLLM's apply."""
    import inspect  # noqa: PLC0415

    plan, missing = [], []
    for p in inspect.signature(fn).parameters.values():
        if p.name in _GEMM_ARGS:
            plan.append((p.name, _GEMM_ARGS[p.name]))
        elif p.default is p.empty:
            missing.append(p.name)
    return plan, (f"gptq_gemm_rdna3 takes {missing}, which this route has no value for"
                  if missing else "")


def _gemm_plan():
    """plan_call() for this process's vLLM, made once."""
    if not _GEMM_PLAN:
        from vllm import _custom_ops as ops  # noqa: PLC0415

        _GEMM_PLAN.append(plan_call(ops.gptq_gemm_rdna3))
    return _GEMM_PLAN[0]


def _vllm_gptq(x, w_q, w_zp, w_s, g_idx):
    """RDNA3W4A16LinearKernel.apply_weights' own GEMM, for the batches hk leaves, called with
    this vLLM's own parameters by name."""
    from vllm import _custom_ops as ops  # noqa: PLC0415

    have = {"x": x, "w_q": w_q, "w_zp": w_zp, "w_s": w_s, "g_idx": g_idx}
    plan, _ = _gemm_plan()
    return ops.gptq_gemm_rdna3(**{n: (have[v] if isinstance(v, str) else v) for n, v in plan})


def _weights(kern, layer):
    """(w_q, w_s, w_zp, g_idx) by the names the kernel was given, as its _get_weight_params
    reads them; g_idx None where this vLLM's kernel has none."""
    def get(attr):
        name = getattr(kern, attr, None)
        return getattr(layer, name, None) if name else None

    return get("w_q_name"), get("w_s_name"), get("w_zp_name"), get("w_gidx_name")


def _why_not(self) -> str:
    """'' if hk can hold this layer's weights, else why not."""
    c = self.quant_config
    if type(getattr(self, "kernel", None)).__name__ != "RDNA3W4A16LinearKernel":
        return f"kernel {type(getattr(self, 'kernel', None)).__name__}"
    if getattr(c, "desc_act", False):
        return "act-order (desc_act)"
    if str(getattr(c, "quant_type", "")) != "uint4b8":
        return f"quant type {getattr(c, 'quant_type', '?')}"
    if c.group_size <= 0 or c.group_size % 16:
        return f"group {c.group_size}"
    if not getattr(self.kernel, "w_q_name", None) or not getattr(self.kernel, "w_s_name", None):
        return "kernel names no weight parameters (w_q_name / w_s_name)"
    plan, problem = _gemm_plan()
    if problem:
        return problem
    return ""


def process_weights_after_loading(self, layer) -> None:
    """hk's layout of the codes, then vLLM's (which shuffles them in place)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    why = _why_not(self)
    s = getattr(layer, "scales", None)
    if not why and (s is None or s.dtype not in (torch.float16, torch.bfloat16)):
        why = f"scales {getattr(s, 'dtype', None)}"
    if not why:
        qw = layer.qweight.data                     # [K/8, N], codes packed along K
        k8, n = qw.shape
        if n % 16 or (k8 * 8) % 32:
            why = f"shape {n}x{k8 * 8}"
    if not why:
        q = getattr(layer, "qweight", None)
        if q is None or s is None:
            why = "no qweight / scales on the layer"
    if why:
        REFUSED[why] += 1
        _ORIG_PWAL(self, layer)
        return
    group = self.quant_config.group_size
    # The stored zero points (GPTQ v1: one short), checked to be 8 -- vLLM's for uint4b8.
    qz = getattr(layer, "qzeros", None)
    qz = qz.data if qz is not None and qz.dim() == 2 else torch.full(
        (k8 * 8 // group, n // 8), 0x77777777, device=qw.device, dtype=torch.int32)
    try:
        hq, hs, _ = hk_w4.repack_gptq(qw, qz, layer.scales.data, group, zero_offset=1,
                                      dtype=layer.scales.dtype, symmetric=True)
    except ValueError:
        REFUSED["zero points not 8"] += 1
        _ORIG_PWAL(self, layer)
        return
    _ORIG_PWAL(self, layer)
    w_q, w_s, w_zp, g_idx = _weights(self.kernel, layer)
    plan, _ = _gemm_plan()
    if w_q is None or w_s is None or w_zp is None or (
            any(n == "b_g_idx" for n, _ in plan) and g_idx is None):
        REFUSED["vLLM's kernel holds no zero points / g_idx here"] += 1
        return                              # vLLM's layer, untouched: its own apply runs
    layer.hk_qw = torch.nn.Parameter(hq, requires_grad=False)
    layer.hk_s = torch.nn.Parameter(hs, requires_grad=False)
    # the op's schema keeps a zs slot; the symmetric form reads none
    layer.hk_zs = torch.nn.Parameter(hs.new_empty(0), requires_grad=False)


def apply(self, layer, x, bias=None):
    if not hasattr(layer, "hk_qw"):
        return _ORIG_APPLY(self, layer, x, bias)
    w_q, w_s, w_zp, g_idx = _weights(self.kernel, layer)
    out = PATCH(x.reshape(-1, x.shape[-1]), w_q, w_zp, w_s, g_idx, layer.hk_qw, layer.hk_s,
                layer.hk_zs, self.quant_config.group_size)
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], layer.hk_qw.shape[0])


def apply_patch() -> str:
    """Route vLLM's GPTQ linear layers. Returns what happened, in one line."""
    global _ORIG_PWAL, _ORIG_APPLY
    if not _common.importable("vllm"):
        return "vllm is not importable; gptq not patched"
    if _ORIG_APPLY is not None:
        return "gptq: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "gptq: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    PATCH.define(gptq_w4a16)
    cls = auto_gptq.AutoGPTQLinearMethod
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    return (f"gptq -> hk for 4-bit symmetric GPTQ layers on RDNA3W4A16LinearKernel "
            f"(hk's symmetric form), 16-bit batches of {MIN_M} to 64 tokens (compile cache "
            f"{_own_compile_cache('hk-gptq')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is None:
        return "gptq: not patched"
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    auto_gptq.AutoGPTQLinearMethod.process_weights_after_loading = _ORIG_PWAL
    auto_gptq.AutoGPTQLinearMethod.apply = _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "gptq -> vllm"


def status() -> str:
    if _ORIG_APPLY is None:
        return "not patched"
    refused = ", ".join(f"{k}: {v}" for k, v in REFUSED.items())
    return PATCH.summary() + (f"; layers left on vLLM's path: {refused}" if refused else "")
