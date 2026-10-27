"""vLLM's AWQ linear layers on hk's 4-bit decode GEMM (`HK_VLLM_INT4=1`).

On ROCm, vLLM 0.31 runs an AutoAWQ checkpoint through `AutoAWQLinearMethod`:
`ops.awq_gemm` below 256 tokens, and above that a dequantize of the whole
weight and a `torch.matmul` on every call. The patch wraps that method's two
entry points:

  * `process_weights_after_loading` runs vLLM's, then writes hk's layout of
    the same weights beside them (`hk.ops.w4a16.repack_awq`): a second copy
    of the codes, the size of the first, kept so that every batch hk does not
    take still runs vLLM's own path on vLLM's own tensors;
  * `apply` makes one call of the opaque custom op `hk_vllm::w4a16`. Inside
    it -- where the token count is a number, not a symbolic shape torch.compile
    would guard on -- fp16 or bf16 batches of up to 64 tokens (16 on
    gfx950) take hk's kernel and the rest vLLM's path.

On ROCm vLLM's AWQ path is its Triton kernel (the C++ one is not built
there, so VLLM_USE_TRITON_AWQ=1), and that kernel takes fp16 only: an AWQ
model is served in fp16, and so are these numbers.

Like the linear-layer patch it changes what the compiled graph calls, so it
gives the engine its own compile-cache root, `hk-int4` under the current one
(`vllm_gemm._own_compile_cache`).

    VLLM_USE_TRITON_AWQ=1 HK_PATCH_VLLM=1 HK_VLLM_INT4=1 \\
        vllm serve Qwen/Qwen3-8B-AWQ --dtype float16
"""

from __future__ import annotations

from typing import Callable, Optional

from ..runtime import hostpath as _hostpath
from . import _common, engine

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None


def _fake(x, qweight, scales, qzeros, hk_qw, hk_s, hk_zs, group):
    return x.new_empty((x.shape[0], hk_qw.shape[0]))


PATCH = engine.Op(
    "hk_vllm::w4a16",
    "w4a16(Tensor x, Tensor qweight, Tensor scales, Tensor qzeros, Tensor hk_qw, "
    "Tensor hk_s, Tensor hk_zs, int group) -> Tensor",
    _fake, paths=("hk", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


#: The most tokens hk's kernel takes, by arch. On gfx950 the batches past 16
#: run a slower schedule of it, which loses to the dequantize and hipBLASLt
#: GEMM vLLM's compiled graph runs there (docs/history/42).
_HK_MAX_M = {"gfx950": 16}


def _cdna() -> bool:
    from ..target import current_arch  # noqa: PLC0415

    return current_arch() == "gfx950"


def _vllm_awq(x, qweight, scales, qzeros):
    """AutoAWQLinearMethod.apply's own arithmetic, for the batches hk leaves.
    Its `x.shape[0] >= 256` test sees torch.compile's symbolic size, which
    takes the dequantize branch at every batch; on gfx950 that branch is also
    the faster one at every batch, so it is the path here."""
    import torch  # noqa: PLC0415
    from vllm import _custom_ops as ops  # noqa: PLC0415

    if x.shape[0] >= 256 or _cdna():
        return torch.matmul(x, ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0))
    return ops.awq_gemm(x, qweight, scales, qzeros, 8)


def w4a16(x, qweight, scales, qzeros, hk_qw, hk_s, hk_zs, group: int):
    """y = x @ W.T for a 4-bit W held both ways: hk's kernel where it takes
    the batch, vLLM's path on vLLM's tensors otherwise."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    if not _hostpath.ENABLED:
        return _w4a16_legacy(x, qweight, scales, qzeros, hk_qw, hk_s, hk_zs, group)
    m, k = x.shape
    run = (hk_w4.plan(m, hk_qw.shape[0], k, group, x.dtype)
           if x.dtype == hk_s.dtype and m <= _max_m() else None)
    if run is not None:
        PATCH.count("hk")
        out = torch.empty((m, hk_qw.shape[0]), device=x.device, dtype=hk_s.dtype)
        run(x if x.is_contiguous() else x.contiguous(), hk_qw, hk_s, hk_zs, out)
        return out
    PATCH.count("fallback")
    return _vllm_awq(x, qweight, scales, qzeros)


def _w4a16_legacy(x, qweight, scales, qzeros, hk_qw, hk_s, hk_zs, group: int):
    """`w4a16` before the host fast paths, line for line (HK_HOST_FASTPATH=0)."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415
    from ..target import current_arch  # noqa: PLC0415

    m, k = x.shape
    if (x.dtype == hk_s.dtype and x.dtype in (torch.bfloat16, torch.float16)
            and m <= _HK_MAX_M.get(current_arch(), 64)
            and hk_w4.takes(m, hk_qw.shape[0], k, group)):
        PATCH.count("hk")
        return hk_w4.matmul(x, hk_qw, hk_s, hk_zs, group)
    PATCH.count("fallback")
    return _vllm_awq(x, qweight, scales, qzeros)


_MAX_M: list = []


def _max_m() -> int:
    if not _MAX_M:
        from ..target import current_arch  # noqa: PLC0415

        _MAX_M.append(_HK_MAX_M.get(current_arch(), 64))
    return _MAX_M[0]


def process_weights_after_loading(self, layer) -> None:
    """vLLM's, then hk's layout of the same codes beside them."""
    import torch  # noqa: PLC0415

    from ..ops import w4a16 as hk_w4  # noqa: PLC0415

    _ORIG_PWAL(self, layer)
    group = self.quant_config.group_size
    if layer.scales.dtype not in (torch.bfloat16, torch.float16) or group % 64:
        return
    qw, s, zs = hk_w4.repack_awq(layer.qweight.data, layer.qzeros.data,
                                 layer.scales.data, group, dtype=layer.scales.dtype)
    layer.hk_qw = torch.nn.Parameter(qw, requires_grad=False)
    layer.hk_s = torch.nn.Parameter(s, requires_grad=False)
    layer.hk_zs = torch.nn.Parameter(zs, requires_grad=False)


def apply(self, layer, x, bias=None):
    if not hasattr(layer, "hk_qw"):
        return _ORIG_APPLY(self, layer, x, bias)
    out = PATCH(x.reshape(-1, x.shape[-1]), layer.qweight, layer.scales,
                layer.qzeros, layer.hk_qw, layer.hk_s, layer.hk_zs,
                self.quant_config.group_size)
    if bias is not None:
        out.add_(bias)
    return out.reshape(*x.shape[:-1], layer.hk_qw.shape[0])


def apply_patch() -> str:
    """Route vLLM's AWQ linear layers. Returns what happened, in one line."""
    global _ORIG_PWAL, _ORIG_APPLY
    if not _common.importable("vllm"):
        return "vllm is not importable; int4 not patched"
    if _ORIG_APPLY is not None:
        return "int4: already patched"
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "int4: not patched -- not a ROCm platform"
    from vllm.model_executor.layers.quantization import auto_awq  # noqa: PLC0415

    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    PATCH.define(w4a16)
    cls = auto_awq.AutoAWQLinearMethod
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply = apply
    return (f"int4 -> hk for AWQ layers, 16-bit batches up to {_max_m()} tokens "
            f"(compile cache {_own_compile_cache('hk-int4')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is None:
        return "int4: not patched"
    from vllm.model_executor.layers.quantization import auto_awq  # noqa: PLC0415

    auto_awq.AutoAWQLinearMethod.process_weights_after_loading = _ORIG_PWAL
    auto_awq.AutoAWQLinearMethod.apply = _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "int4 -> vllm"


def status() -> str:
    return "not patched" if _ORIG_APPLY is None else PATCH.summary()
