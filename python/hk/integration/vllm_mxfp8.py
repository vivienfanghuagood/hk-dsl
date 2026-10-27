"""vLLM's block-scaled MXFP8 linear layers on hk's MXFP8 GEMM (`HK_VLLM_MXFP8=1`, gfx950).

DeepSeek-V4/V4.1 checkpoints carry 32x32 weight-scale blocks; on gfx950 vLLM serves them with
`RocmDotScaledMxfp8LinearKernel`, whose `apply_weights` quantizes the activation
(`mxfp8_e4m3_quantize`) and calls `rocm_mxfp8_block32_gemm`. The patch wraps that class's two
entry points, as `vllm_int4` does AWQ's:

  * `process_weights_after_loading` runs vLLM's, then lays the weight's block scales out the two
    ways hk's kernels read them beside vLLM's (per row: AITER's 32-row layout for the eight-warp
    kernel, packed dwords for the four-warp one) -- a few MB a model;
  * `apply_weights` makes one call of the opaque op `hk_vllm::mxfp8_linear` on the bf16
    activation. Inside it -- where M is a number -- the shapes K2 measured hk faster on
    (k2-mxfp8-gemm c15: 1.00-1.48x the best of vLLM, FlyDSL and hipBLASLt, bit-identical to
    vLLM's kernel at all 18 points) take hk's schedule, the rest vLLM's quant and kernel.

On hk's path one launch quantizes the activation and writes its scales in the layout the GEMM
reads (`mxfp8.quant`). That is bit for bit vLLM's `mxfp8_e4m3_quantize` (AITER's per_1x32
rule) followed by `pack_scales`, and it replaces those two launches (k2-mxfp8-gemm c35).

    HK_PATCH_VLLM=1 HK_VLLM_MXFP8=1 vllm serve <a DeepSeek-V4.1 MXFP8 checkpoint>
"""

from __future__ import annotations

from typing import Callable, Optional

from ..runtime import hostpath as _hostpath
from . import _common, engine

_ORIG_PWAL: Optional[Callable] = None
_ORIG_APPLY: Optional[Callable] = None

#: (N, K) of DeepSeek-V4.1-Flash's prefill projections (TP2), measured in k2-mxfp8-gemm, plus
#: the indexer's wq_b (4096, 1280), measured in the MI355X scoreboard.
MEASURED = {(1792, 5120), (2304, 5120), (5120, 1152), (5120, 4096), (16384, 1280), (12800, 6144),
            (4096, 1280)}
#: The smallest M measured; below it vLLM's kernel runs (unless MIN_M_NK says otherwise).
MIN_M = 2048
#: (N, K) whose smallest M is not MIN_M: where the eight-warp kernel the route picks for them
#: ("8w", whole 256 x 256 blocks) was measured faster than vLLM's, FlyDSL's, AITER Triton's and
#: hipBLASLt's kernels, route level with the activation quant (the MI355X scoreboard's harness),
#: bit-identical to vLLM's at every M:
#: - (4096, 1280), indexer wq_b: 1.08-1.20x from 2825 to 4096, 1.06-1.17x above. At 2048 (128
#:   blocks on 256 CUs) FlyDSL's bmm is faster (0.81x), so not from MIN_M. 2304-2824 were
#:   1.12-1.19x as well: the bound could come down to 2304, not yet checked end to end.
#: - (12800, 6144), engram wkv: 1.39x at 1024 (hipBLASLt the best there), 1.75-1.79x at 1025,
#:   1.29x at 1536, 1.39x at 2047. Below it: 1.38-1.45x at 768-1023 (likewise a candidate for
#:   a lower bound), 1.02x at 512, a loss at 64.
MIN_M_NK = {(4096, 1280): 2825, (12800, 6144): 1024}
#: Route hk-able calls to hk (True) or leave every call to vLLM (False): an in-process A/B.
ENABLED = True
#: Take any (N, K) of whole 256 x 128 tiles at M >= MIN_M on the eight-warp kernel, not only the
#: measured TP2 shapes (HK_MXFP8_ANY=1; e.g. a TP1 run, whose projections are other shapes).
ANY = __import__("os").environ.get("HK_MXFP8_ANY", "0") not in ("0", "", "off", "no")

#: Four-warp schedules where they won (k2-mxfp8-gemm c15): small N at M 2048.
_SMALL_N = {1792: dict(block_m=128, block_n=128, split_m=1, split_n=1, stages=3),
            2304: dict(block_m=128, split_m=1, stages=3)}


def _fake(x, w, ws, hk_wa, hk_wp, out_dtype):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=out_dtype)


PATCH = engine.Op(
    "hk_vllm::mxfp8_linear",
    "mxfp8_linear(Tensor x, Tensor w, Tensor ws, Tensor hk_wa, Tensor hk_wp, "
    "ScalarType out_dtype) -> Tensor",
    _fake, paths=("hk", "fallback"), graph="safe")

STATS = PATCH.stats


def schedule(m: int, n: int, k: int):
    """("8w" | "4w", sched) for a shape hk takes, else None."""
    if not ENABLED or m < MIN_M_NK.get((n, k), MIN_M) or n % 256 or k % 128 or (
            (n, k) not in MEASURED and not ANY):
        return None
    if m < 5120 and n in _SMALL_N:
        return "4w", _SMALL_N[n]
    return "8w", {}


def mxfp8_linear(x, w, ws, hk_wa, hk_wp, out_dtype):
    """MXFP8-quantized bf16 x [M, K] @ w.T for a 32x32 block-scaled w: hk's quant and schedule
    where it was measured faster, vLLM's quant and kernel otherwise -- the same bits either way."""
    import torch  # noqa: PLC0415

    from ..ops import mxfp8  # noqa: PLC0415

    m, k = x.shape
    n = w.shape[0]
    plan = schedule(m, n, k) if out_dtype == torch.bfloat16 and hk_wa.numel() else None
    if plan is not None:
        kind, sched = plan
        xq, xp = mxfp8.quant(x)           # vLLM's quant + pack_scales, in one launch
        PATCH.count("hk")
        if kind == "8w":
            return mxfp8.gemm_8w_tail(xq, w, xp, hk_wa, hk_wp, **sched)
        return mxfp8.gemm_4w(xq, w, xp, hk_wp, **sched)
    PATCH.count("fallback")
    quantize, gemm = _vllm_path() if _hostpath.ENABLED else _vllm_path_legacy()
    xq, xs = quantize(x)
    return gemm(xq, xs, w, ws, out_dtype)


def _vllm_path_legacy():
    """`_vllm_path` as the fallback's body imported it before the host fast paths: every call."""
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (  # noqa: PLC0415
        rocm_mxfp8_block32_gemm)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: PLC0415
        mxfp8_e4m3_quantize)

    return mxfp8_e4m3_quantize, rocm_mxfp8_block32_gemm


_VLLM: list = []


def _vllm_path():
    """(mxfp8_e4m3_quantize, rocm_mxfp8_block32_gemm): vLLM's own quant and kernel, imported once
    -- every call M < MIN_M takes them, decode's included."""
    if not _VLLM:
        from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (  # noqa: PLC0415
            rocm_mxfp8_block32_gemm)
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: PLC0415
            mxfp8_e4m3_quantize)

        _VLLM.extend((mxfp8_e4m3_quantize, rocm_mxfp8_block32_gemm))
    return _VLLM


def process_weights_after_loading(self, layer) -> None:
    """vLLM's, then hk's two layouts of a block-scaled weight's scales beside vLLM's."""
    import torch  # noqa: PLC0415

    from ..ops import mxfp8  # noqa: PLC0415

    _ORIG_PWAL(self, layer)
    w, ws = layer.weight, layer.weight_scale
    n, k = w.shape
    if ws.shape[0] == n or w.element_size() != 1 or n % 256 or k % 128:
        return                       # not the block-scaled path, or a shape hk does not take
    rows = mxfp8.expand_block_scales(ws.data.view(torch.uint8))[:n]
    layer.hk_wa = torch.nn.Parameter(mxfp8.aiter_scales(rows), requires_grad=False)
    layer.hk_wp = torch.nn.Parameter(mxfp8.pack_scales(rows), requires_grad=False)


def apply_weights(self, layer, x, bias=None):
    if not hasattr(layer, "hk_wa"):
        return _ORIG_APPLY(self, layer, x, bias)
    x2d = x.reshape(-1, x.shape[-1])
    out = PATCH(x2d, layer.weight, layer.weight_scale, layer.hk_wa, layer.hk_wp, x.dtype)
    out = out.reshape(*x.shape[:-1], layer.weight.shape[0])
    if bias is not None:
        out = out + bias
    return out


def _cls():
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_native  # noqa: PLC0415

    return rocm_native.RocmDotScaledMxfp8LinearKernel


def apply_patch() -> str:
    """Route vLLM's block-scaled MXFP8 linear layers. Returns what happened, in one line."""
    global _ORIG_PWAL, _ORIG_APPLY
    if not _common.importable("vllm"):
        return "vllm is not importable; mxfp8 not patched"
    if _ORIG_APPLY is not None:
        return "mxfp8: already patched"
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() != "gfx950":
        return f"mxfp8: not patched -- gfx950's scaled MFMA, not {current_arch()}"
    from .vllm_gemm import _own_compile_cache  # noqa: PLC0415

    PATCH.define(mxfp8_linear)
    cls = _cls()
    _ORIG_PWAL, _ORIG_APPLY = cls.process_weights_after_loading, cls.apply_weights
    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    own = ", ".join(f"{n}x{k} from {m}" for (n, k), m in sorted(MIN_M_NK.items()))
    return (f"mxfp8 -> hk for block-scaled layers of {len(MEASURED)} measured (N, K) at M >= "
            f"{MIN_M} ({own}) (compile cache {_own_compile_cache('hk-mxfp8')})")


def revert() -> str:
    global _ORIG_PWAL, _ORIG_APPLY
    if _ORIG_APPLY is None:
        return "mxfp8: not patched"
    cls = _cls()
    cls.process_weights_after_loading, cls.apply_weights = _ORIG_PWAL, _ORIG_APPLY
    _ORIG_PWAL = _ORIG_APPLY = None
    return "mxfp8 -> vllm"


def status() -> str:
    return "not patched" if _ORIG_APPLY is None else PATCH.summary()
