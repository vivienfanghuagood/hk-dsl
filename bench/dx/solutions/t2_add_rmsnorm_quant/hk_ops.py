"""hk's tuned building blocks: hk.ops.fused_add_rms_norm (vLLM's in-place
form: residual <- x + residual, x <- RMSNorm(residual) * weight) and then
hk.ops.quantize (per-row symmetric int8). Two kernels, the normed row through
memory in bf16 between them; hk has no fused add + norm + quant."""

import hk
from hk.target import current_arch

KIND = "hk_ops"
LABEL = "hk.ops.fused_add_rms_norm + hk.ops.quantize (two kernels, in place)"


def supports(x, residual, weight, eps):
    if x.shape[0] < 16 and current_arch() != "gfx950":
        return f"hk.ops.quantize refuses fewer than 16 rows on {current_arch()} (T={x.shape[0]})"
    return True


def solve(x, residual, weight, eps):
    hk.ops.fused_add_rms_norm(x, residual, weight, eps)
    q, scales = hk.ops.quantize(x)
    return q, scales.view(-1, 1), residual
