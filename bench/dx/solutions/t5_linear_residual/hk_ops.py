"""hk's tuned GEMM, then the residual add in torch: hk.ops.matmul (routed per
shape on gfx950, the tuned RDNA GEMM on gfx1100) and `+ residual` in place on
its output. hk has no GEMM with an epilogue."""

import hk

KIND = "hk_ops"
LABEL = "hk.ops.matmul + torch residual add"


def solve(x, weight, residual):
    return hk.ops.matmul(x, weight).add_(residual)
