"""hk.templates.gemm with the residual as the accumulator's starting value
(init_from): the GEMM hk.ops.matmul runs for the shape, its accumulators
loaded from the residual before the K loop where the kernel can, so the add
is free and the result is rounded once (docs/templates/gemm.md)."""

import hk

KIND = "template"
LABEL = "hk.templates.gemm(epilogue=residual)"


def solve(x, weight, residual):
    return hk.templates.gemm(x, weight, epilogue=hk.templates.gemm.residual,
                             residual=residual)
