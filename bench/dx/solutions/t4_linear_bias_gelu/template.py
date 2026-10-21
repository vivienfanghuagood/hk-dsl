"""hk.templates.gemm with a bias + GELU(tanh) epilogue: the GEMM
hk.ops.matmul runs for the shape (the tuned RDNA kernel on gfx1100, the
routed CDNA kernels on gfx950), the bias and the GELU applied to its fp32
accumulator in registers before the one store (docs/templates/gemm.md)."""

import hk

KIND = "template"
LABEL = "hk.templates.gemm(epilogue=bias + gelu_tanh)"


def bias_gelu(acc, ctx):
    acc += ctx.bias                              # (N,): one value per column
    return hk.templates.gemm.gelu_tanh(acc)      # = F.gelu(approximate="tanh")


def solve(x, weight, bias):
    return hk.templates.gemm(x, weight, epilogue=bias_gelu, bias=bias)
