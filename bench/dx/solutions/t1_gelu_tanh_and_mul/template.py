"""hk.templates.elementwise with a GELU(tanh)-and-mul hook. The two halves of
x go in as strided views, read in place; the template picks the plan (strips,
the run, warps, the grid order) for the GPU it runs on."""

import hk

KIND = "template"
LABEL = "hk.templates.elementwise(lambda g, u: gelu(g) * u)"

gelu_and_mul = hk.templates.elementwise(lambda g, u: hk.math.gelu(g) * u)


def solve(x):
    i = x.shape[-1] // 2
    return gelu_and_mul(x[:, :i], x[:, i:])
