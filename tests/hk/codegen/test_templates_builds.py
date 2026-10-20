"""The kernels hk.templates.elementwise and rowwise generate compile, for
gfx1100 and gfx950, without spilling at the plans they pick: the gate/up
halves, a broadcast operand, the norms, the fused add + quantize and a
cross-entropy streamed over 151936 columns. Needs hipcc, not a GPU."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import templates as T  # noqa: E402

B = hk.bf16


def _cases(arch):
    M = T.meta
    gelu = T.elementwise(lambda g, u: hk.math.gelu(g) * u, arch=arch)
    bias = T.elementwise(lambda x, b, s: hk.math.silu(x + b) * s, arch=arch)

    @T.rowwise(arch=arch)
    def rms(x, w, *, EPS=1e-6):
        return x * (T.row_mean(x * x) + EPS).rsqrt() * w

    @T.rowwise(arch=arch, outputs=(hk.i8, T.per_row(hk.fp32), hk.bf16))
    def add_rms_quant(x, r, w, *, EPS=1e-6):
        h = x + r
        y = h * (T.row_mean(h * h) + EPS).rsqrt() * w
        amax = T.row_max(abs(y)).maximum(1e-10)
        return y * (127.0 / amax), amax / 127.0, h

    @T.rowwise(arch=arch)
    def softmax(x):
        e = (x - T.row_max(x)).exp()
        return e / T.row_sum(e)

    @T.rowwise(arch=arch, outputs=T.per_row(hk.fp32))
    def ce(logits, label):
        return (T.row_logsumexp(logits) - T.pick(logits, label)) * (label != -100).to_int()

    half = (M((4096, 14336), B, (28672, 1)),) * 2
    return [
        ("gelu_halves", gelu, half),
        ("gelu_ragged", gelu, (M((37, 9217), B),) * 2),
        ("bias_scale", bias, (M((64, 4096), B), M((4096,), B), M((64, 1), hk.fp32))),
        ("rms_4096", rms, (M((2048, 4096), B), M((4096,), B))),
        ("rms_8192_long", rms, (M((8192, 8192), B), M((8192,), B))),
        ("rms_ragged", rms, (M((7, 4100), B), M((4100,), B))),
        ("add_rms_quant", add_rms_quant, (M((2048, 4096), B), M((2048, 4096), B), M((4096,), B))),
        ("softmax_8192", softmax, (M((2048, 8192), B),)),
        ("softmax_stream", softmax, (M((64, 151936), B),)),
        ("ce_stream", ce, (M((64, 151936), B), M((64,), hk.i32))),
        ("ce_32000", ce, (M((8192, 32000), B), M((8192,), hk.i32))),
    ]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
@pytest.mark.parametrize("case", range(11))
def test_template_kernels_build_without_spilling(arch, case):
    name, t, args = _cases(arch)[case]
    k, c = t.kernel(*args)
    built = k.build(**c).kernels
    assert built, f"{name}: no resource remarks parsed"
    for r in built:
        assert r.scratch == 0 and not r.spills, f"{name} on {arch}: {r.line()}"
