"""hk.map built by hipcc on every target with strips: gfx950 (include/hk/cdna), gfx1100 and
gfx1201 (include/hk/rdna/map.cuh over the trees' strip_each). Every op a map region prints --
integer, bit, mask, select, conversions, the math functions, hk.rcp -- in one kernel, without
spilling. Compile-only: hipcc, no GPU (tests/hk/gpu/test_map.py runs them)."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def _every_op(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], n: hk.Tensor[hk.i32],
              o: hk.Tensor[hk.i32], f: hk.Tensor[hk.fp32]):
    I = hk.RS(hk.i32, 1, 512, 8)
    F = hk.RS(hk.fp32, 1, 512, 8)
    row = hk.arch.block_idx.x
    a = x.at[row, 0].load(I)
    v = y.at[row, 0].load(F)
    k = n.at[0].load_scalar()
    m = (a < 0) & ~(a % 3 == 0) | (v >= 0.5) ^ (a != k)
    o.at[row, 0].store(hk.where(m, (a // 7 + k) << 2, hk.max(a >> 3, -5) ^ ~a)
                       + hk.math.popcount(a) - hk.math.ctz(a) + hk.min(a, 9) * -a)
    f.at[row, 0].store(hk.map(
        lambda e, i: hk.math.fma(e, e, hk.math.rcp(hk.math.exp2(-e)))
        + hk.math.ceil(hk.math.log2(abs(e) + 1.0)) - hk.math.floor(hk.math.sqrt(abs(e)))
        + e.to(hk.bf16).to(hk.fp32) * i.to(hk.fp32) / 3.0
        + (e.bitcast(hk.i32) & 0xFF).to(hk.fp32) + hk.where(e != e, 0.0, hk.max(e, 0.25)),
        v, a))


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100", "gfx1201"])
def test_every_map_op_builds(arch):
    k = hk.kernel(_every_op, arch=arch, warps=1, name=f"map_ops_{arch}",
                  grid=lambda p: (p.x.rows, 1, 1))
    tr = k.trace()
    src = scaffold.device(tr)
    assert src.count("kittens::map_cpp(") == 2          # each store's expression, fused
    b = hc.build(k.with_defines(src), arch, name=tr.name, kind="co")
    r = b.kernels[0]
    assert r.scratch == 0 and not r.spills, r.line()
