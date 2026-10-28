"""An LDS tile narrower (or shorter) than the include/cdna4 st shape its read's fragment picks.

A 32x16 bf16 tile read as the 32x16 (run 8) row operand of the 32x32x16 MFMA picks st_32x32, whose
subtiles are 32 columns wide: the tile holds none of them, and include/cdna4's copies loop over
cols / 32 = 0 subtiles, so its stores and loads compiled to nothing and the kernel read whatever the LDS
held (tests/hk/gpu/test_lds_narrow_tile.py has the numbers). The translator now takes the fragment's own
shape when that one tiles the allocation, and refuses one that no shape tiles.

Tracing only: no hipcc, no GPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402


def _narrow(cols: int):
    st = hk.ST(hk.bf16, 32, cols)

    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        s = hk.lds.alloc(st, name="s")
        s.store(x.at[0, 0].load(hk.RT(hk.bf16, 32, cols, "row")))
        hk.arch.barrier(drain=True)
        p = s.load(hk.RT(hk.bf16, 32, cols, "row", frag=(32, 16)))
        one = hk.full(hk.RT(hk.bf16, 32, cols, "row", frag=(32, 16)), 1.0)
        acc = hk.zeros(hk.RT(hk.fp32, 32, 32, "col", frag=(32, 32)))
        acc += p @ one.T
        o.at[0, 0].store(acc)

    return hk.kernel(k, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1), name=f"narrow_{cols}")


def test_a_tile_narrower_than_its_fragment_s_shape_takes_one_that_tiles_it():
    src = scaffold.bare(_narrow(16).trace())
    assert "kittens::st<bf16, 32, 16, kittens::ducks::st_shape::st_32x16>" in src
    assert "st<bf16, 32, 16, kittens::ducks::st_shape::st_32x32>" not in src


def test_a_tile_wide_enough_keeps_the_fragment_s_shape():
    src = scaffold.bare(_narrow(32).trace())
    assert "kittens::st<bf16, 32, 32, kittens::ducks::st_shape::st_32x32>" in src
