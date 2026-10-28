"""Tile.make_causal is a deprecated alias (docs/design/dsl-surface.md, D5): it warns and emits
the fill that covers it -- tril with diagonal 0, or triu with transposed=True -- on a copy, so
the tile itself is unchanged, as before. Its own dialect ops are gone."""

from __future__ import annotations

import warnings

import pytest

import hk
from hk import fp32


def _src(transposed: bool, arch: str) -> str:
    def body(s: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        t = s[hk.tile_coord(0, 0, 0, 0)].load(hk.RT(fp32, 32, 32, "col"))
        m = t.make_causal(float("-inf"), transposed=transposed)
        o[hk.tile_coord(0, 0, 0, 0)] = m
        o[hk.tile_coord(0, 1, 0, 0)] = t

    k = hk.kernel(body, arch=arch, grid=lambda p: (1, 1, 1),
                  name=f"mc_{int(transposed)}_{arch}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        src = k.source()
    assert any(issubclass(x.category, DeprecationWarning) and "make_causal" in str(x.message)
               for x in w)
    return src


@pytest.mark.parametrize("arch", ["gfx1100", "gfx950"])
@pytest.mark.parametrize("transposed, fill", [(False, "tril"), (True, "triu")])
def test_make_causal_is_the_diagonal_fill(arch, transposed, fill):
    src = _src(transposed, arch)
    assert f"kittens::{fill}(" in src
    assert "make_causal" not in src
