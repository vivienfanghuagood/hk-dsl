"""gfx950: a window of an LDS tile must be whole subtiles of the layout the tile is declared with.

include/cdna4's `st_subtile` points at the layout's subtile holding the window's first element
and keeps no offset inside it. A window narrower or shorter than that subtile therefore reads, or
is copied into, the subtile's first rows and columns whatever its index -- and HipKittens says
nothing. The translation used to check this only for a tile that names its layout
(`hk.lds.alloc(..., shape=...)`). A tile whose layout its reads' fragment picks -- a 32x16
stride-4 col read, P.V's V^T on gfx950, picks st_32x16 -- went unchecked: a paged decode copying
V's two 16-row page bands into such a tile computed with V all ones about 0.5, not 1.0 (#217).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402


def _bare(body):
    return scaffold.bare(hk.kernel(body, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace())


def test_16_row_bands_of_a_tile_read_as_stride4_col_operands_are_refused():
    """#217's kernel: V's 32-key block filled as two 16-row page bands, then read as 32x16 col
    (stride-4) operands. The reads lay the tile out as st_32x16; a 16-row band is half a
    subtile of it."""
    def body(v: hk.Tensor[hk.bf16], p: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        v_s = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="v_s")
        g = hk.group(1)
        for band in range(2):
            g.load(v_s.subtile(16, 128, band, 0), v.at[0, 0, band, 0])
        hk.arch.barrier()
        pt = hk.zeros(hk.RT(hk.fp32, 32, 16, "col")).as_operand(hk.bf16)   # P^T, as P.V reads it
        acc = hk.zeros(hk.RT(hk.fp32, 16, 16, "col"))
        vt = v_s.subtile(32, 16, 0, 1).load(hk.RT(hk.bf16, 32, 16, "col"))
        acc += vt.T @ pt
        o.at[0, 0] = acc

    with pytest.raises(Exception, match=r"16x128 window of an LDS tile laid out as st_32x16 "
                                        r"\(the layout its reads' fragment picks"):
        _bare(body)


def test_whole_32_row_copies_of_the_same_tile_build():
    """The same tile filled whole (one 32-row copy) and read as 32x16 col operands: every
    window is whole st_32x16 subtiles."""
    def body(v: hk.Tensor[hk.bf16], p: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        v_s = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="v_s")
        hk.group(1).load(v_s, v.at[0, 0])
        hk.arch.barrier()
        pt = hk.zeros(hk.RT(hk.fp32, 32, 16, "col")).as_operand(hk.bf16)   # P^T, as P.V reads it
        acc = hk.zeros(hk.RT(hk.fp32, 16, 16, "col"))
        vt = v_s.subtile(32, 16, 0, 1).load(hk.RT(hk.bf16, 32, 16, "col"))
        acc += vt.T @ pt
        o.at[0, 0] = acc

    assert "st_shape::st_32x16>" in _bare(body)


def test_a_window_narrower_than_a_named_layout_is_still_refused():
    """The named-layout case the check covered before (st_16x64's 64-column subtiles)."""
    def body(x: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 64, 64), name="s", shape="16x64")
        hk.group(1).load(s, x.at[0, 0])
        hk.arch.barrier()
        x.at[0, 0] = s.subtile(64, 32, 0, 1).load(hk.RT(hk.bf16, 64, 32, "row"))

    with pytest.raises(Exception, match="is not whole 16x64 subtiles"):
        _bare(body)


def test_an_unread_tile_of_another_tiles_type_takes_its_layout_and_is_checked_too():
    """Two allocations of one type share the type's layout. The one only copied into (never read
    as a fragment) is declared with the layout the other's reads picked, and its windows are held
    to it as well."""
    def body(v: hk.Tensor[hk.bf16], p: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="a")
        b = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="b")
        g = hk.group(1)
        g.load(a, v.at[0, 0])
        g.load(b.subtile(16, 128, 1, 0), v.at[0, 0, 1, 0])      # half a subtile of st_32x16
        hk.arch.barrier()
        pt = hk.zeros(hk.RT(hk.fp32, 32, 16, "col")).as_operand(hk.bf16)   # P^T, as P.V reads it
        acc = hk.zeros(hk.RT(hk.fp32, 16, 16, "col"))
        acc += a.subtile(32, 16, 0, 0).load(hk.RT(hk.bf16, 32, 16, "col")).T @ pt
        o.at[0, 0] = acc

    with pytest.raises(Exception, match=r"16x128 window of an LDS tile laid out as st_32x16"):
        _bare(body)
