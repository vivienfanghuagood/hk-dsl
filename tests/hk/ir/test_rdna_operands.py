"""RDNA's lowering of the CDNA forms of P.V (#212, #213), CPU only."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402


def _src(body, arch):
    return scaffold.bare(hk.kernel(body, arch=arch, warps=1, grid=lambda p: (1, 1, 1)).trace())


def _pv(p_form, v_layout):
    def body(k: hk.Tensor[hk.bf16], q: hk.Tensor[hk.bf16], v: hk.Tensor[hk.bf16],
             o: hk.Tensor[hk.fp32]):
        v_s = hk.lds.alloc(hk.ST(hk.bf16, 32, 16), name="v_s")
        hk.group(1).load(v_s, v[hk.elem_coord(0, 0, 0, 0)])
        hk.arch.barrier(drain=True)
        s = hk.zeros(hk.RT(hk.fp32, 32, 16, "col"))
        s += (k[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(hk.bf16, 32, 32, "row"))
              @ q[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(hk.bf16, 16, 32, "row")).T)
        p = s.as_operand(hk.bf16) if p_form == "as_operand" else s.to(hk.bf16)
        acc = hk.zeros(hk.RT(hk.fp32, 16, 16, "col"))
        acc += v_s.load(hk.RT(hk.bf16, 32, 16, v_layout)).T @ p
        o[hk.elem_coord(0, 0, 0, 0)] = acc
    return body


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_as_operand_traces_on_rdna_as_the_accumulators_cast(arch):
    """#212: refused on gfx1100 before ('gfx1100's register tiles have no fragment shape')."""
    src = _src(_pv("as_operand", "col"), arch)
    assert "mma" in src


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_a_col_layout_lds_operand_is_read_row_major_on_rdna(arch):
    """#213: a col-layout 16-bit LDS read is b128 row reads plus a register transpose, not 16
    ds_read_u16 a tile (the budget pass's slow-smem-layout warning, before) -- the same C++ as
    an explicit row read, its wait and transpose_sep."""
    src = _src(_pv("cast", "col"), arch)
    assert "slow-smem-layout" not in src
    assert "kittens::transpose_sep(" in src


def test_as_operand_still_refuses_what_it_refused():
    """A non-col or non-accumulator tile is still refused on RDNA."""
    def body(x: hk.Tensor[hk.bf16]):
        t = x[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(hk.bf16, 16, 16, "row"))
        t.as_operand(hk.bf16)

    with pytest.raises(TypeError, match="col-layout accumulator"):
        _src(body, "gfx1100")
