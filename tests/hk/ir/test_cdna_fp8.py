"""fp8 e4m3 operands for gfx950's MFMA: v_mfma_f32_16x16x128_f8f6f4 (the
scaled MFMA at unit scales). An fp8 operand is a 16x128 (row) fragment; the
fragment pass names it, the C++ spells it rt_16x128, and the accumulator
stays the 16x16 fp32 one."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import fp8e4m3, fp32  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.target import get_target  # noqa: E402
from hk.types import MxScalesType  # noqa: E402


def fp8_mma_kernel(arch="gfx950", name="fp8_mma_t"):
    def body(a, b, c):
        ta = a[hk.elem_coord(0, 0, hk.arch.block_idx.x * 16, 0)].load(hk.RT(fp8e4m3, 16, 128))
        tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, 128))
        acc = hk.zeros(hk.RT(fp32, 16, 16, "col"))
        acc += ta @ tb.T
        c[hk.elem_coord(0, 0, hk.arch.block_idx.x * 16, 0)] = acc

    body.__name__ = name
    body.__annotations__ = {"a": hk.Tensor[fp8e4m3], "b": hk.Tensor[fp8e4m3],
                            "c": hk.Tensor[fp32]}
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (p.a.rows // 16, 1, 1))


def test_gfx950_has_an_fp8_mma_of_k_128():
    s = get_target("gfx950").mma["fp8e4m3"]
    assert (s.m, s.n, s.k, s.operand_vgprs) == (16, 16, 128, 8)


def test_an_fp8_operand_is_a_16x128_fragment():
    src = scaffold.bare(fp8_mma_kernel().trace())
    assert ("kittens::rt<fp8e4m3, 16, 128, kittens::ducks::rt_layout::row, "
            "kittens::ducks::rt_shape::rt_16x128>") in src
    assert "kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col>" in src


def test_gfx1100_still_refuses_fp8_mma():
    with pytest.raises(Exception, match="fp8"):
        fp8_mma_kernel(arch="gfx1100", name="fp8_mma_rdna").trace()


# -- block-scaled (OCP MX) --------------------------------------------------


def mx_mma_kernel(K=256, name="mx_mma_t"):
    """C = (sa . A)(sb . B)^T for a 16-row strip of A against 32 rows of B."""
    def body(a, b, sa, sb, c):
        r0 = hk.arch.block_idx.x * 16
        ta = a[hk.elem_coord(0, 0, r0, 0)].load(hk.RT(fp8e4m3, 16, K))
        tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 32, K))
        s_a = hk.load_mx_scales(sa, 16, K // 32, r0, 0)
        s_b = hk.load_mx_scales(sb, 32, K // 32, 0, 0)
        acc = hk.mma("mma_ABt", ta.scaled(s_a), tb.scaled(s_b),
                     hk.zeros(hk.RT(fp32, 16, 32, "col")))
        c[hk.elem_coord(0, 0, r0, 0)] = acc

    body.__name__ = name
    body.__annotations__ = {"a": hk.Tensor[fp8e4m3], "b": hk.Tensor[fp8e4m3],
                            "sa": hk.Tensor[hk.i8], "sb": hk.Tensor[hk.i8],
                            "c": hk.Tensor[fp32]}
    return hk.kernel(body, arch="gfx950", warps=1, name=name,
                     grid=lambda p: (p.a.rows // 16, 1, 1))


def test_mx_scales_and_the_scaled_mma_reach_the_cpp():
    src = scaffold.bare(mx_mma_kernel().trace())
    assert "kittens::hk_mxs<16, 8>" in src and "kittens::hk_mxs<32, 8>" in src
    assert "kittens::load_mx_scales(" in src and "kittens::mma_ABt_mx<0, 0>(" in src


def test_mx_shapes_are_checked_while_tracing():
    def body(a, sa, c):
        ta = a[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, 256))
        s = hk.load_mx_scales(sa, 16, 4, 0, 0)          # 128 wide, not 256
        c[hk.elem_coord(0, 0, 0, 0)] = hk.mma("mma_ABt", ta.scaled(s), ta.scaled(s),
                                              hk.zeros(hk.RT(fp32, 16, 16, "col")))

    body.__annotations__ = {"a": hk.Tensor[fp8e4m3], "sa": hk.Tensor[hk.i8],
                            "c": hk.Tensor[fp32]}
    with pytest.raises(TypeError, match="holds the scales of a 16x256"):
        hk.kernel(body, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace()
    with pytest.raises(ValueError, match="128-wide K steps"):
        MxScalesType(16, 6)
