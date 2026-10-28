"""hk.arch.agpr: what the C++ emitter makes of an accumulator held in AGPRs
(include/hk/cdna_agpr.cuh), and what it refuses."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402

TA = hk.RT(hk.bf16, 32, 128, "row")
TS = hk.RT(hk.fp32, 32, 32, "col")


def _src(body, arch="gfx950"):
    return hk.kernel(body, arch=arch, warps=1, grid=lambda p: (1, 1, 1)).source()


def test_an_agpr_accumulator_runs_its_mmas_on_agprs_and_copies_for_the_rest():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        hk.arch.agpr(s)
        s += a @ a.T
        s.__imul__(2.0)
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    src = _src(k)
    assert re.search(r"hkagpr::acc<\w+> \w+;", src)
    assert "hkagpr::zero(" in src and "hkagpr::mma_ABt(" in src
    assert "kittens::mma_ABt(" not in src
    # the multiply and the store each read a drained VGPR copy and write it back
    assert src.count("hkagpr::drain();") == 2
    assert src.count("hkagpr::read(") == 2 and src.count("hkagpr::write(") == 2
    # the AGPR budget: a 32x32 fp32 tile is 16 a lane
    assert 'asm volatile("" ::: "a15");' in src


def test_a_column_scale_runs_in_place_a_base_tile_at_a_time():
    def k(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        a = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        s = hk.zeros(TS)
        hk.arch.agpr(s)
        s += a @ a.T
        m = s.max(axis=0)
        s.__imul__(hk.as_vec(m, TS))
        o[hk.tile_coord(0, 0, 0, 0)].store(s)

    src = _src(k)
    assert "hkagpr::mul_col(" in src


def test_agpr_is_cdna_and_takes_a_fresh_zero_tile():
    def rdna(x: hk.Tensor[hk.bf16]):
        hk.arch.agpr(hk.zeros(hk.RT(hk.fp32, 16, 16, "col")))

    with pytest.raises(TypeError, match="CDNA"):
        _src(rdna, arch="gfx1100")

    def not_zeros(x: hk.Tensor[hk.bf16]):
        t = x[hk.tile_coord(0, 0, 0, 0)].load(TA)
        hk.arch.agpr(t)

    with pytest.raises(TypeError, match="hk.zeros just made"):
        _src(not_zeros)


def test_the_attention_option_is_off_by_default():
    from hk.ops import attn_cdna

    assert "hkagpr::" not in attn_cdna.attn_kernel("agpr_off", head_dim=128, arch="gfx950").source()
    src = attn_cdna.attn_kernel("agpr_t", head_dim=128, causal=False, acc="agpr",
                                arch="gfx950").source()
    assert "hkagpr::mma_AtB(" in src and 'asm volatile("" ::: "a63");' in src
    with pytest.raises(ValueError, match="acc="):
        attn_cdna.attn_kernel("agpr_bad", head_dim=128, acc="sgpr", arch="gfx950")
