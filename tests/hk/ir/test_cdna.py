"""gfx942 / gfx950: the facts the targets carry, and what the translation does
differently for them.

Each one is something that compiled and computed the wrong answer, or would
have, before it was written down: a wave64 kernel launched with 32 threads a
warp; an MFMA operand spelled as a 16x16 tile; occupancy read off the VGPRs
alone when the AGPRs share the file; an LDS read the CDNA library never waits
for; a global->LDS DMA the barrier does not order (measured on an MI355X: a
different wrong GEMM every run).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.runtime.resources import parse  # noqa: E402
from hk.target import get_target  # noqa: E402

CDNA = ("gfx942", "gfx950")


@pytest.mark.parametrize("arch", CDNA)
def test_a_warp_is_64_threads(arch):
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch=arch, warps=4, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, hk.arch.warp_id())
        o[idx] = a[idx].load(t)

    tr = k.trace()
    assert tr.threads == 256
    src = scaffold.bare(tr)
    assert "dim3 block() const { return dim3(256); }" in src
    assert "__launch_bounds__(256, 1)" in src
    assert '#include "hk/cdna.cuh"' in src


def test_rdna_sources_do_not_include_the_cdna_shim():
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        o[idx] = a[idx].load(t)

    assert "hk/cdna.cuh" not in scaffold.bare(k.trace())


def test_the_register_file_is_512_with_a_granule_of_8():
    for arch in CDNA:
        t = get_target(arch)
        assert [t.occupancy(v) for v in (64, 65, 72, 73, 128, 129, 168, 169, 256)] == \
            [8, 7, 7, 6, 4, 3, 3, 2, 2]


def test_agprs_are_charged_with_the_vgprs():
    t = get_target("gfx950")
    # The sweep's points: 64 + 1 AGPR is a step down; AGPRs start at a
    # multiple of four, so 129 + 128 is 260, past the 256 of two waves.
    assert t.occupancy(t.charged_vgprs(64, 1)) == 7
    assert t.occupancy(t.charged_vgprs(129, 128)) == 1
    assert t.occupancy(t.charged_vgprs(128, 128)) == 2
    # RDNA has no AGPRs to charge.
    assert get_target("gfx1100").charged_vgprs(64, 16) == 64


def test_the_gate_reads_agprs_off_the_remarks():
    log = "\n".join([
        "k.cpp:1:1: remark: Function Name: _Z1kv [-Rpass-analysis=kernel-resource-usage]",
        "k.cpp:1:1: remark:     VGPRs: 64 [-Rpass-analysis=kernel-resource-usage]",
        "k.cpp:1:1: remark:     AGPRs: 16 [-Rpass-analysis=kernel-resource-usage]",
        "k.cpp:1:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]",
    ])
    (r,) = parse(log, get_target("gfx950"))
    assert (r.vgpr, r.agpr, r.vgpr_rounded, r.occ_by_reg) == (64, 16, 80, 6)


def test_tile_prices_follow_the_fragment():
    t = get_target("gfx950")
    assert t.tile_vgprs(16, 16, 16) == 2      # 256 bf16 over 64 lanes
    assert t.tile_vgprs(16, 16, 64) == 8
    assert t.tile_vgprs(32, 16, 16) == 4      # the 16x16 fp32 accumulator
    g = get_target("gfx942")
    assert (g.tile_vgprs(16, 32, 32), g.tile_vgprs(32, 32, 16)) == (8, 8)
    # Unchanged where the fragment is 16x16x16 with mirrored operands.
    assert get_target("gfx1100").tile_vgprs(16, 16, 16) == 8


def _mma(arch):
    ta = hk.RT(hk.bf16, 32, 32, "row")
    tb = hk.RT(hk.bf16, 16, 32, "row")
    tc = hk.RT(hk.fp32, 32, 16, "col")

    @hk.kernel(arch=arch, grid=lambda p: (1, 1, 1))
    def mma(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
        acc = hk.zeros(tc)
        acc += a.at[0, 0].load(ta) @ b.at[0, 0].load(tb).T
        c.at[0, 0] = acc

    return scaffold.bare(mma.trace())


def test_gfx950_operands_name_their_16x32_fragment():
    src = _mma("gfx950")
    assert ("rt<bf16, 32, 32, kittens::ducks::rt_layout::row, "
            "kittens::ducks::rt_shape::rt_16x32>") in src
    # The accumulator keeps the library's 16x16 default.
    assert "rt<float, 32, 16, kittens::ducks::rt_layout::col>;" in src


def test_gfx942_operands_are_the_16x16_tiles_hk_already_writes():
    assert "rt_shape" not in _mma("gfx942")


def _lds_gemm(arch, wait=True):
    st = hk.ST(hk.bf16, 64, 32)
    ta = hk.RT(hk.bf16, 32, 32, "row")
    tc = hk.RT(hk.fp32, 32, 32, "col")

    @hk.kernel(arch=arch, warps=2, grid=lambda p: (1, 1, 1))
    def lds_gemm(a: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
        g = hk.group(2)
        As = hk.lds.alloc(st, name="As")
        g.load(As, a.at[0, 0])
        hk.arch.barrier()
        at = As.subtile(32, 32, hk.arch.warp_id(), 0).load(ta, wait=wait)
        if not wait:
            hk.lds.wait_for(at)
        acc = hk.zeros(tc)
        acc += at @ at.T
        c.at[0, 0] = acc

    return scaffold.bare(lds_gemm.trace())


def test_cdna_waits_for_the_lds_reads_the_library_does_not():
    for arch in CDNA:
        src = _lds_gemm(arch)
        assert "kittens::load(lds, sub);\n    kittens::lds_wait_for<0>(lds);" in src
        assert "load<false>" not in src


def test_cdna_waits_for_the_global_to_lds_dma_before_the_barrier():
    src = _lds_gemm("gfx950")
    i = src.index("kittens::group<2>::load(As")
    assert src.index("kittens::vm_wait<0>();", i) < src.index("__builtin_amdgcn_s_barrier", i)


def test_gfx950_shared_tiles_take_the_fragment_of_what_is_read_out_of_them():
    assert "kittens::st<bf16, 64, 32, kittens::ducks::st_shape::st_16x32>" in _lds_gemm("gfx950")
    assert "st_shape" not in _lds_gemm("gfx942")


def test_unwaited_lds_reads_are_refused_until_the_count_is_per_target():
    with pytest.raises(TypeError, match="wait=False"):
        _lds_gemm("gfx950", wait=False)


@pytest.mark.parametrize("arch", CDNA)
def test_cdna_globals_carry_a_row_stride(arch):
    # include/cdna{3,4}'s gl has rows exactly `cols` apart; hk/cdna.cuh's
    # strided_gl has the stride the launchers pass and the code-object
    # layout table names.
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch=arch, warps=1, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        o[idx] = a[idx].load(t)

    src = scaffold.device(k.trace())
    assert "kittens::strided_gl<bf16> a;" in src
    assert "offsetof(decltype(globals::a), row_stride_internal)" in src
    assert "kittens::gl<" not in src


def test_rdna_globals_are_the_library_gl():
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch="gfx1100", warps=1, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        o[idx] = a[idx].load(t)

    src = scaffold.bare(k.trace())
    assert "kittens::gl<bf16, -1, -1, -1, -1> a;" in src
    assert "strided_gl" not in src

