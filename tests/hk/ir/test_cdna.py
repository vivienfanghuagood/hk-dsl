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


@pytest.mark.parametrize("arch", CDNA + ("gfx1100", "gfx1201"))
def test_the_cdna_entry_point_takes_the_stream(arch):
    """On the null stream a launch is ordered against nothing torch did on
    its own streams and a CUDA graph cannot capture it; on CDNA the pybind
    entry point takes the stream Kernel.__call__ passes (torch's current
    one). RDNA's is unchanged."""
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(arch=arch, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, 0, 0)
        o[idx] = a[idx].load(t)

    src = scaffold.pybind_module(k.trace())
    cdna = arch in CDNA
    assert scaffold.takes_stream(arch) == cdna
    assert ("bind_function_on_stream<launch_on>(m, \"k\"" in src) == cdna
    assert ("bind_function<launch>(m, \"k\"" in src) == (not cdna)


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
    # an fp8 16x128 operand is 8 VGPRs, not priced as fp32 (experiment 2's hk arm: a register
    # prefetch refused at an estimated 576 VGPRs where ~200 were needed)
    assert t.tile_vgprs(8, 16, 128) == 8 and t.tile_vgprs(8, 128, 256) == 128
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


def _two_loads(arch, depth=1):
    st = hk.ST(hk.bf16, 32, 32)
    ta = hk.RT(hk.bf16, 32, 32, "row")
    tc = hk.RT(hk.fp32, 32, 32, "col")

    @hk.kernel(arch=arch, warps=1, grid=lambda p: (1, 1, 1))
    def two(a: hk.Tensor[hk.bf16], c: hk.Tensor[hk.bf16]):
        As = hk.lds.alloc(st, name="As")
        Bs = hk.lds.alloc(st, name="Bs")
        g = hk.group(1)
        g.load(As, a.at[0, 0])
        g.load(Bs, a.at[0, 0])
        hk.arch.barrier()
        x = As.load(ta, wait=False)
        ys = [Bs.load(ta, wait=False) for _ in range(depth)]
        hk.lds.wait_for(x)
        acc = hk.zeros(tc)
        acc += x @ x.T
        for y in ys:
            hk.lds.wait_for(y)
            acc += y @ y.T
        c.at[0, 0] = acc

    return two.trace()


def test_unwaited_lds_reads_wait_for_the_target_count():
    # gfx950: a 32x32 operand is two 16x32 fragments, one ds_read_b128 each,
    # so retiring x leaves y's two in flight. gfx942: one ds_read_b64 per
    # 16x16, four. (include/rdna3's count would be 8.)
    for arch, n in (("gfx950", 2), ("gfx942", 4)):
        src = scaffold.bare(_two_loads(arch))
        assert f"kittens::lds_wait_for<{n}>(lds);" in src, arch
        assert "kittens::lds_wait_for<0>(lds1);" in src, arch


def test_the_ir_names_the_fragment_and_the_count():
    ir = _two_loads("gfx950").dump()
    assert "!hk.rt<bf16, 32x32, row, frag 16x32>" in ir
    assert "lds_loads = 2" in ir
    assert "frag" not in _two_loads("gfx942").dump()


def test_a_wait_deeper_than_the_counter_is_the_counter():
    # Eight loads of two ds_reads behind x: lgkmcnt(16) does not encode in
    # gfx9's 4 bits, and a wave cannot have more than 15 outstanding anyway.
    src = scaffold.bare(_two_loads("gfx950", depth=8))
    assert "kittens::lds_wait_for<15>(lds);" in src


def test_an_accumulator_cast_into_an_operand_is_refused_on_gfx950():
    # Attention's S -> P: a 16x16 accumulator fragment is not a 16x32
    # operand; on gfx950 that is a relayout across lanes, not a cast.
    tq = hk.RT(hk.bf16, 32, 32, "row")
    ts = hk.RT(hk.fp32, 32, 32, "col")

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1))
    def sp(q: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        x = q.at[0, 0].load(tq)
        s = hk.zeros(ts)
        s += x @ x.T
        p = s.to(hk.bf16)
        acc = hk.zeros(ts)
        acc += x @ p
        o.at[0, 0] = acc

    with pytest.raises(Exception, match="fragments"):
        sp.trace()


def test_cdna_lds_stores_and_group_loads_end_with_the_lds_counter_empty():
    # hk-lds-pipeline takes both to retire every LDS op, as include/rdna3's
    # do; include/cdna4's vectorised store returns with its ds_writes in
    # flight, and the DMA counts on vmcnt only.
    t = hk.RT(hk.bf16, 16, 32)
    st = hk.ST(hk.bf16, 16, 32)

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(st, name="s")
        s.store(a.at[0, 0].load(t))
        hk.arch.barrier()
        o.at[0, 0] = s.load(t)

    src = scaffold.bare(k.trace())
    i = src.index("kittens::store(s, ")
    assert src.index("kittens::lds_wait<0>();", i) < src.index("__builtin_amdgcn_s_barrier", i)
    src = _lds_gemm("gfx950")
    i = src.index("kittens::vm_wait<0>();")
    assert src.index("kittens::lds_wait<0>();", i) < src.index("__builtin_amdgcn_s_barrier", i)


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


def test_a_kernel_for_another_arch_is_refused_before_it_loads(monkeypatch):
    # A gfx1100 code object launched on a gfx950 crashed the process.
    from hk.runtime import arch as rta  # noqa: PLC0415

    monkeypatch.setattr(rta, "device_arch", lambda device: "gfx950")
    rta.check_device("gfx950", "kernel 'k'", device=0)
    with pytest.raises(rta.ArchMismatchError, match="built for gfx1100, but GPU 0 is gfx950"):
        rta.check_device("gfx1100", "kernel 'k'", device=0)
    monkeypatch.setattr(rta, "device_arch", lambda device: None)
    rta.check_device("gfx1100", "kernel 'k'", device=0)   # no GPU: nothing to check


def test_gfx950_elementwise_and_row_tiles_take_16x32_fragments():
    # 16-byte runs a lane instead of 8: bf16 add went 0.58x -> 0.92x torch.
    from hk.ops import elementwise, norm  # noqa: PLC0415

    def ir(k, arch, **c):
        return hk.kernel(k.fn, arch=arch, warps=k.warps, grid=k.grid_fn,
                         name=k.name).trace(**c).dump()

    add = elementwise.KERNELS["add_bf16"]
    assert "frag 16x32" in ir(add, "gfx950", COLS=256)
    assert "frag" not in ir(add, "gfx942") and "frag" not in ir(add, "gfx1100")
    rms = norm.KERNELS["rms_bf16_w4"]
    assert "!hk.rt<f32, 16x64, row, frag 16x32>" in ir(rms, "gfx950", TPW=1)
    # Folded rows reduce through a 16x16 tile: no fragment there.
    assert "frag" not in ir(rms, "gfx950", TPW=1, FOLD=16)


def test_a_fragment_on_a_target_without_tile_shapes_is_refused():
    t = hk.RT(hk.bf16, 16, 64, frag=(16, 32))

    @hk.kernel(arch="gfx942", warps=1, grid=lambda p: (1, 1, 1))
    def k(a: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        o.at[0, 0] = a.at[0, 0].load(t)

    with pytest.raises(Exception, match="no fragment shape"):
        k.trace()


def test_the_spill_gate_refuses_memory_spills_not_sgpr_lane_spills():
    """An SGPR spill with ScratchSize 0 went to VGPR lanes (v_writelane): no memory traffic,
    nothing a hand-placed s_waitcnt can reorder against. A VGPR spill, or any scratch, is a wrong
    answer and is refused (K1 c32: a per-expert buffer resource spilled 8 SGPRs to lanes)."""
    import pytest as _pytest

    from hk.runtime.resources import KernelResources, ResourceError, gate

    gate([KernelResources(name="k", mangled="k", vgpr=180, sgpr=104, sgpr_spill=8)])
    for bad in (dict(sgpr_spill=8, scratch=16), dict(vgpr_spill=2, scratch=8), dict(scratch=4)):
        with _pytest.raises(ResourceError, match="ScratchSize"):
            gate([KernelResources(name="k", mangled="k", vgpr=180, **bad)])


def test_a_bitcast_keeps_the_element_width():
    import pytest as _pytest

    def k(x: hk.Tensor[hk.fp32], y: hk.Tensor[hk.bf16]):
        y.at[0, 0] = x.at[0, 0].load(hk.RT(hk.fp32, 16, 64)).bitcast(hk.bf16)

    with _pytest.raises(TypeError, match="keeps the element width"):
        hk.kernel(k, arch="gfx950", grid=lambda p: (1, 1, 1)).trace()
