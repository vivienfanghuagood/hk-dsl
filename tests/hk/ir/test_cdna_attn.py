"""gfx950's attention and what it is written with: a 32x32 accumulator, the
operand read off an accumulator's registers (a 16x32 fragment with a lane run
of 4), a register subtile, the wave vote on the running max, and the pin that
holds a phase's results to it. Then the kernel's shape, and hk.ops.attn's
route to it on gfx950.

Tracing only: no hipcc, no GPU. The build gate is
tests/hk/codegen/test_cdna_attn_builds.py; the numerics are
tests/hk/gpu/test_attn.py, which runs on the kernel on gfx950.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.ops import attn_cdna  # noqa: E402

fp32, bf16 = hk.fp32, hk.bf16
T_S = hk.RT(fp32, 64, 32, "col", frag=(32, 32))
T_P = hk.RT(bf16, 64, 32, "col", frag=(16, 32, 4))
T_K = hk.RT(bf16, 64, 64, "row")
T_Q = hk.RT(bf16, 32, 64, "row")


def _trace(fn, arch="gfx950", warps=8):
    return hk.kernel(fn, arch=arch, warps=warps, grid=lambda p: (1, 1, 1)).trace()


def _scores(q, k):
    """Sᵀ = K·Qᵀ in a 32x32 accumulator, the way the kernel computes it."""
    s = hk.zeros(T_S)
    s += k.at[0, 0].load(T_K) @ q.at[0, 0].load(T_Q).T
    return s


# -- the language -------------------------------------------------------------


def test_the_operand_fragment_with_a_run_of_4_is_its_own_type():
    """16x32 is the operand fragment; a lane holds 8 consecutive rows of it,
    or with `stride 4` two runs of 4 -- which is what a 32x32 accumulator's
    lane holds, so the two are different tiles. A fragment's own run is
    spelled without one."""
    assert T_P != hk.RT(bf16, 64, 32, "col", frag=(16, 32))
    assert hk.RT(bf16, 64, 32, "col", frag=(16, 32, 8)) == hk.RT(bf16, 64, 32, "col", frag=(16, 32))

    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        s = _scores(q, kk)
        p = hk.zeros(T_P)
        s.as_operand(bf16, into=p)

    tr = _trace(k)
    assert "!hk.rt<bf16, 64x32, col, frag 16x32 stride 4>" in tr.dump()
    assert "kittens::ducks::rt_shape::rt_16x32_4>" in scaffold.bare(tr)


def test_an_accumulator_is_read_as_an_operand_in_place():
    """No conversion op, no register moves: the shim's as_operand converts
    fp32 to bf16 pairs into the operand's registers, which are the
    accumulator's rows."""
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        s = _scores(q, kk)
        p = hk.zeros(T_P)
        s.as_operand(bf16, into=p)

    tr = _trace(k)
    assert re.search(r"hk\.as_operand %\w+ into %\w+ : !hk\.rt<f32, 64x32, col, frag 32x32> "
                     r"into !hk\.rt<bf16, 64x32, col, frag 16x32 stride 4>", tr.dump())
    assert re.search(r"kittens::as_operand\(\w+, \w+\);", scaffold.bare(tr))


def test_a_32x32_accumulator_is_wave64_only():
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        _scores(q, kk)

    with pytest.raises(TypeError, match="32x32"):
        _trace(k, "gfx1100", warps=4)


def test_a_register_subtile_is_the_tile_s_own_registers():
    """exp2 into the second half writes the tile: a reference, not a copy."""
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        s = _scores(q, kk)
        s.subtile(32, 1).exp2_()

    src = scaffold.bare(_trace(k))
    m = re.search(r"auto &(\w+) = kittens::subtile_inplace<32>\((\w+), 1\);", src)
    assert m
    assert re.search(rf"kittens::exp2\({m.group(1)}, {m.group(1)}\);", src)


def test_the_vote_is_one_scalar_a_wave():
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16], o: hk.Tensor[fp32]):
        s = _scores(q, kk)
        m = hk.zeros(hk.row_vec(T_S))
        m.max_(s.max(axis=0))
        m0 = hk.zeros(hk.row_vec(T_S))
        if m.any_above(m0, 8.0) > 0:
            s.exp2_()
        o.at[0, 0] = hk.transpose(s)

    tr = _trace(k)
    assert "hk.any_above" in tr.dump()
    src = scaffold.bare(tr)
    assert re.search(r"> \(float\)8\.0\);", src) and "__any(" in src


def test_a_pin_binds_each_value_it_names():
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        s = _scores(q, kk)
        m = hk.zeros(hk.row_vec(T_S))
        hk.arch.pin(s, m)

    src = scaffold.bare(_trace(k))
    assert len(re.findall(r"kittens::lds_bind\(\w+\);", src)) == 2


def test_a_pin_takes_register_values_only():
    def k(q: hk.Tensor[bf16], kk: hk.Tensor[bf16]):
        s = _scores(q, kk)
        hk.arch.pin(s.max(axis=0))

    with pytest.raises(TypeError, match="register tiles and vectors"):
        _trace(k)


# -- the kernel ---------------------------------------------------------------


@pytest.mark.parametrize("d", attn_cdna.HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True])
def test_the_kernel_traces_with_its_copies_and_masks(d, causal):
    tr = attn_cdna.attn_kernel(f"attn_t_{d}_{int(causal)}", head_dim=d, causal=causal,
                               arch="gfx950").trace()
    src = scaffold.bare(tr)
    # Three copies before the pingpong, two in its prologue, two a step
    # (two steps an iteration); the last two steps copy nothing.
    assert src.count("kittens::group_load_async<8>(") == 3 + 2 + 4
    # P·V reads V with P's permutation: hk-fragments gives the MMA's other
    # operand the stride of the one read off the accumulator.
    assert f"!hk.rt<bf16, 64x{d}, col, frag 16x32 stride 4>" in tr.dump()
    assert ("kittens::triu(" in src) == causal
    assert ("kittens::lower_fill(" in src) == (not causal)
    # One division a query at the end, not one an element.
    assert "kittens::div_col(" not in src


def test_k_and_v_share_a_shared_tile_type_but_not_its_lds_layout():
    """K's blocks are read as row operands and V's as the transposed,
    stride-4 one: one !hk.st<bf16, 64x128> each, two st_shapes, so each
    allocation gets the layout its reads want."""
    src = scaffold.bare(attn_cdna.attn_kernel("attn_t_shapes", arch="gfx950").trace())
    shapes = set(re.findall(r"kittens::st<bf16, 64, 128, (kittens::ducks::st_shape::\w+)>", src))
    assert len(shapes) == 2, shapes


def test_causal_first_half_warps_skip_the_last_two_steps():
    """The region is a loop of 0 or 1 trips (a branch may not hold a barrier),
    and the other has the steps' seven barriers too."""
    src = scaffold.bare(attn_cdna.attn_kernel("attn_t_skip", causal=True,
                                              arch="gfx950").trace())
    plain = scaffold.bare(attn_cdna.attn_kernel("attn_t_noskip", causal=True, skip=False,
                                                arch="gfx950").trace())
    assert src.count("__builtin_amdgcn_s_barrier()") > plain.count("__builtin_amdgcn_s_barrier()")


def test_a_kernel_per_scale_named_by_its_bits():
    a = attn_cdna.kernel_for(128, False)
    assert a is attn_cdna.kernel_for(128, False, 128 ** -0.5)
    b = attn_cdna.kernel_for(128, False, 0.1)
    assert a is not b and a.name == "attn_fwd_d128_cdna"
    assert re.fullmatch(r"attn_fwd_d128_cdna_s[0-9a-f]{16}", b.name)


def test_hk_attention_routes_to_it_on_gfx950(monkeypatch):
    from hk import target
    from hk.ops import attn

    monkeypatch.setattr(target, "current_arch", lambda: "gfx950")
    assert attn._kernel_for(64, True, None).name == "attn_fwd_d64_causal_cdna"
    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    assert not attn._kernel_for(64, True, None).name.endswith("_cdna")


# -- the spatial numbering ----------------------------------------------------


@pytest.mark.parametrize("batch,h,h_kv,n_t", [
    (1, 128, 128, 128), (2, 40, 40, 17), (1, 64, 8, 32), (4, 8, 1, 64), (1, 28, 4, 9),
    (3, 24, 24, 5), (1, 8, 8, 1), (1, 2, 1, 3)])
def test_the_spatial_numbering_is_a_bijection_one_cluster_an_xcd_at_a_time(batch, h, h_kv, n_t):
    """Every (batch, head, query tile) once, whatever the total's remainder
    mod 8; each XCD's workgroups (w = d, d + 8, ...) a contiguous run of
    clusters in order, so its resident ones read as few K/V streams as the
    split allows; the XCDs' loads differ by at most one."""
    group = h // h_kv
    total = batch * h * n_t
    got = [attn_cdna._spatial(w, batch, h_kv, group, n_t) for w in range(total)]
    assert {(b, hd, t) for b, _, hd, t in got} == {
        (b, hd, t) for b in range(batch) for hd in range(h) for t in range(n_t)}
    assert all(hd // group == kv for _, kv, hd, _ in got)
    loads = [len(range(d, total, 8)) for d in range(8)]
    assert max(loads) - min(loads) <= 1
    for d in range(8):
        clusters = [(got[w][0], got[w][1]) for w in range(d, total, 8)]
        # a cluster is entered once and left for good
        runs = [c for i, c in enumerate(clusters) if i == 0 or c != clusters[i - 1]]
        assert len(runs) == len(set(runs))


@pytest.mark.parametrize("d", attn_cdna.HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True])
def test_the_spatial_kernel_is_the_same_kernel_on_a_flat_grid(d, causal):
    """Only the numbering differs: the same copies and masks, one more name."""
    base = scaffold.bare(attn_cdna.attn_kernel(f"attn_t_bf_{d}_{int(causal)}", head_dim=d,
                                               causal=causal, arch="gfx950").trace())
    sp = attn_cdna.attn_kernel(f"attn_t_sp_{d}_{int(causal)}", head_dim=d, causal=causal,
                               mapping="spatial", arch="gfx950")
    src = scaffold.bare(sp.trace())
    for op in ("kittens::group_load_async<8>(", "kittens::triu(", "kittens::lower_fill(",
               "__builtin_amdgcn_s_barrier()"):
        assert src.count(op) == base.count(op), op
    assert "blockIdx.z" in base and "blockIdx.z" not in src


def test_the_spatial_paged_prefill_traces_and_is_its_own_kernel():
    a = attn_cdna.paged_kernel_for(128)
    b = attn_cdna.paged_kernel_for(128, mapping="spatial")
    assert a is not b and b.name == "paged_prefill_d128_cdna_spatial"
    assert b is attn_cdna.paged_kernel_for(128, mapping="spatial")
    attn_cdna.attn_kernel("paged_t_sp", causal=True, paged=True, mapping="spatial",
                          arch="gfx950").trace()


def test_an_unknown_mapping_is_refused():
    with pytest.raises(ValueError, match="mapping"):
        attn_cdna.attn_kernel("attn_t_bad", mapping="head_first", arch="gfx950")


def test_the_route_picks_spatial_where_it_was_measured_to_win():
    """attn_cdna.mapping_for against the MI355X sweep (tools/hk-bench/attn_mapping.py):
    block-first where it holds at most two KV streams an XCD (GQA-8, 16 KV heads) and for
    short causal grids; spatial from five streams an XCD on."""
    m = attn_cdna.mapping_for
    assert m(1, 8, 32768, False) == m(1, 8, 32768, True) == "block_first"
    assert m(8, 1, 32768, True) == "block_first"
    assert m(1, 16, 131072, False) == "block_first"
    assert m(1, 40, 8192, False) == "spatial"
    assert m(1, 40, 8192, True) == "block_first"        # -10% measured
    assert m(2, 40, 8192, True) == "block_first"        # -2% measured (10 rounds)
    assert m(1, 64, 8192, True) == "block_first"        # -2% measured
    assert m(1, 48, 16384, True) == m(1, 96, 8192, True) == "spatial"   # +1.5%, +1.9%
    assert m(1, 64, 8192, False) == m(1, 128, 8192, True) == "spatial"


def test_hk_attention_routes_by_the_mapping_on_gfx950(monkeypatch):
    from hk import target
    from hk.ops import attn

    monkeypatch.setattr(target, "current_arch", lambda: "gfx950")
    assert attn._kernel_for(128, False, None, mapping="spatial").name == \
        "attn_fwd_d128_cdna_mappingspatial"
    assert attn._kernel_for(128, False, None).name == "attn_fwd_d128_cdna"
    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    with pytest.raises(ValueError, match="gfx950"):
        attn._kernel_for(128, False, None, mapping="spatial")
