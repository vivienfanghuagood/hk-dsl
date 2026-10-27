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


def test_causal_pairs_walk_two_blocks_on_half_the_grid():
    """`pairs`: the walk traced twice (the second a loop of 0 or 1 trips), K's and V's
    buffers allocated once, and cdiv(nqb, 2) workgroups a head; kernel_for turns it on
    for causal on the default numbering only, and names only its absence."""
    one = scaffold.bare(attn_cdna.attn_kernel("attn_t_one", causal=True, arch="gfx950").trace())
    two = attn_cdna.attn_kernel("attn_t_pairs", causal=True, pairs=True, arch="gfx950")
    src = scaffold.bare(two.trace())
    assert src.count("kittens::group_load_async<8>(") == 2 * one.count(
        "kittens::group_load_async<8>(")

    class _T:
        def __init__(self, rows, depth, batch):
            self.rows, self.depth, self.batch = rows, depth, batch

    class _P:
        def __init__(self, n):
            self.q = _T(n, 32, 2)
    for n, nqb in ((2048, 8), (2049, 9), (256, 1)):
        assert two.grid_fn(_P(n)) == (32, -(-nqb // 2), 2)
    # Not for full attention, nor the paged prefill, nor the spatial numbering.
    for kw in ({"causal": False}, {"causal": True, "mapping": "spatial"}):
        a = scaffold.bare(attn_cdna.attn_kernel("attn_t_np", pairs=True, arch="gfx950",
                                                **kw).trace())
        b = scaffold.bare(attn_cdna.attn_kernel("attn_t_np", arch="gfx950", **kw).trace())
        assert a == b, kw
    assert attn_cdna.kernel_for(128, True).name == "attn_fwd_d128_causal_cdna"
    assert attn_cdna.kernel_for(128, True, pairs=False).name == "attn_fwd_d128_causal_cdna_pairs0"
    assert attn_cdna.kernel_for(128, False, pairs=False).name == "attn_fwd_d128_cdna"
    assert attn_cdna.kernel_for(128, True, mapping="spatial").name == \
        "attn_fwd_d128_causal_cdna_mappingspatial"


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


# -- the spatial numbering (installed by hk.dsl.placement) --------------------


def _spatial_55(w, batch, h_kv, group, n_t):
    """PR #55's hand-written spatial numbering, the oracle the placement pass
    must reproduce: XCD d (w % 8) one contiguous range of cluster-major units,
    (batch, KV head), then the query tile, then the query head in the group."""
    total = batch * h_kv * group * n_t
    die, base = w % 8, total // 8
    u = die * base + min(die, total % 8) + w // 8
    cluster, r = divmod(u, group * n_t)
    head_kv = cluster % h_kv
    return cluster // h_kv, head_kv * group + r % group, r // group


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch,h,h_kv,n", [
    (1, 128, 128, 32768), (2, 40, 40, 4300), (1, 64, 8, 8192), (4, 8, 1, 16384),
    (1, 28, 4, 2304), (3, 24, 24, 1280), (1, 8, 8, 256), (1, 2, 1, 768)])
def test_mapping_spatial_is_pr55_s_numbering(batch, h, h_kv, n, causal):
    """The decode the spatial kernel computes (evaluated from its emitted ops)
    is, workgroup by workgroup, the one PR #55 wrote by hand -- and so covers
    every (batch, head, query tile) once, one cluster an XCD at a time."""
    from hk.dsl import placement as P

    tr = attn_cdna.attn_kernel(f"attn_t_map_{int(causal)}", causal=causal, mapping="spatial",
                               arch="gfx950").trace()
    ext = {"q": (batch, h, n, 128), "k": (batch, h_kv, n, 128), "v": (batch, h_kv, n, 128),
           "o": (batch, h, n, 128)}
    total = P.check_emitted(tr, ext)
    dec = P.emitted_decode(tr.module, tr.name)
    group, n_t = h // h_kv, -(-n // attn_cdna.Q_TILE)
    for w in range(total):
        x, y, z = dec(w, 0, 0, ext)
        assert (z, (x % h_kv) * group + x // h_kv, y) == _spatial_55(w, batch, h_kv, group, n_t)


@pytest.mark.parametrize("n_tiles", [1, 7, 40])
def test_the_spatial_paged_prefill_is_pr55_s_numbering(n_tiles):
    from hk.dsl import placement as P

    tr = attn_cdna.attn_kernel("paged_t_map", causal=True, paged=True, mapping="spatial",
                               arch="gfx950").trace()
    h, h_kv = 32, 8
    ext = {"q": (1, 1, 4096, h * 128), "k_cache": (64, h_kv, 32, 128),
           "v_cache": (64, h_kv, 32, 128), "block_table": (1, 1, 4, 64),
           "seq_lens": (1, 1, 1, 4), "q_start": (1, 1, 1, 5), "tiles": (1, 1, n_tiles, 2),
           "o": (1, 1, 4096, h * 128)}
    total = P.check_emitted(tr, ext)
    dec = P.emitted_decode(tr.module, tr.name)
    for w in range(total):
        x, y, _ = dec(w, 0, 0, ext)
        _, head, t = _spatial_55(w, 1, h_kv, h // h_kv, n_tiles)
        assert ((x % h_kv) * (h // h_kv) + x // h_kv, y) == (head, t)


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
    """attn_cdna.mapping_for -- hk.dsl.placement.choose on the kernel's own analysis --
    against the MI355X sweep (tools/hk-bench/attn_mapping.py): block-first where it holds
    at most two KV streams an XCD (GQA-8, 16 KV heads) and for short causal grids;
    spatial from five streams an XCD on."""
    m = attn_cdna.mapping_for
    assert m(1, 64, 8, 32768, False) == m(1, 64, 8, 32768, True) == "block_first"
    assert m(8, 8, 1, 32768, True) == "block_first"
    assert m(1, 16, 16, 131072, False) == "block_first"
    assert m(1, 40, 40, 8192, False) == "spatial"
    assert m(1, 40, 40, 8192, True) == "block_first"        # -10% measured
    assert m(2, 40, 40, 8192, True) == "block_first"        # -2% measured (10 rounds)
    assert m(1, 64, 64, 8192, True) == "block_first"        # -2% measured
    assert m(1, 48, 48, 16384, True) == m(1, 96, 96, 8192, True) == "spatial"   # +1.5%, +1.9%
    assert m(1, 64, 64, 8192, False) == m(1, 128, 128, 8192, True) == "spatial"


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
