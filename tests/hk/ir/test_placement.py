"""hk.dsl.placement: the numbering function, the analysis, the rewrite, the policy.

No GPU: `place` and the decodes are integer functions checked over whole
grids on the host, and the decode a placed kernel computes is evaluated from
its own emitted ops (hk.dsl.placement.emitted_decode). The GPU half --
outputs bitwise equal under a placement, and the dispatch policy every
placement relies on -- is tests/hk/gpu/test_placement.py and
tests/hk/gpu/test_comm.py::test_workgroups_are_dealt_to_the_xcds_by_id_mod_8_in_every_wave.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk import i32  # noqa: E402
from hk.compiler import mlir as M  # noqa: E402
from hk.dsl import placement as P  # noqa: E402
from hk.ops import attn_cdna  # noqa: E402

TOTALS = list(range(0, 130)) + [1151 * 6, 1152 * 4, 16384, 16387]


# -- place ---------------------------------------------------------------------


@pytest.mark.parametrize("xcds", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("chunk,tail", [(0, "spread"), (1, "stay"), (16, "stay"), (32, "spread"),
                                        (64, "stay"), ("even", "stay"), ("even", "spread")])
def test_place_is_a_bijection_for_every_total(xcds, chunk, tail):
    for total in TOTALS:
        got = sorted(P.place_host(w, total, xcds, chunk, tail) for w in range(total))
        assert got == list(range(total)), (xcds, chunk, tail, total)


@pytest.mark.parametrize("xcds", [2, 4, 6, 8])
def test_contiguous_place_gives_each_xcd_one_balanced_range(xcds):
    for total in TOTALS:
        per = {}
        for w in range(total):
            per.setdefault(w % xcds, []).append(P.place_host(w, total, xcds))
        sizes = [len(v) for v in per.values()]
        assert not sizes or max(sizes) - min(sizes) <= 1
        for v in per.values():
            assert v == list(range(v[0], v[0] + len(v)))


def test_place_reproduces_the_hand_written_remaps():
    """chunked(c, "stay") is gemm_cdna's / mxfp8's xcd_chunk remap, chunked("even",
    "stay") gemm_cdna_dp's xcd_chunk=0, and contiguous ranges moe_a8w4's where it applied
    (total a multiple of 8) -- so migrating them changes no workgroup's work or order."""
    def gemm(w, blocks, c):
        per_round = 8 * c
        limit = (blocks // per_round) * per_round
        local = w // 8
        moved = (local // c) * per_round + (w % 8) * c + local % c
        stays = min(max(w - limit, 0), 1)
        return moved + stays * (w - moved)

    def moe(w, total):
        if total % 8:
            return w
        return (w % 8) * (total // 8) + w // 8

    for total in TOTALS:
        for c in (16, 32, 64):
            assert all(P.place_host(w, total, 8, c, "stay") == gemm(w, total, c)
                       for w in range(total)), (total, c)
        even = max(total // 8, 1)
        assert all(P.place_host(w, total, 8, "even", "stay") == gemm(w, total, even)
                   for w in range(total)), total
        if total % 8 == 0:
            assert all(P.place_host(w, total, 8) == moe(w, total) for w in range(total))


# -- analysis --------------------------------------------------------------------


def _analyze(k, **consts):
    tr = k.trace(**consts)
    with M.context():
        return P.analyze(tr.module, tr.name)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_s_key_is_batch_and_kv_head(causal):
    a = _analyze(attn_cdna.attn_kernel(f"pl_attn_{int(causal)}", causal=causal, arch="gfx950"))
    assert a.kind == "cluster"
    assert [repr(d) for d in a.key] == ["z", "x % g.k.depth()"]
    assert a.nonuniform == causal          # causal: the walk's length follows the query block
    assert not a.order_dependent


def test_a_kernel_whose_reads_are_its_own_has_nothing_to_place():
    @hk.kernel(arch="gfx950", warps=4, grid=lambda p: (hk.cdiv(p.x.rows, 64), 1, 1))
    def pl_copy(x: hk.Tensor[hk.bf16], y: hk.Tensor[hk.bf16]):
        r = hk.arch.block_idx.x * 64
        t = x.at[r, 0].load(hk.RT(hk.bf16, 64, 64))
        y.at[r, 0].store(t)

    a = _analyze(pl_copy)
    assert a.kind == "none" and not a.key


def test_atomics_make_a_kernel_order_dependent():
    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.out.cols, 4, 1))
    def pl_atomic(src: hk.Tensor[i32], out: hk.Tensor[i32]):
        v = src.at[hk.arch.block_idx.y].load_scalar()
        out.at[hk.arch.block_idx.x].atomic_max_scalar(v)

    a = _analyze(pl_atomic)
    assert "hk.atomic_scalar" in a.order_dependent
    c = P.choose(a, None, (64, 4, 1), {}, 8, 32)
    assert not c.placed and "order-dependent" in c.reason


def test_a_key_read_through_memory_is_flagged_as_a_heuristic():
    """MoE-like: the block's expert comes from a table in memory, so its
    weights' key is data, and the analysis says the key is a guess."""
    @hk.kernel(arch="gfx950", warps=4, grid=lambda p: (p.tab.cols, 8, 1))
    def pl_indirect(tab: hk.Tensor[i32], w: hk.Tensor[hk.bf16], y: hk.Tensor[hk.bf16]):
        b = hk.arch.block_idx.x
        e = tab.at[b].load_scalar()
        for kk in hk.range(0, 8):
            t = w.at[e * 64, kk * 64].load(hk.RT(hk.bf16, 64, 64))
            y.at[b * 64, hk.arch.block_idx.y * 64].store(t)

    a = _analyze(pl_indirect)
    assert any(r.indirect for r in a.reads)
    assert a.kind == "cluster" and "heuristic" in a.note


# -- rewrite ---------------------------------------------------------------------


def _ext(B, H, HKV, N):
    return {"q": (B, H, N, 128), "k": (B, HKV, N, 128), "v": (B, HKV, N, 128),
            "o": (B, H, N, 128)}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,HKV,N", [(1, 40, 40, 8192), (2, 24, 3, 777), (1, 9, 9, 300),
                                       (3, 16, 8, 1000), (1, 128, 128, 2048)])
def test_the_placed_attention_decodes_like_the_hand_written_spatial(causal, B, H, HKV, N):
    """The pass, given nothing but the traced kernel, installs PR #55's spatial
    numbering (contiguous ranges of (batch, KV head) clusters, then the query
    tile, then the head in the group): workgroup by workgroup the same (batch,
    query head, query tile), and every grid point once."""
    k = attn_cdna.attn_kernel(f"pl_rw_{int(causal)}", causal=causal, arch="gfx950").placed()
    tr = k.trace()
    ext = _ext(B, H, HKV, N)
    n = P.check_emitted(tr, ext)
    dec = P.emitted_decode(tr.module, tr.name)
    group, nq = H // HKV, -(-N // attn_cdna.Q_TILE)
    for w in range(n):
        x, y, z = dec(w, 0, 0, ext)
        total = B * H * nq
        u = P.place_host(w, total, 8)
        cl, r = divmod(u, group * nq)
        b, head, t = cl // HKV, (cl % HKV) * group + r % group, r // group
        assert (z, (x % HKV) * group + x // HKV, y) == (b, head, t), w


def test_the_placed_kernel_is_the_same_kernel_on_a_flat_grid():
    from hk.codegen import scaffold

    base = scaffold.bare(attn_cdna.attn_kernel("pl_same", arch="gfx950").trace())
    k = attn_cdna.attn_kernel("pl_same", arch="gfx950").placed()
    src = scaffold.bare(k.trace())
    assert k.name == "pl_same_x8cl0p"
    for op in ("kittens::group_load_async<8>(", "__builtin_amdgcn_s_barrier()"):
        assert src.count(op) == base.count(op)
    assert "blockIdx.y" in base and "blockIdx.y" not in src and "blockIdx.z" not in src


def test_an_explicit_key_and_the_chunked_placement():
    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.out.cols, 6, 1),
               placement=hk.Placement.cluster(key=(hk.Digit.of("y"),)))
    def pl_key(out: hk.Tensor[i32]):
        out.at[hk.arch.block_idx.x].atomic_max_scalar(hk.arch.block_idx.y)

    tr = pl_key.trace()
    assert [repr(d) for d in tr.plan.order] == ["y", "z", "x"]
    assert P.check_emitted(tr, {"out": (1, 1, 1, 37)}) == 37 * 6

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.out.cols, 3, 1),
               placement=hk.Placement.chunked(16))
    def pl_chunk(out: hk.Tensor[i32]):
        out.at[hk.arch.block_idx.x].atomic_max_scalar(hk.arch.block_idx.y)

    tr = pl_chunk.trace()
    assert tr.grid == tr.logical_grid             # the grid stays; x is renumbered in place
    dec = P.emitted_decode(tr.module, tr.name)
    for total in (5, 128, 300, 1000):
        e = {"out": (1, 1, 1, total)}
        assert P.check_emitted(tr, e) == total * 3
        for s_ in range(3):
            assert [dec(w, s_, 0, e) for w in range(total)] == \
                [(P.place_host(w, total, 8, 16, "stay"), s_, 0) for w in range(total)]


def test_a_cluster_placement_without_a_key_names_what_is_missing():
    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.out.cols, 1, 1),
               placement=hk.Placement.cluster())
    def pl_nokey(out: hk.Tensor[i32]):
        out.at[hk.arch.block_idx.x].atomic_max_scalar(1)

    with pytest.raises(ValueError, match="key="):
        pl_nokey.trace()


def test_one_l2_is_the_identity(monkeypatch):
    monkeypatch.setenv("HK_XCDS", "1")
    k = attn_cdna.attn_kernel("pl_one", arch="gfx950").placed()
    assert k.trace().plan is None


# -- policy -------------------------------------------------------------------------


#: (B, H_KV, N, causal) points PR #55 measured, and what attn_cdna.mapping_for routes.
MEASURED = [(1, h, n, c) for h in (16, 40, 64, 128) for n in (8192, 16384, 32768, 65536, 131072)
            for c in (False, True)] + [
    (1, 8, 32768, False), (8, 1, 32768, True), (1, 24, 65536, True), (1, 32, 32768, False),
    (1, 32, 65536, True), (1, 40, 12288, True), (1, 48, 8192, True), (1, 48, 16384, True),
    (2, 40, 8192, True), (1, 56, 12288, True), (1, 96, 8192, True), (2, 64, 24576, True)]


def _measured_route(B, HKV, N, causal):
    """PR #55's routing rule, fitted to its 74 measured points (worst routed / block-first
    1.000): spatial from five KV streams an XCD, not for causal grids under 12 rounds."""
    if -(-HKV // 8) < 5:
        return "block_first"
    if causal and B * HKV * -(-N // 256) / 256 < 12:
        return "block_first"
    return "spatial"


@pytest.mark.parametrize("B,HKV,N,causal", MEASURED)
def test_the_policy_chooses_as_the_measured_route(B, HKV, N, causal):
    """hk.dsl.placement.choose, from the analysis alone, picks what PR #55's
    measurements routed at every measured MHA/GQA point -- one workgroup a CU,
    32 an XCD."""
    k = attn_cdna.attn_kernel(f"pl_pol_{int(causal)}", causal=causal, arch="gfx950")
    a = _analyze(k)
    plan = P.make_plan(P.Placement.cluster(), a, 8)
    H = HKV if HKV >= 16 else 64 if HKV == 8 else 8
    grid = (H, -(-N // attn_cdna.Q_TILE), B)
    c = P.choose(a, plan, grid, {"g.k.depth()": HKV}, 8, 32)
    assert c.placed == (_measured_route(B, HKV, N, causal) == "spatial"), c
    assert attn_cdna.mapping_for(B, H, HKV, N, causal) == _measured_route(B, HKV, N, causal)


# -- the XCD count and hk.check --------------------------------------------------------


def test_the_xcd_count_comes_from_the_device_or_the_arch(monkeypatch, tmp_path):
    for i, (v, x) in enumerate([(90500, 4), (90500, 4), (110000, 1)]):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "properties").write_text(f"gfx_target_version {v}\nnum_xcc {x}\n")
    monkeypatch.setattr(P, "_KFD_NODES", str(tmp_path))
    monkeypatch.delenv("HK_XCDS", raising=False)
    P._XCD_CACHE.clear()
    assert P.xcd_count("gfx950") == 4            # an MI350P / a DPX partition
    assert P.xcd_count("gfx942") == 8            # no node of it: the default
    monkeypatch.setenv("HK_XCDS", "6")
    assert P.xcd_count("gfx950") == 6
    P._XCD_CACHE.clear()


def test_hk_check_placement_on_a_raw_kernel(tmp_path):
    f = tmp_path / "k.py"
    f.write_text(textwrap.dedent('''
        import hk
        from hk import i32

        @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.out.cols, 3, p.out.batch),
                   placement=hk.Placement.cluster(key=(hk.Digit.of("z"),)))
        def kernel(out: hk.Tensor[i32]):
            out.at[hk.arch.block_idx.x].atomic_max_scalar(hk.arch.block_idx.y)
    '''))
    r = subprocess.run([sys.executable, "-m", "hk.check", str(f), "--placement",
                        "--inputs", "i32[2,1,1,45]"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "a bijection over 270 workgroups" in r.stdout


def test_placement_values_are_checked():
    with pytest.raises(ValueError):
        hk.Placement("cluster", tail="anywhere")
    with pytest.raises(ValueError):
        hk.Placement.chunked(-1)
    with pytest.raises(TypeError, match="placement"):
        hk.kernel(lambda out: None, placement="spatial")


def test_the_suffix_names_the_placement():
    assert hk.Placement.cluster().suffix(8) == "_x8cl0p"
    assert hk.Placement.chunked(32).suffix(8) == "_x8ch32s"
    assert hk.Placement.chunked("even").suffix(4) == "_x4chEs".replace("E", "e")
    assert hk.Placement.none().suffix(8) == ""
    assert list(itertools.islice(iter(P.AXES), 3)) == ["x", "y", "z"]


# -- M2: the MoE and MXFP8 kernels on it -----------------------------------------------


def test_xcd_range_deals_every_unit_once_balanced():
    o = P._IntOps()
    for xcds in (1, 2, 4, 6, 8):
        for total in TOTALS:
            ranges = [P._xcd_range(o, d, total, xcds) for d in range(xcds)]
            units = [u for first, n in ranges for u in range(first, first + n)]
            assert units == list(range(total)), (xcds, total)
            assert max(n for _, n in ranges) - min(n for _, n in ranges) <= 1


def test_moe_sorted_kernels_renumber_contiguously_for_every_total():
    """moe_a8w4's grid kernels: contiguous ranges an XCD -- what the hand-written remap did
    where total % 8 == 0, and now also where it fell back to the identity."""
    from hk.ops import moe_a8w4 as Mo

    k = Mo._sorted_ring(1, "pl_moe_s1", group_rows=2304, mt=4, pp=3, k=5120, warps=4)
    tr = k.trace()
    assert tr.plan is not None and tr.plan.kind == "chunked"
    dec = P.emitted_decode(tr.module, tr.name)
    for blocks in (1152, 1153, 1322, 1920):
        e = {n: (1, 1, 1, 1) for n, _ in k._tensors}
        e["seids"], e["out"] = (1, 1, 1, blocks), (1, 1, blocks * 64, 1152)
        total = blocks * 6
        assert [dec(w, 0, 0, e)[0] for w in range(total)] == \
            [P.place_host(w, total, 8) for w in range(total)]


def test_mx4w_renumbers_as_before():
    from hk.ops import mxfp8

    k = mxfp8.mx4w_kernel("pl_mx4w")
    tr = k.trace()
    dec = P.emitted_decode(tr.module, tr.name)
    for m, n in ((8192, 5120), (1000, 4096), (256, 7168)):
        e = {nm: (1, 1, 1, 1) for nm, _ in k._tensors}
        e["a"], e["b"] = (1, 1, m, 4096), (1, 1, n, 4096)
        total = tr.logical_grid[0].value(e)
        assert total == -(-m // 256) * (n // 256)
        assert [dec(w, 0, 0, e)[0] for w in range(total)] == \
            [P.place_host(w, total, 8, 32, "stay") for w in range(total)]
