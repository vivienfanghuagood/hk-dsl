"""gfx950's paged attention: the split-KV decode (hk.ops.paged_cdna) and the
paged prefill (hk.ops.attn_cdna's `paged`), traced; the scalar loads they
read the page table with; and hk.ops.paged and hk.ops.paged_prefill routing
to them on gfx950 and nowhere else.

Tracing only: no hipcc, no GPU. The build gate is
tests/hk/codegen/test_cdna_paged_builds.py; the numerics are
tests/hk/gpu/test_paged*.py, which run these kernels on gfx950.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.ops import attn_cdna, paged, paged_cdna, paged_prefill  # noqa: E402


def _src(k):
    return scaffold.bare(k.trace())


@pytest.mark.parametrize("d", paged_cdna.HEAD_DIMS)
@pytest.mark.parametrize("mode", ["plain", "split", "fused"])
def test_the_decode_traces_in_every_mode(d, mode):
    k = paged_cdna.decode_kernel(f"pc_t_{mode}_{d}", head_dim=d, splits=4, mode=mode,
                                 arch="gfx950")
    src = _src(k)
    # Pᵀ read in place as the 32x16 stride-4 operand, V read with its run.
    assert "kittens::as_operand(" in src
    assert "rt_32x16_4" in src
    # K and V by DMA, non-temporal: two copies before the ring and two a
    # page.
    assert src.count("kittens::group_load_async<1, true>(") == 2 * (1 + 2)
    # Laid out for the copies' runs, not by the fragments' own layouts.
    assert "st_shape::st_16x64> (&Ks)" in src and "st_shape::st_32x32> (&Vs)" in src


def test_the_decode_reads_its_table_through_the_scalar_cache():
    """A vector load's wait is in vmcnt's queue behind every copy issued
    before it; the page index and the length are scalar loads."""
    src = _src(paged_cdna.decode_kernel("pc_t_scalar", splits=8, arch="gfx950"))
    assert "kittens::load_scalar(" not in src
    assert src.count("kittens::load_scalar_invariant(") >= 3


def test_a_page_s_copy_is_bounded_to_the_sequence():
    """A page past the split copies no rows (its index is never followed),
    and the tail of the last page reads zero."""
    src = _src(paged_cdna.decode_kernel("pc_t_rows", splits=2, arch="gfx950"))
    assert re.search(r"kittens::group_load_async<1, true>\(\w+, g\.k_cache, \w+, \w+\);", src)


def test_the_merge_folds_the_splits_that_saw_pages():
    """One warp a query row, a strip of head_dim; rounds of MERGE_CHUNK
    splits, as many as the length gives the walk."""
    src = _src(paged_cdna.merge_kernel("pc_t_merge", splits=32, arch="gfx950"))
    assert "kittens::rs<float, 1, 128, 2>" in src
    assert src.count("kittens::load_scalar_invariant(") == 1
    assert re.search(r"for \(int \w+ = 0; \w+ < \w+; \+\+\w+\)", src)


def test_a_layout_a_read_cannot_use_is_refused():
    def k(x: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="s", shape="32x32")
        hk.group(1).load(s, x.at[0, 0])
        hk.arch.barrier()
        r = s.load(hk.RT(hk.bf16, 32, 128, "row"))
        hk.zeros(hk.RT(hk.fp32, 32, 32, "col")).__iadd__(r @ r.T)

    with pytest.raises(Exception, match="not one an LDS tile laid out as st_32x32"):
        scaffold.bare(hk.kernel(k, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace())

    def bad(x: hk.Tensor[hk.bf16]):
        hk.lds.alloc(hk.ST(hk.bf16, 32, 128), shape="8x8")

    with pytest.raises(TypeError, match="one of"):
        hk.kernel(bad, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace()


def test_nontemporal_is_the_async_copy_s():
    def k(x: hk.Tensor[hk.bf16]):
        s = hk.lds.alloc(hk.ST(hk.bf16, 32, 128), name="s")
        hk.group(1).load(s, x.at[0, 0], nontemporal=True)

    with pytest.raises(TypeError, match="wait=False"):
        hk.kernel(k, arch="gfx950", warps=1, grid=lambda p: (1, 1, 1)).trace()


def test_the_decode_is_refused_where_it_has_no_fragment():
    k = paged_cdna.decode_kernel("pc_t_1100", arch="gfx1100")
    with pytest.raises(Exception):
        k.trace()


def test_a_kernel_per_mode_and_split_built_once():
    a = paged_cdna.kernel_for(128, 8, "fused")
    assert a is paged_cdna.kernel_for(128, 8, "fused")
    assert a.name == "paged_decode_d128_fused_s8_cdna"
    assert paged_cdna.kernel_for(128, 1, "plain").name == "paged_decode_d128_cdna"


@pytest.mark.parametrize("d", attn_cdna.HEAD_DIMS)
def test_the_paged_prefill_copies_a_block_as_its_two_pages(d):
    k = attn_cdna.attn_kernel(f"pp_t_{d}", head_dim=d, causal=True, paged=True,
                              arch="gfx950")
    tr = k.trace()
    src = scaffold.bare(tr)
    # The dense kernel's 3 + 2 + 4 block copies, two pages each; a page is
    # a whole 8-warp copy at D = 128 and half of one at D = 64.
    warps = 8 if d == 128 else 4
    assert src.count(f"kittens::group_load_async<{warps}>(") == 2 * (3 + 2 + 4)
    # A block's halves: Sᵀ written a 32-row subtile at a time, P·V read off
    # P's two halves.
    assert "subtile_inplace<32>" in src
    assert "kittens::triu(" in src and "kittens::lower_fill(" not in src
    assert f"!hk.rt<bf16, 32x{d}, col, frag 16x32 stride 4>" in tr.dump()


def test_the_paged_prefill_is_causal():
    with pytest.raises(ValueError, match="causal"):
        attn_cdna.attn_kernel("pp_t_nc", causal=False, paged=True, arch="gfx950")


def test_the_dense_kernel_is_not_the_paged_one():
    dense = _src(attn_cdna.attn_kernel("pp_t_dense", causal=True, arch="gfx950"))
    assert "block_table" not in dense and "load_scalar" not in dense


def test_paged_and_paged_prefill_route_to_them_on_gfx950(monkeypatch):
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx950")
    assert paged.KERNELS[128].name == "paged_decode_d128_cdna"
    assert paged.split_kernel(64, 4).name == "paged_decode_d64_split_s4_cdna"
    assert paged.fused_kernel(128, 8).name == "paged_decode_d128_fused_s8_cdna"
    assert paged_prefill.kernel_for(128, 4).name == "paged_prefill_d128_cdna"
    assert paged.merge_kernel(128, 8).name == "paged_merge_d128_s8_cdna"
    # The plan aims at paged_cdna.TARGET_WORKGROUPS one-warp workgroups.
    assert paged.plan_splits(1, 8, 1280) == 32 and paged.plan_splits(16, 8, 1280) == 2
    assert paged.plan_splits(64, 8, 1280) == 1
    with pytest.raises(ValueError, match="ablate"):
        paged.fused_kernel(128, 8, ablate=("qk",))
    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    assert paged.KERNELS[128].name == "paged_decode_d128"
    assert paged.fused_kernel(128, 8).name == "paged_decode_fused_d128_s8"
    assert paged.merge_kernel(128, 8).name == "paged_merge_d128_s8"
    assert paged_prefill.kernel_for(128, 4).name == "paged_prefill_d128_g4"


def test_the_work_list_on_gfx950_is_its_tile_longest_first(monkeypatch):
    torch = pytest.importorskip("torch")
    from hk import target

    monkeypatch.setattr(target, "current_arch", lambda: "gfx950")
    wl = paged_prefill.work_list([(0, 600), (1, 5)], "cpu").tolist()
    assert wl == [[0, 2], [0, 1], [0, 0], [1, 0]]
    monkeypatch.setattr(target, "current_arch", lambda: "gfx1100")
    wl = paged_prefill.work_list([(0, 600), (1, 5)], "cpu").tolist()
    assert wl == [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]]
    del torch


def test_kernels_iterates_the_rdna_kernels():
    """paged.KERNELS still lists what it always listed: the source dumps
    and coverage walk it."""
    assert sorted(paged.KERNELS) == [64, 128]
    assert all(not k.name.endswith("_cdna") for k in paged.KERNELS.values())
