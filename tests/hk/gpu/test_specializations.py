"""The type-selected specializations in include/hk/cdna.cuh (docs/design/dsl-surface.md §2.5, E4)
give the generic path's results bit for bit. Each one is chosen by type alone, so each is tested
here against the generic result -- the path it replaces, or an exact reference where the generic
result is defined bit for bit (a round-to-nearest-even conversion, an exact sum):

- the 32x32 fp32 -> bf16 tile store (v_permlane32_swap, attention's O) and the 16x16 one;
- the 16-column column reduction (permlane swaps, the paged decode's softmax) -- max, and a sum
  of small integers, which is exact in any order;
- the col-layout masks with -inf (asm v_cmp / v_cndmask into SGPR pairs, or a sign-bit add)
  against the same masks with a finite value (the plain select);
- neg_infty on a col-layout tile (asm v_mov), against -inf.

The 16x16 fp32 -> bf16 scatter (v_permlane16_swap, K1 c35) is held to torch's rounding bit for
bit by test_scatter_add.py::test_sort_ids_scatter_each_pair_to_its_choice_major_row."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("include/hk/cdna.cuh's specializations, measured and tested on gfx950",
                allow_module_level=True)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("frag", [(32, 32), (16, 16)], ids=["32x32", "16x16"])
def test_an_accumulator_stores_to_bf16_as_torch_rounds_it(frag):
    """Oᵀ [64, 32] fp32 (col layout) stored transposed into bf16 [32, 64]: the 32x32 fragment's
    store trades runs with v_permlane32_swap; the value is torch's round to nearest even."""
    T = hk.RT(fp32, 64, 32, "col", frag=frag)

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1), name=f"spec_store_{frag[0]}")
    def store(x: hk.Tensor[fp32], o: hk.Tensor[bf16]):
        t = x.at[0, 0].load(T)
        o.at[0, 0].store(hk.transpose(t))

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(64, 32, device="cuda", generator=g) * 3
    o = torch.empty(32, 64, device="cuda", dtype=torch.bfloat16)
    store(x, o)
    assert torch.equal(o, x.T.to(torch.bfloat16))


@pytest.mark.parametrize("op", ["max", "sum"])
def test_the_16_column_column_reduction(op):
    """A col-layout fp32 tile of 16-column base tiles reduced down its rows (the specialization:
    four lanes combined by v_permlane32_swap and v_permlane16_swap). Small integers, so a sum is
    exact in any order and equals torch's."""
    T = hk.RT(fp32, 64, 32, "col")

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1), name=f"spec_colred_{op}")
    def red(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        t = x.at[0, 0].load(T)
        v = t.max(axis=0) if op == "max" else t.sum(axis=0)
        o.at[0, 0].store(v.value())

    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randint(-50, 50, (64, 32), device="cuda", generator=g).float()
    o = torch.empty(1, 32, device="cuda")
    red(x, o)
    want = x.amax(0) if op == "max" else x.sum(0)
    assert torch.equal(o[0], want)


MASKS = ["lower_fill_", "upper_fill_", "triu_", "tril_"]


@pytest.mark.parametrize("frag", [(32, 32), (16, 16)], ids=["32x32", "16x16"])
@pytest.mark.parametrize("mask", MASKS)
def test_a_mask_with_neg_inf_masks_what_the_select_masks(mask, frag):
    """A col-layout fp32 tile masked with -inf (the specialized integer / SGPR-pair path) and with
    a finite value (the plain select), at every offset: the same elements masked, the others
    untouched, bit for bit."""
    T = hk.RT(fp32, 64, 32, "col", frag=frag)
    FIN = -12345.0

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (p.o.batch, 1, 1),
               name=f"spec_mask_{mask}_{frag[0]}")
    def fill(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        idx = hk.arch.block_idx.x - 40
        a = x.at[0, 0].load(T)
        b = x.at[0, 0].load(T)
        getattr(a, mask)(idx, float("-inf"))
        getattr(b, mask)(idx, FIN)
        o[hk.tile_coord(hk.arch.block_idx.x, 0, 0, 0)].store(a)
        o[hk.tile_coord(hk.arch.block_idx.x, 1, 0, 0)].store(b)

    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(64, 32, device="cuda", generator=g)
    o = torch.empty(80, 2, 64, 32, device="cuda")
    fill(x, o)
    ninf, fin = o[:, 0], o[:, 1]
    assert torch.equal(torch.isneginf(ninf), fin == FIN)
    keep = fin != FIN
    assert torch.equal(ninf[keep], fin[keep])
    assert torch.equal(fin[keep], x.expand_as(fin)[keep])
    assert keep.any() and (~keep).any()


def test_neg_infty_on_a_col_tile():
    T = hk.RT(fp32, 64, 32, "col", frag=(32, 32))

    @hk.kernel(arch="gfx950", warps=1, grid=lambda p: (1, 1, 1), name="spec_ninf")
    def ninf(o: hk.Tensor[fp32]):
        o.at[0, 0].store(hk.neg_infty(T))

    o = torch.zeros(64, 32, device="cuda")
    ninf(o)
    assert torch.isneginf(o).all()
