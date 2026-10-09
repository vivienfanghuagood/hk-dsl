"""The architecture facts, pinned where a kernel author would look for them.

`target/gfx1100.py` exists so that the facts that decide a kernel's shape are
data rather than folklore. A fact only earns that if someone consults it, and
the honest history of this file is that at least one of these was rediscovered
the expensive way -- by writing the optimization the fact forbids, compiling
it, and watching it spill. So each test below names the mistake it answers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

import pytest  # noqa: E402

from hk.target.gfx1100 import GFX1100  # noqa: E402


# ------------------------------------------------- what a register tile costs

def test_a_16_bit_tile_costs_as_much_as_a_32_bit_one():
    """The mistake: "hold the tile in bf16, it is half the registers."

    It is not. On wave32 a WMMA operand is v16bf16 -- the whole K=16 vector in
    one lane -- and lanes l and l+16 must hold identical data, so the wave
    stores the tile twice over. Eight VGPRs a lane per 16x16 base tile, which
    is exactly what the fp32 accumulator costs, and `rt_base`'s own
    static_assert says so.

    This is why `hk.ops.quant` holds an fp32 tile for a bf16 tensor. The
    narrowing version does not merely fail to save: it loses, because it still
    needs an fp32 temporary to reduce over. Measured at TPW=2, 145 VGPRs
    against 109, and at TPW=4 it spills outright.
    """
    for rows, cols in ((16, 16), (16, 64), (64, 128)):
        assert GFX1100.tile_vgprs(16, rows, cols) == GFX1100.tile_vgprs(32, rows, cols)
    assert GFX1100.tile_vgprs(16, 16, 64) == 32
    assert GFX1100.mma_operands_mirrored


def test_the_mirrored_operand_is_what_makes_them_equal():
    """Stated twice, in the shape table and in the flag, so pin them together.

    If a future part has a non-mirrored 16-bit operand the flag goes false and
    the two costs come apart; a test that only checked the equality would then
    be wrong in a way nobody notices.
    """
    bf16 = GFX1100.mma["bf16"]
    assert bf16.operand_vgprs == bf16.accum_vgprs
    assert (bf16.operand_vgprs == bf16.accum_vgprs) == GFX1100.mma_operands_mirrored


# ------------------------------------------------------ what occupancy costs

@pytest.mark.parametrize("vgprs,occ", [(240, 6), (241, 5), (216, 7), (64, 16)])
def test_the_granule_is_24(vgprs, occ):
    """The mistake: "256 VGPRs is the budget for occupancy 6." It is 240.

    gfx1100 charges VGPRs in blocks of 24, so 241 rounds to 264 and 1536/264
    is 5. Verified against LLVM's own Occupancy remark at 42 points.
    """
    assert GFX1100.occupancy(vgprs) == occ


def test_headroom_is_what_is_left_before_the_next_cliff():
    """A kernel at 207 VGPRs is at occupancy 7 with room to spare; the number
    that matters when deciding whether one more live tile fits is how much."""
    assert GFX1100.occupancy(207) == 7
    assert GFX1100.vgpr_headroom(207) == 9   # 216 is the last block at occ 7
    assert GFX1100.occupancy(207 + 10) < 7


# -------------------------------------------------------- what the ISA lacks

def test_there_is_no_global_to_lds_dma():
    """The mistake: writing a staging pass as a copy. gfx11 has no
    vmem-to-lds-load-insts, so every staged byte goes through VGPRs and any
    pass that stages must emit a register buffer."""
    assert GFX1100.has_global_to_lds_dma is False


def test_only_row_layout_reads_shared_memory_fast():
    """A col-layout bf16 operand degrades from 2 ds_read_b128 to 16 scalar
    ds_read_u16. Not wrong, just 8x the instructions -- which is why the
    verifier rejects it at trace time rather than leaving it to a benchmark."""
    assert GFX1100.fast_smem_layouts == ("row",)
