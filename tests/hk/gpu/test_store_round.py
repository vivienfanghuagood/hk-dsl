"""A store that narrows fp32 to bf16 rounds to nearest even, as torch does.

The exact cases are ties: 1 + 2^-8 sits halfway between two bf16 values, and
an even mantissa wins whichever side it is on; a NaN stays a NaN. Then an add
of random bf16 rows, a fifth of whose fp32 sums are ties, is torch's bit for
bit. Before this, RDNA's stores rounded ties away from zero and that add
differed from torch at 10.7% of its elements (docs/history/36-the-agent-loop.md).
"""

from __future__ import annotations

import pytest

import hk

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

T = hk.RT(hk.fp32, 16, 64)


@hk.kernel(warps=4, grid=lambda p: (hk.cdiv(p.o.cols, 256), hk.cdiv(p.o.rows, 16), 1))
def add_rows(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.fp32], o: hk.Tensor[hk.bf16]):
    r0 = hk.arch.block_idx.y * 16
    c0 = (hk.arch.block_idx.x * 4 + hk.arch.warp_id()) * 64
    x = a.at[r0, c0].load(T, bounded=True)
    x += b.at[r0, c0].load(T, bounded=True)
    o.at[r0, c0].store(x, bounded=True)


def _add(a, b):
    o = torch.empty_like(a)
    add_rows(a, b, o)
    return o


def test_ties_round_to_even_both_ways_and_both_signs():
    ulp = 2.0 ** -8                     # half a bf16 ulp at 1.0 is 2^-8
    base = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125, 3.0, 2.0 ** -120],
                        device="cuda")
    tie = torch.tensor([ulp, ulp, -ulp, -ulp, 2.0 ** -7, 2.0 ** -128],
                       device="cuda")
    a = base.repeat(16, 43)[:, :256].to(torch.bfloat16).contiguous()
    b = tie.repeat(16, 43)[:, :256].contiguous()
    want = (a.float() + b).to(torch.bfloat16)  # torch: nearest even
    got = _add(a, b)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # 1 + 2^-8 goes down to 1.0 (even), 1.0078125 + 2^-8 up to 1.015625.
    assert got[0, 0].item() == 1.0 and got[0, 1].item() == 1.015625
    assert got[0, 2].item() == -1.0 and got[0, 3].item() == -1.015625
    assert got[0, 4].item() == 3.0


def test_nan_stays_nan_and_infinity_stays_infinity():
    a = torch.zeros(16, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(16, 256, device="cuda")
    b[:, 0] = float("nan")
    b[:, 1] = float("inf")
    b[:, 2] = -float("inf")
    b[:, 3] = 3.4e38                    # rounds up past bf16's largest: inf
    got = _add(a, b)
    assert torch.isnan(got[:, 0]).all()
    assert (got[:, 1] == float("inf")).all() and (got[:, 2] == -float("inf")).all()
    assert torch.equal(got[:, 3], b[:, 3].to(torch.bfloat16))


@pytest.mark.parametrize("rows", [16, 2047])
def test_a_residual_add_is_torchs_bit_for_bit(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    a = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    r = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    got = _add(a, r.float())
    assert torch.equal(got.view(torch.int16), (a + r).view(torch.int16))


def test_a_strip_store_rounds_to_even():
    from hk.target import current_arch, get_target

    if get_target(current_arch()).kittens_define not in ("KITTENS_RDNA3", "KITTENS_RDNA4"):
        pytest.skip("these strips are include/rdna3's and include/rdna4's")
    S = hk.RS(hk.fp32, 1, 256, 8)

    @hk.kernel(warps=1, grid=lambda p: (p.o.rows, 1, 1))
    def strip_add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        row = hk.arch.block_idx.x
        v = a.at[row, 0].load(S)
        v += b.at[row, 0].load(S)
        o.at[row, 0].store(v)

    g = torch.Generator(device="cuda").manual_seed(7)
    a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, generator=g)
    o = torch.empty_like(a)
    strip_add(a, b, o)
    assert torch.equal(o.view(torch.int16), (a + b).view(torch.int16))
