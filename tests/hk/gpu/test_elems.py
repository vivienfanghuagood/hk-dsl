"""Element-wise exchange on gfx950: tagged mailbox pairs between CTAs (Global.post_elems /
take_elems, one element a pair or runs of pairs of pairs, bf16 two a pair), an fp32 LDS tile
addressed element by element (SharedTile.load_elems / store_elems), and store_elems(runs=True)
-- the hand-offs of DeepSeek-V4.1's K1 in hk (~/hk-dx/kda/mk-decode/m1/k1_hk.py)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import i32  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="gfx950's")]

S = hk.RS(hk.fp32, 1, 512, 8)


def _hand_off(runs: bool, bf16: bool):
    """CTA 1 posts row 0 of x with tag `tag` (every 9th element masked out unless runs); CTA 0 takes
    it -- the two CTAs run together, so the take really waits on the post."""
    def body(x: hk.Tensor[hk.fp32], box: hk.Tensor[i32, "device"], tg: hk.Tensor[i32],
             o: hk.Tensor[hk.fp32]):
        c = hk.iota(S)
        tag = tg.at[0].load_scalar()
        if runs:
            idx = hk.map_cpp("x0 / 2" if bf16 else "x0", c)      # a run's first index: its pairs follow
        else:
            idx = hk.map_cpp("x0 % 9 == 4 ? -1 : x0", c)
        if hk.uniform(hk.arch.block_idx.x == 1):
            box.post_elems(x.at[0, 0].load(S), idx, tag, runs=runs, bf16=bf16)
        if hk.uniform(hk.arch.block_idx.x == 0):
            o.at[0, 0].store(box.take_elems(idx, tag, timeout_s=5.0, runs=runs, bf16=bf16))
    return hk.kernel(body, warps=1, name=f"hand_off_{int(runs)}{int(bf16)}", grid=lambda p: (2, 1, 1))


@pytest.mark.parametrize("runs,bf16", [(False, False), (True, False), (True, True)])
def test_a_post_reaches_the_take_with_its_tag(runs, bf16):
    k = _hand_off(runs, bf16)
    g = torch.Generator(device="cpu").manual_seed(9)
    x = torch.randn(1, 512, generator=g)
    if bf16:
        x = x.bfloat16().float()             # bf16 pairs carry the top halves: values already bf16
    box = torch.zeros(2 * 512, dtype=torch.int32, device="cuda")
    o = torch.full((1, 512), -3.0, device="cuda")
    for tag in (5, 6):                       # a second round over the first one's pairs
        x_ = x * tag
        k(x_.cuda(), box, torch.tensor([tag], dtype=torch.int32, device="cuda"), o)
        torch.cuda.synchronize()
        want = x_.clone()
        if not runs:
            want[0, torch.arange(512) % 9 == 4] = 0.0          # masked: ready at 0, nothing posted
        assert torch.equal(o.cpu(), want), tag
        if not runs:
            tags = box.view(-1, 2)[:, 1].cpu()
            assert torch.equal(tags[torch.arange(512) % 9 == 4], torch.zeros(57, dtype=torch.int32))


def test_an_fp32_lds_tile_is_addressed_element_by_element():
    st = hk.ST(hk.fp32, 16, 64)

    def body(x: hk.Tensor[hk.fp32], o: hk.Tensor[hk.fp32]):
        s = hk.lds.alloc(st, name="s")
        w = hk.arch.warp_id()
        c = hk.iota(S)
        s.store_elems(x.at[w, 0].load(S), hk.map_cpp("x1 * 512 + x0", c, w))
        hk.arch.barrier(drain=True)
        # warp w reads the other warp's row, reversed; an index past the tile reads 0
        got = s.load_elems(hk.map_cpp("x0 == 7 ? 4096 : (1 - x1) * 512 + 511 - x0", c, w))
        o.at[w, 0].store(got)

    k = hk.kernel(body, warps=2, name="lds_elems", grid=lambda p: (1, 1, 1))
    x = torch.randn(2, 512, generator=torch.Generator(device="cpu").manual_seed(10))
    o = torch.zeros(2, 512, device="cuda")
    k(x.cuda(), o)
    torch.cuda.synchronize()
    want = x.flip(0).flip(1).clone()
    want[:, 7] = 0.0
    assert torch.equal(o.cpu(), want)


def test_store_elems_runs_is_one_store_a_run_and_packs_fp8():
    def body(x: hk.Tensor[hk.fp32], o8: hk.Tensor[hk.fp8e4m3], o32: hk.Tensor[hk.fp32]):
        c = hk.iota(S)
        v = x.at[0, 0].load(S)
        # runs to their mirrored run slot: run r of 64 -> slot 63 - r (8 elements apart); run 5 masked
        idx = hk.map_cpp("(x0 / 8) == 5 ? -1 : (63 - x0 / 8) * 8", c)
        o8.store_elems(v, idx, runs=True)
        o32.store_elems(v, idx, runs=True)

    k = hk.kernel(body, warps=1, name="store_runs", grid=lambda p: (1, 1, 1))
    x = torch.randn(1, 512, generator=torch.Generator(device="cpu").manual_seed(11)) * 4
    o8 = torch.zeros(1, 512, dtype=torch.float8_e4m3fn, device="cuda")
    o32 = torch.full((1, 512), -1.0, device="cuda")
    k(x.cuda(), o8, o32)
    torch.cuda.synchronize()
    runs = x.view(64, 8).flip(0)                               # slot s holds run 63 - s
    want32 = runs.clone()
    want32[58] = -1.0                                          # run 5 -> slot 58: masked
    assert torch.equal(o32.cpu().view(64, 8), want32)
    want8 = runs.to(torch.float8_e4m3fn).view(torch.uint8).clone()
    want8[58] = 0
    assert torch.equal(o8.cpu().view(torch.uint8).view(64, 8), want8)
