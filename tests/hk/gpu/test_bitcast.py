"""Tile.bitcast: the same registers read as another type of the same width."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp16  # noqa: E402

pytestmark = pytest.mark.gpu


@hk.kernel(grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))
def bitcast_bf16_fp16(x: hk.Tensor[bf16], y: hk.Tensor[fp16]):
    idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 64)
    y[idx] = x[idx].load(hk.RT(bf16, 16, 64)).bitcast(fp16)


def test_a_bf16_tile_read_as_fp16_keeps_every_bit():
    x = torch.randn(32, 128, device="cuda").to(torch.bfloat16)
    x[0, :3] = torch.tensor([float("inf"), -0.0, float("nan")])
    y = torch.zeros(32, 128, device="cuda", dtype=torch.float16)
    bitcast_bf16_fp16(x, y)
    assert torch.equal(y.view(torch.int16), x.view(torch.int16))
