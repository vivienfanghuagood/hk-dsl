"""int8 into half / bf16 register tiles on RDNA: a register cast of an i8 tile, and a load of an
int8 global straight into a half / bf16 tile (bounded and not). Every int8 value is exact in either
type, so the results are bitwise torch's cast, and rows past a bound read zero."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp16, i8, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not current_arch().startswith("gfx1"),
                                 reason="RDNA's register layouts")]

TORCH = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _kernels(dt):
    def cast(x: hk.Tensor[i8], y: hk.Tensor[dt], n: hk.Tensor[i32]):
        r0, c0 = hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 16
        idx = hk.elem_coord(0, 0, r0, c0)
        left = n[hk.elem_coord(0, 0, 0, 0)].load_scalar() - r0
        y[idx].store(x[idx].load(hk.RT(i8, 16, 16, "row"), bounded=left).to(dt))

    def widen(x: hk.Tensor[i8], y: hk.Tensor[dt], n: hk.Tensor[i32]):
        r0, c0 = hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 16
        idx = hk.elem_coord(0, 0, r0, c0)
        left = n[hk.elem_coord(0, 0, 0, 0)].load_scalar() - r0
        y[idx].store(x[idx].load(hk.RT(dt, 16, 16, "row"), bounded=left))

    for f in (cast, widen):
        # the annotations are strings (from __future__ import annotations): give this dtype's
        f.__annotations__.update(x=hk.Tensor[i8], y=hk.Tensor[dt], n=hk.Tensor[i32])
    grid = lambda p: (p.x.cols // 16, p.x.rows // 16, 1)  # noqa: E731
    return {name: hk.kernel(f, warps=1, grid=grid, name=f"t_i8_{name}_{dt.name}")
            for name, f in (("cast", cast), ("widen", widen))}


@pytest.mark.parametrize("dt", [fp16, bf16], ids=lambda d: d.name)
@pytest.mark.parametrize("path", ["cast", "widen"])
@pytest.mark.parametrize("rows_valid", [48, 37])
def test_int8_into_a_16_bit_tile_is_exact(dt, path, rows_valid):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(-128, 128, (48, 64), device="cuda", generator=g, dtype=torch.int8)
    x[0, :4] = torch.tensor([-128, 127, 0, -1], dtype=torch.int8)
    y = torch.full((48, 64), 7.0, device="cuda", dtype=TORCH[dt.name])
    n = torch.tensor([rows_valid], device="cuda", dtype=torch.int32)
    _kernels(dt)[path](x, y, n)
    want = x.to(TORCH[dt.name])
    want[rows_valid:] = 0
    assert torch.equal(y, want)


def test_an_int8_row_at_a_4_byte_offset():
    """Rows starting 4 bytes past a 16-byte boundary (an int8 KV cache's V, after its K and a
    float scale): the widening load reads them as they are."""
    dt = fp16
    g = torch.Generator(device="cuda").manual_seed(1)
    base = torch.randint(-128, 128, (32, 16 * 5 + 4), device="cuda", generator=g,
                         dtype=torch.int8)
    x = base[:, 4:4 + 64]                       # row stride 84 B, first element at +4 B
    y = torch.zeros(32, 64, device="cuda", dtype=torch.float16)
    n = torch.tensor([32], device="cuda", dtype=torch.int32)
    _kernels(dt)["widen"](x, y, n)
    assert torch.equal(y, x.to(torch.float16))
