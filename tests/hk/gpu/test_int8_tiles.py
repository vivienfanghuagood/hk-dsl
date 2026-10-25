"""i8 register tiles straight to and from global memory on gfx1100.

The elementwise paths of HipKittens' rdna3 global<->register copies wrote a lane's elements two to
a word (`data[e >> 1].x/.y`). rt<int8>'s words hold four (int8_4), so an i8 tile loaded straight
from global memory came back scrambled and partly unwritten. It still built, ran and was
deterministic: the SageAttention-INT8 rerun's c0 lost a debug round to it (cosine 0.53). Every
element is now addressed by its index in the packed word, whatever the packing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import i8, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx1100":
    pytest.skip("gfx1100's int8 WMMA tiles", allow_module_level=True)

pytestmark = pytest.mark.gpu


def _round_trip():
    ty = hk.RT(i8, 16, 64, "row")

    def i8_round_trip(x: hk.Tensor[i8], y: hk.Tensor[i8]):
        idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 64)
        y[idx] = x[idx].load(ty)

    return hk.kernel(i8_round_trip, warps=1, name="i8_round_trip",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))


def _qk_from_global():
    a_t, b_t, c_t = hk.RT(i8, 16, 16, "row"), hk.RT(i8, 16, 16, "row"), hk.RT(i32, 16, 16, "col")

    def i8_mma_from_global(a: hk.Tensor[i8], b: hk.Tensor[i8], c: hk.Tensor[i32]):
        acc = hk.zeros(c_t)
        for kk in range(a.cols // 16):
            acc += a[hk.elem_coord(0, 0, 0, kk * 16)].load(a_t) @ b[hk.elem_coord(0, 0, 0, kk * 16)].load(b_t).T
        c[hk.elem_coord(0, 0, 0, 0)] = acc

    return hk.kernel(i8_mma_from_global, warps=1, name="i8_mma_from_global", grid=lambda p: (1, 1, 1))


def test_an_i8_tile_goes_global_to_registers_to_global_unchanged():
    x = torch.randint(-128, 128, (64, 256), device="cuda", dtype=torch.int8)
    y = torch.zeros_like(x)
    _round_trip()(x, y)
    assert torch.equal(y, x)


def test_an_i8_product_of_tiles_loaded_from_global_is_exact():
    a = torch.randint(-128, 128, (16, 64), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (16, 64), device="cuda", dtype=torch.int8)
    c = torch.zeros(16, 16, device="cuda", dtype=torch.int32)
    _qk_from_global()(a, b, c)
    assert torch.equal(c.cpu(), a.cpu().int() @ b.cpu().int().T)


@pytest.mark.parametrize("m,n,k", [(256, 256, 512), (384, 512, 1024)])
def test_the_gemm_on_i8_operands_is_exact(m, n, k):
    """hk.ops.gemm's schedule with op_dtype=i8: v_wmma_i32_16x16x16_iu8, i32 accumulators,
    raw i32 C -- bit-exact against the integer product."""
    from hk.ops import gemm

    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8, generator=g)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8, generator=g)
    c = torch.zeros(m, n, device="cuda", dtype=torch.int32)
    gemm.gemm_kernel(f"gemm_i8_t_{m}_{n}_{k}", op_dtype=hk.i8)(a, b, c)
    assert torch.equal(c.cpu(), a.cpu().int() @ b.cpu().int().T)
