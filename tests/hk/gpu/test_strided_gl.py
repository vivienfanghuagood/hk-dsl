"""Strided globals on the GPU: g.stage reads an engine's layout in place.

vLLM's two paged-cache layouts, staged one (page, head) tile at a time into LDS and written out
row-major: ROCM_ATTN's x-split K (blocks, heads, head_dim / 8, block, 8) -- a token's row is
head_dim / 8 granules, `block * 8` elements apart -- and the flash layout (blocks, block, heads,
head_dim), a head's rows heads * head_dim apart. Both must come out bit for bit as the token-major
K, and rows past a tensor's last read zero even where the granule stride puts them inside the
tile's span.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.runtime.torch_ext import launcher  # noqa: E402

pytestmark = pytest.mark.gpu

BLOCK, D = 16, 128


def _copy_kernel(name, rows=BLOCK, cols=D):
    st = hk.ST(hk.bf16, rows, cols)
    ty = hk.RT(hk.bf16, rows, cols)

    def body(src: hk.Tensor[hk.bf16, "strided"], o: hk.Tensor[hk.bf16]):
        g = hk.group(4)
        s = hk.lds.alloc(st)
        buf = g.stage_buffer(s)
        page, head = hk.arch.block_idx.x, hk.arch.block_idx.y
        g.stage(buf, src[hk.elem_coord(page, head, 0, 0)])
        hk.arch.vm_wait(0)
        g.commit(s, buf, wait=True)
        hk.arch.barrier()
        if hk.arch.warp_id() == 0:
            o[hk.tile_coord(0, 0, page * src.depth + head, 0)] = s.load(ty)

    body.__name__ = name
    return hk.kernel(body, warps=4, grid=lambda p: (p.src.batch, p.src.depth, 1), name=name)


@pytest.mark.parametrize("heads", [1, 2, 8])
def test_rocm_x_split_k(heads):
    g = torch.Generator(device="cuda").manual_seed(heads)
    nb = 7
    k = torch.randn(nb, heads, BLOCK, D, device="cuda", dtype=torch.bfloat16, generator=g)
    # (blocks, heads, d/8, block, 8): what vLLM's reshape_and_cache writes
    xs = k.view(nb, heads, BLOCK, D // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    view = xs.permute(0, 1, 3, 2, 4)                  # (blocks, heads, block, d/8, 8)
    o = torch.full((nb * heads * BLOCK, D), float("nan"), device="cuda", dtype=torch.bfloat16)
    launcher(_copy_kernel("strided_copy_xsplit"))(view, o)
    torch.cuda.synchronize()
    assert torch.equal(o.view(nb, heads, BLOCK, D), k)


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_flash_layout(heads):
    g = torch.Generator(device="cuda").manual_seed(10 + heads)
    nb = 5
    f = torch.randn(nb, BLOCK, heads, D, device="cuda", dtype=torch.bfloat16, generator=g)
    view = f.permute(0, 2, 1, 3)                      # (blocks, heads, block, d)
    o = torch.full((nb * heads * BLOCK, D), float("nan"), device="cuda", dtype=torch.bfloat16)
    launcher(_copy_kernel("strided_copy_flash"))(view, o)
    torch.cuda.synchronize()
    assert torch.equal(o.view(nb, heads, BLOCK, D), view)


def test_rows_past_the_end_read_zero():
    # 10 of a page's 16 rows exist; in the x-split layout rows 10..15 sit inside the tile's span
    g = torch.Generator(device="cuda").manual_seed(3)
    nb, heads, live = 3, 2, 10
    xs = torch.randn(nb, heads, D // 8, BLOCK, 8, device="cuda", dtype=torch.bfloat16, generator=g)
    view = xs.permute(0, 1, 3, 2, 4)[:, :, :live]     # (blocks, heads, 10, d/8, 8)
    o = torch.full((nb * heads * BLOCK, D), float("nan"), device="cuda", dtype=torch.bfloat16)
    launcher(_copy_kernel("strided_copy_xsplit"))(view, o)
    torch.cuda.synchronize()
    got = o.view(nb, heads, BLOCK, D)
    want = view.reshape(nb, heads, live, D)
    assert torch.equal(got[:, :, :live], want)
    assert torch.count_nonzero(got[:, :, live:]) == 0
