"""Rows by index, atomics and random draws, against torch (and Triton's
tl.rand, which the draws are bit-identical to)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

T16 = hk.RT(hk.bf16, 16, 64)
T32 = hk.RT(hk.fp32, 16, 64)


def _gather_kernel(ty):
    @hk.kernel(grid=lambda p: (p.o.cols // 64, hk.cdiv(p.o.rows, 16), 1),
               name=f"idx_gather_{ty.dtype.name}")
    def body(x: hk.Tensor[hk.bf16], ids: hk.Tensor[hk.i32],
             o: hk.Tensor[hk.bf16]):
        at = hk.arch.block_idx.y * 16
        col = hk.arch.block_idx.x * 64
        t = x.gather(ty, ids, at, col=col)
        o[hk.elem_coord(0, 0, at, col)].store(t, bounded=True)
    return body


def _scatter_kernel(ty):
    @hk.kernel(grid=lambda p: (p.x.cols // 64, hk.cdiv(p.x.rows, 16), 1),
               name=f"idx_scatter_{ty.dtype.name}")
    def body(x: hk.Tensor[hk.bf16], ids: hk.Tensor[hk.i32],
             o: hk.Tensor[hk.bf16]):
        at = hk.arch.block_idx.y * 16
        col = hk.arch.block_idx.x * 64
        t = x[hk.elem_coord(0, 0, at, col)].load(ty, bounded=True)
        o.scatter(t, ids, at, col=col)
    return body


def _ids(n, rows, g):
    """A shuffled index list with a -1 and an out-of-range entry in it."""
    ids = torch.randperm(rows, generator=g, device="cuda")[:n].to(torch.int32)
    ids[3] = -1
    ids[n // 2] = rows + 7
    return ids


def _want_gather(x, ids):
    ok = (ids >= 0) & (ids < x.shape[0])
    out = torch.zeros(ids.shape[0], x.shape[1], device="cuda", dtype=x.dtype)
    out[ok] = x[ids[ok].long()]
    return out


@pytest.mark.parametrize("ty", [T16, T32], ids=["bf16_tile", "fp32_tile"])
@pytest.mark.parametrize("n", [16, 45])
def test_gather_reads_the_indexed_rows(ty, n):
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(100, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = _ids(n, 100, g)
    out = torch.full((n, 128), 7.0, device="cuda", dtype=torch.bfloat16)
    _gather_kernel(ty)(x, ids, out)
    torch.cuda.synchronize()
    assert torch.equal(out, _want_gather(x, ids))


@pytest.mark.parametrize("ty", [T16, T32], ids=["bf16_tile", "fp32_tile"])
def test_scatter_writes_the_indexed_rows(ty):
    g = torch.Generator(device="cuda").manual_seed(5)
    n, rows = 48, 100
    x = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    ids = _ids(n, rows, g)
    out = torch.full((rows, 128), 7.0, device="cuda", dtype=torch.bfloat16)
    _scatter_kernel(ty)(x, ids, out)
    torch.cuda.synchronize()
    want = torch.full_like(out, 7.0)
    ok = (ids >= 0) & (ids < rows)
    want[ids[ok].long()] = x[ok]
    assert torch.equal(out, want)


@hk.kernel(grid=lambda p: (8, 1, 1))
def idx_atomic(x: hk.Tensor[hk.fp32], acc: hk.Tensor[hk.fp32],
               cnt: hk.Tensor[hk.i32]):
    t = x[hk.elem_coord(0, 0, 0, 0)].load(T32)
    acc[hk.elem_coord(0, 0, 0, 0)].atomic_add(t)
    cnt[hk.elem_coord(0, 0, 0, 0)].atomic_add_scalar(1)
    cnt[hk.elem_coord(0, 0, 0, 1)].atomic_max_scalar(hk.arch.block_idx.x)


def test_atomics_sum_across_workgroups():
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32)
    acc = torch.zeros_like(x)
    cnt = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    idx_atomic(x, acc, cnt)
    torch.cuda.synchronize()
    torch.testing.assert_close(acc, 8 * x, rtol=1e-6, atol=1e-5)
    assert cnt.tolist() == [[8, 7]]   # one add per warp, eight warps


@hk.kernel(grid=lambda p: (p.o.cols // 64, p.o.rows // 16, 1))
def idx_rand(o: hk.Tensor[hk.fp32]):
    r0 = hk.arch.block_idx.y * 16
    c0 = hk.arch.block_idx.x * 64
    t = hk.rand(T32, 1234, r0 * o.cols + c0, o.cols)
    o[hk.elem_coord(0, 0, r0, c0)].store(t)


def test_rand_draws_what_triton_draws():
    o = torch.empty(64, 256, device="cuda", dtype=torch.float32)
    idx_rand(o)
    torch.cuda.synchronize()
    assert 0.0 <= o.min().item() and o.max().item() < 1.0
    assert abs(o.mean().item() - 0.5) < 0.02
    triton = pytest.importorskip("triton")
    tl = triton.language

    @triton.jit
    def _tl_rand(out, n, seed, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + offs, tl.rand(seed, offs), mask=offs < n)

    want = torch.empty_like(o)
    _tl_rand[(o.numel() // 1024,)](want, o.numel(), 1234, BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.equal(o, want)
