"""Scattered stores and atomic adds by row index on gfx950, against torch: AITER's MoE sort ids
(token | choice << 24, padding past the tokens), `scatter(..., add=True)` from a register tile, and
`scatter_add_lds` from a warp's LDS tile (row-contiguous lanes)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp32, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("scatter(add) and scatter_add_lds are written for gfx950's atomics", allow_module_level=True)

pytestmark = pytest.mark.gpu

T32 = hk.RT(fp32, 16, 64)
M, TOPK, C = 40, 3, 128


def _sort_ids(g):
    """Every (token, choice) once, shuffled, in 16-row blocks, with padding rows (token M)."""
    pairs = torch.arange(M * TOPK, device="cuda", dtype=torch.int32)
    pairs = pairs[torch.randperm(pairs.numel(), device="cuda", generator=g)]
    ids = (pairs // TOPK) | ((pairs % TOPK) << 24)
    pad = torch.full((16 - ids.numel() % 16,), M, device="cuda", dtype=torch.int32)
    return torch.cat([ids, pad])


def _rows_kernel(add, sort_topk, out_dtype):
    def scat(x, ids, o):
        at = hk.arch.block_idx.y * 16
        col = hk.arch.block_idx.x * 64
        t = x[hk.elem_coord(0, 0, at, col)].load(T32)
        o.scatter(t, ids, at, col=col, sort_topk=sort_topk, add=add)

    scat.__annotations__ = {"x": hk.Tensor[fp32], "ids": hk.Tensor[i32], "o": hk.Tensor[out_dtype]}

    return hk.kernel(scat, arch="gfx950", warps=1,
                     name=f"scat_{int(add)}_{sort_topk}_{out_dtype.name}",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))


def _lds_kernel():
    def scat_lds(x: hk.Tensor[fp32], ids: hk.Tensor[i32], o: hk.Tensor[bf16]):
        ws = hk.lds.alloc(hk.ST(bf16, 16, 64), name="ws")
        at = hk.arch.block_idx.y * 16
        col = hk.arch.block_idx.x * 64
        t = x[hk.elem_coord(0, 0, at, col)].load(T32)
        ws.store(t.to(bf16))
        o.scatter_add_lds(ws, ids, at, col=col, sort_topk=TOPK)

    return hk.kernel(scat_lds, arch="gfx950", warps=1, name="scat_lds",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))


def _want(x, ids, add, dtype):
    tok, ch = (ids & 0xFFFFFF).long(), (ids >> 24).long()
    live = tok < M
    if add:
        out = torch.zeros(M, C, device="cuda")
        out.index_add_(0, tok[live], x[live])
    else:
        out = torch.zeros(TOPK * M, C, device="cuda")
        out[ch[live] * M + tok[live]] = x[live]
    return out.to(dtype)


def test_sort_ids_scatter_each_pair_to_its_choice_major_row():
    g = torch.Generator(device="cuda").manual_seed(1)
    ids = _sort_ids(g)
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    out = torch.zeros(TOPK * M, C, device="cuda", dtype=torch.bfloat16)
    _rows_kernel(False, TOPK, bf16)(x, ids.view(1, -1), out)
    assert torch.equal(out, _want(x, ids, False, torch.bfloat16))


@pytest.mark.parametrize("dtype", [fp32, bf16], ids=["fp32", "bf16"])
def test_an_atomic_scatter_sums_each_tokens_choices(dtype):
    g = torch.Generator(device="cuda").manual_seed(2)
    ids = _sort_ids(g)
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    td = torch.float32 if dtype is fp32 else torch.bfloat16
    out = torch.zeros(M, C, device="cuda", dtype=td)
    _rows_kernel(True, TOPK, dtype)(x, ids.view(1, -1), out)
    want = _want(x, ids, True, torch.float32)
    tol = 1e-5 if dtype is fp32 else 2e-2      # bf16 adds round each partial sum
    assert ((out.float() - want).abs().max() / want.abs().max()).item() < tol


def test_an_lds_tile_adds_into_each_tokens_row():
    g = torch.Generator(device="cuda").manual_seed(3)
    ids = _sort_ids(g)
    x = torch.randn(ids.numel(), C, device="cuda", generator=g)
    out = torch.zeros(M, C, device="cuda", dtype=torch.bfloat16)
    _lds_kernel()(x, ids.view(1, -1), out)
    want = _want(x, ids, True, torch.float32)
    assert ((out.float() - want).abs().max() / want.abs().max()).item() < 2e-2
