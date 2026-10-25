"""g.load(..., gather=(ids, at)): an async LDS copy whose rows come by AITER-style sort ids."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("gfx950's LDS DMA", allow_module_level=True)

import hk  # noqa: E402
from hk.types import bf16, i32  # noqa: E402


def _kernel():
    st = hk.ST(bf16, 32, 64)
    rt = hk.RT(bf16, 32, 64, "row")

    def body(x: hk.Tensor[bf16], ids: hk.Tensor[i32], out: hk.Tensor[bf16]):
        g = hk.group(4)
        s = hk.lds.alloc(st, name="s")
        at = hk.arch.block_idx.x * 32
        g.load(s, x.at[0, 64], wait=False, gather=(ids, at))
        g.wait_for(s)
        hk.arch.barrier()
        t = s.load(rt)
        if hk.arch.warp_id() == 0:
            out[hk.elem_coord(0, 0, at, 0)].store(t)

    body.__name__ = "gather_load_t"
    return hk.kernel(body, arch="gfx950", warps=4, name="gather_load_t",
                     grid=lambda p: (p.ids.cols // 32, 1, 1))


def test_gathered_rows():
    n, m = 100, 64
    x = torch.randn(n, 128, device="cuda").to(torch.bfloat16)
    tok = torch.randint(0, n, (m,), device="cuda", dtype=torch.int32)
    tok[5] = n                     # padding: past the tensor, reads zero
    tok[40] = n
    ids = (tok | (torch.randint(0, 6, (m,), device="cuda", dtype=torch.int32) << 24)).view(1, -1)
    out = torch.empty(m, 64, device="cuda", dtype=torch.bfloat16)
    _kernel()(x, ids, out)
    ref = torch.zeros(m, 64, device="cuda", dtype=torch.bfloat16)
    live = tok < n
    ref[live] = x[tok[live].long(), 64:]
    assert torch.equal(out, ref)



def _kernel_staged():
    st = hk.ST(bf16, 32, 64)
    rt = hk.RT(bf16, 32, 64, "row")

    def body(x: hk.Tensor[bf16], ids: hk.Tensor[i32], out: hk.Tensor[bf16]):
        g = hk.group(4)
        s = hk.lds.alloc(st, name="s")
        at = hk.arch.block_idx.x * 32
        g.gather_prep(ids, at, 32)
        g.load(s, x.at[0, 64], wait=False, gather=(ids, at), staged=32)
        g.wait_for(s)
        hk.arch.barrier()
        t = s.load(rt)
        if hk.arch.warp_id() == 0:
            out[hk.elem_coord(0, 0, at, 0)].store(t)

    body.__name__ = "gather_load_staged_t"
    return hk.kernel(body, arch="gfx950", warps=4, name="gather_load_staged_t",
                     grid=lambda p: (p.ids.cols // 32, 1, 1))


def test_gathered_rows_staged():
    n, m = 100, 64
    x = torch.randn(n, 128, device="cuda").to(torch.bfloat16)
    tok = torch.randint(0, n, (m,), device="cuda", dtype=torch.int32)
    tok[7] = n
    ids = (tok | (torch.randint(0, 6, (m,), device="cuda", dtype=torch.int32) << 24)).view(1, -1)
    out = torch.empty(m, 64, device="cuda", dtype=torch.bfloat16)
    _kernel_staged()(x, ids, out)
    ref = torch.zeros(m, 64, device="cuda", dtype=torch.bfloat16)
    live = tok < n
    ref[live] = x[tok[live].long(), 64:]
    assert torch.equal(out, ref)
