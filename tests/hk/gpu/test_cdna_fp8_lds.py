"""fp8 e4m3 operands copied into LDS by gfx950's asynchronous DMA (buffer_load ... lds), plain and
gathered by index (AITER's MoE sort ids, rows staged in LDS by g.gather_prep), feeding the MFMA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import fp8e4m3, fp32, i32  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's DMA and fp8 MFMA", allow_module_level=True)

pytestmark = pytest.mark.gpu


def _rel(out, ref):
    return ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def _fp8(*shape, g):
    return (torch.randn(*shape, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)


def _fp8_dma_kernel(gather):
    def fp8_dma(a: hk.Tensor[fp8e4m3], ids: hk.Tensor[i32], b: hk.Tensor[fp8e4m3],
                c: hk.Tensor[fp32]):
        g = hk.group(4)
        sa = hk.lds.alloc(hk.ST(fp8e4m3, 64, 128), name="sa")
        if gather:
            g.gather_prep(ids, 0, 64)
            g.load(sa, a.at[0, 0], wait=False, gather=(ids, 0), staged=64)
        else:
            g.load(sa, a.at[0, 0], wait=False)
        g.wait_for(sa)
        hk.arch.barrier()
        w = hk.arch.warp_id()
        ta = sa.subtile(16, 128, w, 0).load(hk.RT(fp8e4m3, 16, 128))
        tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(fp8e4m3, 16, 128))
        acc = hk.zeros(hk.RT(fp32, 16, 16, "col"))
        acc += ta @ tb.T
        c[hk.elem_coord(0, 0, w * 16, 0)] = acc

    return hk.kernel(fp8_dma, arch="gfx950", warps=4, name=f"fp8_dma_{int(gather)}",
                     grid=lambda p: (1, 1, 1))


@pytest.mark.parametrize("gather", [False, True], ids=["dma", "gathered"])
def test_fp8_rows_copied_into_lds_asynchronously_feed_the_mfma(gather):
    g = torch.Generator(device="cuda").manual_seed(7 + gather)
    a, b = _fp8(100, 128, g=g), _fp8(16, 128, g=g)
    rows = torch.randperm(100, device="cuda", generator=g)[:64].to(torch.int32)
    rows[5] = 100                                                    # padding: reads zero
    ids = (rows | (torch.randint(0, 6, (64,), device="cuda", generator=g, dtype=torch.int32) << 24)
           if gather else torch.zeros(64, device="cuda", dtype=torch.int32))
    c = torch.zeros(64, 16, device="cuda")
    _fp8_dma_kernel(gather)(a, ids.view(1, -1), b, c)
    src = a.float()
    if gather:
        src = torch.cat([src, torch.zeros(1, 128, device="cuda")])[rows.long()]
    else:
        src = src[:64]
    ref = src @ b.float().T
    assert _rel(c, ref) < 1e-4
