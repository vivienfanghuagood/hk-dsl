"""hk.acc_strip: an MFMA's 16x16 fp32 result as the strip of the same registers (gfx950)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import fp32, i32  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="gfx950's")]


def test_acc_strip_is_the_accumulators_registers_in_place():
    """An MFMA result as a strip: element 4 l + e is C[4 (l / 16) + e][l % 16] -- the same values the
    tile's LDS round trip gives at those positions."""
    T_B = hk.RT(hk.bf16, 16, 32, "row")
    T_C = hk.RT(fp32, 16, 16, "col")

    def body(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[fp32]):
        c = hk.mma("mma_ABt", a.at[0, 0].load(T_B), b.at[0, 0].load(T_B), hk.zeros(T_C))
        q = hk.iota(hk.RS(i32, 1, 256, 4))
        o.store_elems(hk.acc_strip(c), hk.map_cpp("((x0 / 4) / 16 * 4 + x0 % 4) * 16 + (x0 / 4) % 16", q, dtype=i32))

    k = hk.kernel(body, arch="gfx950", warps=1, name="acc_strip", grid=lambda p: (1, 1, 1))
    g = torch.Generator(device="cpu").manual_seed(31)
    a = torch.randn(16, 32, generator=g).bfloat16()
    b = torch.randn(16, 32, generator=g).bfloat16()
    o = torch.zeros(16, 16, device="cuda")
    k(a.cuda(), b.cuda(), o)
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    assert ((o.cpu() - ref).abs().max() / ref.abs().max()).item() < 1e-5
