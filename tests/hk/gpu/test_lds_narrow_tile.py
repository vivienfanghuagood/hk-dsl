"""An LDS tile narrower than the st shape its read picks (32x16, read as the 32x16 row operand that picks
st_32x32): its copies compiled to nothing before the fix (the MFMA read stale LDS: 5e37 for a row sum of
~4); now it round-trips. acc[m, n] = sum_k x[m, k]."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's st shapes (include/cdna4)", allow_module_level=True)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("cols", [16, 32])
def test_a_narrow_lds_tile_round_trips(cols):
    st = hk.ST(hk.bf16, 32, cols)

    def narrow_lds(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        s = hk.lds.alloc(st, name="s")
        s.store(x.at[0, 0].load(hk.RT(hk.bf16, 32, cols, "row")))
        hk.arch.barrier(drain=True)
        p = s.load(hk.RT(hk.bf16, 32, cols, "row", frag=(32, 16)))
        one = hk.full(hk.RT(hk.bf16, 32, cols, "row", frag=(32, 16)), 1.0)
        acc = hk.zeros(hk.RT(hk.fp32, 32, 32, "col", frag=(32, 32)))
        acc += p @ one.T
        o.at[0, 0].store(acc)

    k = hk.kernel(narrow_lds, warps=1, grid=lambda p: (1, 1, 1), name=f"narrow_lds_{cols}")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(32, cols, device="cuda", generator=g).bfloat16()
    o = torch.zeros(32, 32, device="cuda")
    k(x, o)
    torch.cuda.synchronize()
    ref = x.float().sum(1, keepdim=True).expand(32, 32)
    assert torch.allclose(o, ref, rtol=1e-2, atol=1e-2), (o[0, :4], ref[0, 0])
