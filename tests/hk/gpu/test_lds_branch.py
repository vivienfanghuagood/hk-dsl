"""A tile allocated in LDS inside a runtime branch holds what is stored to it.

Before hk-lds-alloc counted allocations inside branches, the launch asked for
the kernel's own block's LDS only and a branch's tiles lay past its end: every
read of them returned 0 and every write was dropped, without an error. This
kernel puts 17 KiB in the branch ahead of the tile it reads back, so on such a
build the tile is well past any rounding of the launch's 1 KiB and `o` comes
back all zeros. The IR half is tests/hk/ir/test_lds_alloc.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import fp32  # noqa: E402

pytestmark = pytest.mark.gpu

T = hk.RT(fp32, 16, 16, "row")
T_COL = hk.RT(fp32, 16, 16, "col")


def _kernel():
    def lds_branch_roundtrip(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        a = hk.lds.alloc(hk.ST(fp32, 16, 16), name="a")          # the kernel's own block: 1 KiB
        t = x.at[0, 0].load(T)
        a.store(t)
        if hk.arch.warp_id() == 0:
            pad = hk.lds.alloc(hk.ST(fp32, 64, 64), name="pad")  # 16 KiB, in the branch
            b = hk.lds.alloc(hk.ST(fp32, 16, 16), name="b")
            pad.store(hk.zeros(hk.RT(fp32, 64, 64, "row")))
            b.store(t + a.load(T))
            # read back in the other layout: every lane reads what other lanes wrote, so the
            # values have to go through LDS (a same-lane read can be forwarded by the compiler)
            o.at[0, 0] = b.load(T_COL)

    return hk.kernel(lds_branch_roundtrip, warps=4, name="lds_branch_roundtrip",
                     grid=lambda p: (1, 1, 1))


def test_a_branch_tile_reads_back_what_was_stored():
    x = torch.randn(16, 16, device="cuda", dtype=torch.float32)
    o = torch.full((16, 16), float("nan"), device="cuda", dtype=torch.float32)
    _kernel()(x, o)
    torch.cuda.synchronize()
    assert torch.equal(o, x + x)
