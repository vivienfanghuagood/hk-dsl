"""The LDS a launch asks for counts every allocation, those inside a branch too.

The generated C++ allocates LDS with a bump allocator along the path the kernel
takes, so a tile allocated inside a runtime `if` starts where the path's earlier
allocations end. hk-lds-alloc used to size the launch from the kernel's own
block only: a branch's tiles then sat past the end of the launch's LDS, where
reads return 0 and writes are dropped -- no error, a wrong answer. The GPU half
of this is tests/hk/gpu/test_lds_branch.py.
"""

import pytest

import hk
from hk import fp32

# A power-of-two size is placed at a multiple of itself (the pass rounds at
# least as much as the C++ allocator's 16 B).
S1K = hk.ST(fp32, 16, 16)        # 1 KiB
S2K = hk.ST(fp32, 16, 32)        # 2 KiB
T = hk.RT(fp32, 16, 16, "row")


def _src(body, arch):
    return hk.kernel(body, arch=arch, warps=4, grid=lambda p: (1, 1, 1),
                     name=body.__name__).source()


def _lds(src):
    line = next(ln for ln in src.splitlines() if "// LDS    :" in ln)
    return int(line.split(":")[1].split()[0])


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100"])
def test_an_allocation_inside_a_branch_is_in_the_launch(arch):
    def lds_in_branch(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        a = hk.lds.alloc(S1K, name="a")
        t = x.at[0, 0].load(T)
        a.store(t)
        if hk.arch.warp_id() == 0:
            b = hk.lds.alloc(S1K, name="b")
            b.store(t)
            o.at[0, 0] = b.load(T)

    assert _lds(_src(lds_in_branch, arch)) == 2048


def test_the_two_arms_of_a_branch_share_their_start():
    # a 1 KiB first, then [1024, 3072) in one arm and [2048, 4096) in the other: a
    # path takes one arm, so 4096 (the arms one after the other would be 6144)
    def lds_two_arms(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        a = hk.lds.alloc(S1K, name="a")
        t = x.at[0, 0].load(T)
        a.store(t)
        if hk.arch.warp_id() == 0:
            b = hk.lds.alloc(S1K, name="b")
            c = hk.lds.alloc(S1K, name="c")
            b.store(t)
            c.store(b.load(T))
            o.at[0, 0] = c.load(T)
        else:
            d = hk.lds.alloc(S2K, name="d")
            d.store(hk.zeros(hk.RT(fp32, 16, 32, "row")))

    assert _lds(_src(lds_two_arms, "gfx950")) == 4096


def test_a_branch_inside_a_branch_counts():
    def lds_nested(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        a = hk.lds.alloc(S1K, name="a")
        t = x.at[0, 0].load(T)
        a.store(t)
        if hk.arch.warp_id() < 2:
            if hk.arch.warp_id() == 0:
                b = hk.lds.alloc(S2K, name="b")
                b.store(hk.zeros(hk.RT(fp32, 16, 32, "row")))
                o.at[0, 0] = a.load(T)

    assert _lds(_src(lds_nested, "gfx950")) == 4096           # b at 2048


def test_an_allocation_inside_a_runtime_loop_is_refused():
    # every iteration would bump the allocator again, past any size known up front
    def lds_in_loop(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        a = hk.lds.alloc(S1K, name="a")
        t = x.at[0, 0].load(T)
        a.store(t)
        for _ in hk.range(x.cols // 16):
            b = hk.lds.alloc(S1K, name="b")
            b.store(t)
            o.at[0, 0] = b.load(T)

    with pytest.raises(Exception, match="inside a runtime loop"):
        _src(lds_in_loop, "gfx950")
