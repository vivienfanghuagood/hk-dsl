"""hk.map and the integer / mask / bit operations on strips, on the GPU, against torch: C's
semantics where Python's differ (`//` and `%` truncate, `>>` is arithmetic), hk.where with
scalar arms, popcount, ctz (32 at 0), rcp within an ulp, the bf16 round trip, and fma as one
rounding where `a * b + c` is two.
"""

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402


def _target():
    from hk.target import current_arch, get_target

    return get_target(current_arch())


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(_target().wave != 64,
                                 reason="hk.map lowers on gfx950 (RDNA strips: S7b)")]

S = hk.RS(hk.i32, 1, 512, 8)
F = hk.RS(hk.fp32, 1, 512, 8)
ROWS = 64


def _ints(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(-(1 << 31), (1 << 31) - 1, (ROWS, 512), generator=g, dtype=torch.int64)
    x = x.to(torch.int32)
    x[0, :8] = torch.tensor([0, 1, -1, 7, -7, (1 << 31) - 1, -(1 << 31), 64], dtype=torch.int32)
    return x


def _c_div(a, b):
    return torch.div(a, b, rounding_mode="trunc")


def test_integer_operators_are_cs():
    def ints(x: hk.Tensor[hk.i32], q: hk.Tensor[hk.i32], r: hk.Tensor[hk.i32],
             s: hk.Tensor[hk.i32], b: hk.Tensor[hk.i32], n: hk.Tensor[hk.i32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        k = n.at[0].load_scalar()
        q.at[row, 0].store(a // 7 + k)
        r.at[row, 0].store(a % -5)
        s.at[row, 0].store((a >> 3) ^ (a << 2))
        b.at[row, 0].store(~a & 0xFFFF | ((a >> 4) - k) * 3)   # no overflow: C++'s int

    k = hk.kernel(ints, warps=1, name="map_ints", grid=lambda p: (p.x.rows, 1, 1))
    x = _ints(1)
    outs = [torch.zeros(ROWS, 512, dtype=torch.int32, device="cuda") for _ in range(4)]
    k(x.cuda(), *outs, torch.tensor([5], dtype=torch.int32).cuda())
    torch.cuda.synchronize()
    q, r, s, b = (o.cpu() for o in outs)
    x64 = x.long()
    wrap = lambda v: (v & 0xFFFFFFFF).to(torch.int64).where(  # noqa: E731
        (v & 0xFFFFFFFF) < (1 << 31), (v & 0xFFFFFFFF) - (1 << 32)).to(torch.int32)
    assert torch.equal(q, wrap(_c_div(x64, 7) + 5))
    assert torch.equal(r, wrap(x64 - _c_div(x64, -5) * -5))
    assert torch.equal(s, wrap((x64 >> 3) ^ (x64 << 2)))
    assert torch.equal(b, wrap((~x64 & 0xFFFF) | (((x64 >> 4) - 5) * 3)))


def test_comparisons_masks_and_where():
    def masks(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], o: hk.Tensor[hk.i32],
              f: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        v = y.at[row, 0].load(F)
        m = (a < 0) & ~(a % 3 == 0) | (v >= 0.5)
        o.at[row, 0].store(hk.where(m, a, -1))
        f.at[row, 0].store(hk.where(v != v, 0.0, hk.where(a > 0, v, 2.5)))   # NaN -> 0

    k = hk.kernel(masks, warps=1, name="map_masks", grid=lambda p: (p.x.rows, 1, 1))
    x = _ints(2)
    g = torch.Generator(device="cpu").manual_seed(2)
    y = torch.randn(ROWS, 512, generator=g)
    y[1, :4] = torch.tensor([float("nan"), 0.5, -0.0, float("inf")])
    o = torch.zeros(ROWS, 512, dtype=torch.int32, device="cuda")
    f = torch.zeros(ROWS, 512, device="cuda")
    k(x.cuda(), y.cuda(), o, f)
    torch.cuda.synchronize()
    m = (x < 0) & ~(torch.fmod(x, 3) == 0) | (y >= 0.5)
    assert torch.equal(o.cpu(), torch.where(m, x, torch.full_like(x, -1)))
    want = torch.where(y.isnan(), torch.zeros_like(y), torch.where(x > 0, y, torch.full_like(y, 2.5)))
    assert torch.equal(f.cpu().view(torch.int32), want.view(torch.int32))


def test_bit_functions_and_float_functions():
    def bits(x: hk.Tensor[hk.i32], y: hk.Tensor[hk.fp32], p: hk.Tensor[hk.i32],
             z: hk.Tensor[hk.i32], r: hk.Tensor[hk.fp32], t: hk.Tensor[hk.fp32],
             fm: hk.Tensor[hk.fp32], mu: hk.Tensor[hk.fp32]):
        row = hk.arch.block_idx.x
        a = x.at[row, 0].load(S)
        v = y.at[row, 0].load(F)
        p.at[row, 0].store(hk.math.popcount(a))
        z.at[row, 0].store(hk.math.ctz(a))
        r.at[row, 0].store(hk.math.rcp(v))
        t.at[row, 0].store(hk.map(lambda e: e.to(hk.bf16).to(hk.fp32), v))
        fm.at[row, 0].store(hk.math.fma(v, v, v))
        mu.at[row, 0].store(hk.map(lambda e: e * e + e, v))

    k = hk.kernel(bits, warps=1, name="map_bits", grid=lambda p: (p.x.rows, 1, 1))
    x = _ints(3)
    g = torch.Generator(device="cpu").manual_seed(3)
    y = torch.randn(ROWS, 512, generator=g) * torch.exp(torch.randn(ROWS, 512, generator=g) * 3)
    outs = [torch.zeros(ROWS, 512, dtype=torch.int32, device="cuda") for _ in range(2)]
    outs += [torch.zeros(ROWS, 512, device="cuda") for _ in range(4)]
    k(x.cuda(), y.cuda(), *outs)
    torch.cuda.synchronize()
    p, z, r, t, fm, mu = (o.cpu() for o in outs)
    u = x.long() & 0xFFFFFFFF
    want_p = torch.zeros_like(u)
    for i in range(32):
        want_p += (u >> i) & 1
    assert torch.equal(p, want_p.int())
    low = u & -u                                    # the lowest set bit
    want_z = torch.where(u == 0, torch.full_like(u, 32), torch.log2(low.double()).round().long())
    assert torch.equal(z, want_z.int())
    exact = 1.0 / y.double()
    ulp = (torch.nextafter(exact.float().abs(), torch.tensor(float("inf"))) - exact.float().abs())
    assert ((r.double() - exact).abs() <= ulp.double()).all()
    assert torch.equal(t.view(torch.int32), y.to(torch.bfloat16).float().view(torch.int32))
    one = (y.double() * y.double() + y.double()).float()
    two = (y * y) + y
    assert torch.equal(fm.view(torch.int32), one.view(torch.int32))
    assert torch.equal(mu.view(torch.int32), two.view(torch.int32))
    assert not torch.equal(one, two)                # the inputs tell the two apart
