"""hk.gemv_w4 and the small-batch form of hk.ops.w4a16 built on it (`w4a16_gemv_kernel`): a few
tokens against 4-bit weights without WMMA, each lane decoding its own row.

Against a float64 reference (both forms, GPTQ's and AWQ's groups, 1-4 tokens), exact probes, 200
launches into NaN-filled outputs, every allocation end-fenced, and no scratch or spill in any
instantiation. The VOPD gate (hk.runtime.compile) rebuilds these kernels without dual issue where
the compiler pairs registers into gfx11's interlock hazard; the 200-launch test is the one that
would catch a pairing it missed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk.ops import w4a16  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    hk.target.current_arch() not in w4a16.GEMV_ARCHES,
    reason="the small-batch form is measured on " + ", ".join(w4a16.GEMV_ARCHES))]

LAUNCHES = 200


def _gptq_sym(n, k, group, g, q=None, s=None):
    """A symmetric GPTQ checkpoint (zero point 8) in hk's symmetric layout, and its codes/scales."""
    if q is None:
        q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    if s is None:
        s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device="cuda", dtype=torch.int32)
    qw, sc, zs = w4a16.repack_gptq(qweight, qzeros, s, group, dtype=torch.float16, symmetric=True)
    return q, s, (qw, sc, zs)


def _awq(n, k, group, g):
    """An AWQ checkpoint (zero points) in hk's layout, and the float64 W [N, K] it means."""
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    qw = w4a16._interleave(w4a16._pack_k(q.t().contiguous()))
    zs = (z.double() * s.double()).half()
    w = (q.double() * s.double().repeat_interleave(group, 0)
         - zs.double().repeat_interleave(group, 0)).t()
    return qw, s.t().contiguous(), zs.t().contiguous(), w


def _within_two_ulps_a_row(got, want):
    top = want.abs().amax(dim=1, keepdim=True)
    ulp = torch.pow(2.0, torch.floor(torch.log2(top.clamp_min(2.0 ** -14)))) * 2.0 ** -10
    return ((got.double() - want).abs() / ulp).max().item()


@pytest.mark.parametrize("group", [32, 64, 128])
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_the_symmetric_form_is_within_two_ulps_a_row_of_float64(m, group):
    g = torch.Generator(device="cuda").manual_seed(300 + 10 * m + group)
    n, k = 272, 2048
    q, s, (qw, sc, zs) = _gptq_sym(n, k, group, g)
    assert w4a16.gemv_takes(m, n, k, group, "fp16")
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    got = w4a16.matmul(x, qw, sc, zs, group)
    w = ((q - 8).double() * s.double().repeat_interleave(group, 0)).t()
    assert _within_two_ulps_a_row(got, x.double() @ w.t()) <= 2


@pytest.mark.parametrize("group", [32, 128])
@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_the_asymmetric_form_is_within_two_ulps_a_row_of_float64(m, group):
    g = torch.Generator(device="cuda").manual_seed(400 + 10 * m + group)
    n, k = 144, 1024
    qw, s, zs, w = _awq(n, k, group, g)
    assert w4a16.gemv_takes(m, n, k, group, "fp16")
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    got = w4a16.matmul(x, qw, s, zs, group)
    assert _within_two_ulps_a_row(got, x.double() @ w.t()) <= 2


def test_the_tile_is_the_mma_accumulator_layout_and_tokens_past_m_are_zero():
    """One warp, no reduction: hk.gemv_w4's tile, transposed and stored as an mma accumulator would
    be, is the product for the tokens x has and 0 for the rest of `tokens`."""
    g = torch.Generator(device="cuda").manual_seed(7)
    n, k, group = 16, 128, 32
    q, s, (qw, sc, _) = _gptq_sym(n, k, group, g)

    @hk.kernel(warps=1, grid=lambda p: (1, 1, 1))
    def probe(x: hk.Tensor[hk.fp16], qw: hk.Tensor[hk.i32], s: hk.Tensor[hk.fp16],
              c: hk.Tensor[hk.fp32]):
        acc = hk.gemv_w4(x, qw, s, None, 0, 0, 64, 2, group=32, tokens=4, zero=8, s_blocked=True)
        c.at[0, 0].store(hk.transpose(acc))

    x = torch.randn(2, k, device="cuda", dtype=torch.float16, generator=g)
    c = torch.full((16, 16), float("nan"), device="cuda")
    probe(x, qw, sc, c)
    w = ((q - 8).double() * s.double().repeat_interleave(group, 0)).t()
    want = x.double() @ w.t()
    assert _within_two_ulps_a_row(c[:2], want) <= 2
    assert torch.equal(c[2:].cpu(), torch.zeros(14, 16))


def test_exact_probes():
    """Codes all 9 with scales 1 (W = 1) and x all 1: every output is exactly K. Codes all 8: W = 0,
    exactly 0. The asymmetric form with zs = q * s: exactly 0."""
    n, k, group = 64, 1024, 32
    for code, want in ((9, float(k)), (8, 0.0)):
        q = torch.full((k, n), code, device="cuda", dtype=torch.int32)
        s = torch.ones(k // group, n, device="cuda", dtype=torch.float16)
        _, _, (qw, sc, zs) = _gptq_sym(n, k, group, None, q=q, s=s)
        for m in (1, 2, 3, 4):
            x = torch.ones(m, k, device="cuda", dtype=torch.float16)
            got = w4a16.matmul(x, qw, sc, zs, group)
            assert torch.equal(got.cpu(), torch.full((m, n), want, dtype=torch.float16))
    q = torch.full((n, k), 5, dtype=torch.int32, device="cuda")
    qw = w4a16._interleave(w4a16._pack_k(q))
    s = torch.full((n, k // 128), 0.25, device="cuda", dtype=torch.float16)
    zs = s * 5
    x = torch.randn(4, k, device="cuda", dtype=torch.float16)
    assert torch.equal(w4a16.matmul(x, qw, s, zs, 128).cpu(), torch.zeros(4, n, dtype=torch.float16))


@pytest.mark.parametrize("sym", [True, False], ids=["gptq", "awq"])
def test_the_same_bits_on_every_launch_into_nan_filled_outputs(sym):
    g = torch.Generator(device="cuda").manual_seed(11)
    n, k = 5120, 6144
    if sym:
        _, _, (qw, s, zs) = _gptq_sym(n, k, 32, g)
        group = 32
    else:
        qw, s, zs, _ = _awq(n, k, 128, g)
        group = 128
    for m in (1, 2, 3, 4):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
        first, differ, nan = None, 0, 0
        for _ in range(LAUNCHES):
            out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float16)
            w4a16.matmul(x, qw, s, zs, group, out=out)
            torch.cuda.synchronize()
            nan += bool(out.isnan().any())
            if first is None:
                first = out.clone()
            else:
                differ += not torch.equal(out, first)
        assert (nan, differ) == (0, 0), (m, nan, differ)


def test_every_allocation_end_fenced():
    """The last row of x and the last 16-row block of W end flush against an unmapped page."""
    from _fence import fenced

    fenced('''
    from hk.ops import w4a16
    g = torch.Generator(device="cuda").manual_seed(3)
    for n, k, group in ((16, 1024, 32), (48, 2048, 128)):
        q = torch.randint(0, 16, (n, k), device="cuda", generator=g, dtype=torch.int32)
        qw = w4a16._interleave(w4a16._pack_k(q))
        s = torch.rand(n // 16, k // group * 16, device="cuda", generator=g).half()
        for m in (1, 2, 3, 4):
            x = torch.randn(m, k, device="cuda", generator=g).half()
            y = step(f"gemv sym N{n} K{k} M{m}", lambda: w4a16.matmul(x, qw, s, None, group))
            ref = x.float() @ w4a16.dequant(qw, s, None, group).float().t()
            close(f"sym N{n} M{m}", y.float(), ref, atol=2e-2, rtol=2e-2)
    ''')


def test_no_instantiation_spills_or_uses_scratch():
    for tokens in (1, 2, 3, 4):
        for group, sym in ((32, True), (64, True), (128, True), (32, False), (128, False)):
            b = w4a16.gemv_kernel_for(tokens, group, sym).build()
            for r in b.kernels:
                assert (r.scratch, r.vgpr_spill) == (0, 0), (tokens, group, sym, r)
