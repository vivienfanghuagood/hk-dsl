"""4-bit weights: the repacks against the checkpoint formats' own meaning, and
the decode kernel against the dequantized product."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import w4a16  # noqa: E402


def _awq_checkpoint(n, k, group, g):
    """A random AutoAWQ checkpoint and the W it means."""
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()

    def pack(v):          # [R, C] -> [R, C/8], column c in nibble AWQ_ORDER[c]
        r, c = v.shape
        v = v.reshape(r, c // 8, 8).to(torch.int64)
        shift = torch.tensor([4 * o for o in w4a16.AWQ_ORDER], device=v.device)
        words = (v << shift).sum(-1)
        return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)

    w = ((q - z.repeat_interleave(group, 0)).float()
         * s.float().repeat_interleave(group, 0)).t()      # [N, K]
    return pack(q), pack(z), s, w


def test_awq_repack_means_what_the_checkpoint_means():
    g = torch.Generator(device="cuda").manual_seed(0)
    n, k, group = 256, 512, 128
    qweight, qzeros, scales, w = _awq_checkpoint(n, k, group, g)
    qw, s, zs = w4a16.repack_awq(qweight, qzeros, scales, group)
    assert qw.shape == (n, k // 8) and s.shape == (n, k // group)
    got = w4a16.dequant(qw, s, zs, group).float()
    assert (got - w).abs().max().item() <= 2e-2 * w.abs().max().item()


def test_gptq_repack_means_what_the_checkpoint_means():
    g = torch.Generator(device="cuda").manual_seed(1)
    n, k, group = 128, 256, 64
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(1, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    zst = (z - 1).to(torch.int64).reshape(k // group, n // 8, 8)
    qzeros = (zst << shifts).sum(-1)
    qzeros = torch.where(qzeros >= 2 ** 31, qzeros - 2 ** 32, qzeros).to(torch.int32)
    qw, sc, zs = w4a16.repack_gptq(qweight, qzeros, s, group)
    w = ((q - z.repeat_interleave(group, 0)).float() * s.float().repeat_interleave(group, 0)).t()
    got = w4a16.dequant(qw, sc, zs, group).float()
    assert (got - w).abs().max().item() <= 2e-2 * w.abs().max().item()


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("m", [1, 7, 16, 33, 64])
def test_the_decode_kernel_matches_the_dequantized_product(m, dtype):
    g = torch.Generator(device="cuda").manual_seed(m)
    n, k, group = 512, 2048, 128
    td = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    qweight, qzeros, scales, w = _awq_checkpoint(n, k, group, g)
    qw, s, zs = w4a16.repack_awq(qweight, qzeros, scales, group, dtype=td)
    assert w4a16.takes(m, n, k, group)
    x = torch.randn(m, k, device="cuda", dtype=td, generator=g)
    got = w4a16.matmul(x, qw, s, zs, group).float()
    want = x.float() @ w4a16.dequant(qw, s, zs, group).float().t()
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err


@pytest.mark.parametrize("m", [1, 4, 16, 33, 64])
def test_the_decode_kernel_takes_gptq_groups_of_32(m):
    """A GPTQ checkpoint's 32-column groups: the load narrows to one group
    (w4a16.schedule_for), and the product is still the dequantized one."""
    g = torch.Generator(device="cuda").manual_seed(100 + m)
    n, k, group = 512, 2048, 32
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device="cuda", dtype=torch.int32)
    qw, sc, zs = w4a16.repack_gptq(qweight, qzeros, s, group, dtype=torch.float16)
    assert w4a16.takes(m, n, k, group) and w4a16.schedule_for(m, k, group)["slices"] == 2
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    got = w4a16.matmul(x, qw, sc, zs, group).float()
    w = ((q - 8).float() * s.float().repeat_interleave(group, 0)).t()
    want = x.float() @ w.t()
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err


# ---------------------------------------------------------------- symmetric (zs None, s in 16-row blocks)


def _gptq_sym(n, k, group, g, q=None, s=None):
    """A symmetric GPTQ v1 checkpoint (every stored zero 7, i.e. zero point 8) for codes q [K, N] and
    scales s [K/group, N] (random when None), and its symmetric repack."""
    if q is None:
        q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    if s is None:
        s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device="cuda", dtype=torch.int32)
    return q, s, qweight, qzeros, w4a16.repack_gptq(qweight, qzeros, s, group, dtype=torch.float16,
                                                    symmetric=True)


def test_the_symmetric_repack_refuses_other_zero_points():
    g = torch.Generator(device="cuda").manual_seed(5)
    _, s, qweight, qzeros, (qw, sc, zs) = _gptq_sym(64, 128, 32, g)
    assert zs is None and sc.shape == (64 // 16, 128 // 32 * 16)
    assert torch.equal(w4a16.unblock_scales(sc), s.t().contiguous())
    bad = qzeros.clone()
    bad[0, 0] = 0x77777778                      # one zero point 9
    with pytest.raises(ValueError, match="not all 8"):
        w4a16.repack_gptq(qweight, bad, s, 32, symmetric=True)


@pytest.mark.parametrize("m", [1, 2, 4, 7, 16, 33, 64])
def test_the_symmetric_kernel_is_within_an_ulp_a_row_of_float64(m):
    """fp16 activations, GPTQ's 32-column groups: against the float64 product, each output row within
    two fp16 ulps of its largest |value| (one rounding of each weight, fp32 accumulation, one rounding
    of the output)."""
    g = torch.Generator(device="cuda").manual_seed(200 + m)
    n, k, group = 512, 2048, 32
    q, s, _, _, (qw, sc, zs) = _gptq_sym(n, k, group, g)
    assert w4a16.takes(m, n, k, group)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, generator=g)
    got = w4a16.matmul(x, qw, sc, zs, group).double()
    w = ((q - 8).double() * s.double().repeat_interleave(group, 0)).t()       # [N, K]
    want = x.double() @ w.t()
    top = want.abs().amax(dim=1, keepdim=True)
    ulp = torch.pow(2.0, torch.floor(torch.log2(top.clamp_min(2.0 ** -14)))) * 2.0 ** -10
    assert ((got - want).abs() <= 2 * ulp).all(), ((got - want).abs() / ulp).max().item()


def test_every_code_and_scale_dequantizes_to_one_rounding_of_the_exact_product():
    """q = 0..15 against scales from subnormal to large, read back through one-hot rows of x: each
    output is one weight, which must be (q - 8) * s rounded once to nearest even -- and exactly 0 at
    q = 8."""
    n, k, group = 64, 128, 32
    scales = torch.tensor([2.0 ** -24, 3 * 2.0 ** -20, 6.0e-5, 1.0e-3, 0.0371, 0.5, 7.0, 100.0, 1000.0],
                          dtype=torch.float16)
    q = (torch.arange(k).view(k, 1) + torch.arange(n).view(1, n)) % 16
    s = scales[(torch.arange(k // group).view(-1, 1) * 7 + torch.arange(n).view(1, -1)) % len(scales)]
    q, s = q.to(torch.int32).cuda(), s.cuda()
    _, _, _, _, (qw, sc, zs) = _gptq_sym(n, k, group, None, q=q, s=s)
    want = ((q - 8).double() * s.double().repeat_interleave(group, 0)).half().t()   # [N, K], once
    for k0 in range(0, k, 64):
        x = torch.zeros(64, k, device="cuda", dtype=torch.float16)
        x[torch.arange(64), k0 + torch.arange(64)] = 1.0
        got = w4a16.matmul(x, qw, sc, zs, group)                                   # [64, N]
        assert torch.equal(got.t().view(torch.int16), want[:, k0:k0 + 64].contiguous().view(torch.int16))


def test_all_zero_codes_are_exactly_minus_eight_s_times_the_sums():
    """Every code 0: W = -8 s, so with power-of-two scales and small non-negative integer x the output
    is exact. (x of both signs can cancel to 0 within a WMMA's dot product, which then leaves a
    residue of a few 2^-18 -- the asymmetric path the same: the matrix core's accumulation, not
    the decode.)"""
    g = torch.Generator(device="cuda").manual_seed(9)
    n, k, group = 128, 1024, 32
    q = torch.zeros(k, n, device="cuda", dtype=torch.int32)
    e = torch.randint(-8, -2, (k // group, n), device="cuda", generator=g)
    s = torch.pow(2.0, e.float()).half()
    _, _, _, _, (qw, sc, zs) = _gptq_sym(n, k, group, g, q=q, s=s)
    x = torch.randint(0, 4, (4, k), device="cuda", generator=g).half()
    got = w4a16.matmul(x, qw, sc, zs, group)
    xs = x.double().reshape(4, k // group, group).sum(-1)                         # [4, G]
    want = (xs @ (-8.0 * s.double())).half()                                      # [4, N]
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
