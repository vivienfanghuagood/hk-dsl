"""hk.ops.mxfp8 and gemm_cdna's mx mode on gfx950: the MXFP8 GEMM schedules (K2) against the
scaled fp32 product -- and, where vLLM's kernel is there to compare with, bit for bit."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's scaled f8f6f4 MFMA", allow_module_level=True)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("m,n,k", [(256, 256, 512), (512, 384, 1280)])
def test_the_mxfp8_gemm_matches_the_scaled_product(m, n, k):
    from hk.ops import mxfp8

    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    blocks = torch.randint(124, 131, (n // 32, k // 32), device="cuda", dtype=torch.uint8,
                           generator=g)
    sb = mxfp8.expand_block_scales(blocks)
    out = mxfp8.gemm(a, b, sa, sb)
    af = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    bf = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ref = af @ bf.T
    assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 5e-3


@pytest.mark.parametrize("m,n,k,bm,bn", [(300, 512, 1152, 128, 256), (512, 256, 1280, 256, 128)])
def test_the_mx_pingpong_matches_the_scaled_product(m, n, k, bm, bn):
    """gemm_cdna's pingpong with fp8 operands and per-K-tile scales: a ragged
    M and an odd count of 128-wide K-tiles (the pad tile)."""
    from hk.ops import mxfp8

    g = torch.Generator(device="cuda").manual_seed(m + k)
    a = torch.randn(m, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    sb = torch.randint(124, 131, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    out = mxfp8.gemm_pp(a, b, sa, sb, block_n=bn, block_m=bm)
    af = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    bf = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ref = af @ bf.T
    assert ((out.float() - ref).abs().max() / ref.abs().max()).item() < 5e-3


@pytest.mark.parametrize("m,n,k", [(300, 512, 1152), (512, 256, 1280), (256, 768, 512)])
def test_the_mx_pingpong_direct_scales_match_the_ocp_ones(m, n, k):
    """mx_scales="direct" (packed A, AITER-layout B: no VALU on a loaded scale) computes what the
    four-warp mx4w kernel does, bit for bit -- ragged M, an odd K-tile count (B's scales padded).
    (The byte-scale pingpong spills at 256 x 256.)"""
    from hk.ops import mxfp8

    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    sa = torch.randint(118, 137, (m, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    sb = torch.randint(118, 137, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    ref = mxfp8.gemm_4w(a, b, mxfp8.pack_scales(sa), mxfp8.pack_scales(sb), rolled=True)
    out = mxfp8.gemm_8w(a, b, mxfp8.pack_scales(sa), mxfp8.aiter_scales(sb))
    assert torch.equal(out, ref)
    af = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    bf = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    full = af @ bf.T
    assert ((out.float() - full).abs().max() / full.abs().max()).item() < 5e-3


@pytest.mark.parametrize("m,n,k", [(300, 512, 1152), (128, 256, 1280)])
def test_the_mx_pingpong_128_row_blocks_match_the_four_warp_kernel(m, n, k):
    """gemm_8w(block_m=128): both operands' scales in AITER's 32-row layout ("direct32")."""
    from hk.ops import mxfp8

    g = torch.Generator(device="cuda").manual_seed(m + n + k + 1)
    a = torch.randn(m, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
    sa = torch.randint(118, 137, (m, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    sb = torch.randint(118, 137, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
    ref = mxfp8.gemm_4w(a, b, mxfp8.pack_scales(sa), mxfp8.pack_scales(sb), rolled=True)
    pad = -m % 32
    sa32 = torch.cat([sa, sa.new_full((pad, k // 32), 127)]) if pad else sa
    out = mxfp8.gemm_8w(a, b, mxfp8.aiter_scales(sa32), mxfp8.aiter_scales(sb), block_m=128)
    assert torch.equal(out, ref)
