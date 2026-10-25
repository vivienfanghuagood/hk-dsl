"""fp8 e4m3 MFMA on gfx950 against an fp32 matmul of the same fp8 values."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's f8f6f4 MFMA", allow_module_level=True)

pytestmark = pytest.mark.gpu


def _ir():
    """The kernels tests/hk/ir/test_cdna_fp8.py traces (same file name, so by path)."""
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "ir" / "test_cdna_fp8.py"
    spec = importlib.util.spec_from_file_location("ir_test_cdna_fp8", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fp8_mma_matches_fp32():
    fp8_mma_kernel = _ir().fp8_mma_kernel

    g = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn(64, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(16, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    c = torch.zeros(64, 16, device="cuda")
    fp8_mma_kernel(name="fp8_mma_gpu")(a, b, c)
    ref = a.float() @ b.float().T
    assert ((c - ref).abs().max() / ref.abs().max()).item() < 1e-4


def test_mx_scaled_mma_matches_the_scaled_fp32_product():
    mx_mma_kernel = _ir().mx_mma_kernel

    g = torch.Generator(device="cuda").manual_seed(1)
    M, N, K = 64, 32, 256
    a = (torch.randn(M, K, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (M, K // 32), device="cuda", dtype=torch.uint8, generator=g)
    sb = torch.randint(124, 131, (N, K // 32), device="cuda", dtype=torch.uint8, generator=g)
    c = torch.zeros(M, N, device="cuda")
    mx_mma_kernel(K, name="mx_mma_gpu")(a, b, sa.view(torch.int8), sb.view(torch.int8), c)
    af = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    bf = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    ref = af @ bf.T
    assert ((c - ref).abs().max() / ref.abs().max()).item() < 1e-4
