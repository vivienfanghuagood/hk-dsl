"""fp8 e4m3 MFMA on gfx950 against an fp32 matmul of the same fp8 values."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("gfx950's f8f6f4 MFMA", allow_module_level=True)

pytestmark = pytest.mark.gpu


def test_fp8_mma_matches_fp32():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ir"))
    from test_cdna_fp8 import fp8_mma_kernel

    g = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn(64, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    b = (torch.randn(16, 128, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    c = torch.zeros(64, 16, device="cuda")
    fp8_mma_kernel(name="fp8_mma_gpu")(a, b, c)
    ref = a.float() @ b.float().T
    assert ((c - ref).abs().max() / ref.abs().max()).item() < 1e-4
