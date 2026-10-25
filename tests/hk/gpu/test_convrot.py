"""hk.ops.convrot, the ConvRot W8A8 linear on gfx1100 (hk.ops.gemm's int8 WMMA, op_dtype=i8): against
its own fp32 oracle (convrot.reference: the same rotation, quantization and dequant in torch) and
against the unquantized product; repeatable bit for bit; the rotation orthogonal."""

from __future__ import annotations

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx1100":
    pytest.skip("gfx1100's int8 WMMA", allow_module_level=True)

from hk.ops import convrot  # noqa: E402


@pytest.mark.parametrize("group", [16, 64, 256])
@pytest.mark.parametrize("bias", [False, True])
def test_convrot_linear_matches_its_oracle(group, bias):
    torch.manual_seed(0)
    m, n, k = 1024, 768, 1024
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None
    wq, ws = convrot.prepare_weight(w, group)
    y = convrot.linear(x, wq, ws, group, bias=b)
    ref = convrot.reference(x, wq, ws, group, bias=b)
    scale = ref.float().abs().max()
    assert (y.float() - ref.float()).abs().max() / scale < 2e-2
    exact = x.float() @ w.float().t() + (b.float() if bias else 0)
    assert (y.float() - exact).abs().max() / exact.abs().max() < 6e-2   # W8A8's own error
    assert all(torch.equal(convrot.linear(x, wq, ws, group, bias=b), y) for _ in range(3))


def test_convrot_rotation_is_orthogonal():
    torch.manual_seed(0)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    rx, rw = convrot.rotate(x, 64), convrot.rotate(w, 64)
    lhs = rx.float() @ rw.float().t()
    rhs = x.float() @ w.float().t()
    assert (lhs - rhs).abs().max() / rhs.abs().max() < 1e-2
