"""The row and pointwise ops on strips on gfx950: quantize at any row count,
the elementwise ops bit for bit against torch at any size, silu_mul on a
gate/up output's halves read in place, and hk.ops.rotary (vLLM's
rotary_embedding, NeoX) with and without positions. The norms on strips are
test_phase2.py's and test_rms_norm_weighted.py's; the layout itself is
test_strip.py's.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402
from hk.ops import elementwise, fused, quant  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(current_arch() not in quant.STRIP_ARCHS,
                                 reason="the row and pointwise ops run on strips on gfx950")]

BF16 = torch.bfloat16


def _rel(got, want):
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("m,n", [(1, 4096), (3, 4100), (7, 1000), (32, 5120), (5, 96),
                                 (17, 65), (2, 1001), (2048, 8192)])
@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
def test_quantize_any_rows(m, n, dtype):
    x = torch.randn(m, n, device="cuda", dtype=dtype) * 3
    q, s = hk.ops.quantize(x)
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    want = (xf * (127.0 / amax)).round().clamp(-128, 127)
    # test_phase2's budget: one step at most, on under 1% of elements. A bf16
    # row puts many products exactly on .5, and the last fp32 bit of
    # 127/amax decides them (0.22% of a 1x4096 row measured).
    assert (q.float() - want).abs().max().item() <= 1
    assert (q.float() != want).float().mean().item() < 0.01
    torch.testing.assert_close(s, (amax / 127).squeeze(-1), rtol=1e-6, atol=0)


def test_quantize_a_zero_row_is_zero():
    x = torch.zeros(3, 512, device="cuda", dtype=BF16)
    q, s = hk.ops.quantize(x)
    assert not q.any() and (s > 0).all()


@pytest.mark.parametrize("shape", [(1, 4096), (3, 1001), (17,), (2, 3, 4097), (2048, 8192)])
@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
def test_binary_ops_are_torchs_bit_for_bit(shape, dtype):
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    for op, want in (("add", a + b), ("sub", a - b), ("mul", a * b),
                     ("maximum", torch.maximum(a, b)), ("minimum", torch.minimum(a, b))):
        assert torch.equal(getattr(hk.ops, op)(a, b), want), op


@pytest.mark.parametrize("shape", [(1, 4096), (3, 1001), (2048, 5120)])
def test_unary_ops(shape):
    a = torch.randn(*shape, device="cuda", dtype=BF16).clamp(-8, 8)
    assert torch.equal(hk.ops.neg(a), -a)
    assert torch.equal(hk.ops.abs(a), a.abs())
    assert torch.equal(hk.ops.relu(a), torch.relu(a))
    assert _rel(hk.ops.exp(a), torch.exp(a.float())) < 1e-2


def test_the_plan_depends_on_the_bytes_and_the_arity():
    """Two warps of eight chunks for a two-input op moving 24-64 MB, two of
    one chunk past 257 MB, eight warps of one chunk otherwise
    (elementwise.STRIP_PLANS)."""
    big = elementwise._strip_plan("add", "bf16", 2048 * 4096, 2)
    one = elementwise._strip_plan("exp", "bf16", 2048 * 4096, 2)
    huge = elementwise._strip_plan("add", "bf16", 8192 * 8192, 2)
    assert big[0].warps == 2 and big[1] == 8 * 64 * 8
    assert one[0].warps == 8 and one[1] == 64 * 8
    assert huge[0].warps == 2 and huge[1] == 64 * 8


@pytest.mark.parametrize("m,i", [(1, 14336), (32, 17408), (2048, 28672), (5, 1000), (3, 72)])
def test_silu_mul_reads_the_halves_in_place(m, i):
    x = torch.randn(m, 2 * i, device="cuda", dtype=BF16)
    gate, up = x[:, :i], x[:, i:]
    want = F.silu(gate.float()) * up.float()
    assert _rel(hk.ops.silu_mul(gate, up), want) < 1e-2
    a = gate.contiguous()
    hk.ops.silu_mul(a, up, out=a)
    assert _rel(a, want) < 1e-2


def _ref_rotary(x, cs, positions):
    d = x.shape[-1]
    rows = cs[positions.long()] if positions is not None else cs[:x.shape[0]]
    c, s = rows[:, None, :d // 2], rows[:, None, d // 2:]
    x1, x2 = x[..., :d // 2].float(), x[..., d // 2:].float()
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1)


@pytest.mark.parametrize("t,h,d", [(1, 32, 128), (32, 40, 128), (2048, 64, 128), (7, 8, 256),
                                   (5, 3, 64),
                                   # the 16-byte form past 64 MB: ten one-warp
                                   # workgroups a token, and eight warps
                                   (8192, 40, 128), (4096, 64, 128)])
@pytest.mark.parametrize("pos", [False, True])
def test_rotary(t, h, d, pos):
    x = torch.randn(t, h, d, device="cuda", dtype=BF16)
    ang = torch.rand(t + 9, d // 2, device="cuda") * 6.28
    cs = torch.cat((ang.cos(), ang.sin()), -1).contiguous()
    p = torch.randint(0, t + 9, (t,), device="cuda", dtype=torch.int32) if pos else None
    got = hk.ops.rotary(x, cs, positions=p)
    assert _rel(got, _ref_rotary(x, cs, p)) < 1e-2
    xi = x.clone()
    hk.ops.rotary(xi, cs, positions=p, out=xi)
    assert torch.equal(xi, got)


def test_rotary_by_zero_is_a_copy():
    """cos 1 and sin 0: a kernel that pairs the wrong halves passes a random
    table by tolerance and fails this outright."""
    x = torch.randn(4, 8, 128, device="cuda", dtype=BF16)
    cs = torch.cat((torch.ones(4, 64), torch.zeros(4, 64)), -1).cuda()
    assert torch.equal(hk.ops.rotary(x, cs), x)


def test_rotary_refuses_what_it_cannot_take():
    x = torch.randn(4, 8, 96, device="cuda", dtype=BF16)
    with pytest.raises(ValueError, match="head dim 96"):
        hk.ops.rotary(x, torch.zeros(4, 96, device="cuda"))
    x = torch.randn(4, 8, 128, device="cuda", dtype=BF16)
    with pytest.raises(ValueError, match="positions must be 4 int32s"):
        hk.ops.rotary(x, torch.zeros(9, 128, device="cuda"),
                      positions=torch.zeros(4, device="cuda", dtype=torch.int64))
    assert fused.STRIP_ARCHS == ("gfx950",)
