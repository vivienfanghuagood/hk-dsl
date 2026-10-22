"""hk.schedules on the GPU: a schedule's launch() and hk.templates.gemm's
schedule= compute A @ B.T -- the shipped schedules, the composite and
split-K ones, and points of the space around them."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
import hk.schedules as S  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = pytest.mark.gpu

ARCH = current_arch()

if ARCH == "gfx950":
    CASES = [(4096, 4096, 4096), (300, 151936, 4096), (2048, 12288, 4096),
             (16, 4096, 4096), (1, 6144, 4096), (1024, 4096, 14336)]
else:
    CASES = [(4096, 4096, 4096), (77, 6144, 4096), (16, 4096, 4096), (1, 4096, 4096)]


def _ab(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return a, b


def _err(got, ref):
    return ((got.float() - ref).abs().max() / ref.abs().max()).item()


@pytest.mark.parametrize("op", ["matmul", "template"])
@pytest.mark.parametrize("m,n,k", CASES)
def test_the_shipped_schedule_launches_right(m, n, k, op):
    s = S.lookup(m, n, k, op=op)
    a, b = _ab(m, n, k)
    ref = a.float() @ b.float().t()
    assert _err(s.launch(a, b), ref) < 2e-2, s
    assert _err(hk.templates.gemm(a, b, schedule=s), ref) < 2e-2, s


@pytest.mark.parametrize("m,n,k", CASES[:3])
def test_the_neighbourhood_launches_right(m, n, k):
    s = S.lookup(m, n, k)
    pts = list(S.space(m, n, k, around=s, radius=1))[:8]
    assert pts
    a, b = _ab(m, n, k)
    ref = a.float() @ b.float().t()
    for t in pts:
        assert _err(t.launch(a, b), ref) < 2e-2, t


def test_an_epilogue_on_an_explicit_split_schedule():
    m, n, k = 256, 4096, 4096
    s = S.Dp(block_m=64, block_n=128, stages=3, xcd_chunk=0, splits=4) \
        if ARCH == "gfx950" else S.Rdna(block_m=64)
    a, b = _ab(m, n, k)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.gelu(a.float() @ b.float().t() + bias.float(), approximate="tanh")
    got = hk.templates.gemm(a, b, epilogue=hk.templates.gemm.bias_gelu, bias=bias, schedule=s)
    assert _err(got, ref) < 2e-2


def test_an_illegal_schedule_is_refused_before_launch():
    a, b = _ab(64, 4096, 4096)
    bad = S.Dp(block_n=96) if ARCH == "gfx950" else S.Rdna(block_n=96)
    with pytest.raises(S.Illegal):
        bad.launch(a, b)
    with pytest.raises(S.Illegal):
        hk.templates.gemm(a, b, schedule=bad)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128])
def test_attention_schedules_launch_right(d, causal):
    n = 1024
    g = torch.Generator(device="cuda").manual_seed(d + causal)
    q, k, v = (torch.randn(1, 8, n, d, device="cuda", dtype=torch.bfloat16, generator=g)
               for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal)
    base = S.lookup_attention(n, d, causal)
    pts = list(S.space_attention(n, d, causal=causal, around=base))[:4]
    for s in pts:
        got = s.launch(q, k, v, causal=causal)
        assert (got.float() - ref).abs().max().item() < 2e-2 * ref.abs().max().item() + 1e-2, s
