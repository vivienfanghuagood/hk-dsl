"""gfx950's pingpong past its two block sizes: blocks of 160, 192 and 320
rows or columns (a half that is whole copies a warp, not a thread), and the
persistent stream-K forms -- every block's K cut into runs (stream_k=1), and
whole waves first with only the rest cut (stream_k=2).

The stream-K shapes are chosen for its edges: fewer blocks than workgroups
(each block shared by many), a part wave after whole ones, a ragged last block
of M, and a K of an odd number of K-tile pairs. A cut block is summed by
whichever of its workgroups finishes last, in a fixed order: the result must
be the same bits on every call.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
import hk.schedules as S  # noqa: E402
from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("the pingpong is gfx950's", allow_module_level=True)

pytestmark = pytest.mark.gpu


def _ab(m, n, k):
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return a, b


def _err(got, ref):
    return ((got.float() - ref).abs().max() / ref.abs().max()).item()


def _check(s, m, n, k, repeat=1):
    why = s.check(m, n, k, "gfx950")
    if why:
        pytest.skip(f"{s} at {m}x{n}x{k}: {why}")
    a, b = _ab(m, n, k)
    ref = a.float() @ b.float().t()
    room = torch.full((m + 8, n), 7.0, device="cuda", dtype=torch.bfloat16)
    s.launch(a, b, room[:m])
    first = room[:m].clone()
    assert _err(first, ref) < 2e-2, f"{s} {m}x{n}x{k}"
    assert torch.equal(room[m:], torch.full_like(room[m:], 7.0)), "wrote past M"
    for _ in range(repeat):
        room[:m].fill_(float("nan"))
        s.launch(a, b, room[:m])
        assert torch.equal(room[:m], first), f"{s} {m}x{n}x{k}: not the same bits twice"


WIDE = [(S.PingPong(block_n=192, warps_m=4, warps_n=2), 1000, 3072, 1088),
        (S.PingPong(block_m=192, warps_m=2, warps_n=4), 1000, 2048, 1024),
        (S.PingPong(block_n=160, warps_m=8, warps_n=1), 1024, 2560, 1024),
        (S.PingPong(block_m=160, warps_m=1, warps_n=8), 1000, 2048, 1024),
        (S.PingPong(block_m=128, block_n=320, warps_m=4, warps_n=2), 512, 2560, 1024),
        (S.PingPong(block_n=192, warps_m=4, warps_n=2, splits=3), 512, 3072, 3072)]


@pytest.mark.parametrize("s,m,n,k", WIDE, ids=lambda x: str(x) if isinstance(x, S.PingPong) else None)
def test_a_wide_block_is_right(s, m, n, k):
    _check(s, m, n, k)


SK = [S.PingPong(stream_k=1), S.PingPong(stream_k=2), S.PingPong(block_m=128, stream_k=1),
      S.PingPong(block_n=128, warps_m=4, warps_n=2, stream_k=2)]
SK_SHAPES = [(64, 1024, 4096),       # 4 blocks of 256: 64 workgroups a block
             (300, 3072, 2048),      # a ragged last block of M
             (4096, 5120, 1152),     # one whole wave and 64 blocks left; 9 K-tile pairs
             (1000, 2048, 1280)]


@pytest.mark.parametrize("s", SK, ids=str)
@pytest.mark.parametrize("m,n,k", SK_SHAPES)
def test_stream_k_is_right_and_the_same_every_call(s, m, n, k):
    _check(s, m, n, k, repeat=3)


def test_an_epilogue_runs_once_on_a_cut_blocks_sum():
    """bias + GELU on stream-K: applied to the summed block, not to each
    workgroup's share."""
    m, n, k = 300, 3072, 2048
    a, b = _ab(m, n, k)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.gelu(a.float() @ b.float().t() + bias.float(), approximate="tanh")
    got = hk.templates.gemm(a, b, epilogue=hk.templates.gemm.bias_gelu, bias=bias,
                            schedule=S.PingPong(stream_k=1))
    assert _err(got, ref) < 2e-2
