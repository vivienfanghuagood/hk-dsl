"""Every kernel hk.ops.gemm routes to on gfx950, against fp32, whatever the
table routes: the pingpong with and without split-K, the four-warp kernel's
blocks and splits, and the decode GEMV.

The shapes are chosen for the places each one has an edge: a ragged last
block of M, an odd number of 64-wide K-tiles (the pad tile), a K that splits
into shares of an odd tile count, and an N that 256 does not divide (the
pingpong's second, 128-wide launch).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx950":
    pytest.skip("hk.ops.gemm_cdna is gfx950's", allow_module_level=True)

from hk.ops import gemm_cdna  # noqa: E402

SHAPES = [(1, 256, 128), (16, 384, 192), (100, 512, 640), (257, 640, 1152),
          (513, 1024, 2048), (1000, 1536, 4096),
          # 96-wide blocks and 256-deep K-tiles, five of them (a pad tile).
          (33, 1152, 1280),
          # 224-wide blocks.
          (130, 1792, 768),
          # The pingpong's last 128 columns, split along K.
          (300, 1408, 1024)]


def _check(name, m, n, k):
    if not gemm_cdna.fits(name, m, n, k):
        pytest.skip(f"{name} does not take {m}x{n}x{k}")
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
    room = torch.full((m + 8, n), 7.0, device="cuda", dtype=torch.bfloat16)
    out = room[:m]
    gemm_cdna.run(name, a, b, out)
    ref = a.float() @ b.float().t()
    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 1e-2, f"{name} {m}x{n}x{k}: rel err {err:.3g}"
    assert torch.equal(room[m:], torch.full_like(room[m:], 7.0)), "wrote past M"


@pytest.mark.parametrize("name", list(gemm_cdna.CANDIDATES))
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_every_candidate_is_right(name, m, n, k):
    _check(name, m, n, k)


@pytest.mark.parametrize("name", ["ppw", "ppwd"])
@pytest.mark.parametrize("m,n,k,cus", [(300, 1408, 1024, 2), (257, 1792, 640, 4)])
def test_the_wave_split_is_right(name, m, n, k, cus, monkeypatch):
    """Whole waves of 256x256 blocks and the rest of N apart; on two or four
    CUs a test-sized N has both."""
    monkeypatch.setattr(gemm_cdna, "CUS", cus)
    assert 0 < gemm_cdna.wave_cut(m, n) < n
    _check(name, m, n, k)


def test_the_router_is_right_where_it_routes():
    for (m, n, k) in [(7, 4096, 4096), (300, 4096, 14336), (2048, 4096, 14336),
                      (4096, 4224, 1024)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
        got = gemm_cdna.matmul(a, b)
        ref = a.float() @ b.float().t()
        err = ((got.float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, f"{gemm_cdna.route(m, n, k)} {m}x{n}x{k}: rel err {err:.3g}"
