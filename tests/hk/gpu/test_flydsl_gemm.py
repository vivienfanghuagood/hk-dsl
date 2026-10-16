"""FlyDSL's GEMM as a path of `torch.ops.hk_vllm.linear`, where FlyDSL is
installed (FLYDSL_ROOT names its checkout); skipped, saying why, where not.

What the routing tests cannot see: that the call the routes make computes
x @ W.T at the Ms FlyDSL's tiles divide, that an M no tile divides goes to
the table's own path, and that the launch is on torch's stream with no host
sync, so a CUDA graph captures and replays it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.integration import engine, flydsl_gemm, vllm_gemm  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a Radeon"),
    pytest.mark.skipif(not flydsl_gemm.available(), reason=flydsl_gemm.status()),
]

RTOL, ATOL = 2e-2, 1e-2
MS = (64, 96, 128, 192, 1024, 4096)


@pytest.fixture(scope="module", autouse=True)
def op():
    vllm_gemm.register_op()
    return torch.ops.hk_vllm.linear


@pytest.fixture
def routes(request):
    """A table preferring FlyDSL at exactly MS, at the tile it picks there."""
    n, k = request.param

    vllm_gemm.set_routes(engine.RouteTable({(n, k): {
        "runs": [[1, 64, "hk_decode"], [80, 10 ** 6, "hk"]],
        "prefer": {"flydsl": [[m, m, flydsl_gemm.variant(m, n, k)] for m in MS]}}}))
    yield n, k
    vllm_gemm.set_routes(None)


def _wx(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return x, w


def _took(fn):
    before = dict(vllm_gemm.STATS)
    out = fn()
    torch.cuda.synchronize()
    return out, {p: vllm_gemm.STATS[p] - before[p] for p in before}


SHAPES = [(6144, 4096), (4096, 12288)]


@pytest.mark.parametrize("routes", SHAPES, indirect=True)
@pytest.mark.parametrize("m", MS)
def test_flydsl_computes_x_wt_where_the_table_prefers_it(op, routes, m):
    n, k = routes
    assert vllm_gemm.route_for(m, n, k) == "flydsl"
    x, w = _wx(m, n, k)
    y, took = _took(lambda: op(x, w, None))
    assert took["flydsl"] == 1
    torch.testing.assert_close(y.float(), x.float() @ w.float().t(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("routes", SHAPES[:1], indirect=True)
@pytest.mark.parametrize("m", [1000, 1030, 1056])
def test_an_m_at_another_tile_or_none_takes_the_tables_path(op, routes, m):
    n, k = routes
    x, w = _wx(m, n, k)
    y, took = _took(lambda: op(x, w, None))
    assert took["flydsl"] == 0 and took["hk"] == 1
    torch.testing.assert_close(y.float(), x.float() @ w.float().t(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("routes", SHAPES[:1], indirect=True)
@pytest.mark.parametrize("m", [128, 1024])
def test_flydsl_replays_from_a_cuda_graph(op, routes, m):
    # Captured under the op's graph="safe" contract: a host sync inside
    # FlyDSL's launch would raise GraphCaptureError here.
    n, k = routes
    x, w = _wx(m, n, k)
    op(x, w, None)                                 # compile before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = op(x, w, None)
    assert vllm_gemm.PATCH.captured["flydsl"] >= 1
    x2, _ = _wx(m, n, k, seed=1)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y.float(), x2.float() @ w.float().t(), rtol=RTOL, atol=ATOL)
