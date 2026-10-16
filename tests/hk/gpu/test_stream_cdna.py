"""hk kernels on gfx950 launch on torch's current stream.

A launch on the null stream is ordered against nothing torch does on its own
streams, and a CUDA graph cannot capture it (the capture fails with
"operation not permitted when stream is capturing"). On CDNA the pybind entry
point takes the stream (`scaffold.takes_stream`) and `Kernel.__call__` passes
torch's current one, so every hk op -- here the GEMM's routes, its split-K
pair of launches and the decode GEMV, and a norm -- is captured and replayed
like any torch op. Built with the null-stream entry point instead, the
capture comes out empty and every replay test here fails.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch, get_target  # noqa: E402

if get_target(current_arch()).wave != 64:
    pytest.skip("the stream-taking pybind entry point is CDNA's",
                allow_module_level=True)

from hk.codegen import scaffold  # noqa: E402
from hk.ops import gemm_cdna, norm  # noqa: E402

#: One shape per kind of route: the decode GEMV, the four-warp kernel, the
#: pingpong, and each with split-K (two launches and a reduce).
ROUTES = [("decode", 4, 4096, 4096), ("dp64", 256, 4096, 4096),
          ("dp128k4", 512, 4096, 4096), ("pp", 2048, 4096, 1024),
          ("ppk2", 1024, 4096, 4096)]


def _ab(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5
    return a, b


def _err(got, a, b):
    ref = a.float() @ b.float().t()
    return ((got.float() - ref).abs().max() / ref.abs().max()).item()


def test_the_entry_point_takes_the_stream():
    k = gemm_cdna.kernel_for(gemm_cdna.DEFAULT)
    src = k.source()
    assert "bind_function_on_stream<launch_on>" in src
    assert scaffold.takes_stream(current_arch())


@pytest.mark.parametrize("name,m,n,k", ROUTES)
def test_a_route_replays_from_a_cuda_graph(name, m, n, k):
    if not gemm_cdna.fits(name, m, n, k):
        pytest.skip(f"{name} does not take {m}x{n}x{k}")
    a, b = _ab(m, n, k, 0)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        gemm_cdna.run(name, a, b, out)            # compiles, loads, warms
    torch.cuda.current_stream().wait_stream(side)
    assert _err(out, a, b) < 1e-2

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm_cdna.run(name, a, b, out)
    a2, b2 = _ab(m, n, k, 1)
    a.copy_(a2)
    b.copy_(b2)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _err(out, a2, b2) < 1e-2, f"{name}: the replay did not compute"


def test_the_router_and_a_norm_replay_from_one_graph():
    m, n, k = 64, 4096, 4096
    a, b = _ab(m, n, k, 2)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    y = gemm_cdna.matmul(a, b)
    z = norm.rms_norm(y, w)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemm_cdna.matmul(a, b)
        z = norm.rms_norm(y, w)
    a.copy_(_ab(m, n, k, 3)[0])
    graph.replay()
    torch.cuda.synchronize()
    ref = a.float() @ b.float().t()
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    err = ((z.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 3e-2, f"rel err {err:.3g}"
