"""A torch op over a device-only code object (hk.runtime.codeobj), against
the same kernel registered through libtorch (hk.runtime.torch_ext): the same
bytes out, a strided slice read as the C++ op reads it, and a capture that
replays over new inputs.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import elementwise, norm  # noqa: E402
from hk.runtime import codeobj, torch_ext  # noqa: E402

pytestmark = pytest.mark.gpu


def _rmsw():
    return norm.KERNELS["rmsw_bf16_w4"].specialize(TPW=1, FOLD=16, EXACT=1)


def test_the_code_object_op_writes_what_the_library_op_writes(monkeypatch):
    k = _rmsw()
    monkeypatch.setenv("HK_TORCH_OP", "library")
    lib = torch_ext.torch_op(k, name="co_test_rmsw_lib")
    co = codeobj.torch_op(k, name="co_test_rmsw_co")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(300 * 16, 256, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    w = torch.rand(16, 256, device="cuda", dtype=torch.bfloat16, generator=g)
    a, b = torch.empty_like(x), torch.full_like(x, 7.0)
    lib(x, w, a)
    co(x, w, b)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def _gemm_with_all_its_lds():
    """(kernel, LDS bytes, (M, N, K)): on gfx11 the two-buffer GEMM, 2 x
    (128 + 128) x 64 bf16, the whole of a workgroup's LDS; on gfx950 the
    pingpong GEMM's 128 KB, twice what a launch gets without the attribute."""
    from hk.target import current_arch  # noqa: PLC0415

    if current_arch() == "gfx950":
        from hk.ops import gemm_cdna  # noqa: PLC0415

        return gemm_cdna.kernel_for(gemm_cdna.DEFAULT), 128 * 1024, (512, 512, 512)
    from hk.ops import gemm  # noqa: PLC0415

    return gemm.gemm_kernel("co_test_gemm_lds2", lds_buffers=2), 64 * 1024, (256, 384, 512)


def test_a_kernel_with_all_its_lds_launches():
    """A GEMM that asks for all the LDS it is allowed, at launch, with no
    attribute set."""
    k, lds, (m, n, kk) = _gemm_with_all_its_lds()
    co = codeobj.load(k)
    assert co.shmem == lds
    a = torch.randn(m, kk, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, kk, device="cuda", dtype=torch.bfloat16)
    want, got = (torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                 for _ in range(2))
    k(a, b, want)
    shapes, ptrs, rows = codeobj.tensor_args(co, (a, b, got), "gemm")
    co.launch(0, torch.cuda.current_stream().cuda_stream, shapes, ptrs, rows)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


def test_a_slice_of_the_last_axis_is_read_in_place():
    k = elementwise.KERNELS["add_bf16"]
    co = codeobj.torch_op(k, name="co_test_add")
    g = torch.Generator(device="cuda").manual_seed(1)
    wide = torch.randn(128, 2 * 512, device="cuda", dtype=torch.bfloat16,
                       generator=g)
    x, y = wide[:, :512], wide[:, 512:]
    assert not x.is_contiguous()
    out = torch.empty(128, 512, device="cuda", dtype=torch.bfloat16)
    co(x, y, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x + y)


def test_it_is_captured_and_replays_over_new_inputs():
    k = elementwise.KERNELS["add_bf16"]
    co = codeobj.torch_op(k, name="co_test_add_graph")
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        co(x, y, out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        co(x, y, out)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, x + y)


def test_a_kernel_call_can_launch_the_code_object(monkeypatch):
    """HK_LAUNCH=codeobj: `Kernel.__call__` over the code object, on torch's
    stream, with the pybind module never built."""
    import hk  # noqa: PLC0415

    monkeypatch.setenv("HK_LAUNCH", "codeobj")
    t = hk.RT(hk.bf16, 16, 64)

    @hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
    def co_test_scale(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
        v = x[idx].load(t)
        o[idx] = v + v

    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    co_test_scale(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x + x)
    assert not co_test_scale._mod_cache


def test_a_wrong_dtype_is_refused_before_launch():
    co = codeobj.torch_op(elementwise.KERNELS["add_bf16"], name="co_test_add_dt")
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="bf16"):
        co(x, x, torch.empty_like(x))


@pytest.mark.parametrize("preload", [False, True], ids=["flat", "preloaded"])
def test_a_flat_kernel_launches_as_a_code_object(preload):
    """A noalias strip kernel, flat arguments on CDNA (the struct on RDNA),
    preloaded or not: the code object, packed by its layout table, writes
    what the pybind module writes, a ragged row bounded."""
    from hk.target import current_arch, get_target  # noqa: PLC0415

    t = get_target(current_arch())
    if not t.strips:
        pytest.skip(f"no strips on {t.arch}")
    k = elementwise.STRIP_KERNELS["strip_add_bf16_w4"]
    k = k.preloaded() if preload else k
    cw = t.wave * 8
    co = codeobj.torch_op(k, name=f"co_test_strip_add_{int(preload)}", CW=cw, RUN=8,
                          EXACT=0)
    n = 4 * cw * 5 + 24
    x = torch.randn(1, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    a, b = torch.full_like(x, 7.0), torch.full_like(x, 7.0)
    k(x, y, a, CW=cw, RUN=8, EXACT=0)
    co(x, y, b)
    torch.cuda.synchronize()
    assert torch.equal(a, b)
    assert torch.equal(a, (x.float() + y.float()).to(torch.bfloat16))
