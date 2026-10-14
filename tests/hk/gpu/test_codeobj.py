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


def test_a_kernel_with_all_64_kb_of_lds_launches():
    """The default GEMM double-buffers 2 x (128 + 128) x 64 bf16: the whole
    of a gfx11 workgroup's LDS, asked for at launch with no attribute set."""
    from hk.ops import gemm  # noqa: PLC0415

    k = gemm.KERNELS["gemm_bf16_128x128"]
    co = codeobj.load(k)
    assert co.shmem == 64 * 1024
    a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(384, 512, device="cuda", dtype=torch.bfloat16)
    want, got = (torch.empty(256, 384, device="cuda", dtype=torch.bfloat16)
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
