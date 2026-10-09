"""`torch.ops.hk.*`: the surface a framework actually calls.

Needs a Radeon and torch. What is being checked is not "does the kernel
compute a+b" -- tests/hk/gpu/test_elementwise.py does that -- but the four
properties that make an op usable inside vLLM or SGLang and that are each
invisible until something else breaks:

  * the output argument is really written through,
  * the launch lands on torch's current stream, not the default one,
  * the op survives `torch.compile` (so it needs its Meta implementation),
  * it can be captured into a CUDA graph.
"""

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a Radeon"
)


@pytest.fixture(scope="module")
def add_op():
    return hk.torch_op(hk.ops.elementwise.KERNELS["add_bf16"],
                       ns="hk_test", name="add_bf16")


def _inputs(rows=256, cols=512):
    a = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    return a, b, torch.empty_like(a)


def test_the_op_is_registered_under_its_namespace(add_op):
    assert add_op is torch.ops.hk_test.add_bf16
    assert callable(add_op)


def test_it_writes_through_the_output_argument(add_op):
    a, b, o = _inputs()
    assert add_op(a, b, o) is None          # the schema returns ()
    torch.testing.assert_close(o, a + b)


def test_building_it_twice_returns_the_same_op(add_op):
    again = hk.torch_op(hk.ops.elementwise.KERNELS["add_bf16"],
                        ns="hk_test", name="add_bf16")
    assert again is add_op


def test_it_runs_on_torchs_stream(add_op):
    # The op is enqueued on a side stream with no synchronisation of its own;
    # if it had gone to the default stream instead, the only thing ordering it
    # against the fill below would be luck.
    a, b, o = _inputs()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        a.fill_(2.0)
        b.fill_(3.0)
        add_op(a, b, o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    assert torch.all(o.float() == 5.0)


def test_a_non_contiguous_input_is_refused_rather_than_read_wrong(add_op):
    a, b, o = _inputs()
    with pytest.raises(RuntimeError, match="contiguous"):
        add_op(a.t(), b.t(), o.t())


def test_the_wrong_dtype_is_refused(add_op):
    a, b, _ = _inputs()
    o = torch.empty(a.shape, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must be"):
        add_op(a, b, o)


def test_torch_compile_can_trace_it(add_op):
    def f(a, b):
        o = torch.empty_like(a)
        torch.ops.hk_test.add_bf16(a, b, o)
        return o * 2

    a, b, _ = _inputs()
    want = f(a, b)
    got = torch.compile(f, fullgraph=True)(a, b)
    torch.testing.assert_close(got, want)


def test_it_can_be_captured_in_a_cuda_graph(add_op):
    a, b, o = _inputs()
    # Warm up on a side stream, which is what torch's own graph helpers do:
    # capture refuses to start while the kernel is still being JIT-resolved.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            add_op(a, b, o)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        add_op(a, b, o)

    a.fill_(1.5)
    b.fill_(2.5)
    g.replay()
    torch.cuda.synchronize()
    assert torch.all(o.float() == 4.0)
