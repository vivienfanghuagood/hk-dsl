"""The PyTorch-op recipe of docs/torch-ops.md, as written there: a kernel of your own, made a
`torch.library` custom op by `hk.torch_op`, then used eagerly, under `torch.compile` and inside a
CUDA graph. Each property is checked for both registrations: over the device-only code object
(the default) and the libtorch one (`HK_TORCH_OP=library`).

- `torch.library.opcheck` passes: the schema's out-parameter annotation, the Meta (fake)
  implementation, and the autograd and AOT-dispatch registration checks.
- `torch._dynamo.explain` sees one graph and no graph break, and `torch.compile(fullgraph=True)`
  gives eager's answer. The same holds with `dynamic=True` over several shapes, and with
  `mode="reduce-overhead"` (Inductor's CUDA graphs) replaying it.
- A `torch.cuda.CUDAGraph` captures the op, and a replay reads the inputs as they are then.

Inductor does not lower or fuse the op: it stays an opaque call in the compiled graph. The docs
say so.
"""

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@hk.kernel(grid=lambda p: (p.o.rows * hk.cdiv(p.o.cols, p.CW), 1, 1))
def gelu_mul(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *, CW=512):
    S = hk.RS(hk.fp32, 1, CW, 8)
    per_row = hk.cdiv(o.cols, CW)
    row = hk.arch.block_idx.x // per_row
    c0 = (hk.arch.block_idx.x - row * per_row) * CW
    x = a.at[row, c0].load(S, bounded=True)
    x.gelu(into=x)
    x *= b.at[row, c0].load(S, bounded=True)
    o.at[row, c0].store(x, bounded=True)


@pytest.fixture(scope="module", params=["codeobj", "library"])
def op(request):
    mp = pytest.MonkeyPatch()
    mp.setenv("HK_TORCH_OP", request.param)
    try:
        yield hk.torch_op(gelu_mul, ns=f"hk_recipe_{request.param}", name="gelu_mul")
    finally:
        mp.undo()


def _ref(a, b):
    return torch.nn.functional.gelu(a.float(), approximate="tanh") * b.float()


def _inputs(rows=64, cols=1000):
    a = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    return a, torch.randn_like(a), torch.empty_like(a)


def test_the_schema_takes_the_output_as_a_mutated_argument(op):
    s = str(op.default._schema)
    assert s.endswith("gelu_mul(Tensor a, Tensor b, Tensor(a!) o) -> ()"), s


def test_eager(op):
    a, b, o = _inputs()
    assert op(a, b, o) is None
    torch.testing.assert_close(o.float(), _ref(a, b), rtol=2e-2, atol=2e-2)


def test_opcheck_passes(op):
    a, b, o = _inputs()
    torch.library.opcheck(op, (a, b, o))


def _f(op):
    def f(a, b):
        o = torch.empty_like(a)
        op(a, b, o)
        return (o * 2).sum(-1)
    return f


def test_dynamo_sees_one_graph_and_no_break(op):
    a, b, _ = _inputs()
    ex = torch._dynamo.explain(_f(op))(a, b)
    assert (ex.graph_count, ex.graph_break_count) == (1, 0), ex.break_reasons


@pytest.mark.parametrize("kw", [dict(fullgraph=True), dict(fullgraph=True, dynamic=True),
                                dict(fullgraph=True, mode="reduce-overhead")],
                         ids=["fullgraph", "dynamic", "reduce-overhead"])
def test_torch_compile_gives_eagers_answer(op, kw):
    torch._dynamo.reset()
    f = _f(op)
    g = torch.compile(f, **kw)
    for shape in ((64, 1000), (128, 2048), (64, 1000), (3, 4100)):
        a, b, _ = _inputs(*shape)
        torch.testing.assert_close(g(a, b), f(a, b), rtol=1e-2, atol=1e-1)


def test_a_cuda_graph_captures_it_and_a_replay_reads_the_inputs_then(op):
    a, b, o = _inputs()
    s = torch.cuda.Stream()                 # warm up off the capture stream, as torch's helpers do
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(a, b, o)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        op(a, b, o)
    for _ in range(2):
        a.copy_(torch.randn_like(a))
        b.copy_(torch.randn_like(b))
        o.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(o.float(), _ref(a, b), rtol=2e-2, atol=2e-2)
