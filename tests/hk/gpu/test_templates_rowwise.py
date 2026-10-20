"""hk.templates.elementwise and hk.templates.rowwise on the GPU, against torch:
the gate/up halves read in place, broadcast operands, batched and ragged
shapes, in place, several outputs and int8; RMSNorm, the fused add (its
residual bit for bit), softmax, LayerNorm, quantization, and a cross-entropy
over a vocabulary too wide to hold (the multi-pass schedule), int64 labels and
ignore_index included. Runs on gfx1100 and gfx950.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import hk  # noqa: E402
from hk import templates as T  # noqa: E402

pytestmark = pytest.mark.gpu

BF16 = torch.bfloat16


def _close(got, want, rtol=2e-2, atol=1e-2):
    torch.testing.assert_close(got.float(), want.float(), rtol=rtol, atol=atol)


# ---------------------------------------------------------------- elementwise

silu_mul = T.elementwise(lambda g, u: hk.math.silu(g) * u)
gelu_mul = T.elementwise(lambda g, u: hk.math.gelu(g) * u)
add = T.elementwise(lambda a, b: a + b)


@pytest.mark.parametrize("m,i", [(1, 14336), (37, 9216), (3, 1001), (64, 4096), (2, 64)])
@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
def test_gate_up_halves_in_place(m, i, dtype):
    x = torch.randn(m, 2 * i, device="cuda", dtype=dtype)
    g, u = x[:, :i], x[:, i:]
    _close(silu_mul(g, u), F.silu(g.float()) * u.float())
    _close(gelu_mul(g, u), F.gelu(g.float(), approximate="tanh") * u.float())


@pytest.mark.parametrize("shape", [(1, 4096), (3, 1001), (17,), (2, 3, 4097), (513, 8192)])
@pytest.mark.parametrize("dtype", [BF16, torch.float16, torch.float32])
def test_an_add_is_torchs_bit_for_bit(shape, dtype):
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    assert torch.equal(add(a, b), a + b)


def test_broadcast_operands():
    k = T.elementwise(lambda x, b, s, t: x * s + b * t)
    x = torch.randn(2, 5, 4096, device="cuda", dtype=BF16)
    b = torch.randn(4096, device="cuda", dtype=BF16)
    s = torch.randn(2, 5, 1, device="cuda", dtype=torch.float32)
    t = torch.full((1,), 0.5, device="cuda", dtype=torch.float32)
    want = x.float() * s + b.float() * t
    _close(k(x, b, s, t), want)
    assert k.plan(x, b, s, t).flat is False


def test_a_batched_strided_slice():
    x = torch.randn(3, 7, 2 * 1536, device="cuda", dtype=BF16)
    g, u = x[..., :1536], x[..., 1536:]
    _close(gelu_mul(g, u), F.gelu(g.float(), approximate="tanh") * u.float())


def test_in_place_and_a_partial_overlap():
    x = torch.randn(64, 4096, device="cuda", dtype=BF16)
    y = torch.randn_like(x)
    want = x + y
    got = add(x, y, out=x)
    assert got is x and torch.equal(x, want)
    z = torch.randn(1, 8192, device="cuda", dtype=BF16)
    a, b = z[:, :4096], z[:, 2048:6144]          # overlapping views
    want = a + b
    add(a, b, out=z[:, 1024:5120])                # overlaps both: read before written
    assert torch.equal(z[:, 1024:5120], want)


def test_several_outputs_int8_and_a_constexpr():
    k = T.elementwise(lambda x, *, A=100.0: (x * A, x + 1.0), outputs=(hk.i8, hk.fp32))
    x = torch.randn(9, 1000, device="cuda", dtype=BF16)
    q, f = k(x)
    assert q.dtype == torch.int8 and f.dtype == torch.float32
    want = (x.float() * 100.0).round().clamp(-128, 127)
    assert (q.float() - want).abs().max().item() <= 1
    _close(f, x.float() + 1.0, rtol=0, atol=0)
    q2, _ = k(x, A=1.0)
    assert (q2.float() - x.float().round().clamp(-128, 127)).abs().max().item() <= 1


@pytest.mark.parametrize("m,i", [(1, 14336), (37, 9216), (5, 1001), (300, 4096)])
def test_hk_ops_silu_mul_takes_the_halves_on_every_gpu(m, i):
    """The same arguments on gfx1100 as on gfx950: the two halves of a
    gate/up output, read in place, at any row count (gfx1100's tile kernel
    wanted contiguous operands and 16 rows)."""
    x = torch.randn(m, 2 * i, device="cuda", dtype=BF16)
    g, u = x[:, :i], x[:, i:]
    want = F.silu(g.float()) * u.float()
    _close(hk.ops.silu_mul(g, u), want)
    out = torch.empty(m, i, device="cuda", dtype=BF16)
    hk.ops.silu_mul(g, u, out)
    _close(out, want)
    assert torch.equal(out, silu_mul(g, u))       # the template's kernel, the same answer


# ---------------------------------------------------------------- rowwise


@T.rowwise
def rms_norm(x, w, *, EPS=1e-6):
    return x * (T.row_mean(x * x) + EPS).rsqrt() * w


@T.rowwise(outputs=2)
def add_rms_norm(x, r, w, *, EPS=1e-6):
    h = x + r
    return h * (T.row_mean(h * h) + EPS).rsqrt() * w, h


@T.rowwise
def softmax(x):
    e = (x - T.row_max(x)).exp()
    return e / T.row_sum(e)


@T.rowwise
def layernorm(x, *, EPS=1e-5):
    xc = x - T.row_mean(x)
    return xc * (T.row_mean(xc * xc) + EPS).rsqrt()


@T.rowwise(outputs=(hk.i8, T.per_row(hk.fp32)))
def quantize(x):
    amax = T.row_max(abs(x)).maximum(1e-12)
    return x * (127.0 / amax), amax / 127.0


@T.rowwise(outputs=T.per_row(hk.fp32))
def cross_entropy(logits, label):
    m = T.row_max(logits)
    lse = m + T.row_sum((logits - m).exp()).log()
    return (lse - T.pick(logits, label)) * (label != -100).to_int()


SHAPES = [(1, 4096), (7, 4100), (32, 5120), (2048, 8192), (5, 1000), (3, 64)]


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("dtype", [BF16, torch.float32])
def test_rms_norm(m, n, dtype):
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)
    xf = x.float()
    want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    _close(rms_norm(x, w), want)
    _close(rms_norm(x, w, EPS=0.5), xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 0.5)
           * w.float())


@pytest.mark.parametrize("m,n", SHAPES)
def test_the_fused_add_in_place_rounds_the_residual_like_torch(m, n):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    r = torch.randn_like(x)
    w = torch.randn(n, device="cuda", dtype=BF16)
    h = x + r
    hf = h.float()
    want = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    y, h2 = add_rms_norm(x, r, w, out=(x, r))
    assert y is x and h2 is r
    assert torch.equal(r, h)
    _close(x, want)


@pytest.mark.parametrize("m,n", SHAPES + [(3, 151936), (2, 70001)])
@pytest.mark.parametrize("dtype", [BF16, torch.float32])
def test_softmax(m, n, dtype):
    x = torch.randn(m, n, device="cuda", dtype=dtype) * 3
    _close(softmax(x), torch.softmax(x.float(), -1), rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("m,n", SHAPES)
def test_layernorm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=BF16) + 3.0
    _close(layernorm(x), F.layer_norm(x.float(), (n,), eps=1e-5))


@pytest.mark.parametrize("m,n", SHAPES)
def test_quantize(m, n):
    x = torch.randn(m, n, device="cuda", dtype=BF16) * 3
    q, s = quantize(x)
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    want = (xf * (127.0 / amax)).round().clamp(-128, 127)
    assert q.dtype == torch.int8 and s.shape == (m,)
    assert (q.float() - want).abs().max().item() <= 1
    torch.testing.assert_close(s, (amax / 127).squeeze(-1), rtol=1e-6, atol=0)


@T.rowwise(outputs=T.per_row(hk.fp32))
def cross_entropy_lse(logits, label):
    loss = T.row_logsumexp(logits) - T.pick(logits, label)
    return loss * (label != -100).to_int()


@T.rowwise
def softmax_lse(x):
    return (x - T.row_logsumexp(x)).exp()


@pytest.mark.parametrize("m,n", [(5, 1000), (64, 4096), (3, 151936), (2, 70001)])
def test_softmax_from_the_online_logsumexp(m, n):
    x = torch.randn(m, n, device="cuda", dtype=BF16) * 3
    _close(softmax_lse(x), torch.softmax(x.float(), -1), rtol=2e-2, atol=1e-5)


@pytest.mark.parametrize("m,v", [(16, 4096), (7, 1000), (5, 151936), (3, 128256), (8, 32000)])
@pytest.mark.parametrize("labels_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("form", ["max_sum", "lse"])
def test_cross_entropy(m, v, labels_dtype, form):
    k = cross_entropy if form == "max_sum" else cross_entropy_lse
    logits = (2 * torch.randn(m, v, device="cuda")).to(BF16)
    labels = torch.randint(0, v, (m,), device="cuda", dtype=labels_dtype)
    labels[1] = -100
    labels[0] = v - 1
    got = k(logits, labels)
    want = F.cross_entropy(logits.float(), labels.long(), reduction="none")
    torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-3)
    assert got[1].item() == 0.0


def test_a_long_row_writes_full_outputs_in_its_last_pass():
    x = torch.randn(2, 151936, device="cuda", dtype=BF16)
    assert softmax.plan(x).steps > 0
    y = softmax(x)
    _close(y, torch.softmax(x.float(), -1), atol=1e-6)
    _close(y.float().sum(-1), torch.ones(2, device="cuda"), atol=1e-2)
