"""The templates pages' examples run as written and give torch's answer: each block is taken from
docs/templates/*.md, given the tensors its text names, and checked against a torch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _doc_blocks  # noqa: E402

pytestmark = pytest.mark.gpu

DOCS = Path(__file__).resolve().parents[3] / "docs" / "templates"
ROWWISE = DOCS / "elementwise-rowwise.md"
GEMM = DOCS / "gemm.md"
ROWS, COLS = 64, 2048
TOL = dict(rtol=2e-2, atol=2e-2)


def bf16(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def rms(h, w, eps):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w


def test_first_block_gelu_and_mul_and_rms_norm():
    d = COLS
    gate_up, x, weight = bf16(ROWS, 2 * d), bf16(ROWS, COLS), bf16(COLS)
    ns = _doc_blocks.run(ROWWISE, "gelu_and_mul = T.elementwise",
                         {"gate_up": gate_up, "d": d, "x": x, "weight": weight})
    torch.testing.assert_close(ns["y"].float(), rms(x.float(), weight.float(), 1e-6), **TOL)
    y = ns["gelu_and_mul"](gate_up[:, :d], gate_up[:, d:])
    g, u = gate_up[:, :d].float(), gate_up[:, d:].float()
    torch.testing.assert_close(y.float(), F.gelu(g, approximate="tanh") * u, **TOL)


def test_elementwise_examples():
    ns = _doc_blocks.run(ROWWISE, "silu_mul = T.elementwise", {"hk": __import__("hk"), "T": __import__("hk").templates})
    g, u = bf16(ROWS, COLS), bf16(ROWS, COLS)
    torch.testing.assert_close(ns["silu_mul"](g, u).float(), F.silu(g.float()) * u.float(), **TOL)
    x, b = bf16(ROWS, COLS), bf16(COLS)
    torch.testing.assert_close(ns["bias_gelu"](x, b).float(), F.gelu(x.float() + b.float(), approximate="tanh"), **TOL)
    s = bf16(ROWS, 1)
    torch.testing.assert_close(ns["scaled"](x, s, A=0.5).float(), x.float() * s.float() * 0.5, **TOL)


@pytest.fixture(scope="module")
def rowwise():
    import hk  # noqa: PLC0415

    x, r, w = bf16(ROWS, COLS), bf16(ROWS, COLS), bf16(COLS)
    x0, r0 = x.clone(), r.clone()
    ns = _doc_blocks.run(ROWWISE, "def add_rms_norm", {"hk": hk, "T": hk.templates, "x": x, "r": r, "w": w})
    ns.update(x0=x0, r0=r0, w=w)
    return ns


def test_add_rms_norm_in_place(rowwise):
    h = rowwise["x0"].float() + rowwise["r0"].float()
    torch.testing.assert_close(rowwise["r"].float(), h, **TOL)
    torch.testing.assert_close(rowwise["y"].float(), rms(h, rowwise["w"].float(), 1e-6), **TOL)
    assert rowwise["y"].data_ptr() == rowwise["x"].data_ptr()      # out=(x, r): in place


def test_softmax_layernorm(rowwise):
    x = bf16(ROWS, COLS)
    torch.testing.assert_close(rowwise["softmax"](x).float(), torch.softmax(x.float(), -1), rtol=2e-2, atol=1e-4)
    xc = x.float() - x.float().mean(-1, keepdim=True)
    torch.testing.assert_close(rowwise["layernorm"](x).float(),
                               xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + 1e-5), **TOL)


def test_quantize(rowwise):
    x = bf16(ROWS, COLS)
    q, scale = rowwise["quantize"](x)
    amax = x.float().abs().amax(-1).clamp_min(1e-12)
    torch.testing.assert_close(scale.float().reshape(-1), amax / 127.0, rtol=1e-3, atol=1e-8)
    deq = q.float() * scale.float().reshape(-1, 1)
    assert (deq - x.float()).abs().max() <= (amax / 127.0).max() * 1.01   # within one step


def test_add_rms_quant(rowwise):
    x, r, w = bf16(ROWS, COLS), bf16(ROWS, COLS), bf16(COLS)
    q, scale, h = rowwise["add_rms_quant"](x, r, w)
    href = x.float() + r.float()
    torch.testing.assert_close(h.float(), href, **TOL)
    y = rms(href, w.float(), 1e-6)
    deq = q.float() * scale.float().reshape(-1, 1)
    assert (deq - y).abs().max() <= y.abs().amax() / 127.0 * 1.05


def test_cross_entropy(rowwise):
    logits = bf16(ROWS, COLS)
    label = torch.randint(0, COLS, (ROWS,), device="cuda")
    label[::7] = -100
    loss = rowwise["cross_entropy"](logits, label)
    ref = F.cross_entropy(logits.float(), label, ignore_index=-100, reduction="none")
    torch.testing.assert_close(loss.float().reshape(-1), ref, rtol=1e-2, atol=1e-2)


def test_router(rowwise):
    E, k = 64, 4
    logits = torch.randn(ROWS, E, device="cuda", dtype=torch.float32).bfloat16()
    col = torch.arange(E, device="cuda", dtype=torch.float32)
    ids, ws = rowwise["router"](k)(logits, col)
    p = torch.softmax(logits.float(), -1)
    top = p.topk(k, -1).values
    ids = ids.reshape(ROWS, k).long()
    # bf16 logits tie now and then; the page's router takes the first column of a tie, topk any one
    torch.testing.assert_close(p.gather(-1, ids), top, rtol=0, atol=0)
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()             # k distinct experts
    torch.testing.assert_close(ws.reshape(ROWS, k).float(), top / top.sum(-1, keepdim=True), rtol=1e-3, atol=1e-4)


def test_kernel_source_and_build():
    import hk  # noqa: PLC0415

    ns = _doc_blocks.run(ROWWISE, "gelu_and_mul = T.elementwise", {"gate_up": bf16(8, 128), "d": 64,
                                                                  "x": bf16(8, 256), "weight": bf16(256)})
    _doc_blocks.run(ROWWISE, "kern, consts = gelu_and_mul.kernel", ns)
    assert ns["kern"] is not None and ns["consts"] is not None


def test_gemm_bias_gelu():
    import hk  # noqa: PLC0415

    x, w, b = bf16(128, 512), bf16(256, 512), bf16(256)
    ns = _doc_blocks.run(GEMM, "def bias_gelu(acc, ctx)", {"x": x, "w": w, "b": b})
    ref = F.gelu(F.linear(x.float(), w.float(), b.float()), approximate="tanh")
    torch.testing.assert_close(ns["y"].float(), ref, rtol=2e-2, atol=1e-1)
