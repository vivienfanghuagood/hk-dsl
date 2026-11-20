"""Numerics for the training attention (hk.ops.attn_bwd.attention): o, dq, dk and dv against
float64 autograd.

Needs a gfx1201 or gfx1100 Radeon (attn_bwd.ARCHES) and torch. The reference is softmax(q k^T) v with
autograd in float64 on the same bf16 inputs and the same bf16 upstream gradient, GQA by repeating
the KV heads (so dk and dv come out summed over each group, as the kernels sum them). Shapes are
small -- the head count and batch are only grid dimensions -- and the lengths divide nothing
(300, 517, 1000) as well as the round ones, so the last tile of every kernel, which is moved back
to end on the last row, is checked.

The bound is relative L2 (|got - want| / |want| over the whole tensor): bf16 operands rounded
twice and summed over S terms land near 3e-3 on these shapes, which is what bf16 SDPA itself
measures against the same reference; 1e-2 leaves room and still fails a mask off by one row.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import attn_bwd  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = pytest.mark.skipif(current_arch() not in attn_bwd.ARCHES,
                                reason="the attention backward is built for gfx1201 and gfx1100")

TOL = 1e-2


def _ref(q, k, v, do, causal, scale):
    """o, dq, dk, dv in float64 autograd; (B, S, H, D) in, GQA by repeating KV heads."""
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    g = q.shape[2] // k.shape[2]
    kk = k64.repeat_interleave(g, dim=2).transpose(1, 2)
    vv = v64.repeat_interleave(g, dim=2).transpose(1, 2)
    s = (q64.transpose(1, 2) @ kk.transpose(-1, -2)) * scale
    if causal:
        n = s.shape[-1]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=s.device).triu(1),
                          float("-inf"))
    o = (s.softmax(-1) @ vv).transpose(1, 2)
    o.backward(do.double())
    return o.detach(), q64.grad, k64.grad, v64.grad


def _rel(got, want):
    want = want.double()
    assert want.abs().max().item() > 1e-6, "reference has no magnitude"
    got = got.double()
    assert torch.isfinite(got).all().item(), "non-finite output"
    return ((got - want).norm() / want.norm()).item()


def _check(q, k, v, causal, scale=None):
    d = q.shape[-1]
    sc = d ** -0.5 if scale is None else scale
    g = torch.Generator(device="cuda").manual_seed(1)
    do = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    qr, kr, vr = (t.detach().requires_grad_(True) for t in (q, k, v))
    o = attn_bwd.attention(qr, kr, vr, causal=causal, scale=scale)
    o.backward(do)
    want = _ref(q, k, v, do, causal, sc)
    errs = {n: _rel(a, b) for n, a, b in zip(("o", "dq", "dk", "dv"),
                                             (o, qr.grad, kr.grad, vr.grad), want)}
    bad = {n: e for n, e in errs.items() if not e < TOL}
    assert not bad, f"rel-L2 above {TOL}: {bad} (all: {errs})"
    return errs


def _bshd(b, s, h, d, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16, generator=g)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("b,s,h,h_kv", [
    (1, 192, 2, 2),    # one tile, exactly
    (2, 300, 4, 2),    # GQA 2, ragged
    (1, 517, 4, 1),    # GQA 4 (MQA), ragged
    (1, 1000, 2, 2),
])
def test_attention_bwd(b, s, h, h_kv, d, causal):
    q, k, v = _bshd(b, s, h, d, 2), _bshd(b, s, h_kv, d, 3), _bshd(b, s, h_kv, d, 4)
    _check(q, k, v, causal)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_scale(causal):
    q, k, v = (_bshd(1, 256, 2, 128, i) for i in (5, 6, 7))
    _check(q, k, v, causal, scale=0.2)


def test_fused_qkv_views():
    """q, k and v as slices of one (B, S, H + 2 H_kv, D) projection output: read in place."""
    b, s, h, h_kv, d = 2, 384, 4, 2, 128
    qkv = _bshd(b, s, h + 2 * h_kv, d, 8)
    q, k, v = qkv[:, :, :h], qkv[:, :, h:h + h_kv], qkv[:, :, h + h_kv:]
    assert not q.is_contiguous()
    _check(q, k, v, True)


def test_sbhd():
    """Megatron's (S, B, H, D) permuted to (B, S, H, D): copied once, same answer."""
    s, b, h, d = 320, 2, 2, 64
    q, k, v = (_bshd(s, b, h, d, i).permute(1, 0, 2, 3) for i in (9, 10, 11))
    _check(q, k, v, True)


def test_sbhd_one_batch_in_place():
    """Megatron's micro-batch of one: SBHD q, k, v (and o's gradient) are read where they are."""
    s, h, d = 384, 4, 128
    qkv = _bshd(s, 1, 3 * h, d, 12)                    # (S, B=1, 3H, D), one projection output
    q, k, v = (qkv[:, :, i * h:(i + 1) * h].permute(1, 0, 2, 3) for i in range(3))
    for t in (q, k, v):
        assert attn_bwd._bnhd(t).data_ptr() == t.data_ptr()
    _check(q, k, v, True)


@pytest.mark.parametrize("h,h_kv", [(4, 4), (4, 2)])
def test_sbhd_batches_folded(h, h_kv):
    """Megatron's SBHD at B > 1: read as one batch of B * H heads, no copy of q / k, and o comes
    back in SBHD memory; GQA's head map holds across the fold."""
    s, b, d = 320, 3, 128
    q = _bshd(s, b, h, d, 13).permute(1, 0, 2, 3)
    k = _bshd(s, b, h_kv, d, 14).permute(1, 0, 2, 3)
    v = _bshd(s, b, h_kv, d, 15).permute(1, 0, 2, 3)
    assert attn_bwd._is_sbhd(q) and attn_bwd._fold(q).data_ptr() == q.data_ptr()
    o = attn_bwd.attention(q, k, v, causal=True)
    assert o.transpose(0, 1).is_contiguous()
    _check(q, k, v, True)


def test_rejects():
    q = _bshd(1, 256, 4, 128, 0)
    assert "head_dim" in attn_bwd.why_unsupported(q[..., :96], q[..., :96], q[..., :96])
    assert "GQA" in attn_bwd.why_unsupported(q, q[:, :, :3], q[:, :, :3])
    assert "below" in attn_bwd.why_unsupported(q[:, :128], q[:, :128], q[:, :128])
    assert "bf16" in attn_bwd.why_unsupported(q.half(), q.half(), q.half())
    with pytest.raises(ValueError):
        attn_bwd.attention(q.float(), q.float(), q.float())
