"""The softmax top-k that stands in for sgl_kernel's on gfx1100: the routing
SGLang's own torch path computes, written into the caller's tensors."""

import pytest

torch = pytest.importorskip("torch")

from hk.integration import sglang_moe  # noqa: E402


def reference(logits, k, renormalize, bias=None, cap=0.0):
    x = logits.double()
    if cap:
        x = cap * torch.tanh(x / cap)
    p = torch.softmax(x, -1)
    ids = (p if bias is None else p + bias.double()).topk(k, -1).indices
    w = p.gather(-1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w, ids


@pytest.mark.parametrize("tokens,experts,k", [(1, 64, 8), (37, 64, 8), (256, 128, 8),
                                              (5, 8, 2)])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("extra", [{}, {"cap": 30.0}, {"bias": True}])
def test_it_matches_softmax_then_topk(tokens, experts, k, renormalize, extra):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(tokens * experts)
    # fp32, not the router's bf16: bf16 logits tie often enough that which of
    # two equal experts torch.topk keeps would decide the test.
    logits = torch.randn(tokens, experts, device=dev, generator=g)
    bias = (torch.randn(experts, device=dev, generator=g) * 0.1
            if extra.get("bias") else None)
    w = torch.full((tokens, k), -1.0, device=dev)
    ids = torch.full((tokens, k), -1, device=dev, dtype=torch.int32)

    sglang_moe.topk_softmax(w, ids, logits, renormalize,
                            moe_softcapping=extra.get("cap", 0.0), correction_bias=bias)

    ref_w, ref_ids = reference(logits, k, renormalize, bias, extra.get("cap", 0.0))
    order, ref_order = ids.long().sort(-1), ref_ids.sort(-1)
    assert torch.equal(order.values, ref_order.values)
    assert torch.allclose(w.double().gather(-1, order.indices),
                          ref_w.gather(-1, ref_order.indices), atol=1e-6)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
