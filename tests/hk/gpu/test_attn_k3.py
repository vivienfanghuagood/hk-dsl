"""hk.ops.attn_k3, gfx1100's DiT attention (K3), against fp32 math SDPA: ragged and cross lengths,
both layouts ((B, H, N, D) and (B, H, N, D) views of (B, N, H, D) memory), a value tensor at a
storage offset; deterministic, exactly 1.0 for V = 1, no NaN. And hk.ops.sdpa's HK_SDPA_K3 route."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.target import current_arch  # noqa: E402

if current_arch() != "gfx1100":
    pytest.skip("hk.ops.attn_k3 is gfx1100's", allow_module_level=True)

pytestmark = pytest.mark.gpu

from hk.ops import attn_k3 as K3  # noqa: E402

import _attn_ref  # noqa: E402

# (B, H, N_q, N_kv): whole and ragged tiles, cross attention, N_kv below one block
CASES = [(1, 2, 512, 512), (2, 3, 1485, 1485), (1, 4, 640, 1001), (1, 2, 128, 33),
         (2, 2, 4096, 4096)]


def _make(b, h, nq, nkv, bnhd, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def t(n):
        if bnhd:
            return torch.randn((b, n, h, 128), device="cuda", dtype=torch.bfloat16,
                               generator=g).transpose(1, 2)
        return torch.randn((b, h, n, 128), device="cuda", dtype=torch.bfloat16, generator=g)
    return t(nq), t(nkv), t(nkv)


def _check(q, k, v, bnhd):
    assert K3.why(q, k, v) == "", K3.why(q, k, v)
    o = K3.attention(q, k, v).clone()
    _attn_ref.check(o, _attn_ref.attention64(q, k, v))
    assert not torch.isnan(o).any()
    assert all(torch.equal(o, K3.attention(q, k, v)) for _ in range(3))
    assert (K3.attention(q, k, torch.ones_like(v)) == 1).all()       # no rounding bias
    assert o.transpose(1, 2).is_contiguous() if bnhd else o.is_contiguous()


@pytest.mark.parametrize("bnhd", [False, True], ids=["bhnd", "bnhd"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "x".join(map(str, c)))
def test_k3_matches_fp32_sdpa(case, bnhd):
    _check(*_make(*case, bnhd), bnhd)


@pytest.mark.parametrize("case", [(1, 3, 1001, 1001), (2, 2, 777, 777)],
                         ids=lambda c: "x".join(map(str, c)))
def test_k3_reads_a_value_tensor_at_a_storage_offset(case):
    q, k, v = _make(*case, False)
    v = torch.stack([torch.zeros_like(v), v])[1]
    assert v.storage_offset() > 0
    _check(q, k, v, False)


def test_sdpa_routes_non_causal_d128_to_k3_when_enabled(monkeypatch):
    import hk.ops.sdpa as S

    monkeypatch.setattr(S, "_K3", True)
    q, k, v = _make(1, 2, 640, 640, True)
    before = S.STATS["k3"]
    o = S.scaled_dot_product_attention(q, k, v)
    assert S.STATS["k3"] == before + 1
    assert torch.equal(o, K3.attention(q, k, v))
    S.scaled_dot_product_attention(q, k, v, is_causal=True)            # causal: hk.ops.attn's
    assert S.STATS["k3"] == before + 1
