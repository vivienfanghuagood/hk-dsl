import pytest
import torch

from hk2.kernels.attention import attention


def ref(q, k, v, causal):
    g = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(g, 1).float()
    v = v.repeat_interleave(g, 1).float()
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k, v, is_causal=causal)


@pytest.mark.parametrize("B,H,Hkv,S,D", [(1, 2, 2, 128, 128), (2, 4, 2, 300, 64), (1, 2, 1, 1000, 128),
                                         (1, 1, 1, 64, 64)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_attention(B, H, Hkv, S, D, causal, dt):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=dt)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=dt)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=dt)
    out = attention(q, k, v, causal=causal)
    torch.testing.assert_close(out.float(), ref(q, k, v, causal), atol=2e-2, rtol=2e-2)
