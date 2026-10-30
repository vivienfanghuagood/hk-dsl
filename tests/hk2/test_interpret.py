"""The interpreter runs the same tile IR on the CPU: it must agree with torch and with the compiled kernels."""
import torch

import hk2 as hk
from hk2.kernels import attention as attn
from hk2.kernels import rowwise

from test_basic import causal_fill, cond_scale, matmul, softmax, vadd


def test_vadd():
    x, y = torch.randn(3000, device="cuda"), torch.randn(3000, device="cuda")
    out = torch.empty_like(x)
    vadd[(3,)](x, y, out, interpret=True)
    torch.testing.assert_close(out, x + y)


def test_matmul_vs_compiled():
    a = torch.randn(100, 96, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(72, 96, device="cuda", dtype=torch.bfloat16)
    c1 = torch.empty(100, 72, device="cuda", dtype=torch.bfloat16)
    c2 = torch.empty_like(c1)
    matmul[(2, 2)](a, b, c1, TRANS_B=True, interpret=True)
    matmul[(2, 2)](a, b, c2, TRANS_B=True)
    torch.testing.assert_close(c1.float(), (a.float() @ b.float().T), atol=0.1, rtol=2e-2)
    torch.testing.assert_close(c1, c2, atol=0.1, rtol=2e-2)


def test_softmax_where_if():
    x = torch.randn(37, 1000, device="cuda")
    out = torch.empty_like(x)
    softmax[(10,)](x, out, interpret=True)
    torch.testing.assert_close(out, torch.softmax(x, -1), atol=1e-6, rtol=1e-5)
    m = torch.empty(96, 96, device="cuda")
    causal_fill[(3, 3)](m, interpret=True)
    torch.testing.assert_close(m, torch.ones(96, 96, device="cuda").tril())
    o = torch.empty(1000, device="cuda")
    cond_scale[(4,)](x.view(-1)[:1000].contiguous(), o, 1, interpret=True)
    torch.testing.assert_close(o, x.view(-1)[:1000] * 2)


def test_attention_vs_compiled():
    q, k, v = (torch.randn(1, 2, 80, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o1 = torch.empty_like(q)
    attn.attn_fwd[(1, 2, 1)](q, k, v, o1, 0.125, BM=128, BN=64, D=64, CAUSAL=True, GROUP=1, interpret=True)
    o2 = attn.attention(q, k, v, causal=True, sm_scale=0.125)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True,
                                                           scale=0.125)
    torch.testing.assert_close(o1.float(), ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(o1.float(), o2.float(), atol=2e-2, rtol=2e-2)


def test_print(capsys):
    @hk.jit
    def show(x, BLOCK: hk.constexpr = 4):
        t = hk.load(x, 0, BLOCK)
        hk.print("t =", t)
        hk.store(x, 0, t + 1)

    x = torch.arange(4, device="cuda", dtype=torch.float32)
    show[(1,)](x, interpret=True)
    assert "t =" in capsys.readouterr().out
    torch.testing.assert_close(x, torch.arange(1, 5, device="cuda", dtype=torch.float32))
    show[(1,)](x)  # compiled: print is ignored
    torch.testing.assert_close(x, torch.arange(2, 6, device="cuda", dtype=torch.float32))
