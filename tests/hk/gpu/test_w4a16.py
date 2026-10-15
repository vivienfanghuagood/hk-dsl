"""4-bit weights: the repacks against the checkpoint formats' own meaning, and
the decode kernel against the dequantized product."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import w4a16  # noqa: E402


def _awq_checkpoint(n, k, group, g):
    """A random AutoAWQ checkpoint and the W it means."""
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(0, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()

    def pack(v):          # [R, C] -> [R, C/8], column c in nibble AWQ_ORDER[c]
        r, c = v.shape
        v = v.reshape(r, c // 8, 8).to(torch.int64)
        shift = torch.tensor([4 * o for o in w4a16.AWQ_ORDER], device=v.device)
        words = (v << shift).sum(-1)
        return torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)

    w = ((q - z.repeat_interleave(group, 0)).float()
         * s.float().repeat_interleave(group, 0)).t()      # [N, K]
    return pack(q), pack(z), s, w


def test_awq_repack_means_what_the_checkpoint_means():
    g = torch.Generator(device="cuda").manual_seed(0)
    n, k, group = 256, 512, 128
    qweight, qzeros, scales, w = _awq_checkpoint(n, k, group, g)
    qw, s, zs = w4a16.repack_awq(qweight, qzeros, scales, group)
    assert qw.shape == (n, k // 8) and s.shape == (n, k // group)
    got = w4a16.dequant(qw, s, zs, group).float()
    assert (got - w).abs().max().item() <= 2e-2 * w.abs().max().item()


def test_gptq_repack_means_what_the_checkpoint_means():
    g = torch.Generator(device="cuda").manual_seed(1)
    n, k, group = 128, 256, 64
    q = torch.randint(0, 16, (k, n), device="cuda", generator=g, dtype=torch.int32)
    z = torch.randint(1, 16, (k // group, n), device="cuda", generator=g, dtype=torch.int32)
    s = (torch.rand(k // group, n, device="cuda", generator=g) * 0.02 + 0.005).half()
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (q.to(torch.int64).reshape(k // 8, 8, n) << shifts.view(1, 8, 1)).sum(1)
    qweight = torch.where(qweight >= 2 ** 31, qweight - 2 ** 32, qweight).to(torch.int32)
    zst = (z - 1).to(torch.int64).reshape(k // group, n // 8, 8)
    qzeros = (zst << shifts).sum(-1)
    qzeros = torch.where(qzeros >= 2 ** 31, qzeros - 2 ** 32, qzeros).to(torch.int32)
    qw, sc, zs = w4a16.repack_gptq(qweight, qzeros, s, group)
    w = ((q - z.repeat_interleave(group, 0)).float() * s.float().repeat_interleave(group, 0)).t()
    got = w4a16.dequant(qw, sc, zs, group).float()
    assert (got - w).abs().max().item() <= 2e-2 * w.abs().max().item()


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("m", [1, 7, 16, 33, 64])
def test_the_decode_kernel_matches_the_dequantized_product(m, dtype):
    g = torch.Generator(device="cuda").manual_seed(m)
    n, k, group = 512, 2048, 128
    td = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    qweight, qzeros, scales, w = _awq_checkpoint(n, k, group, g)
    qw, s, zs = w4a16.repack_awq(qweight, qzeros, scales, group, dtype=td)
    assert w4a16.takes(m, n, k, group)
    x = torch.randn(m, k, device="cuda", dtype=td, generator=g)
    got = w4a16.matmul(x, qw, s, zs, group).float()
    want = x.float() @ w4a16.dequant(qw, s, zs, group).float().t()
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 2e-2, err
