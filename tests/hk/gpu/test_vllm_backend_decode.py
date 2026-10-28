"""The decoder backend's two vLLM-side pieces, against torch: the cache write
into its head-major pages, and the small-batch decode it hands to vLLM's
Triton kernel on those pages (vllm_backend.TRITON_DECODE_MAX).

Needs vLLM; skipped without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from hk.integration import vllm_backend  # noqa: E402
from hk.ops import paged  # noqa: E402

import _attn_ref  # noqa: E402

pytestmark = pytest.mark.gpu

PAGE = paged.KV_BLOCK


def test_the_cache_write_lands_head_major_and_skips_padding():
    """vLLM's reshape_and_cache_flash through the transposed view, bit for
    bit against a torch scatter; slot -1 is a CUDA graph's padding token and
    must leave the cache untouched."""
    from vllm import _custom_ops as ops

    # Newer vLLM registers its cache ops from _C_stable_libtorch, which its
    # ROCm platform module imports -- an engine has; a bare _custom_ops does not.
    import contextlib
    import importlib
    with contextlib.suppress(ImportError):
        importlib.import_module("vllm._C_stable_libtorch")

    g = torch.Generator(device="cuda").manual_seed(1)
    nb, hkv, d, n = 64, 8, 128, 300
    key = torch.randn(n, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
    value = torch.randn(key.shape, device=key.device, dtype=key.dtype, generator=g)
    slots = torch.randperm(nb * PAGE, device="cuda", generator=g)[:n]
    slots[::7] = -1
    want = torch.full((2, nb, hkv, PAGE, d), 7.0, device="cuda",
                      dtype=torch.bfloat16)
    live = slots >= 0
    s = slots[live]
    want[0][s // PAGE, :, s % PAGE] = key[live]
    want[1][s // PAGE, :, s % PAGE] = value[live]

    got = torch.full_like(want, 7.0)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    ops.reshape_and_cache_flash(key, value, got[0].transpose(1, 2),
                                got[1].transpose(1, 2), slots, "auto", one, one)
    assert torch.equal(got, want)


def _case(seq_lens, hkv=8, group=4, d=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(seq_lens)
    max_pages = -(-max(seq_lens) // PAGE)
    nb = r * max_pages + 3
    k = torch.randn(nb, hkv, PAGE, d, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    v = torch.randn(k.shape, device=k.device, dtype=k.dtype, generator=g)
    perm = torch.randperm(nb, generator=g, device="cuda")[: r * max_pages]
    table = perm.reshape(r, max_pages).to(torch.int32).contiguous()
    q = torch.randn(r, hkv * group, d, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return q, k, v, table, lens


def _reference(q, k, v, table, lens, group):
    """Exact decode attention in float64 (_attn_ref)."""
    r, h, d = q.shape
    out = torch.zeros(r, h, d, device="cuda", dtype=torch.float32)
    for i in range(r):
        n = int(lens[i])
        pages = table[i, : -(-n // PAGE)].tolist()
        for kh in range(k.shape[1]):
            kk = torch.cat([k[p, kh] for p in pages])[:n]
            vv = torch.cat([v[p, kh] for p in pages])[:n]
            heads = slice(kh * group, (kh + 1) * group)
            out[i, heads] = _attn_ref.softmax_v64(q[i, heads], kk, vv, d ** -0.5)
    return out


@pytest.mark.parametrize("seq_lens", [[1], [37], [1024], [5, 700, 33, 2048],
                                      [4096] * 8])
def test_the_small_batch_decode_on_hk_pages(seq_lens):
    """Triton's kernel reading hk's head-major cache gives the attention,
    for one request to TRITON_DECODE_MAX, ragged and short of a page."""
    q, k, v, table, lens = _case(seq_lens)
    r, h, d = q.shape
    out = torch.empty_like(q)
    qsl = torch.arange(r + 1, device="cuda", dtype=torch.int32)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    assert vllm_backend.triton_decode(q, k, v, table, lens, qsl, out, d ** -0.5,
                                      max(seq_lens), one, one)
    want = _reference(q, k, v, table, lens, h // k.shape[1])
    _attn_ref.check(out, want)
