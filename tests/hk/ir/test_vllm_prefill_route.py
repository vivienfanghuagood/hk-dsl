"""hk.integration.vllm_prefill's host-side checks, without vLLM or a GPU: which TRITON_ATTN
caches the route reads (and in which order), which pages it takes, and that an argument it
does not know refuses the call."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from hk.integration import vllm_prefill as vp  # noqa: E402

NB, PAGE, HKV, D = 5, 544, 2, 64


def _halves(full, d):
    return full[..., :d], full[..., d:]


def test_planes_the_older_triton_attn_cache():
    kv = torch.zeros(NB, 2, PAGE, HKV, D)
    k, v = kv.unbind(1)
    got, order = vp._flash_kv(k, v)
    assert order == "planes" and tuple(got.shape) == (NB, 2, PAGE, HKV * D)
    assert got.data_ptr() == kv.data_ptr()


def test_tokens_a_heads_k_and_v_side_by_side_token_major():
    """vLLM's per-layer (B, H, N, 2 * head_size) cache laid out [B, N, H, C], as TRITON_ATTN's
    forward splits it: transpose(1, 2), then split(head_size, -1)."""
    phys = torch.zeros(NB, PAGE, HKV, 2 * D)
    logical = phys.permute(0, 2, 1, 3)                       # (B, H, N, 2d)
    k, v = _halves(logical.transpose(1, 2), D)
    got, order = vp._flash_kv(k, v)
    assert order == "tokens" and tuple(got.shape) == (NB, 1, PAGE, HKV * 2 * D)
    phys[3, 7, 1, D + 5] = 1.0                                # V, kv head 1, token 7, col 5
    assert got[3, 0, 7, 1 * 2 * D + D + 5] == 1.0


def test_heads_the_same_head_major():
    phys = torch.zeros(NB, HKV, PAGE, 2 * D)                  # [B, H, N, C]
    k, v = _halves(phys.transpose(1, 2), D)
    got, order = vp._flash_kv(k, v)
    assert order == "heads" and tuple(got.shape) == (NB, HKV, PAGE, 2 * D)
    phys[2, 1, 9, 3] = 1.0                                    # K, kv head 1, token 9, col 3
    assert got[2, 1, 9, 3] == 1.0


def test_one_kv_head_is_read_as_tokens():
    """MiniCPM-V-4.7 at TP2: one kv head a rank, where token- and head-major are one layout."""
    phys = torch.zeros(NB, 1, PAGE, 2 * 256)
    k, v = _halves(phys.transpose(1, 2), 256)
    assert vp._flash_kv(k, v)[1] == "tokens"


def test_a_padded_block_is_refused():
    phys = torch.zeros(NB, PAGE * HKV * 2 * D + 64)[:, : PAGE * HKV * 2 * D]
    phys = phys.view(NB, PAGE, HKV, 2 * D)
    k, v = _halves(phys, D)
    assert vp._flash_kv(k, v) is None


@pytest.mark.parametrize("page,ok", [(16, True), (32, True), (48, True), (64, True), (544, True),
                                     (784, True), (400, True), (8, False), (24, False)])
def test_pages_hk_takes(page, ok):
    why = vp.page_why_not(page)
    assert (why == "") == ok
    if not ok:
        assert f"page {page}" in why and "a multiple of 16 tokens" in why


def test_an_argument_hk_does_not_know_refuses_the_call(monkeypatch):
    def unified_attention(q, k, v, out, causal=True, new_feature=None, use_td=False):
        pass

    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    assert vp._unknown_set({"q": 1, "new_feature": None, "use_td": True}) == ""
    assert vp._unknown_set({"q": 1, "new_feature": torch.zeros(2)}) == "new_feature"
    assert vp._unknown_set({"q": 1, "never_heard_of": 3}) == "never_heard_of"
