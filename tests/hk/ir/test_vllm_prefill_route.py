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


def _caf_old(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
             max_seq_len, max_input_len, k_scale, v_scale, alibi_slopes=None,
             sliding_window=None, sm_scale=None, skip_decode=False, fp8_out_scale=None,
             sinks=None, is_block_table_ptr=False, causal=True):
    return "vllm"


def _caf_new(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
             max_seq_len, max_input_len, k_scale, v_scale, alibi_slopes=None,
             sliding_window=None, sm_scale=None, skip_decode=False, fp8_out_scale=None,
             sinks=None, is_block_table_ptr=False, interleaved_v_pack_factor=0, causal=True):
    return "vllm"


@pytest.mark.parametrize("orig", [_caf_old, _caf_new])
def test_the_rocm_attn_standin_takes_either_signature(monkeypatch, orig):
    """vLLM's context_attention_fwd before and after interleaved_v_pack_factor: the stand-in
    shows the signature it rebinds and binds a call to it by name; the parameter at its default
    is not a refusal, a packed V cache is (that call falls back, the engine's other calls and
    routes are untouched)."""
    monkeypatch.setattr(vp, "_P_SIG", inspect.signature(orig))
    monkeypatch.setattr(vp, "_ORIG", orig)
    monkeypatch.setattr(vp.context_attention_fwd, "__signature__", inspect.signature(orig),
                        raising=False)
    assert list(inspect.signature(vp._GUARDED).parameters) == list(
        inspect.signature(orig).parameters)
    assert vp._p_unknown({"q": 1, "causal": True}) == ""
    if "interleaved_v_pack_factor" in inspect.signature(orig).parameters:
        assert vp._p_unknown({"q": 1, "interleaved_v_pack_factor": 0}) == ""
        assert "interleaved_v_pack_factor=8" in vp._p_unknown({"interleaved_v_pack_factor": 8})
        # a refused call goes to vLLM's kernel with the caller's arguments, as they came
        args = dict(q=None, k=None, v=None, o=None, kv_cache_dtype="auto", k_cache=None,
                    v_cache=None, b_loc=None, b_start_loc=None, b_seq_len=None, max_seq_len=1,
                    max_input_len=1, k_scale=None, v_scale=None, interleaved_v_pack_factor=8)
        before = vp.PATCH.stats["fallback"]
        assert vp.context_attention_fwd(**args) == "vllm"
        assert vp.PATCH.stats["fallback"] == before + 1


def test_the_contract_checks_only_what_the_rocm_attn_standin_reads():
    from hk.integration import contract

    have = set(inspect.signature(_caf_new).parameters)
    assert vp._P_NEED <= have and vp._P_NEED <= set(inspect.signature(_caf_old).parameters)
    assert "same_parameters(fn, vllm_prefill.context_attention_fwd" not in inspect.getsource(
        contract._vllm_prefill)

