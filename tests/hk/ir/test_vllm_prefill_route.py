"""hk.integration.vllm_prefill's host-side checks, without vLLM or a GPU: which TRITON_ATTN
caches the route reads (and in which order), which pages it takes, and that an argument it
does not know refuses the call."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

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
    src = inspect.getsource(contract.judge_vllm_prefill)
    assert "same_parameters" not in src and "same_signature" not in src



# ---- the guard: batches hk is not faster on go to vLLM's kernel, unchanged --------------------

def _triton_call(reqs, d=128, hkv=2, group=8, page=16):
    """unified_attention's arguments for `reqs` = [(context, query length)], on CPU: what the
    route reads on the host, and caches in TRITON_ATTN's planes order."""
    n = sum(q for _, q in reqs)
    blocks = sum(-(-(c + q) // page) for c, q in reqs) + 1
    kv = torch.zeros(blocks, 2, page, hkv, d, dtype=torch.bfloat16)
    k, v = kv.unbind(1)
    starts = [0]
    for _, q in reqs:
        starts.append(starts[-1] + q)
    pages = max(-(-(c + q) // page) for c, q in reqs)
    return dict(q=torch.zeros(n, hkv * group, d, dtype=torch.bfloat16), k=k, v=v,
                out=torch.zeros(n, hkv * group, d, dtype=torch.bfloat16),
                cu_seqlens_q=torch.tensor(starts, dtype=torch.int32),
                max_seqlen_q=max(q for _, q in reqs),
                seqused_k=torch.tensor([c + q for c, q in reqs], dtype=torch.int32),
                max_seqlen_k=max(c + q for c, q in reqs), softmax_scale=d ** -0.5, causal=True,
                window_size=(-1, -1),
                block_table=torch.zeros(len(reqs), pages, dtype=torch.int32), softcap=0,
                q_descale=None, k_descale=None, v_descale=None)


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_a_declined_batch_is_vllms_call_unchanged(monkeypatch, capsys, arch):
    """A batch the guard declines reaches vLLM's unified_attention with the caller's own
    arguments -- the same tensors, nothing launched by hk before it -- so its output is stock's;
    counted as "declined", and said once a process."""
    from hk.integration import switches

    seen = []

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, alibi_slopes=None, sinks=None):
        seen.append(dict(q=q, k=k, v=v, out=out, block_table=block_table))
        out.fill_(3.0)

    monkeypatch.setattr(switches, "arch", lambda: arch)
    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    monkeypatch.setattr(vp, "_DECLINED_SAID", [])
    monkeypatch.setattr(vp, "_flash_op", lambda *a, **k: pytest.fail("hk's kernel was built"))
    a = _triton_call([(0, 128), *[(16384, 1)] * 8])        # a small chunk beside 8 long decodes
    assert vp.u_why_not(a) == ""
    before = vp.PATCH.stats["declined"]
    vp.unified_attention(**a)
    assert vp.PATCH.stats["declined"] == before + 1
    assert len(seen) == 1 and all(seen[0][n] is a[n] for n in ("q", "k", "v", "out", "block_table"))
    assert torch.all(a["out"] == 3.0)
    vp.unified_attention(**a)
    err = capsys.readouterr().err
    assert err.count("[hk] prefill: a batch went to vLLM's kernel") == 1
    assert "declined" in err and "HK_VLLM_PREFILL_GUARD=0" in err


def test_the_guard_off_declines_nothing(monkeypatch):
    monkeypatch.setattr(vp, "GUARD", False)
    assert vp.guard_why_not(9, 128, 16385, 128, 2, "gfx1100") == ""
    assert vp.guard_why_not(1, 16, 32768, 256, 1, "gfx1201") == ""


def test_an_unmeasured_arch_or_an_unknown_context_is_declined():
    assert "no measured guard" in vp.guard_why_not(1, 4096, 4096, 128, 2, "gfx1151")
    assert vp.guard_why_not(1, 4096, 0, 128, 2, "gfx1201") == "no context length given"


#: The guard's evidence: tools/hk-bench/attention_bench.py prefill, every shape against the faster
#: of TRITON_ATTN's and ROCM_ATTN's kernels, one process, cold KV (W7900D gfx1100, R9700 gfx1201).
SWEEP = Path(__file__).resolve().parents[3] / "tools" / "hk-bench" / "results" / "2026-11-03"
#: the KV split's sweep (tools/hk-bench/results/<this>/<arch>/prefill-kvsplit-sweep.json)
KVSPLIT_SWEEP = "2026-11-04"
#: the bench's models: (head size, kv heads)
DIMS = {"minicpm5": (128, 2), "qwen3.8": (256, 4), "minicpm-v-tp2": (256, 1)}


def _reqs(shape):
    return [(int(c) + int(n), int(n)) for c, n in (x.split(":") for x in shape.split("+"))]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_every_measured_shape_the_guard_takes_is_a_clear_win(arch):
    """The default-on rule's per-shape clause, on the measurements the guard was set from: a
    batch the route takes was measured at least 1.05x the faster stock kernel; and the guard
    still takes the clear wins (long prompts, chunks over a prefix, a big chunk beside decodes)."""
    rows = json.loads((SWEEP / arch / "prefill-guard-sweep.json").read_text())
    taken = []
    for r in rows:
        d, hkv = DIMS[r["model"]]
        reqs = _reqs(r["shape"])
        why = vp.guard_why_not(len(reqs), max(q for _, q in reqs), max(s for s, _ in reqs), d,
                               hkv, arch)
        if not why:
            taken.append(r)
            assert r["hk_speedup"] >= 1.05, (arch, r["model"], r["shape"], r["hk_speedup"])
    assert len(taken) >= len(rows) // 4, f"the guard takes only {len(taken)} of {len(rows)}"
    for model in DIMS:
        assert any(r["model"] == model and r["shape"] == "0:2048+1024:1" for r in taken), model


# ---- exact sizes from vLLM's metadata builder (host arrays, no sync) ---------------------------

def _stats_of(reqs):
    """batch_stats for `reqs` = [(sequence length, query length)], as host arrays."""
    import numpy as np

    qsl = np.cumsum([0] + [q for _, q in reqs])
    return vp.batch_stats(qsl, np.array([s for s, _ in reqs]), len(reqs))


@pytest.mark.parametrize("order", ["decodes first", "decodes last", "interleaved"])
def test_batch_stats_separate_decodes_from_the_prefill(order):
    pre, dec = [(2048 + 512, 512)], [(768, 1)] * 4
    reqs = {"decodes first": dec + pre, "decodes last": pre + dec,
            "interleaved": dec[:2] + pre + dec[2:]}[order]
    st = _stats_of(reqs)
    assert st == dict(reqs=5, decodes=4, empty=0, dec_work=4 * 768,
                      pre_work=512 * 2048 + 512 * 513 / 2, max_q=512, pre_ctx=2560)


class _NoSync:
    """CommonAttentionMetadata's host side, and device fields that must not be read."""

    def __init__(self, reqs, cached):
        import numpy as np

        self.query_start_loc_cpu = np.cumsum([0] + [q for _, q in reqs])
        lens = np.array([s for s, _ in reqs])
        self._seq_lens_cpu = lens if cached else None
        self.seq_lens_cpu_upper_bound = lens
        self.num_reqs = len(reqs)

    @property
    def seq_lens(self):
        raise AssertionError("read the device's seq_lens")

    @property
    def seq_lens_cpu(self):
        raise AssertionError("read seq_lens_cpu (a device copy when nothing is cached)")


@pytest.mark.parametrize("cached", [True, False])
def test_the_hook_returns_vllms_metadata_and_reads_only_host_arrays(monkeypatch, cached):
    monkeypatch.setattr(vp, "_STATS", {})
    q_start = torch.zeros(6, dtype=torch.int32)

    class Md:
        query_start_loc = q_start

    md = Md()

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return md

    reqs = [(2560, 512)] + [(768, 1)] * 4
    got = vp._stats_hook(build)(object(), 0, _NoSync(reqs, cached))
    assert got is md
    assert vp._STATS[id(q_start)] == _stats_of(reqs)


def test_the_exact_sizes_take_a_prompt_beside_short_decodes_the_bound_declines(monkeypatch):
    """A 2048-token chunk beside 8 decodes over 768 tokens: the size bound sees the chunk's own
    context as the longest and declines; the exact sizes take it (hk measured 1.6-2.3x there).
    A 512-token prompt beside 8 decodes over 16k stays declined either way."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    for reqs, taken in (([(4096, 2048)] + [(768, 1)] * 8, True),
                        ([(512, 512)] + [(16384, 1)] * 8, False)):
        a = _triton_call([(s - q, q) for s, q in reqs])
        assert vp._guard(a) != "", "the size bound alone declines both"
        monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): _stats_of(reqs)})
        assert (vp._guard(a) == "") == taken


def test_without_matching_sizes_the_bound_decides(monkeypatch):
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    a = _triton_call([(2048, 2048)] + [(767, 1)] * 8)
    monkeypatch.setattr(vp, "_STATS", {id(object()): _stats_of([(4096, 2048)])})
    assert vp._guard(a) == vp.guard_why_not(9, 2048, 4096, 128, 2, "gfx1100") != ""
    st = _stats_of([(4096, 2048)] + [(768, 1)] * 7)          # a request short: another step's
    monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): st})
    assert vp._guard(a) != ""


@pytest.mark.parametrize("sig,ok", [
    ("(self, common_prefix_len, common_attn_metadata, fast_build=False)", True),
    ("(self, common_attn_metadata, fast_build=False)", False)])
def test_the_hook_goes_on_only_a_builder_it_knows(sig, ok):
    ns = {}
    exec(f"def build{sig}: pass", ns)
    cls = type("B", (), {"build": ns["build"]})
    why = vp.stats_why_not(cls)
    if ok:
        assert why == "" or "CommonAttentionMetadata" in why   # vLLM itself may be absent here
    else:
        assert "not (self, common_prefix_len" in why


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_every_measured_shape_taken_with_exact_sizes_is_a_clear_win(arch):
    """The same evidence through the exact decision: every measured batch it takes is >= 1.05x,
    and it takes the prompt-beside-short-decodes batches the size bound gave up."""
    rows = json.loads((SWEEP / arch / "prefill-guard-sweep.json").read_text())
    taken = 0
    short_dec = 0
    for r in rows:
        d, hkv = DIMS[r["model"]]
        reqs = _reqs(r["shape"])
        if all(q > 1 for _, q in reqs) or all(q == 1 for _, q in reqs):
            continue
        if not vp.exact_why_not(_stats_of(reqs), d, hkv, arch):
            taken += 1
            assert r["hk_speedup"] >= 1.05, (arch, r["model"], r["shape"], r["hk_speedup"])
            short_dec += all(s <= 1024 for s, q in reqs if q == 1)
    assert taken >= 10 and short_dec >= 3, (taken, short_dec)


def test_a_vllm_without_the_host_arrays_keeps_the_size_bound(monkeypatch):
    """A vLLM whose common metadata has no host sequence lengths (only the device's): the hook
    does not go on (said in STATS_WHY_NOT), build is left as it was, and the guard decides with
    #233's size-only bound."""
    import dataclasses
    import sys
    import types

    from hk.integration import switches

    @dataclasses.dataclass
    class CommonAttentionMetadata:
        query_start_loc: object
        query_start_loc_cpu: object
        seq_lens: object
        num_reqs: int

    class TritonAttentionMetadataBuilder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return None

    orig = TritonAttentionMetadataBuilder.build
    mods = {"vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
            "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
            "vllm.v1.attention.backends.utils": types.ModuleType("u"),
            "vllm.v1.attention.backends.triton_attn": types.ModuleType("t")}
    mods["vllm.v1.attention.backends.utils"].CommonAttentionMetadata = CommonAttentionMetadata
    mods["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder = \
        TritonAttentionMetadataBuilder
    mods["vllm.v1.attention.backends"].triton_attn = mods["vllm.v1.attention.backends.triton_attn"]
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(vp, "_ORIG_BUILD", None)
    monkeypatch.setattr(vp, "_STATS", {})
    vp._apply_stats()
    assert vp.STATS_WHY_NOT == "CommonAttentionMetadata has no host sequence lengths"
    assert TritonAttentionMetadataBuilder.build is orig and vp._ORIG_BUILD is None
    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    a = _triton_call([(2048, 2048)] + [(767, 1)] * 8)
    assert vp._guard(a) == vp.guard_why_not(9, 2048, 4096, 128, 2, "gfx1100") != ""


# ---- the split: a mixed batch's prefill rows to hk, its decode rows to vLLM's kernel ----------

def test_split_takes_a_mixed_batch_the_guard_declines_for_its_decodes(monkeypatch):
    """With the step's exact sizes, a prompt beside long decodes -- declined whole for the
    decodes -- is taken split; not without the sizes, not with HK_VLLM_PREFILL_SPLIT=0, not with
    a request of no queries in the batch, not for a prefill too small or decodes too heavy
    (SPLITS), and not when the prefill itself is one the guard declines (a short chunk over a
    long context)."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    reqs = [(2048, 2048)] + [(16384, 1)] * 8
    a = _triton_call([(s - q, q) for s, q in reqs])
    monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): _stats_of(reqs)})
    assert "decodes" in vp._guard(a)
    assert vp.split_takes(a)
    monkeypatch.setattr(vp, "SPLIT", False)
    assert not vp.split_takes(a)
    monkeypatch.setattr(vp, "SPLIT", True)
    monkeypatch.setattr(vp, "_STATS", {})
    assert not vp.split_takes(a)
    small = [(512, 512)] + [(16384, 1)] * 8
    b = _triton_call([(s - q, q) for s, q in small])
    monkeypatch.setattr(vp, "_STATS", {id(b["cu_seqlens_q"]): _stats_of(small)})
    assert not vp.split_takes(b)
    monkeypatch.setattr(vp, "SPLIT_ALL", True)                 # HK_VLLM_PREFILL_SPLIT=all
    assert vp.split_takes(b)
    monkeypatch.setattr(vp, "SPLIT_ALL", False)
    for bad in ([(16416, 32)] + [(16384, 1)] * 8,          # a 32-token chunk over 16k
                [(512, 512)] + [(16384, 1)] * 8,           # a prefill too small for the split
                [(2048, 2048)] + [(65536, 1)] * 8,         # decodes too heavy for it
                [(2048, 2048), (0, 0)] + [(16384, 1)] * 8):  # a request with no queries
        b = _triton_call([(s - q, q) for s, q in bad])
        monkeypatch.setattr(vp, "_STATS", {id(b["cu_seqlens_q"]): _stats_of(bad)})
        assert vp._guard(b) and not vp.split_takes(b)


def test_a_split_batch_gives_vllm_one_query_a_request_and_keeps_only_its_decode_rows(monkeypatch):
    """hk's kernel (here a stand-in writing 5 on the prefill rows, as the decodes=False kernel
    does) runs first; then vLLM's kernel once, on each request's first row as a one-query batch
    -- a prefill request a one-key row -- and only the decodes' answers are copied back. Counted
    as hk and as split."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    monkeypatch.setattr(vp, "VT", False)
    reqs = [(16384, 1), (2048, 2048), (16384, 1), (16384, 1)]
    a = _triton_call([(s - q, q) for s, q in reqs])
    starts = a["cu_seqlens_q"].tolist()
    pre_rows = list(range(starts[1], starts[2]))
    built = []

    def flash_op(d, group, scale, page, order, decodes):
        built.append(decodes)

        def op(q2, kv, v_src, table, used, cu, tiles, o2, *packed):
            assert not packed
            o2.view(a["out"].shape)[pre_rows] = 5.0
        return op, 16

    seen = []

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, alibi_slopes=None, sinks=None):
        seen.append(dict(q=q.clone(), cu=cu_seqlens_q.tolist(), mq=max_seqlen_q,
                         used=seqused_k.tolist(), table=block_table))
        out.copy_(torch.arange(out.shape[0], dtype=out.dtype)[:, None, None] + 100)

    monkeypatch.setattr(vp, "_flash_op", flash_op)
    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): _stats_of(reqs)})
    rows = torch.arange(a["q"].shape[0])
    a["q"][:, 0, 0], a["q"][:, 0, 1] = rows // 64, rows % 64     # a row's index, exact in bf16
    before = dict(vp.PATCH.stats)
    vp.unified_attention(**a)
    assert built == [False]
    assert len(seen) == 1 and seen[0]["mq"] == 1 and seen[0]["cu"] == [0, 1, 2, 3, 4]
    assert seen[0]["table"] is a["block_table"]
    got = (seen[0]["q"][:, 0, 0].float() * 64 + seen[0]["q"][:, 0, 1].float()).int().tolist()
    assert got == starts[:-1]                                     # each request's first row
    assert seen[0]["used"] == [16384, 1, 16384, 16384]
    out = a["out"][:, 0, 0].tolist()
    for i, (s0, s1) in enumerate(zip(starts, starts[1:])):
        want = 5.0 if s1 - s0 > 1 else 100.0 + i
        assert out[s0:s1] == [want] * (s1 - s0), f"request {i}"
    assert vp.PATCH.stats["hk"] == before["hk"] + 1
    assert vp.PATCH.stats["split"] == before["split"] + 1
    assert vp.PATCH.stats["declined"] == before["declined"]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_every_measured_batch_taken_split_is_a_clear_win(monkeypatch, arch):
    """The split's evidence through its decision: the 68-shape set on three models, hk's route
    against the faster of TRITON_ATTN and ROCM_ATTN. Every batch the guard declines and the split
    takes is >= 1.05x; and it takes some on every model."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: arch)
    rows = json.loads((SWEEP.parent / "2026-11-04" / arch / "prefill-split-sweep.json")
                      .read_text())
    taken = {}
    for r in rows:
        d, hkv = DIMS[r["model"]]
        reqs = _reqs(r["shape"])
        st = _stats_of(reqs)
        if not st["decodes"] or not vp.exact_why_not(st, d, hkv, arch):
            continue
        a = _triton_call([(s - q, q) for s, q in reqs], d=d, hkv=hkv, page=16)
        monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): st})
        if vp.split_takes(a):
            u = r["us"]
            sp = min(u["triton"], u["rocm_attn"]) / u["hk"]
            assert sp >= 1.05, (arch, r["model"], r["shape"], sp)
            taken[r["model"]] = taken.get(r["model"], 0) + 1
    assert sorted(taken) == sorted(DIMS) and min(taken.values()) >= 4, taken


def test_the_split_kernels_are_built_with_the_first_ones(monkeypatch):
    """A batch hk takes whole builds the split's decodes=False kernel too (vLLM's warm-up run is
    such a batch), so the first mixed batch of a fresh cache does not compile on the serving
    path; with HK_VLLM_PREFILL_SPLIT=0 it is not built."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    monkeypatch.setattr(vp, "VT", False)
    monkeypatch.setattr(vp, "_STATS", {})

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, alibi_slopes=None, sinks=None):
        pytest.fail("vLLM's kernel was called")

    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    for split, want in ((True, [True, False]), (False, [True])):
        built = []

        def flash_op(d, group, scale, page, order, decodes):
            built.append(decodes)
            return (lambda *args: None), 16

        kvs = []
        monkeypatch.setattr(vp, "_flash_op", flash_op)
        monkeypatch.setattr(vp, "_kvs_ops", lambda *key: kvs.append(key[-2:]))
        monkeypatch.setattr(vp, "SPLIT", split)
        vp.unified_attention(**_triton_call([(0, 4096)]))
        assert built == want, (split, built)
        plan = vp.KVSPLITS[("gfx1100", 128)]
        assert sorted(kvs) == sorted({(sp, w) for w, sp in plan[2:]}), kvs


# ---- the KV split (paged_prefill's kv_splits) for short chunks over long contexts -------------

def test_kvsplit_plan_takes_a_short_chunk_with_exact_sizes(monkeypatch):
    """A chunk shorter than KVSPLITS' bound is taken with the KV split, its tile by its length;
    not without the step's sizes, not with HK_VLLM_PREFILL_KVSPLIT=0, not for a long chunk, not
    with a request of no queries beside decodes, and not on an arch without a measured plan."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: "gfx1100")
    lim, _, (w_short, s_short), (w_long, s_long) = vp.KVSPLITS[("gfx1100", 128)]

    def plan(reqs):
        a = _triton_call(reqs)
        st = [(c + q, q) for c, q in reqs]
        monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): _stats_of(st)})
        return a, vp.kvsplit_plan(a)

    a, p = plan([(16384, 32)])
    assert p == (s_short, w_short)
    assert plan([(16384, 128)])[1] == (s_long, w_long)
    assert plan([(16384, 128), (8192, 1), (8192, 1)])[1] == (s_long, w_long)
    assert plan([(16384, lim)])[1] is None
    assert plan([(16384, 128), (8192, 1), (0, 0)])[1] is None
    monkeypatch.setattr(vp, "_STATS", {})
    assert vp.kvsplit_plan(a) is None
    a, _ = plan([(16384, 32)])
    monkeypatch.setattr(vp, "KVSPLIT", False)
    assert vp.kvsplit_plan(a) is None
    monkeypatch.setattr(vp, "KVSPLIT", True)
    monkeypatch.setattr(switches, "arch", lambda: "gfx1030")
    assert vp.kvsplit_plan(a) is None


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_every_measured_shape_the_kvsplit_takes_is_a_clear_win(monkeypatch, arch):
    """The default-on rule's per-shape clause for the KV split, on its own sweep (hk's route as
    served, against the faster of TRITON_ATTN and ROCM_ATTN): every shape it takes is >= 1.15x,
    and it takes the short chunks over long contexts on every model."""
    from hk.integration import switches

    monkeypatch.setattr(switches, "arch", lambda: arch)
    rows = json.loads((SWEEP.parent / KVSPLIT_SWEEP / arch / "prefill-kvsplit-sweep.json")
                      .read_text())
    taken = {}
    for r in rows:
        d, hkv = DIMS[r["model"]]
        reqs = _reqs(r["shape"])
        a = _triton_call([(s - q, q) for s, q in reqs], d=d, hkv=hkv, page=16)
        monkeypatch.setattr(vp, "_STATS", {id(a["cu_seqlens_q"]): _stats_of(reqs)})
        if vp.kvsplit_plan(a) is None:
            continue
        u = r["us"]
        sp = min(u["triton"], u["rocm_attn"]) / u["hk"]
        assert sp >= 1.15, (arch, r["model"], r["shape"], sp)
        taken.setdefault(r["model"], set()).add(r["shape"])
    for model in DIMS:
        assert "32768:32" in taken.get(model, ()), (model, sorted(taken.get(model, ())))
