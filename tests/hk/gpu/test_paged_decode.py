"""hk.ops.paged_decode against a float64 reference, on vLLM's TRITON_ATTN cache layouts.

The decode step of a serving engine: every request has one query token, attending to its whole
sequence in the paged cache. The cases are the ones that break a split decode: lengths that end
inside a page and inside a block, a sequence shorter than one split's share (empty splits), a
CUDA graph's padding (a request with no query, which must write nothing), hybrid models' pages
(544, 784, 2064 tokens) whose rows past a sequence's end hold NaN, the three cache orders, GQA
groups that do not fill the 16-row tile, a strided query (a fused qkv's rows), head_dim 256
(split across a warp pair), and the exact probe: V all ones gives exactly 1.0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import _attn_ref  # noqa: E402

from hk.ops import paged_decode as pd  # noqa: E402

pytestmark = pytest.mark.gpu


def _rdna() -> bool:
    from hk.target import current_arch

    return current_arch().startswith("gfx1")


rdna_only = pytest.mark.skipif(not _rdna(), reason="hk.ops.paged_decode is RDNA's (WMMA tiles)")

#: lengths that end inside a page, on a page and block edge, past several splits' shares, and
#: one shorter than a block
LENS = [1, 15, 16, 33, 1000, 2063, 4100]
ODD_PAGES = [(16, "planes"), (16, "tokens"), (16, "heads"), (32, "planes"), (48, "tokens"),
             (544, "heads"), (784, "tokens"), (2064, "heads")]


def _case(lens, *, head_dim=128, h_kv=2, group=8, page=16, kv_order="planes", seed=0,
          v_ones=False, nan_tail=False, pad=0, strided_q=False, dtype=torch.bfloat16):
    """lens: each request's sequence length; `pad` requests with no query after them (a CUDA
    graph's padding: query_start_loc repeats, seq_lens 0)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(lens)
    pages = [max(1, -(-n // page)) for n in lens]
    nb = sum(pages) + 3
    k = torch.randn(nb, h_kv, page, head_dim, device="cuda", dtype=dtype, generator=g)
    v = (torch.ones_like(k) if v_ones else
         torch.randn(k.shape, device="cuda", dtype=dtype, generator=g))
    perm = torch.randperm(nb, generator=g, device="cuda")
    table = torch.zeros(r + pad, max(pages), dtype=torch.int32, device="cuda")
    at = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[at:at + n].to(torch.int32)
        at += n
    lens_t = torch.tensor(list(lens) + [0] * pad, device="cuda", dtype=torch.int32)
    if nan_tail:
        live = torch.zeros(nb, page, dtype=torch.bool, device="cuda")
        for i, s in enumerate(lens):
            for j in range(-(-s // page)):
                live[int(table[i, j]), : min(page, s - j * page)] = True
        dead = ~live[:, None, :, None]
        k, v = k.masked_fill(dead, float("nan")), v.masked_fill(dead, float("nan"))
    h = h_kv * group
    if strided_q:
        # the query rows of a fused qkv: (tokens, (h + 2 h_kv) * d), q the first h heads
        qkv = torch.randn(r + pad, (h + 2 * h_kv) * head_dim, device="cuda",
                          dtype=torch.bfloat16, generator=g)
        q = qkv[:, : h * head_dim].view(r + pad, h, head_dim)
    else:
        q = torch.randn(r + pad, h, head_dim, device="cuda", dtype=dtype, generator=g)
    q_start = torch.tensor(list(range(r + 1)) + [r] * pad, device="cuda", dtype=torch.int32)
    return q, k, v, table, lens_t, q_start


def _flash(k, v, kv_order):
    """TRITON_ATTN's cache in `kv_order`, as the walk takes it (vllm_prefill._flash_kv)."""
    nb, h_kv, page, d = k.shape
    if kv_order == "planes":
        fl = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=1)
        return fl.reshape(nb, 2, page, h_kv * d).contiguous()
    both = torch.cat([k, v], dim=-1)
    if kv_order == "tokens":
        return both.transpose(1, 2).contiguous().view(nb, 1, page, h_kv * 2 * d)
    return both.contiguous()


def _reference(q, k, v, table, lens, group):
    page = k.shape[2]
    n, h, d = q.shape
    out = torch.zeros(n, h, d, device="cuda", dtype=torch.float32)
    for i, s in enumerate(lens.tolist()):
        if s == 0:
            continue
        pages = table[i, : -(-s // page)].tolist()
        for hh in range(h):
            kk = torch.cat([k[p, hh // group] for p in pages])[:s]
            vv = torch.cat([v[p, hh // group] for p in pages])[:s]
            out[i, hh] = _attn_ref.softmax_v64(q[i, hh][None], kk, vv, d ** -0.5)[0]
    return out


def _decode(q, k, v, table, lens, q_start, *, kv_order, splits=None, warps=4, stream=None,
            out=None, scratch=None, poison=False, v_both=False):
    n, h, d = q.shape
    h_kv, page = k.shape[1], k.shape[2]
    group = h // h_kv
    reqs = len(lens)
    s = splits or pd.plan_splits(reqs, h_kv)
    from hk.runtime.torch_ext import launcher

    f16 = q.dtype == torch.float16
    walk = launcher(pd.walk_for(d, group, page, kv_order, s, warps=warps, v_both=v_both, fp16=f16))
    kv = _flash(k, v, kv_order)
    out = torch.full_like(q, 7.0) if out is None else out
    o_part, ml = scratch or pd.scratch(reqs * s, h_kv, group, d, q.device, dtype=q.dtype)
    if poison:
        o_part.fill_(float("nan"))
        ml.fill_(float("nan"))
    o4 = out.view(n, h_kv, group, d)
    walk(q.unsqueeze(0), kv, table, lens, q_start, o4, o_part, ml)
    if s > 1:
        launcher(pd.merge_for(d, group, s, fp16=f16))(lens, q_start, o4, o_part, ml)
    return out


def _check(got, want, lens):
    rows = [i for i, s in enumerate(lens.tolist()) if s]
    _attn_ref.check(got[rows], want[rows])


@rdna_only
@pytest.mark.parametrize("head_dim,h_kv,group,page,kv_order", [
    (128, 2, 8, 16, "planes"), (256, 4, 6, 784, "heads"), (256, 1, 8, 544, "tokens")])
@pytest.mark.parametrize("splits", [1, 8])
def test_fp16(head_dim, h_kv, group, page, kv_order, splits):
    """dtype fp16 (fp16-served models): the walk and merge on fp16 q / cache / out match the
    float64 reference, over hybrid models' NaN-tailed pages too."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, dtype=torch.float16,
                                          nan_tail=page > 16)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order, splits=splits)
    assert got.dtype == torch.float16
    _check(got, _reference(q, k, v, table, lens, group), lens)


@rdna_only
@pytest.mark.parametrize("page,kv_order", ODD_PAGES)
def test_every_layout_and_page(page, kv_order):
    q, k, v, table, lens, q_start = _case(LENS, page=page, kv_order=kv_order)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order)
    _check(got, _reference(q, k, v, table, lens, 8), lens)


@rdna_only
@pytest.mark.parametrize("splits", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("warps", [1, 2, 4])
def test_every_split_and_warp_count(splits, warps):
    q, k, v, table, lens, q_start = _case(LENS, seed=splits)
    got = _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=splits, warps=warps)
    _check(got, _reference(q, k, v, table, lens, 8), lens)


@rdna_only
@pytest.mark.parametrize("page,kv_order", [(16, "planes"), (16, "tokens"), (16, "heads")])
@pytest.mark.parametrize("warps", [2, 4])
def test_both_v_bands_in_flight(page, kv_order, warps):
    """`v_both` (plan's small-batch form): a block's two V bands loaded before the softmax."""
    from hk.runtime.resources import ResourceError
    from hk.runtime.torch_ext import launcher

    try:
        launcher(pd.walk_for(128, 8, page, kv_order, 32, warps=warps, v_both=True))
    except ResourceError:
        pytest.skip("both V bands in flight spill under this compiler (clang 22); the route "
                    "takes the one-band walk there")
    q, k, v, table, lens, q_start = _case(LENS, page=page, kv_order=kv_order, seed=7)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order, splits=32, warps=warps,
                  v_both=True)
    _check(got, _reference(q, k, v, table, lens, 8), lens)
    q, k, v, table, lens, q_start = _case(LENS, page=page, kv_order=kv_order, v_ones=True)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order, splits=32, warps=warps,
                  v_both=True)
    assert torch.all(got[: len(LENS)] == 1.0)


@rdna_only
@pytest.mark.parametrize("h_kv,group", [(1, 16), (2, 8), (4, 6), (8, 4), (2, 1)])
def test_every_gqa_group(h_kv, group):
    q, k, v, table, lens, q_start = _case([5, 700, 3000], h_kv=h_kv, group=group)
    got = _decode(q, k, v, table, lens, q_start, kv_order="planes")
    _check(got, _reference(q, k, v, table, lens, group), lens)


@rdna_only
@pytest.mark.parametrize("page,kv_order", [(16, "planes"), (784, "tokens"), (544, "heads"),
                                           (2064, "heads")])
@pytest.mark.parametrize("h_kv,group", [(4, 6), (2, 8), (1, 12)])
@pytest.mark.parametrize("warps", [2, 4])
def test_head_dim_256(page, kv_order, h_kv, group, warps):
    """Qwen3.8 (24/4), Qwen3.6 (16/2), MiniCPM-V's TP2 rank: head_dim split across warp pairs."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=256, h_kv=h_kv, group=group, page=page,
                                          kv_order=kv_order)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order, warps=warps)
    _check(got, _reference(q, k, v, table, lens, group), lens)


@rdna_only
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("splits", [1, 5, 32])
@pytest.mark.parametrize("page,kv_order", [(16, "planes"), (784, "tokens"), (2064, "heads")])
def test_v_ones_is_exactly_one(head_dim, splits, page, kv_order):
    """softmax(S) V with V all ones is 1.0 exactly in any summation order: a lost or
    double-counted key, a mis-weighted split or a bad rounding shows here as != 1."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=2, group=6, page=page,
                                          kv_order=kv_order, v_ones=True)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order, splits=splits)
    assert torch.all(got[: len(LENS)] == 1.0)


@rdna_only
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page,kv_order", [(544, "heads"), (784, "tokens"), (2064, "heads")])
def test_nan_past_the_end_of_hybrid_pages(head_dim, page, kv_order):
    """A hybrid model's attention page shares its block with mamba state: the rows past a
    sequence's end (and the blocks no request holds) can hold NaN. P = 0 times NaN is NaN."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, group=6, page=page,
                                          kv_order=kv_order, nan_tail=True)
    got = _decode(q, k, v, table, lens, q_start, kv_order=kv_order)
    want = _reference(q, k.nan_to_num(), v.nan_to_num(), table, lens, 6)
    _check(got, want, lens)


@rdna_only
@pytest.mark.parametrize("head_dim,group", [(128, 8), (256, 6)])
@pytest.mark.parametrize("splits", [1, 4])
def test_graph_padding_writes_nothing(head_dim, group, splits):
    """A CUDA graph's padded batch: requests with no query (query_start_loc repeats, seq_lens
    0) after the real ones. Their rows -- and the tensor past them -- stay untouched."""
    q, k, v, table, lens, q_start = _case([300, 17], head_dim=head_dim, group=group, pad=3)
    got = _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=splits)
    _check(got, _reference(q, k, v, table, lens, group), lens)
    assert torch.all(got[2:] == 7.0)


@rdna_only
@pytest.mark.parametrize("head_dim,group", [(128, 8), (256, 6)])
def test_a_strided_query(head_dim, group):
    """vLLM's query is a view of the fused qkv projection's rows."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, group=group, strided_q=True)
    assert not q.is_contiguous()
    got = _decode(q, k, v, table, lens, q_start, kv_order="planes")
    _check(got, _reference(q, k, v, table, lens, group), lens)


@rdna_only
@pytest.mark.parametrize("head_dim,group", [(128, 8), (256, 6)])
def test_nan_poisoned_repeats_are_bitwise_and_on_any_stream(head_dim, group):
    """200 calls with the split scratch and the output set to NaN before each: every output
    bitwise the first (nothing reads a stale or unwritten partial), and the same on a side
    stream."""
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, group=group, seed=3)
    s = 6
    scratch = pd.scratch(len(LENS) * s, 2, group, head_dim, q.device)
    out = torch.empty_like(q)
    first = None
    for _ in range(200):
        out.fill_(float("nan"))
        _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=s, out=out,
                scratch=scratch, poison=True)
        if first is None:
            torch.cuda.synchronize()
            first = out.clone()
            _check(first, _reference(q, k, v, table, lens, group), lens)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), first.view(torch.int16))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out2 = torch.full_like(q, float("nan"))
        _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=s, out=out2,
                scratch=scratch, poison=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), first.view(torch.int16))


@rdna_only
def test_a_cuda_graph_replays_other_lengths():
    """Captured once, replayed with other sequence lengths and pages: the split ranges come
    from the device's seq_lens, so every replay is right."""
    q, k, v, table, lens, q_start = _case([4000, 4000, 4000], seed=5)
    out = torch.full_like(q, 7.0)
    s = pd.plan_splits(3, 2)
    scratch = pd.scratch(3 * s, 2, 8, 128, q.device)
    _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=s, out=out, scratch=scratch)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _decode(q, k, v, table, lens, q_start, kv_order="planes", splits=s, out=out,
                scratch=scratch)
    for new in ([1, 70, 4000], [3999, 2, 31]):
        lens.copy_(torch.tensor(new, device="cuda", dtype=torch.int32))
        out.fill_(7.0)
        g.replay()
        torch.cuda.synchronize()
        _check(out, _reference(q, k, v, table, lens, 8), lens)


def test_a_bad_shape_is_refused():
    with pytest.raises(ValueError):
        pd.walk_kernel("t_bad_page", page=24)
    with pytest.raises(ValueError):
        pd.walk_kernel("t_bad_group", group=17)
    with pytest.raises(ValueError):
        pd.walk_kernel("t_bad_warps", head_dim=256, warps=1)


# ---- the route: vllm_prefill's TRITON_ATTN stand-in on a decode-only call, without vLLM -------

def _arch_in_route() -> bool:
    from hk.integration import vllm_prefill as vp
    from hk.target import current_arch

    return current_arch() in vp.DECODE_ARCHES


@pytest.mark.skipif(not _arch_in_route(), reason="the route's arches")
@pytest.mark.parametrize("head_dim,h_kv,group,page,kv_order,reqs", [
    (128, 2, 8, 16, "planes", 1),          # MiniCPM5-2B, with the padding: 4 pairs, the most
    (128, 2, 8, 16, "tokens", 1),          # the head_dim-128 decode takes (DECODE_MAX_PAIRS_D128)
    (256, 4, 6, 784, "heads", 8),          # Qwen3.8-27B
    (256, 2, 8, 2064, "tokens", 2),        # Qwen3.6
])
def test_the_route_takes_a_decode_batch_and_matches(monkeypatch, head_dim, h_kv, group, page,
                                                     kv_order, reqs):
    """HK_VLLM_DECODE=1: unified_attention's stand-in, called as TRITON_ATTN calls it on a
    decode-only batch (with a graph's padding request), takes it on hk's split decode --
    counted as decode, vLLM's kernel never called -- and matches the reference."""
    import inspect

    from hk.integration import vllm_prefill as vp

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, seq_threshold_3D=None,
                          num_par_softmax_segments=None, softmax_segm_output=None,
                          softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
                          sinks=None):
        raise AssertionError("fell back to vLLM's kernel")

    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    monkeypatch.setattr(vp, "DECODE", True)
    monkeypatch.setattr(vp, "DECODE_D128", True)
    lens_l = [37 + 1500 * i for i in range(reqs)]
    q, k, v, table, lens, q_start = _case(lens_l, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, pad=1)
    cache = _flash(k, v, kv_order)
    nb, d = k.shape[0], head_dim
    if kv_order == "planes":
        kc, vc = cache.view(nb, 2, page, h_kv, d).unbind(1)
    elif kv_order == "tokens":
        both = cache.view(nb, page, h_kv, 2 * d)
        kc, vc = both[..., :d], both[..., d:]
    else:
        both = cache.view(nb, h_kv, page, 2 * d)
        kc, vc = both[..., :d].transpose(1, 2), both[..., d:].transpose(1, 2)
    out = torch.full_like(q, 7.0)
    before = vp.PATCH.stats["decode"]
    vp.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=q_start, max_seqlen_q=1,
                         seqused_k=lens, max_seqlen_k=int(lens.max()), softmax_scale=d ** -0.5,
                         causal=True, window_size=(-1, -1), block_table=table, softcap=0,
                         q_descale=None, k_descale=None, v_descale=None)
    assert vp.PATCH.stats["decode"] == before + 1
    _check(out, _reference(q, k, v, table, lens, group), lens)
    assert torch.all(out[reqs:] == 7.0)


@pytest.mark.parametrize("kv_order", ["planes", "tokens"])
def test_a_larger_d128_decode_batch_is_vllms(monkeypatch, kv_order):
    """HK_VLLM_DECODE=1, head_dim 128, 3 requests and a graph's padding request: 8 (request, kv
    head) pairs, past DECODE_MAX_PAIRS_D128 (four requests were 0.98x in serving). The call goes
    to vLLM's unified_attention with the caller's own tensors, counted as a fallback; its output
    is vLLM's."""
    import inspect

    from hk.integration import vllm_prefill as vp

    seen = []

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, seq_threshold_3D=None,
                          num_par_softmax_segments=None, softmax_segm_output=None,
                          softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
                          sinks=None):
        seen.append((q, out))
        out.fill_(3.0)

    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    monkeypatch.setattr(vp, "DECODE", True)
    monkeypatch.setattr(vp, "DECODE_D128", True)
    q, k, v, table, lens, q_start = _case([37, 1537, 3037], head_dim=128, h_kv=2, group=8,
                                          page=16, kv_order=kv_order, pad=1)
    cache = _flash(k, v, kv_order)
    nb = k.shape[0]
    if kv_order == "planes":
        kc, vc = cache.view(nb, 2, 16, 2, 128).unbind(1)
    else:
        both = cache.view(nb, 16, 2, 256)
        kc, vc = both[..., :128], both[..., 128:]
    out = torch.full_like(q, 7.0)
    before = dict(vp.PATCH.stats)
    vp.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=q_start, max_seqlen_q=1,
                         seqused_k=lens, max_seqlen_k=int(lens.max()), softmax_scale=128 ** -0.5,
                         causal=True, window_size=(-1, -1), block_table=table, softcap=0,
                         q_descale=None, k_descale=None, v_descale=None)
    assert vp.PATCH.stats["decode"] == before["decode"]
    assert vp.PATCH.stats["fallback"] == before["fallback"] + 1
    assert len(seen) == 1 and seen[0][0] is q and seen[0][1] is out
    assert torch.all(out == 3.0)


@pytest.mark.parametrize("env,on,d128", [("", True, False), ("1", True, True), ("0", False, False),
                                         ("off", False, False)])
def test_the_decode_switch(monkeypatch, env, on, d128):
    """HK_VLLM_DECODE: unset takes head_dim 256 decodes only, 1 also head_dim 128 (up to the
    pair limit), 0 none."""
    from hk.integration import vllm_prefill as vp

    assert vp.decode_switches(env) == (on, d128)
    monkeypatch.setattr(vp, "DECODE_D128", d128)
    assert vp.decode_takes(2, 2, 128) == d128
    assert not vp.decode_takes(8, 2, 128)
    assert not vp.decode_takes(4, 2, 128)    # 8 pairs: 0.98x in serving at 4k-8k
    assert vp.decode_takes(16, 4, 256)


# ---- kv_order="rocm": vLLM ROCM_ATTN's own cache (x-split K, per-page V^T) --------------------

def _rocm(k, v):
    """ROCM_ATTN's caches of the same values: K (blocks, kv_heads, d / 8, page, 8), V^T
    (blocks, kv_heads, d, page)."""
    nb, h_kv, page, d = k.shape
    k5 = k.reshape(nb, h_kv, page, d // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    return k5, v.transpose(2, 3).contiguous()


def _decode_rocm(q, k, v, table, lens, q_start, *, splits=None, warps=1, out=None,
                 scratch=None, poison=False):
    from hk.ops import paged_prefill as pp
    from hk.runtime.torch_ext import launcher

    n, h, d = q.shape
    h_kv = k.shape[1]
    group = h // h_kv
    reqs = len(lens)
    s = splits or pd.plan_rocm(reqs, h_kv)
    walk = launcher(pd.walk_for(d, group, 16, "rocm", s, warps=warps))
    kc, vt = pp.rocm_views(*_rocm(k, v))
    out = torch.full_like(q, 7.0) if out is None else out
    o_part, ml = scratch or pd.scratch(reqs * s, h_kv, group, d, q.device)
    if poison:
        o_part.fill_(float("nan"))
        ml.fill_(float("nan"))
    o4 = out.view(n, h_kv, group, d)
    walk(q.unsqueeze(0), kc, table, lens, q_start, o4, o_part, ml, vt)
    if s > 1:
        launcher(pd.merge_for(d, group, s))(lens, q_start, o4, o_part, ml)
    return out


@rdna_only
@pytest.mark.parametrize("splits", [1, 3, 8, 48, 96])
@pytest.mark.parametrize("warps", [1, 2])
def test_rocm_every_split_and_warp_count(splits, warps):
    """Lengths ending inside a page and a block, shorter than a split's share (empty splits),
    one shorter than a block."""
    q, k, v, table, lens, q_start = _case(LENS, seed=11)
    got = _decode_rocm(q, k, v, table, lens, q_start, splits=splits, warps=warps)
    _check(got, _reference(q, k, v, table, lens, 8), lens)


@rdna_only
@pytest.mark.parametrize("h_kv,group", [(1, 16), (2, 8), (4, 6), (8, 4), (2, 1)])
def test_rocm_every_gqa_group(h_kv, group):
    q, k, v, table, lens, q_start = _case(LENS, h_kv=h_kv, group=group, seed=12)
    got = _decode_rocm(q, k, v, table, lens, q_start)
    _check(got, _reference(q, k, v, table, lens, group), lens)


@rdna_only
@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 5, 8])
def test_rocm_the_planned_splits(reqs):
    """The plan the route uses (4 one-warp workgroups a WGP of a W7900D), on long and short
    requests together."""
    lens_l = [70000 if i % 2 == 0 else 33 + 977 * i for i in range(reqs)]
    q, k, v, table, lens, q_start = _case(lens_l, seed=13 + reqs)
    got = _decode_rocm(q, k, v, table, lens, q_start, splits=pd.plan_rocm(reqs, 2, wgps=48))
    _check(got, _reference(q, k, v, table, lens, 8), lens)


@rdna_only
@pytest.mark.parametrize("splits", [1, 4, 96])
def test_rocm_graph_padding_writes_nothing(splits):
    q, k, v, table, lens, q_start = _case([300, 17], pad=3)
    got = _decode_rocm(q, k, v, table, lens, q_start, splits=splits)
    _check(got, _reference(q, k, v, table, lens, 8), lens)
    assert torch.all(got[2:] == 7.0)


@rdna_only
def test_rocm_a_strided_query():
    q, k, v, table, lens, q_start = _case(LENS, strided_q=True, seed=14)
    assert not q.is_contiguous()
    got = _decode_rocm(q, k, v, table, lens, q_start)
    _check(got, _reference(q, k, v, table, lens, 8), lens)


@rdna_only
@pytest.mark.parametrize("splits", [1, 5, 96])
def test_rocm_v_ones_is_exactly_one(splits):
    q, k, v, table, lens, q_start = _case(LENS, v_ones=True, seed=15)
    got = _decode_rocm(q, k, v, table, lens, q_start, splits=splits)
    rows = [i for i, s in enumerate(lens.tolist()) if s]
    assert torch.all(got[rows] == 1.0)


@rdna_only
def test_rocm_nan_poisoned_repeats_are_bitwise_and_on_any_stream():
    q, k, v, table, lens, q_start = _case(LENS, seed=16)
    s = 24
    scratch = pd.scratch(len(LENS) * s, 2, 8, 128, q.device)
    out = torch.empty_like(q)
    first = None
    for _ in range(200):
        out.fill_(float("nan"))
        _decode_rocm(q, k, v, table, lens, q_start, splits=s, out=out, scratch=scratch,
                     poison=True)
        if first is None:
            torch.cuda.synchronize()
            first = out.clone()
            _check(first, _reference(q, k, v, table, lens, 8), lens)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), first.view(torch.int16))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out2 = torch.full_like(q, float("nan"))
        _decode_rocm(q, k, v, table, lens, q_start, splits=s, out=out2, scratch=scratch,
                     poison=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), first.view(torch.int16))


@rdna_only
def test_rocm_a_cuda_graph_replays_other_lengths():
    q, k, v, table, lens, q_start = _case([4000, 4000, 4000], seed=17)
    out = torch.full_like(q, 7.0)
    s = pd.plan_rocm(3, 2, wgps=48)
    scratch = pd.scratch(3 * s, 2, 8, 128, q.device)
    from hk.ops import paged_prefill as pp
    from hk.runtime.torch_ext import launcher

    k5, vt0 = _rocm(k, v)
    kc, vt = pp.rocm_views(k5, vt0)
    walk = launcher(pd.walk_for(128, 8, 16, "rocm", s, warps=1))
    merge = launcher(pd.merge_for(128, 8, s))
    o4 = out.view(3, 2, 8, 128)

    def step():
        walk(q.unsqueeze(0), kc, table, lens, q_start, o4, *scratch, vt)
        merge(lens, q_start, o4, *scratch)
    step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for new in ([1, 70, 4000], [3999, 2, 31]):
        lens.copy_(torch.tensor(new, device="cuda", dtype=torch.int32))
        out.fill_(7.0)
        g.replay()
        torch.cuda.synchronize()
        _check(out, _reference(q, k, v, table, lens, 8), lens)


def test_rocm_refuses_what_it_does_not_read():
    with pytest.raises(ValueError):
        pd.walk_kernel("t_rocm_page", kv_order="rocm", page=32)
    with pytest.raises(ValueError):
        pd.walk_kernel("t_rocm_d256", kv_order="rocm", head_dim=256, warps=2)


def test_rocm_plan_keeps_four_workgroups_a_wgp():
    for reqs, want in ((1, 96), (2, 48), (3, 32), (4, 24), (5, 19), (8, 12), (64, 1)):
        assert pd.plan_rocm(reqs, 2, wgps=48) == want
        assert reqs * 2 * pd.plan_rocm(reqs, 2, wgps=48) <= 192 or want == 1


# ---- the route: vllm_prefill's ROCM_ATTN chunked_prefill_paged_decode stand-in -----------------

def _rocm_route_arch() -> bool:
    from hk.integration import vllm_prefill as vp
    from hk.target import current_arch

    return current_arch() in vp.ROCM_DEC_ARCHES


def _cppd_stub(seen):
    def chunked_prefill_paged_decode(query, key, value, output, kv_cache_dtype, key_cache,
                                     value_cache, block_table, query_start_loc, seq_lens,
                                     max_seq_len, max_query_len, k_scale, v_scale,
                                     alibi_slopes=None, sliding_window=None, sm_scale=None,
                                     output_scale=None, sinks=None, is_block_table_ptr=False,
                                     causal=True):
        seen.append((query, output))
        output.fill_(3.0)
    return chunked_prefill_paged_decode


def _call_rocm_route(vp, q, k, v, table, lens, q_start, max_query_len=1):
    k5, vt = _rocm(k, v)
    out = torch.full_like(q, 7.0)
    one = torch.ones(1, device="cuda")
    vp.chunked_prefill_paged_decode(
        query=q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=k5,
        value_cache=vt, block_table=table, query_start_loc=q_start, seq_lens=lens,
        max_seq_len=int(lens.max()), max_query_len=max_query_len, k_scale=one, v_scale=one,
        sliding_window=None, sm_scale=128 ** -0.5)
    return out


@pytest.mark.skipif(not _rocm_route_arch(), reason="the route's arches")
@pytest.mark.parametrize("reqs", [1, 2, 7])
def test_the_rocm_route_takes_a_decode_batch_and_matches(monkeypatch, reqs):
    """ROCM_ATTN's chunked_prefill_paged_decode, as its forward calls it on a decode-only batch
    with a graph's padding request: hk's split decode takes it (counted as decode, vLLM's
    function never called) and matches the reference; the padding row is untouched."""
    import inspect

    from hk.integration import vllm_prefill as vp

    seen = []
    stub = _cppd_stub(seen)
    monkeypatch.setattr(vp, "_ORIG_C", stub)
    monkeypatch.setattr(vp, "_C_SIG", inspect.signature(stub))
    monkeypatch.setattr(vp, "ROCM_DECODE", True)
    q, k, v, table, lens, q_start = _case([37 + 9000 * i for i in range(reqs)], pad=1)
    before = vp.PATCH.stats["decode"]
    out = _call_rocm_route(vp, q, k, v, table, lens, q_start)
    assert not seen
    assert vp.PATCH.stats["decode"] == before + 1
    _check(out, _reference(q, k, v, table, lens, 8), lens)
    assert torch.all(out[reqs:] == 7.0)


@pytest.mark.parametrize("case", ["batch", "prefill", "off"])
def test_the_rocm_route_leaves_the_rest_to_vllm(monkeypatch, case):
    """Past the pair limit (8 requests and a padding one: 18 pairs), a call with prefill in it,
    or the route off (HK_VLLM_ROCM_DECODE unset): vLLM's function, with the caller's own
    tensors."""
    import inspect

    from hk.integration import vllm_prefill as vp

    seen = []
    stub = _cppd_stub(seen)
    monkeypatch.setattr(vp, "_ORIG_C", stub)
    monkeypatch.setattr(vp, "_C_SIG", inspect.signature(stub))
    monkeypatch.setattr(vp, "ROCM_DECODE", case != "off")
    n = 8 if case == "batch" else 2
    q, k, v, table, lens, q_start = _case([100 + i for i in range(n)], pad=1)
    before = dict(vp.PATCH.stats)
    out = _call_rocm_route(vp, q, k, v, table, lens, q_start,
                           max_query_len=2 if case == "prefill" else 1)
    assert len(seen) == 1 and seen[0][0] is q and seen[0][1] is out
    assert torch.all(out == 3.0)
    assert vp.PATCH.stats["decode"] == before["decode"]
