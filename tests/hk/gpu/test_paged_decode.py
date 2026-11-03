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
          v_ones=False, nan_tail=False, pad=0, strided_q=False):
    """lens: each request's sequence length; `pad` requests with no query after them (a CUDA
    graph's padding: query_start_loc repeats, seq_lens 0)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(lens)
    pages = [max(1, -(-n // page)) for n in lens]
    nb = sum(pages) + 3
    k = torch.randn(nb, h_kv, page, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    v = (torch.ones_like(k) if v_ones else
         torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=g))
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
        q = torch.randn(r + pad, h, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
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

    walk = launcher(pd.walk_for(d, group, page, kv_order, s, warps=warps, v_both=v_both))
    kv = _flash(k, v, kv_order)
    out = torch.full_like(q, 7.0) if out is None else out
    o_part, ml = scratch or pd.scratch(reqs * s, h_kv, group, d, q.device)
    if poison:
        o_part.fill_(float("nan"))
        ml.fill_(float("nan"))
    o4 = out.view(n, h_kv, group, d)
    walk(q.unsqueeze(0), kv, table, lens, q_start, o4, o_part, ml)
    if s > 1:
        launcher(pd.merge_for(d, group, s))(lens, q_start, o4, o_part, ml)
    return out


def _check(got, want, lens):
    rows = [i for i, s in enumerate(lens.tolist()) if s]
    _attn_ref.check(got[rows], want[rows])


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
    (128, 2, 8, 16, "planes", 3),          # MiniCPM5-2B
    (128, 2, 8, 16, "planes", 1),          # (with the padding: 4 pairs, 32 splits of 2 warps)
    (128, 2, 8, 16, "tokens", 3),
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
