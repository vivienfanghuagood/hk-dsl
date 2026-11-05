"""Paged prefill against a dense reference: what chunked prefill and prefix
caching need from attention.

A request's new tokens are the suffix of its sequence -- `ctx` tokens are
already in the cache -- and each attends causally to everything up to its own
position. The reference gathers the request's pages and applies that mask by
hand. The cases are the ones that break a kernel that assumes otherwise: a
cached prefix, a chunk shorter than a query tile, a tile that runs into the
next request's tokens (whose rows must come out as that request's answer,
not this one's), and rows that belong to no prefill (left untouched).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import _attn_ref  # noqa: E402

from hk.ops import paged_prefill as pp  # noqa: E402

pytestmark = pytest.mark.gpu

PAGE = pp.KV_BLOCK


def _case(reqs, head_dim=128, h_kv=2, group=4, seed=0):
    """reqs: [(ctx, q_len)]. Every request's whole sequence is in the cache;
    its last q_len tokens are the queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = len(reqs)
    seq = [c + n for c, n in reqs]
    max_pages = max(-(-s // PAGE) for s in seq)
    n_blocks = r * max_pages + 5
    k_cache = torch.randn(n_blocks, h_kv, PAGE, head_dim, device="cuda",
                          dtype=torch.bfloat16, generator=g)
    v_cache = torch.randn(k_cache.shape, device=k_cache.device, dtype=k_cache.dtype, generator=g)
    perm = torch.randperm(n_blocks, generator=g, device="cuda")[: r * max_pages]
    table = perm.reshape(r, max_pages).to(torch.int32).contiguous()
    n_tok = sum(n for _, n in reqs)
    q = torch.randn(n_tok, h_kv * group, head_dim, device="cuda",
                    dtype=torch.bfloat16, generator=g)
    starts = [0]
    for _, n in reqs:
        starts.append(starts[-1] + n)
    q_start = torch.tensor(starts, device="cuda", dtype=torch.int32)
    lens = torch.tensor(seq, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, table, lens, q_start


def _reference(q, k_cache, v_cache, table, lens, q_start, group, page=PAGE):
    PAGE = page  # noqa: N806 -- the cache's page, hk's or ROCM_ATTN's
    n_tok, h, d = q.shape
    out = torch.zeros(n_tok, h, d, device="cuda", dtype=torch.float32)
    for i in range(len(lens)):
        q0, q1, s = int(q_start[i]), int(q_start[i + 1]), int(lens[i])
        n, ctx = q1 - q0, s - (q1 - q0)
        if n == 0:
            continue
        pages = table[i, : -(-s // PAGE)].tolist()
        pos = ctx + torch.arange(n, device="cuda")
        keep = torch.arange(s, device="cuda")[None, :] <= pos[:, None]
        for hh in range(h):
            kk = torch.cat([k_cache[p, hh // group] for p in pages])[:s]
            vv = torch.cat([v_cache[p, hh // group] for p in pages])[:s]
            out[q0:q1, hh] = _attn_ref.softmax_v64(q[q0:q1, hh], kk, vv, d ** -0.5, keep)
    return out


def _run(reqs, prefill=None, **kw):
    group = kw.get("group", 4)
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, **kw)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    prefill = range(len(reqs)) if prefill is None else prefill
    tiles = pp.work_list([(r, reqs[r][1]) for r in prefill], q.device)
    pp.kernel_for(d, group)(q.view(n_tok, h * d), k_cache, v_cache, table,
                            lens, q_start, tiles, out.view(n_tok, h * d))
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    return out, want, q_start


def test_no_prefix_two_tiles_the_second_short():
    got, want, _ = _run([(0, 300)])
    _attn_ref.check(got, want)


def test_a_cached_prefix_and_a_chunk_shorter_than_a_tile():
    got, want, _ = _run([(1000, 77)])
    _attn_ref.check(got, want)


def test_a_tile_that_runs_into_the_next_request():
    # Request 0's only tile covers rows 0..191; rows 100..149 are request 1's
    # and must come out as request 1's answer.
    got, want, _ = _run([(64, 100), (300, 50), (0, 200)])
    _attn_ref.check(got, want)


def test_ragged_batch_with_a_one_token_request():
    got, want, _ = _run([(0, 192), (500, 5), (40, 600), (33, 1)])
    _attn_ref.check(got, want)


def test_rows_of_requests_not_in_the_work_list_are_untouched():
    reqs = [(0, 40), (128, 70), (10, 30)]
    got, want, q_start = _run(reqs, prefill=[1])
    lo, hi = int(q_start[1]), int(q_start[2])
    assert torch.all(got[:lo] == 7.0) and torch.all(got[hi:] == 7.0)
    _attn_ref.check(got[lo:hi], want[lo:hi])


@pytest.mark.parametrize("group", [1, 4, 8])
def test_every_gqa_group(group):
    got, want, _ = _run([(96, 130), (0, 64)], group=group)
    _attn_ref.check(got, want)


def test_head_dim_64():
    got, want, _ = _run([(70, 250)], head_dim=64)
    _attn_ref.check(got, want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's numbering")
@pytest.mark.parametrize("h_kv,group", [(40, 1), (8, 4), (3, 2)])
def test_the_spatial_numbering_is_bitwise_the_default_s(h_kv, group):
    """attn_cdna's mapping="spatial" runs the same tiles in another order: every output
    bit the same as the default numbering's, on a ragged batch with cached prefixes."""
    from hk.ops import attn_cdna

    reqs = [(64, 100), (300, 50), (0, 600), (33, 1)]
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, h_kv=h_kv, group=group)
    n_tok, h, d = q.shape
    tiles = pp.work_list([(r, reqs[r][1]) for r in range(len(reqs))], q.device)
    outs = []
    for mapping in attn_cdna.MAPPINGS:
        out = torch.full_like(q, 7.0)
        attn_cdna.paged_kernel_for(d, mapping=mapping)(
            q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
            out.view(n_tok, h * d))
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    _attn_ref.check(outs[1], want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's KV-split prefill")
@pytest.mark.parametrize("reqs,target", [
    ([(4096, 512)], 256), ([(4100, 300), (0, 600), (33, 1), (9000, 70)], 512),
    ([(2048, 1024), (2048, 1024)], 1024), ([(5000, 40)], 64)])
def test_the_kv_split_prefill(reqs, target):
    """gfx950's prefill with its long walks cut over the cached prefix
    (paged_prefill.split_plan): cut tiles merged from their partials, uncut ones
    written straight, a tile whose rows run past its request, a one-token chunk;
    right against fp32 and the same answer twice."""
    from hk.ops import attn_cdna

    group = 4
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, group=group)
    n_tok, h, d = q.shape
    starts = q_start.tolist()
    tiles, mlist, n_merge, n_slots = pp.split_plan(
        [(i, c, n, starts[i]) for i, (c, n) in enumerate(reqs)], h, q.device, target=target,
        min_blocks=4)
    o_part, ml_part = pp.split_scratch(n_slots, h, d, q.device)
    walk, merge = attn_cdna.split_kernels_for(d)
    outs = []
    for _ in range(2):
        out = torch.full_like(q, 7.0)
        walk(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
             out.view(n_tok, h * d), o_part, ml_part)
        if n_merge:
            merge(mlist, o_part, ml_part, out.view(n_tok, h * d))
        outs.append(out)
    want = _reference(q, k_cache, v_cache, table, lens, q_start, group)
    _attn_ref.check(outs[0], want)
    assert torch.equal(outs[0], outs[1])
    if target >= 256:
        assert n_merge > 0


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="attn_cdna's KV-split prefill")
def test_an_uncut_split_plan_is_bitwise_the_work_list():
    """With nothing cut, the KV-split kernel over split_plan's entries (in its
    longest-first order) writes every bit the single-tile kernel does."""
    from hk.ops import attn_cdna

    reqs = [(64, 600), (300, 50), (0, 1024), (33, 1)]
    q, k_cache, v_cache, table, lens, q_start = _case(reqs, h_kv=1, group=8)
    n_tok, h, d = q.shape
    starts = q_start.tolist()
    tiles, _, n_merge, n_slots = pp.split_plan(
        [(i, c, n, starts[i]) for i, (c, n) in enumerate(reqs)], h, q.device, target=1)
    assert n_merge == 0
    o_part, ml_part = pp.split_scratch(1, h, d, q.device)
    walk, _ = attn_cdna.split_kernels_for(d)
    a = torch.full_like(q, 7.0)
    walk(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start, tiles,
         a.view(n_tok, h * d), o_part, ml_part)
    b = torch.full_like(q, 7.0)
    pp.kernel_for(d, 8)(q.view(n_tok, h * d), k_cache, v_cache, table, lens, q_start,
                        pp.work_list([(i, n) for i, (_, n) in enumerate(reqs)], q.device),
                        b.view(n_tok, h * d))
    assert torch.equal(a, b)


# ---- layout="rocm": vLLM ROCM_ATTN's own cache, read in place -------------------------------

ROCM_PAGE = 16


def _rdna() -> bool:
    from hk.target import current_arch

    return current_arch().startswith("gfx1")


#: layout="rocm" reads K through a strided global, an RDNA-only type so far (#146).
rocm_only = pytest.mark.skipif(not _rdna(), reason="layout='rocm' is RDNA's (strided globals)")


def _rocm_case(reqs, head_dim=128, h_kv=2, group=8, seed=0, v_ones=False, page=ROCM_PAGE):
    """reqs: [(ctx, q_len)] over `page`-token pages. The token-major caches (for the reference)
    and ROCM_ATTN's: K x-split (blocks, kv_heads, d / 8, page, 8), V (blocks, kv_heads, d,
    page)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = [c + n for c, n in reqs]
    max_pages = max(1, max(-(-s // page) for s in seq))
    n_blocks = len(reqs) * max_pages + 3
    k = torch.randn(n_blocks, h_kv, page, head_dim, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    v = (torch.ones_like(k) if v_ones else
         torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=g))
    perm = torch.randperm(n_blocks, generator=g, device="cuda")[: len(reqs) * max_pages]
    table = perm.reshape(len(reqs), max_pages).to(torch.int32).contiguous()
    n_tok = sum(n for _, n in reqs)
    q = torch.randn(n_tok, h_kv * group, head_dim, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    starts = [0]
    for _, n in reqs:
        starts.append(starts[-1] + n)
    q_start = torch.tensor(starts, device="cuda", dtype=torch.int32)
    lens = torch.tensor(seq, device="cuda", dtype=torch.int32)
    k5 = k.view(n_blocks, h_kv, page, head_dim // 8, 8).permute(0, 1, 3, 2, 4).contiguous()
    v4 = v.transpose(2, 3).contiguous()
    return q, (k, v), (k5, v4), table, lens, q_start


def _rocm_run(reqs, page=ROCM_PAGE, **kw):
    group = kw.get("group", 8)
    q, (k, v), (k5, v4), table, lens, q_start = _rocm_case(reqs, page=page, **kw)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    tiles = pp.q_tiles(max(n for _, n in reqs), q.device)
    pp.rocm_kernel_for(d, group, page=page)(q.view(n_tok, h * d), *pp.rocm_views(k5, v4), table,
                                            lens, q_start, tiles, out.view(n_tok, h * d))
    want = _reference(q, k, v, table, lens, q_start, group, page=page)
    return out, want, q_start


def _prefill_rows(reqs, q_start):
    """Rows of requests with more than one query: the ones layout="rocm" computes."""
    s = q_start.tolist()
    return [r for i, (_, n) in enumerate(reqs) if n > 1 for r in range(s[i], s[i + 1])]


@rocm_only
@pytest.mark.parametrize("reqs", [
    [(0, 300)],                                   # no prefix, two tiles, the second short
    [(1000, 77)],                                 # a prefix and a chunk shorter than a tile
    [(64, 100), (300, 50), (0, 200)],             # a tile running into the next request
    [(0, 192), (500, 5), (40, 600), (33, 1)],     # ragged, with a one-token request (a decode)
    [(7, 2), (16, 16), (31, 33), (2047, 2049)],   # ctx / ends at and off the 16-token pages
])
def test_rocm_layout(reqs):
    got, want, q_start = _rocm_run(reqs)
    rows = _prefill_rows(reqs, q_start)
    _attn_ref.check(got[rows], want[rows])
    skip = sorted(set(range(got.shape[0])) - set(rows))
    assert torch.all(got[skip] == 7.0), "a decode's rows are vLLM's paged decode's, not this"


@rocm_only
@pytest.mark.parametrize("group,h_kv", [(1, 4), (4, 2), (8, 2), (8, 1)])
def test_rocm_layout_every_gqa_group(group, h_kv):
    reqs = [(96, 130), (0, 64)]
    got, want, _ = _rocm_run(reqs, group=group, h_kv=h_kv)
    _attn_ref.check(got, want)


@rocm_only
def test_rocm_layout_head_dim_64():
    got, want, _ = _rocm_run([(70, 250)], head_dim=64, group=4)
    _attn_ref.check(got, want)


@rocm_only
def test_rocm_layout_v_ones_is_exactly_one():
    """softmax weights sum to 1: with V all ones every output is exactly 1.0 -- a biased
    bf16 conversion of P (hk-bf16-truncation-bias) shows here and nowhere else."""
    got, _, q_start = _rocm_run([(300, 200), (0, 50)], v_ones=True)
    assert torch.all(got == 1.0)


@rocm_only
def test_rocm_layout_is_deterministic_and_skips_padding():
    """The same answer three times; a request padded into the batch (no queries, no tokens),
    as a captured batch has, computes and reads nothing."""
    reqs = [(500, 300), (0, 0), (40, 77)]
    outs = [_rocm_run(reqs, seed=5)[0] for _ in range(3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


# ---- layout="flash" (TRITON_ATTN's cache) and dsplit (head_dim 256) --------------------------

def _flash_cache(k, v, kv_order="planes"):
    """TRITON_ATTN's cache from the token-major (blocks, kv_heads, page, d) K and V, as
    layout="flash" takes it in `kv_order` (vllm_prefill._flash_kv)."""
    nb, h_kv, page, d = k.shape
    if kv_order == "planes":
        fl = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=1)   # (nb, 2, page, hkv, d)
        return fl.reshape(nb, 2, page, h_kv * d).contiguous()
    both = torch.cat([k, v], dim=-1)                                       # (nb, hkv, page, 2d)
    if kv_order == "tokens":
        return both.transpose(1, 2).contiguous().view(nb, 1, page, h_kv * 2 * d)
    return both.contiguous()


def _nan_past_ends(k, v, table, lens):
    """NaN in every cache row no request has -- past each sequence's end in its last page, and
    the blocks nobody's table holds: what a hybrid model's shared blocks can hold there."""
    page = k.shape[2]
    live = torch.zeros(k.shape[0], page, dtype=torch.bool, device=k.device)
    for r, s in enumerate(lens.tolist()):
        for j in range(-(-s // page)):
            live[int(table[r, j]), : min(page, s - j * page)] = True
    dead = ~live[:, None, :, None]
    return k.masked_fill(dead, float("nan")), v.masked_fill(dead, float("nan"))


def _flash_run(reqs, head_dim=128, h_kv=2, group=8, dsplit=False, layout="flash", seed=0,
               decodes=True, warps=None, page=ROCM_PAGE, v_ones=False, kv_order="planes",
               nan_tail=False, kv_splits=1, pack=False):
    q, (k, v), (k5, v4), table, lens, q_start = _rocm_case(reqs, head_dim=head_dim, h_kv=h_kv,
                                                           group=group, seed=seed, page=page,
                                                           v_ones=v_ones)
    if nan_tail:
        k, v = _nan_past_ends(k, v, table, lens)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    warps = warps or (16 if dsplit else 12)
    kern = pp.paged_prefill_kernel(
        f"t_pp_{layout}_d{head_dim}_g{group}{'_ds' if dsplit else ''}_w{warps}"
        f"{'_dec' if decodes else ''}{'' if page == ROCM_PAGE else f'_p{page}'}"
        f"{'' if kv_order == 'planes' else '_' + kv_order}{f'_ks{kv_splits}' if kv_splits > 1 else ''}"
        f"{'_pk' if pack else ''}",
        head_dim=head_dim, group=group, layout=layout, dsplit=dsplit, warps=warps,
        decodes=decodes, page=page, kv_order=kv_order, kv_splits=kv_splits, pack=pack)
    tile = 16 * (warps // 2 if dsplit else warps) // (group if pack else 1)
    tiles = pp.q_tiles(max(n for _, n in reqs), q.device, tile)
    if layout == "flash":
        kv = _flash_cache(k, v, kv_order)
        packed = [q.unsqueeze(0), out.unsqueeze(0)] if decodes and not dsplit else []
        if kv_splits > 1:
            packed = list(pp.split_parts(kv_splits, len(reqs) * tiles.shape[0] * tile, h, d,
                                         q.device))
        kern(q.view(n_tok, h * d), kv, kv, table, lens, q_start, tiles, out.view(n_tok, h * d),
             *packed)
        if kv_splits > 1:
            pp.merge_kernel(f"t_pp_merge_d{head_dim}_ks{kv_splits}_t{tile}", head_dim=head_dim,
                            kv_splits=kv_splits, q_tile=tile)(
                lens, q_start, tiles, *packed, out.view(n_tok, h * d))
    else:
        kern(q.view(n_tok, h * d), *pp.rocm_views(k5, v4), table, lens, q_start, tiles,
             out.view(n_tok, h * d))
    want = _reference(q, k, v, table, lens, q_start, group, page=page)
    return out, want, q_start


@rocm_only
@pytest.mark.parametrize("reqs", [
    [(0, 300)],
    [(0, 192), (500, 5), (40, 600), (33, 1)],     # with a decode: computed (decodes=True)
    [(7, 2), (16, 16), (31, 33), (2047, 2049)],
])
def test_flash_layout_with_decodes(reqs):
    got, want, _ = _flash_run(reqs)
    _attn_ref.check(got, want)


def _clang_23() -> bool:
    from hk.ops.attn import _clang_major

    return (_clang_major() or 0) >= 23


#: dsplit with the flash layout holds V^T's register staging across the compute as well, and
#: at 256 VGPRs clang 22 (ROCm 7.2.3) spills 1-8 of them -- the build refuses it; clang 23
#: (TheRock 7.14, the production vLLM images) does not. The route falls back where it does not
#: build.
flash_dsplit_builds = pytest.mark.skipif(not _clang_23(), reason="flash dsplit spills under clang 22")


@rocm_only
@pytest.mark.parametrize("layout", [pytest.param("flash", marks=flash_dsplit_builds), "rocm"])
@pytest.mark.parametrize("reqs", [[(0, 300)], [(1000, 77), (0, 1), (40, 600)]])
def test_dsplit_head_dim_256(layout, reqs):
    """d256 with each query row split over a warp pair: MiniCPM-V's / Qwen3.8's full attention."""
    got, want, q_start = _flash_run(reqs, head_dim=256, h_kv=1, group=8, dsplit=True,
                                    layout=layout, decodes=layout == "flash",
                                    warps=16 if layout == "flash" else 12)
    rows = (list(range(got.shape[0])) if layout == "flash" else _prefill_rows(reqs, q_start))
    _attn_ref.check(got[rows], want[rows])


@rocm_only
@flash_dsplit_builds
def test_dsplit_v_ones_is_exactly_one():
    q, (k, v), _, table, lens, q_start = _rocm_case([(300, 200)], head_dim=256, h_kv=1, group=8,
                                                    v_ones=True)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    kern = pp.paged_prefill_kernel("t_pp_flash_d256_g8_ds_w16_dec", head_dim=256, group=8,
                                   layout="flash", dsplit=True, warps=16, decodes=True)
    fl = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=1)
    kv = fl.reshape(k.shape[0], 2, ROCM_PAGE, d).contiguous()
    kern(q.view(n_tok, h * d), kv, kv, table, lens, q_start, pp.q_tiles(200, q.device, 128),
         out.view(n_tok, h * d))
    assert torch.all(out == 1.0)


@rocm_only
def test_dsplit_rocm_v_ones_is_exactly_one():
    q, (k, v), (k5, v4), table, lens, q_start = _rocm_case([(300, 200)], head_dim=256, h_kv=1,
                                                           group=8, v_ones=True)
    n_tok, h, d = q.shape
    out = torch.full_like(q, 7.0)
    kern = pp.paged_prefill_kernel("t_pp_rocm_d256_g8_ds_w12", head_dim=256, group=8,
                                   layout="rocm", dsplit=True, warps=12)
    kern(q.view(n_tok, h * d), *pp.rocm_views(k5, v4), table, lens, q_start,
         pp.q_tiles(200, q.device, 96), out.view(n_tok, h * d))
    assert torch.all(out == 1.0)


@rocm_only
def test_flash_packed_decodes_with_a_strided_q():
    """Decodes GQA-packed (one workgroup a request and kv head) next to prefill chunks, with q a
    slice of a wider row as a fused qkv projection gives it."""
    reqs = [(500, 1), (2047, 1), (0, 300), (16, 1), (1000, 77), (33, 1)]
    q, (k, v), _, table, lens, q_start = _rocm_case(reqs, head_dim=128, h_kv=2, group=8)
    n_tok, h, d = q.shape
    wide = torch.randn(n_tok, h * d + 2 * 2 * d, device="cuda", dtype=torch.bfloat16)
    wide[:, : h * d] = q.reshape(n_tok, h * d)
    qs = wide[:, : h * d].view(n_tok, h, d)                   # row stride (h + 4) * d
    out = torch.full_like(q, 7.0)
    kern = pp.paged_prefill_kernel("t_pp_flash_d128_g8_w12_dec", head_dim=128, group=8,
                                   layout="flash", warps=12, decodes=True)
    fl = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=1)
    kv = fl.reshape(k.shape[0], 2, ROCM_PAGE, 2 * d).contiguous()
    kern(qs.reshape(n_tok, h * d), kv, kv, table, lens, q_start,
         pp.q_tiles(300, q.device, 192), out.view(n_tok, h * d), qs.unsqueeze(0),
         out.unsqueeze(0))
    want = _reference(q, k, v, table, lens, q_start, 8, page=ROCM_PAGE)
    _attn_ref.check(out, want)


@rocm_only
@pytest.mark.parametrize("reqs", [[(0, 300)], [(500, 1), (2047, 1), (0, 300), (1000, 77), (33, 1)]])
@pytest.mark.parametrize("page,kv_order", [(16, "planes"), (544, "tokens"), (784, "tokens")])
def test_flash_vt_pages_is_bitwise_the_flash_layout(reqs, page, kv_order):
    """vt_pages: the batch's V pages packed once into V^T slots (vt_pack_kernel) and staged
    straight -- the same V^T tile in LDS as the register transpose builds, so the same bits."""
    q, (k, v), _, table, lens, q_start = _rocm_case(reqs, head_dim=128, h_kv=2, group=8,
                                                    page=page)
    n_tok, h, d = q.shape
    kv = _flash_cache(k, v, kv_order)
    max_q = max(n for _, n in reqs)
    pt = ("" if page == 16 else f"_p{page}") + ("" if kv_order == "planes" else f"_{kv_order}")
    outs = []
    for vt in (False, True):
        out = torch.full_like(q, 7.0)
        kern = pp.paged_prefill_kernel(f"t_pp_flash_d128_g8_w12_dec{'_vt' if vt else ''}{pt}",
                                       head_dim=128, group=8, layout="flash", warps=12,
                                       decodes=True, vt_pages=vt, page=page, kv_order=kv_order)
        v_src = kv
        if vt:
            span = max(-(-s // page) for s in lens.tolist())
            # NaN where nothing is packed: a read of an unpacked V^T column would show
            v_src = torch.full((len(reqs) * span, 2, d, page), float("nan"), device="cuda",
                               dtype=torch.bfloat16)
            pp.vt_pack_kernel(f"t_vt_pack_d128{pt}", head_dim=128, page=page, kv_order=kv_order)(
                kv, table, lens, pp.vt_pack_span(max(lens.tolist()), q.device, page), v_src)
        kern(q.view(n_tok, h * d), kv, v_src, table, lens, q_start, pp.q_tiles(max_q, q.device, 192),
             out.view(n_tok, h * d), q.unsqueeze(0), out.unsqueeze(0))
        outs.append(out)
    assert torch.equal(outs[0], outs[1])
    _attn_ref.check(outs[1], _reference(q, k, v, table, lens, q_start, 8, page=page))


# ---- pages that are whole KV blocks (hybrid models: MiniCPM-V-4.7's are 544 tokens) ----------

#: requests that start, end and cross page and block boundaries of 544-token pages: a prefix
#: ending mid-page, a chunk crossing into the next page, one exactly a page, a decode
BIG_PAGE_REQS = [(1000, 77), (0, 300), (530, 40), (544, 544), (33, 1)]


@rocm_only
@pytest.mark.parametrize("page", [32, 64, 544])
def test_rocm_layout_pages_of_whole_blocks(page):
    got, want, q_start = _rocm_run(BIG_PAGE_REQS, page=page)
    rows = _prefill_rows(BIG_PAGE_REQS, q_start)
    _attn_ref.check(got[rows], want[rows])


@rocm_only
@pytest.mark.parametrize("page", [32, 544])
@pytest.mark.parametrize("kv_order", ["planes", "tokens", "heads"])
def test_flash_layout_pages_of_whole_blocks(page, kv_order):
    got, want, _ = _flash_run(BIG_PAGE_REQS, page=page, kv_order=kv_order)
    _attn_ref.check(got, want)


@rocm_only
@pytest.mark.parametrize("kv_order", ["tokens", "heads"])
def test_flash_kv_orders_on_pages_of_16(kv_order):
    got, want, _ = _flash_run([(0, 192), (500, 5), (40, 600), (33, 1)], kv_order=kv_order)
    _attn_ref.check(got, want)


@rocm_only
@pytest.mark.parametrize("layout", [pytest.param("flash", marks=flash_dsplit_builds), "rocm"])
def test_dsplit_pages_of_544(layout):
    """MiniCPM-V-4.7's full attention as vLLM lays it out at TP2: d256, one kv head a rank,
    544-token pages, a head's K and V side by side (its image's vLLM)."""
    got, want, q_start = _flash_run(BIG_PAGE_REQS, head_dim=256, h_kv=1, group=8, dsplit=True,
                                    layout=layout, decodes=layout == "flash",
                                    warps=16 if layout == "flash" else 12, page=544,
                                    kv_order="tokens" if layout == "flash" else "planes")
    rows = (list(range(got.shape[0])) if layout == "flash" else
            _prefill_rows(BIG_PAGE_REQS, q_start))
    _attn_ref.check(got[rows], want[rows])


@rocm_only
@flash_dsplit_builds
def test_dsplit_pages_of_544_v_ones_is_exactly_one():
    got, _, _ = _flash_run([(600, 500), (0, 1)], head_dim=256, h_kv=1, group=8, dsplit=True,
                           warps=16, page=544, v_ones=True, kv_order="tokens")
    assert torch.all(got == 1.0)


@rocm_only
@pytest.mark.parametrize("head_dim,kv_order", [
    (128, "planes"), (128, "tokens"),
    pytest.param(256, "tokens", marks=flash_dsplit_builds)])
def test_flash_pages_of_544_never_read_past_the_end(head_dim, kv_order):
    """A hybrid model's page past a sequence's end is another layer's bytes, NaN included: with
    every row no request has set to NaN, the output is still the reference's."""
    ds = head_dim > 128
    got, want, _ = _flash_run(BIG_PAGE_REQS, head_dim=head_dim, h_kv=1 if ds else 2, group=8,
                              dsplit=ds, warps=16 if ds else 12, page=544, kv_order=kv_order,
                              nan_tail=True)
    assert not torch.isnan(got).any()
    _attn_ref.check(got, want)


def test_a_page_not_a_multiple_of_16_is_refused():
    with pytest.raises(ValueError, match="page=40"):
        pp.paged_prefill_kernel("t_pp_flash_p40", layout="flash", page=40)


# ---- pages that are neither (Qwen3.8's 784 tokens = 16 x 49): a block's halves, two lookups ---

#: requests around 784-token page boundaries: a chunk across one, a prefix ending a half-block
#: before one, a sequence of exactly one page (its last block's second half past the page),
#: a decode at a page's first token
ODD_PAGE_REQS = [(1000, 77), (0, 300), (770, 30), (0, 784), (784, 1), (33, 1)]


@rocm_only
@pytest.mark.parametrize("page", [48, 400, 784])
@pytest.mark.parametrize("kv_order", ["planes", "tokens"])
def test_flash_layout_pages_of_half_blocks(page, kv_order):
    got, want, _ = _flash_run(ODD_PAGE_REQS, page=page, kv_order=kv_order)
    _attn_ref.check(got, want)


@rocm_only
@pytest.mark.parametrize("page", [48, 784])
def test_rocm_layout_pages_of_half_blocks(page):
    got, want, q_start = _rocm_run(ODD_PAGE_REQS, page=page)
    rows = _prefill_rows(ODD_PAGE_REQS, q_start)
    _attn_ref.check(got[rows], want[rows])


@rocm_only
@flash_dsplit_builds
@pytest.mark.parametrize("group", [6, 8])
def test_dsplit_pages_of_784(group):
    """Qwen3.8-27B's full attention: d256, 24 query heads over 4 kv heads (6 a group), 784-token
    pages, a head's K and V side by side."""
    got, want, _ = _flash_run(ODD_PAGE_REQS, head_dim=256, h_kv=2, group=group, dsplit=True,
                              warps=16, page=784, kv_order="tokens")
    _attn_ref.check(got, want)


@rocm_only
@pytest.mark.parametrize("case", [
    dict(), pytest.param(dict(head_dim=256, h_kv=1, dsplit=True, warps=16), marks=flash_dsplit_builds)])
def test_pages_of_784_v_ones_and_nan_tails(case):
    got, _, _ = _flash_run([(600, 500), (0, 784), (7, 1)], page=784, kv_order="tokens",
                           v_ones=True, **case)
    assert torch.all(got == 1.0)
    got, want, _ = _flash_run(ODD_PAGE_REQS, page=784, kv_order="tokens", nan_tail=True, **case)
    assert not torch.isnan(got).any()
    _attn_ref.check(got, want)


# ---- the route: vllm_prefill's TRITON_ATTN stand-in, without vLLM ---------------------------

def _route_arch() -> bool:
    from hk.integration.vllm_prefill import ARCHES
    from hk.target import current_arch

    return current_arch() in ARCHES


#: a mixed batch: a prompt, decodes, a chunk after a prefix -- one the guard leaves to vLLM
#: (decodes beside a small prefill), so its kernel case runs with the guard off
MIXED = [(0, 300), (500, 1), (1000, 77), (33, 1)]
#: a prefill-heavy batch with the same kinds of request, one the guard takes on every arch
HEAVY = [(0, 2048), (1000, 77), (33, 1)]


def _route_case(monkeypatch, reqs, kv_order, page, vt, orig):
    """vllm_prefill's unified_attention stand-in on `reqs`, vLLM's kernel replaced by `orig`."""
    import inspect

    from hk.integration import vllm_prefill as vp

    monkeypatch.setattr(vp, "_ORIG_U", orig)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(orig))
    monkeypatch.setattr(vp, "VT", vt)
    q, (k, v), _, table, lens, q_start = _rocm_case(reqs, head_dim=128, h_kv=2, group=8,
                                                    page=page)
    cache = _flash_cache(k, v, kv_order)
    nb, d = k.shape[0], k.shape[3]
    if kv_order == "planes":
        kc, vc = cache.view(nb, 2, page, 2, d).unbind(1)
    else:
        both = cache.view(nb, page, 2, 2 * d)
        kc, vc = both[..., :d], both[..., d:]
    out = torch.full_like(q, 7.0)
    vp.unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=q_start,
                         max_seqlen_q=max(n for _, n in reqs), seqused_k=lens,
                         max_seqlen_k=int(lens.max()), softmax_scale=d ** -0.5, causal=True,
                         window_size=(-1, -1), block_table=table, softcap=0, q_descale=None,
                         k_descale=None, v_descale=None)
    return out, _reference(q, k, v, table, lens, q_start, 8, page=page)


def _vllm_must_not_run(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                       softmax_scale, causal, window_size, block_table, softcap, q_descale,
                       k_descale, v_descale, alibi_slopes=None, sinks=None):
    raise AssertionError("fell back to vLLM's kernel")


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("vt", [False, True])
@pytest.mark.parametrize("kv_order,page", [("planes", 16), ("tokens", 16), ("tokens", 544),
                                           ("tokens", 784)])
@pytest.mark.parametrize("reqs,guard", [(MIXED, False), (HEAVY, True)], ids=["mixed", "heavy"])
def test_vllm_prefill_takes_a_triton_attn_batch_and_matches(monkeypatch, kv_order, page, vt,
                                                            reqs, guard):
    """The route (vllm_prefill): unified_attention's stand-in, called as TRITON_ATTN calls it
    on a batch of prompts, chunks after a prefix and decodes, takes it on hk's kernel -- counted
    as hk, vLLM's kernel never called -- and matches the reference. Both of vLLM's cache orders,
    with and without the V^T pre-pack; vLLM itself is not needed (its signature is the
    stand-in's own). A prefill-heavy batch with the guard on, as served; a mixed one, which the
    guard leaves to vLLM, with it off (HK_VLLM_PREFILL_GUARD=0), so the kernel's decode rows
    stay tested."""
    from hk.integration import vllm_prefill as vp

    monkeypatch.setattr(vp, "GUARD", guard)
    before = vp.PATCH.stats["hk"]
    out, want = _route_case(monkeypatch, reqs, kv_order, page, vt, _vllm_must_not_run)
    assert vp.PATCH.stats["hk"] == before + 1
    _attn_ref.check(out, want)


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
def test_vllm_prefill_leaves_a_mixed_batch_to_vllm(monkeypatch):
    """With the guard on (the default), the mixed batch -- decodes beside a small prefill, which
    hk's kernel is slower on -- goes to vLLM's kernel with the caller's own tensors, counted as
    declined: its output is vLLM's, whatever hk's kernel would have written."""
    from hk.integration import vllm_prefill as vp

    seen = []

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, alibi_slopes=None, sinks=None):
        seen.append((q, out))
        out.fill_(3.0)

    monkeypatch.setattr(vp, "GUARD", True)
    before = dict(vp.PATCH.stats)
    out, _ = _route_case(monkeypatch, MIXED, "planes", 16, False, unified_attention)
    assert vp.PATCH.stats["declined"] == before["declined"] + 1
    assert vp.PATCH.stats["hk"] == before["hk"]
    assert len(seen) == 1 and seen[0][1] is out
    assert torch.all(out == 3.0)


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("vt", [False, True])
@pytest.mark.parametrize("head_dim,h_kv,kv_order,page", [(128, 2, "planes", 16),
                                                         (128, 2, "tokens", 16),
                                                         (256, 1, "tokens", 544)])
def test_vllm_prefill_nan_poisoned_repeats_and_a_side_stream_producer(monkeypatch, head_dim, h_kv,
                                                                      kv_order, page, vt):
    """The default-on rule's repeat and stream checks, on the route as vLLM calls it: 200 calls
    with the output set to NaN before each, every one bitwise the first (nothing left unwritten
    or read stale, and the result is deterministic); then the query and the cache written by a
    producer on a torch stream of its own, the route called after torch's stream waits for it:
    bitwise the same again."""
    import inspect

    from hk.integration import vllm_prefill as vp

    if head_dim == 256 and not _clang_23():
        pytest.skip("flash dsplit spills under clang 22: the route leaves d256 to vLLM there")

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                          softmax_scale, causal, window_size, block_table, softcap, q_descale,
                          k_descale, v_descale, alibi_slopes=None, sinks=None):
        raise AssertionError("fell back to vLLM's kernel")

    monkeypatch.setattr(vp, "_ORIG_U", unified_attention)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(unified_attention))
    monkeypatch.setattr(vp, "VT", vt)
    monkeypatch.setattr(vp, "GUARD", False)  # the mixed batch, so the kernel's decode rows run too
    reqs = MIXED
    q, (k, v), _, table, lens, q_start = _rocm_case(reqs, head_dim=head_dim, h_kv=h_kv, group=8,
                                                    page=page, seed=5)
    cache = _flash_cache(k, v, kv_order)
    nb, d = k.shape[0], k.shape[3]

    def run(q_in, cache_in, out):
        if kv_order == "planes":
            kc, vc = cache_in.view(nb, 2, page, h_kv, d).unbind(1)
        else:
            both = cache_in.view(nb, page, h_kv, 2 * d)
            kc, vc = both[..., :d], both[..., d:]
        vp.unified_attention(q=q_in, k=kc, v=vc, out=out, cu_seqlens_q=q_start, max_seqlen_q=300,
                             seqused_k=lens, max_seqlen_k=int(lens.max()),
                             softmax_scale=d ** -0.5, causal=True, window_size=(-1, -1),
                             block_table=table, softcap=0, q_descale=None, k_descale=None,
                             v_descale=None)

    before = vp.PATCH.stats["hk"]
    out = torch.empty_like(q)
    first = None
    for i in range(200):
        out.fill_(float("nan"))
        run(q, cache, out)
        if first is None:
            torch.cuda.synchronize()
            first = out.clone()
            _attn_ref.check(first, _reference(q, k, v, table, lens, q_start, 8, page=page))
        else:
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), f"call {i}"
    assert vp.PATCH.stats["hk"] == before + 200

    q_host, cache_host = q.cpu(), cache.cpu()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        q2 = q_host.to("cuda", non_blocking=True)
        cache2 = cache_host.to("cuda", non_blocking=True)
        q2.mul_(1.0)
        cache2.mul_(1.0)
        out2 = torch.full_like(q2, float("nan"))
    torch.cuda.current_stream().wait_stream(side)
    run(q2, cache2, out2)
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), first.view(torch.int16))


def _caf_must_not_run(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
                      max_seq_len, max_input_len, k_scale, v_scale, alibi_slopes=None,
                      sliding_window=None, sm_scale=None, skip_decode=False, fp8_out_scale=None,
                      sinks=None, is_block_table_ptr=False, causal=True):
    raise AssertionError("fell back to vLLM's kernel")


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("head_dim,h_kv,page", [(128, 2, 16), (128, 2, 32), (256, 1, 16)])
def test_vllm_prefill_rocm_attn_nan_poisoned_repeats_and_a_side_stream_producer(
        monkeypatch, head_dim, h_kv, page):
    """The default-on rule's repeat and stream checks on ROCM_ATTN's entry point, as
    chunked_prefill_paged_decode calls it (skip_decode: its own kernel writes the decode rows):
    200 calls with the output set to NaN before each, every one taken by hk and bitwise the
    first, the first matching the reference on the prefill rows and leaving the decode rows
    untouched; then the query and the caches written by a producer on a torch stream of its own,
    the route called after torch's stream waits for it: bitwise the same again."""
    import inspect

    from hk.integration import vllm_prefill as vp

    if head_dim == 256 and not _clang_23():
        pytest.skip("rocm dsplit spills under clang 22: the route leaves d256 to vLLM there")
    monkeypatch.setattr(vp, "_ORIG", _caf_must_not_run)
    monkeypatch.setattr(vp, "_P_SIG", inspect.signature(_caf_must_not_run))
    reqs = MIXED
    q, (k, v), (k5, v4), table, lens, q_start = _rocm_case(reqs, head_dim=head_dim, h_kv=h_kv,
                                                           page=page, seed=5)
    rows = _prefill_rows(reqs, q_start)
    dead = sorted(set(range(q.shape[0])) - set(rows))
    max_q = max(n for _, n in reqs)

    def run(q_in, k_in, v_in, out):
        vp.context_attention_fwd(q=q_in, k=None, v=None, o=out, kv_cache_dtype="auto",
                                 k_cache=k_in, v_cache=v_in, b_loc=table, b_start_loc=q_start,
                                 b_seq_len=lens, max_seq_len=int(lens.max()),
                                 max_input_len=max_q, k_scale=None, v_scale=None,
                                 sm_scale=head_dim ** -0.5, skip_decode=True)

    before = vp.PATCH.stats["hk"]
    out = torch.empty_like(q)
    first = None
    for i in range(200):
        out.fill_(float("nan"))
        run(q, k5, v4, out)
        if first is None:
            torch.cuda.synchronize()
            first = out.clone()
            want = _reference(q, k, v, table, lens, q_start, 8, page=page)
            _attn_ref.check(first[rows], want[rows])
            assert torch.isnan(first[dead]).all()
        else:
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), f"call {i}"
    assert vp.PATCH.stats["hk"] == before + 200

    hosts = [t.cpu() for t in (q, k5, v4)]
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        q2, k2, v2 = (t.to("cuda", non_blocking=True) for t in hosts)
        for t in (q2, k2, v2):
            t.mul_(1.0)
        out2 = torch.full_like(q2, float("nan"))
    torch.cuda.current_stream().wait_stream(side)
    run(q2, k2, v2, out2)
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), first.view(torch.int16))


def _unified_ref(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                 softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale,
                 v_descale, alibi_slopes=None, sinks=None):
    """vLLM's unified_attention for these tests, in float32 torch over TRITON_ATTN's cache views
    (blocks, page, kv_heads, head_dim): every request's queries, the last ones of its context."""
    _, page, hkv, d = k.shape
    group = q.shape[1] // hkv
    cu, used = cu_seqlens_q.tolist(), seqused_k.tolist()
    for i, n_k in enumerate(used):
        q0, q1 = cu[i], cu[i + 1]
        if q1 == q0:
            continue
        pages = block_table[i, : -(-n_k // page)].long()
        kk = k[pages].reshape(-1, hkv, d)[:n_k].float().repeat_interleave(group, dim=1)
        vv = v[pages].reshape(-1, hkv, d)[:n_k].float().repeat_interleave(group, dim=1)
        s = torch.einsum("qhd,khd->hqk", q[q0:q1].float(), kk) * softmax_scale
        pos = torch.arange(n_k - (q1 - q0), n_k, device=q.device)
        s.masked_fill_(torch.arange(n_k, device=q.device)[None, :] > pos[:, None], float("-inf"))
        out[q0:q1] = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv).to(out.dtype)


#: a prompt and a chunk after a prefix, beside decodes over 8k tokens: the guard declines it whole
#: for the decodes on gfx1100 and gfx1201 (exact sizes), its prefill rows alone it takes
SPLIT_REQS = [(8191, 1), (0, 2048), (8191, 1), (1000, 1100), (8191, 1), (8191, 1)]


def _split_case(monkeypatch, kv_order, page, vt, head_dim=128, h_kv=2, reqs=SPLIT_REQS, group=8):
    """vllm_prefill's unified_attention on `reqs` with the step's exact sizes given (as the
    TRITON_ATTN builder hook gives them), vLLM's kernel being _unified_ref. Returns run(q, cache,
    out) and the case."""
    import inspect

    from hk.integration import vllm_prefill as vp

    monkeypatch.setattr(vp, "_ORIG_U", _unified_ref)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(_unified_ref))
    monkeypatch.setattr(vp, "VT", vt)
    monkeypatch.setattr(vp, "GUARD", True)
    q, (k, v), _, table, lens, q_start = _rocm_case(reqs, head_dim=head_dim, h_kv=h_kv,
                                                    group=group, page=page, seed=7)
    cache = _flash_cache(k, v, kv_order)
    nb, d = k.shape[0], k.shape[3]
    monkeypatch.setattr(vp, "_STATS", {id(q_start): vp.batch_stats(
        q_start.tolist(), lens.tolist(), len(reqs))})

    def views(c):
        if kv_order == "planes":
            return c.view(nb, 2, page, h_kv, d).unbind(1)
        if kv_order == "heads":
            both = c.view(nb, h_kv, page, 2 * d).transpose(1, 2)
            return both[..., :d], both[..., d:]
        both = c.view(nb, page, h_kv, 2 * d)
        return both[..., :d], both[..., d:]

    def run(q_in, c, out):
        kc, vc = views(c)
        vp.unified_attention(q=q_in, k=kc, v=vc, out=out, cu_seqlens_q=q_start,
                             max_seqlen_q=max(n for _, n in reqs), seqused_k=lens,
                             max_seqlen_k=int(lens.max()), softmax_scale=d ** -0.5, causal=True,
                             window_size=(-1, -1), block_table=table, softcap=0, q_descale=None,
                             k_descale=None, v_descale=None)
    return run, (q, k, v, cache, table, lens, q_start, views)


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("vt", [False, True])
@pytest.mark.parametrize("head_dim,h_kv,kv_order,page", [(128, 2, "planes", 16),
                                                         (128, 2, "tokens", 544),
                                                         (256, 1, "tokens", 544)])
def test_vllm_prefill_splits_a_mixed_batch(monkeypatch, head_dim, h_kv, kv_order, page, vt):
    """A mixed batch the guard declines for its decodes, with the step's exact sizes: hk's kernel
    takes the prefill rows (matching the reference) and vLLM's kernel the decode rows -- bitwise
    its own answer on the batch as one query a request. Counted as hk and as split."""
    from hk.integration import vllm_prefill as vp

    if head_dim == 256 and not _clang_23():
        pytest.skip("flash dsplit spills under clang 22: the route leaves d256 to vLLM there")
    run, (q, k, v, cache, table, lens, q_start, views) = _split_case(
        monkeypatch, kv_order, page, vt, head_dim, h_kv)
    assert vp._guard({"q": q, "k": views(cache)[0], "seqused_k": lens, "cu_seqlens_q": q_start})
    before = dict(vp.PATCH.stats)
    out = torch.full_like(q, float("nan"))
    run(q, cache, out)
    torch.cuda.synchronize()
    assert vp.PATCH.stats["split"] == before["split"] + 1
    assert vp.PATCH.stats["hk"] == before["hk"] + 1
    rows = _prefill_rows(SPLIT_REQS, q_start)
    want = _reference(q, k, v, table, lens, q_start, 8, page=page)
    _attn_ref.check(out[rows], want[rows])
    first = q_start[:-1].long()
    dec = [i for i, (_, n) in enumerate(SPLIT_REQS) if n == 1]
    kc, vc = views(cache)
    ref1 = torch.empty(len(SPLIT_REQS), *q.shape[1:], device=q.device, dtype=q.dtype)
    used1 = torch.where(q_start[1:] - q_start[:-1] == 1, lens, torch.ones_like(lens))
    _unified_ref(q[first], kc, vc, ref1, torch.arange(len(SPLIT_REQS) + 1, device=q.device,
                                                      dtype=torch.int32), 1, used1,
                 int(lens.max()), head_dim ** -0.5, True, (-1, -1), table, 0, None, None, None)
    got_dec = out[first[dec]]
    assert torch.equal(got_dec.view(torch.int16), ref1[dec].view(torch.int16))
    _attn_ref.check(got_dec, want[first[dec]])


#: a short chunk over a long context beside decodes: what a packed KV split takes
PACKED_DEC_REQS = [(8191, 1), (6000, 32), (4000, 1), (8191, 1)]


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("head_dim,h_kv,kv_order,page", [(128, 2, "planes", 16),
                                                         (256, 1, "tokens", 544)])
def test_vllm_prefill_packed_kv_split_beside_decodes(monkeypatch, head_dim, h_kv, kv_order, page):
    """A packed KV-split plan beside decodes (KVS_PACKED, KVS_PACK_MAX_DECODES): the packed split
    takes the prefill rows (matching the reference), vLLM's kernel the decode rows -- bitwise its own
    answer on them as one query a request. Counted as hk, split and kvsplit."""
    from hk.integration import switches, vllm_prefill as vp

    if head_dim == 256 and not _clang_23():
        pytest.skip("flash dsplit spills under clang 22: the route leaves d256 to vLLM there")
    arch, cls = switches.arch(), 256 if head_dim > 128 else 128
    monkeypatch.setattr(vp, "KVS_PACKED", {(arch, cls, 8): ((64, 1024, 4),)})
    monkeypatch.setattr(vp, "KVS_PACK_MAX_DECODES", {})
    monkeypatch.setattr(vp, "KVS_PACK_DEC_RATIO", 1e9)
    run, (q, k, v, cache, table, lens, q_start, views) = _split_case(
        monkeypatch, kv_order, page, False, head_dim, h_kv, reqs=PACKED_DEC_REQS)
    a = {"q": q, "k": views(cache)[0], "seqused_k": lens, "cu_seqlens_q": q_start}
    assert vp.kvsplit_plan(a) == (4, vp.kvs_pack_warps(head_dim, 8), True)
    before = dict(vp.PATCH.stats)
    out = torch.full_like(q, float("nan"))
    run(q, cache, out)
    torch.cuda.synchronize()
    for c in ("hk", "split", "kvsplit"):
        assert vp.PATCH.stats[c] == before[c] + 1, c
    rows = _prefill_rows(PACKED_DEC_REQS, q_start)
    want = _reference(q, k, v, table, lens, q_start, 8, page=page)
    _attn_ref.check(out[rows], want[rows])
    first = q_start[:-1].long()
    dec = [i for i, (_, n) in enumerate(PACKED_DEC_REQS) if n == 1]
    kc, vc = views(cache)
    ref1 = torch.empty(len(PACKED_DEC_REQS), *q.shape[1:], device=q.device, dtype=q.dtype)
    used1 = torch.where(q_start[1:] - q_start[:-1] == 1, lens, torch.ones_like(lens))
    _unified_ref(q[first], kc, vc, ref1, torch.arange(len(PACKED_DEC_REQS) + 1, device=q.device,
                                                      dtype=torch.int32), 1, used1,
                 int(lens.max()), head_dim ** -0.5, True, (-1, -1), table, 0, None, None, None)
    assert torch.equal(out[first[dec]].view(torch.int16), ref1[dec].view(torch.int16))


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
def test_vllm_prefill_split_off_declines_the_batch_whole(monkeypatch):
    """HK_VLLM_PREFILL_SPLIT=0: the same batch goes to vLLM's kernel whole, counted declined."""
    from hk.integration import vllm_prefill as vp

    run, (q, *_rest) = _split_case(monkeypatch, "planes", 16, False)
    monkeypatch.setattr(vp, "SPLIT", False)
    before = dict(vp.PATCH.stats)
    out = torch.empty_like(q)
    run(q, _rest[2], out)
    assert vp.PATCH.stats["declined"] == before["declined"] + 1
    assert vp.PATCH.stats["hk"] == before["hk"] and vp.PATCH.stats["split"] == before["split"]


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
@pytest.mark.parametrize("vt", [False, True])
def test_vllm_prefill_split_nan_poisoned_repeats_and_a_side_stream_producer(monkeypatch, vt):
    """The default-on rule's repeat and stream checks on the split path: 200 calls over a
    NaN-filled output, each split and bitwise the first; then the query and the cache from a
    producer on its own stream, bitwise the same."""
    from hk.integration import vllm_prefill as vp

    run, (q, k, v, cache, table, lens, q_start, views) = _split_case(monkeypatch, "planes", 16,
                                                                     vt)
    before = vp.PATCH.stats["split"]
    out = torch.empty_like(q)
    first = None
    for i in range(200):
        out.fill_(float("nan"))
        run(q, cache, out)
        if first is None:
            torch.cuda.synchronize()
            first = out.clone()
            assert not torch.isnan(first).any()
        else:
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), f"call {i}"
    assert vp.PATCH.stats["split"] == before + 200
    q_host, cache_host = q.cpu(), cache.cpu()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        q2 = q_host.to("cuda", non_blocking=True)
        c2 = cache_host.to("cuda", non_blocking=True)
        q2.mul_(1.0)
        c2.mul_(1.0)
        out2 = torch.full_like(q2, float("nan"))
    torch.cuda.current_stream().wait_stream(side)
    run(q2, c2, out2)
    torch.cuda.synchronize()
    assert torch.equal(out2.view(torch.int16), first.view(torch.int16))


#: short chunks over long contexts (what kv_splits is for), a chunk longer than the tile, a split
#: with no key some rows see, a decode (skipped: decodes=False) and a padded request
KV_SPLIT_REQS = [(4000, 32), (3000, 1), (900, 150), (0, 70), (63, 2), (5, 0)]


@rocm_only
@pytest.mark.parametrize("kv_splits", [2, 5, 16])
@pytest.mark.parametrize("page", [16, 544])
def test_flash_kv_splits(kv_splits, page):
    got, want, q_start = _flash_run(KV_SPLIT_REQS, decodes=False, warps=4, page=page,
                                    kv_splits=kv_splits, nan_tail=page == 544)
    rows = _prefill_rows(KV_SPLIT_REQS, q_start)
    assert not torch.isnan(got[rows]).any()
    _attn_ref.check(got[rows], want[rows])
    rest = sorted(set(range(got.shape[0])) - set(rows))
    assert torch.all(got[rest] == 7.0)


@rocm_only
@flash_dsplit_builds
@pytest.mark.parametrize("kv_splits", [3, 8])
def test_flash_kv_splits_dsplit(kv_splits):
    got, want, q_start = _flash_run(KV_SPLIT_REQS, head_dim=256, h_kv=1, group=8, dsplit=True,
                                    decodes=False, warps=8, page=544, kv_order="tokens",
                                    kv_splits=kv_splits)
    rows = _prefill_rows(KV_SPLIT_REQS, q_start)
    _attn_ref.check(got[rows], want[rows])


#: GQA-packed tiles (pack=True): a tile the kv head's query heads of a run of tokens -- with and
#: without the KV split, tiles of 16 and 32 tokens
@rocm_only
@pytest.mark.parametrize("kv_splits", [1, 4, 16])
@pytest.mark.parametrize("page,group,warps", [(16, 8, 8), (16, 4, 8), (544, 8, 16)])
def test_flash_packed(kv_splits, page, group, warps):
    got, want, q_start = _flash_run(KV_SPLIT_REQS, group=group, decodes=False, warps=warps,
                                    page=page, kv_splits=kv_splits, nan_tail=page == 544,
                                    pack=True)
    rows = _prefill_rows(KV_SPLIT_REQS, q_start)
    assert not torch.isnan(got[rows]).any()
    _attn_ref.check(got[rows], want[rows])
    rest = sorted(set(range(got.shape[0])) - set(rows))
    assert torch.all(got[rest] == 7.0)


@rocm_only
@flash_dsplit_builds
@pytest.mark.parametrize("kv_splits", [1, 8])
@pytest.mark.parametrize("group,warps,page,kv_order", [(8, 16, 544, "tokens"),
                                                       (6, 12, 784, "heads")])
def test_flash_packed_dsplit(kv_splits, group, warps, page, kv_order):
    got, want, q_start = _flash_run(KV_SPLIT_REQS, head_dim=256, h_kv=1, group=group,
                                    dsplit=True, decodes=False, warps=warps, page=page,
                                    kv_order=kv_order, kv_splits=kv_splits, pack=True)
    rows = _prefill_rows(KV_SPLIT_REQS, q_start)
    _attn_ref.check(got[rows], want[rows])


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
def test_a_cache_past_4_gib_goes_to_vllm(monkeypatch):
    """hk's buffer loads address 4 GiB from a tensor's base (vllm_prefill.SPAN_LIMIT): over a
    TRITON_ATTN cache of Qwen3.8's shape past that (4.3 GB, the requests' pages at its far end) a
    decode batch and a prefill chunk beside a decode go to vLLM's kernel -- its answer exactly,
    hk's kernels not run (they read the wrong pages there, or fault)."""
    import inspect

    from hk.integration import vllm_prefill as vp

    free, _ = torch.cuda.mem_get_info()
    if free < (6 << 30):
        pytest.skip("needs 6 GB of free device memory")
    hkv, group, d, page = 4, 6, 256, 16
    nb = (1 << 32) // (2 * page * hkv * d * 2) + 64
    st = torch.zeros(nb, 2, page, hkv, d, device="cuda", dtype=torch.bfloat16)
    k, v = st[:, 0], st[:, 1]
    assert vp.span_why_not(vp._flash_kv(k, v)[0])
    g = torch.Generator(device="cuda").manual_seed(5)
    monkeypatch.setattr(vp, "_ORIG_U", _unified_ref)
    monkeypatch.setattr(vp, "_U_SIG", inspect.signature(_unified_ref))
    for reqs in ([(4000, 1), (3000, 1)], [(4000, 64), (3000, 1)]):
        pages = [-(-(c + n) // page) for c, n in reqs]
        table = torch.zeros(len(reqs), max(pages), dtype=torch.int32, device="cuda")
        last = nb
        for i, p in enumerate(pages):                 # the far end of the cache, past 4 GiB
            table[i, :p] = torch.arange(last - p, last, dtype=torch.int32, device="cuda")
            last -= p
        used = table[table > 0].long().unique()
        st[used] = torch.randn(st[used].shape, device="cuda", dtype=torch.bfloat16, generator=g)
        n_tok = sum(n for _, n in reqs)
        q = torch.randn(n_tok, hkv * group, d, device="cuda", dtype=torch.bfloat16, generator=g)
        cu = torch.tensor([0] + [sum(n for _, n in reqs[: i + 1]) for i in range(len(reqs))],
                          dtype=torch.int32, device="cuda")
        lens = torch.tensor([c + n for c, n in reqs], dtype=torch.int32, device="cuda")
        args = dict(q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=max(n for _, n in reqs),
                    seqused_k=lens, max_seqlen_k=int(lens.max()), softmax_scale=d ** -0.5,
                    causal=True, window_size=(-1, -1), block_table=table, softcap=0,
                    q_descale=None, k_descale=None, v_descale=None)
        want = torch.empty_like(q)
        _unified_ref(out=want, **args)
        before = vp.PATCH.stats["hk"] + vp.PATCH.stats["decode"]
        out = torch.full_like(q, float("nan"))
        vp.unified_attention(out=out, **args)
        torch.cuda.synchronize()
        assert vp.PATCH.stats["hk"] + vp.PATCH.stats["decode"] == before, reqs
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), reqs


#: Qwen3.8's shape: a 64-token chunk beside decodes whose work (0.08 of the chunk's) KVS_DEC_RATIO
#: declined; KVS_DEC_MAX_TOKENS takes it
QWEN_DEC_REQS = [(8000, 64)] + [(6000, 1)] * 7


@pytest.mark.skipif(not _route_arch(), reason="the route is gfx1100's and gfx1201's")
def test_vllm_prefill_kv_split_beside_decodes_by_their_tokens(monkeypatch):
    """The unpacked KV split beside decodes under KVS_DEC_MAX_TOKENS, on Qwen3.8's cache (d256,
    4 kv heads, group 6, 784-token pages, head-major): the split takes the prefill rows (matching
    the reference), vLLM's kernel the decode rows (matching it too). Counted as hk, split and kvsplit."""
    from hk.integration import switches, vllm_prefill as vp

    if not _clang_23():
        pytest.skip("flash dsplit spills under clang 22: the route leaves d256 to vLLM there")
    monkeypatch.setattr(vp, "KVS_PACKED", {})
    monkeypatch.setattr(vp, "KVS_DEC_MAX_TOKENS", {(switches.arch(), 256, 6): 1 << 19})
    run, (q, k, v, cache, table, lens, q_start, views) = _split_case(
        monkeypatch, "heads", 784, False, 256, 4, reqs=QWEN_DEC_REQS, group=6)
    a = {"q": q, "k": views(cache)[0], "seqused_k": lens, "cu_seqlens_q": q_start}
    plan = vp.kvsplit_plan(a)
    assert plan is not None and not plan[2]
    before = dict(vp.PATCH.stats)
    out = torch.full_like(q, float("nan"))
    run(q, cache, out)
    torch.cuda.synchronize()
    for c in ("hk", "split", "kvsplit"):
        assert vp.PATCH.stats[c] == before[c] + 1, c
    rows = _prefill_rows(QWEN_DEC_REQS, q_start)
    want = _reference(q, k, v, table, lens, q_start, 6, page=784)
    _attn_ref.check(out[rows], want[rows])
    first = q_start[:-1].long()
    dec = [i for i, (_, n) in enumerate(QWEN_DEC_REQS) if n == 1]
    # vLLM's kernel's answer on the decode rows (its float32 stand-in here is not bit-reproducible
    # call to call at d256, so against the reference, not bitwise against a second call)
    _attn_ref.check(out[first[dec]], want[first[dec]])
