"""hk.ops.paged refuses, before any launch, a decode whose tensors do not have the shapes its
kernels read (`check_decode`). Each wrong layout below used to be launched as it was. The kernel
then read or wrote past an allocation. The first case is the one that faulted a W7900D: the
unsplit kernel given the fused walk's 4-row query read 12 rows past it.

The tensors live on the CPU: the check reads shapes and dtypes only, and a refused call never
reaches a kernel. No GPU, no compile.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import paged  # noqa: E402

P = paged.KV_BLOCK
T = paged.Q_TILE


def _case(reqs=4, heads=2, rows=T, d=128, blocks=16, pages=4, bf=torch.bfloat16):
    q = torch.zeros(reqs, heads, rows, d, dtype=bf)
    k = torch.zeros(blocks, heads, P, d, dtype=bf)
    return dict(q=q, k_cache=k, v_cache=torch.zeros_like(k),
                block_table=torch.zeros(reqs, pages, dtype=torch.int32),
                seq_lens=torch.ones(reqs, dtype=torch.int32), out=torch.zeros_like(q))


def _plain(c):
    paged.check_decode(c["q"], c["k_cache"], c["v_cache"], c["block_table"], c["seq_lens"],
                       c["out"], rows="tile", head_dim=128)


def _fused(c, scratch=None, splits=1):
    paged.check_decode(c["q"], c["k_cache"], c["v_cache"], c["block_table"], c["seq_lens"],
                       c["out"], scratch=scratch, splits=splits)


def test_the_contract_passes_what_the_kernels_take():
    _plain(_case())
    for rows in (1, 4, T):
        c = _case(rows=rows)
        _fused(c, paged.fused_scratch(4 * 8, 2, 128, "cpu", group=rows), splits=8)


def test_the_unsplit_kernel_refuses_the_fused_walks_4_row_query():
    # what faulted: q and out (requests, kv_heads, 4, head_dim) into paged.KERNELS[128]
    with pytest.raises(ValueError, match=r"q is \(16, 8, 4, 128\).*16-row tile"):
        _plain(_case(reqs=16, heads=8, rows=4))


@pytest.mark.parametrize("bad, match", [
    (dict(rows=T + 4), r"q .* 16-row tile"),
    (dict(d=64), r"q .* head_dim 128"),
])
def test_query_shapes(bad, match):
    with pytest.raises(ValueError, match=match):
        _plain(_case(**bad))


def test_the_fused_walk_refuses_more_rows_than_a_tile():
    with pytest.raises(ValueError, match="1 to 16 query rows"):
        _fused(_case(rows=T + 1))


def test_out_must_be_qs_shape():
    c = _case()
    c["out"] = torch.zeros(4, 2, 4, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out is"):
        _plain(c)


@pytest.mark.parametrize("name, shape", [
    ("k_cache", (16, 1, P, 128)),          # fewer kv heads than q: a head past the cache
    ("v_cache", (16, 2, 16, 128)),         # 16-token pages: a page's second half past the block
    ("k_cache", (16, 2, P, 64)),           # head_dim 64 cache under a d128 query
])
def test_cache_pages_must_match_q(name, shape):
    c = _case()
    c[name] = torch.zeros(*shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=name):
        _fused(c)


def test_block_table_and_seq_lens_cover_every_request():
    c = _case(reqs=4)
    c["seq_lens"] = torch.ones(3, dtype=torch.int32)
    with pytest.raises(ValueError, match="seq_lens"):
        _fused(c)
    c = _case(reqs=4)
    c["block_table"] = torch.zeros(2, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="block_table"):
        _fused(c)
    c = _case()
    c["block_table"] = torch.zeros(4, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="int32"):
        _fused(c)


def test_the_fused_scratch_holds_a_row_a_request_and_split():
    c = _case(reqs=4, rows=4)
    small = paged.fused_scratch(4 * 4, 2, 128, "cpu", group=4)        # planned for 4 splits
    with pytest.raises(ValueError, match="o_part"):
        _fused(c, small, splits=8)
    o_part, ml = paged.fused_scratch(4 * 8, 2, 128, "cpu", group=4)
    with pytest.raises(ValueError, match="ml_part"):
        _fused(c, (o_part, ml[:, :, :1]), splits=8)
    other = paged.fused_scratch(4 * 8, 2, 128, "cpu", group=8)        # another group's rows
    with pytest.raises(ValueError, match="o_part"):
        _fused(c, other, splits=8)


def test_kernels_and_decode_check_before_launching(monkeypatch):
    """The entry points run the check: paged.KERNELS[d](...) and paged.decode(...). A refused
    call reaches neither the kernel nor the walk."""
    monkeypatch.setattr(paged, "_native_cdna", lambda: False)      # RDNA's entry points
    monkeypatch.setattr(paged._arch, "require", lambda *a, **kw: None)
    c = _case(reqs=16, heads=8, rows=4)
    launched = []
    k = paged.KERNELS[128]
    assert k.name == "paged_decode_d128"                            # the kernel's, through
    monkeypatch.setattr(k, "kernel", lambda *a, **kw: launched.append(a))
    with pytest.raises(ValueError, match="16-row tile"):
        k(c["q"], c["k_cache"], c["v_cache"], c["block_table"], c["seq_lens"], c["out"])
    walk = lambda *a: launched.append(a)  # noqa: E731
    with pytest.raises(ValueError, match="o_part"):
        paged.decode(walk, walk, c["q"], c["k_cache"], c["v_cache"], c["block_table"],
                     c["seq_lens"], c["out"], paged.fused_scratch(4, 8, 128, "cpu", group=4), 8)
    assert launched == []
    # and a call that passes goes through, once checked
    ok = paged.fused_scratch(16 * 8, 8, 128, "cpu", group=4)
    paged.decode(walk, walk, c["q"], c["k_cache"], c["v_cache"], c["block_table"],
                 c["seq_lens"], c["out"], ok, 8)
    assert len(launched) == 2
