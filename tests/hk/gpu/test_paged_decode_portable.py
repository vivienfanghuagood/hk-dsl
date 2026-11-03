"""hk.ops.paged_decode_portable -- one decode source for RDNA and gfx950 -- against float64.

The same cases on every target: TRITON_ATTN's layouts, pages of 16 to 2064 tokens (on gfx950,
whole 32-key blocks: 32, 64, 544), hybrid models' pages with NaN past the end, head_dim 128 and
256, split and unsplit, and V all ones giving exactly 1.0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_paged_decode import _case, _check, _flash, _reference  # noqa: E402

from hk.ops import paged_decode as pd  # noqa: E402
from hk.ops import paged_decode_portable as pdp  # noqa: E402

pytestmark = pytest.mark.gpu

def _gfx950() -> bool:
    from hk.target import current_arch

    return current_arch() == "gfx950"


#: gfx950 reads a block of V as one 32-row LDS subtile, so there a page is a whole number of
#: 32-key blocks; pages of 16 (and 784, 2064, whose blocks straddle pages) are RDNA's here
half_bands = pytest.mark.skipif(_gfx950(), reason="gfx950: pages of whole 32-key blocks")

CASES = [
    # head_dim, h_kv, group, page, kv_order, nan_tail
    pytest.param(128, 2, 8, 16, "planes", False, marks=half_bands),      # MiniCPM5-2B
    (128, 2, 8, 32, "planes", False),                                     # ... at --block-size 32
    pytest.param(128, 2, 6, 2064, "heads", True, marks=half_bands),      # Qwen3.6
    (128, 2, 8, 544, "heads", True),                                      # MiniCPM-V-4.7's pages
    pytest.param(256, 4, 6, 784, "heads", True, marks=half_bands),       # Qwen3.8-27B
    (256, 4, 6, 544, "heads", True),
    pytest.param(256, 2, 6, 16, "planes", False, marks=half_bands),
    (256, 2, 6, 64, "tokens", False),
]
LENS = [1, 15, 16, 33, 1000, 2063, 4100]


def _refused_by_this_compiler(head_dim, group) -> bool:
    """gfx1201 under clang 22 (ROCm 7.2.3): head_dim 256 with a GQA group of 6 takes 264 VGPRs and
    the spill gate refuses the walk (ResourceError); clang 23, the vLLM images', builds it."""
    from hk.ops.attn import _clang_major
    from hk.target import current_arch

    return (current_arch() == "gfx1201" and head_dim == 256 and group == 6
            and (_clang_major() or 0) < 23)


REFUSED = "gfx1201, head_dim 256, group 6: spills under clang 22 (clang 23 builds it)"


def _run(q, k, v, table, lens, q_start, kv_order, splits):
    from hk.runtime.torch_ext import launcher

    n, h, d = q.shape
    h_kv, page = k.shape[1], k.shape[2]
    group = h // h_kv
    out = torch.full_like(q, 7.0)
    o_part, ml = pd.scratch(len(lens) * splits, h_kv, group, d, q.device)
    o4 = out.view(n, h_kv, group, d)
    launcher(pdp.walk_for(d, group, page, kv_order, splits))(
        q.unsqueeze(0), _flash(k, v, kv_order), table, lens, q_start, o4, o_part, ml)
    if splits > 1:
        launcher(pd.merge_for(d, group, splits))(lens, q_start, o4, o_part, ml)
    return out


@pytest.mark.parametrize("splits", [1, 5])
@pytest.mark.parametrize("head_dim,h_kv,group,page,kv_order,nan_tail", CASES)
def test_matches_float64(head_dim, h_kv, group, page, kv_order, nan_tail, splits):
    if _refused_by_this_compiler(head_dim, group):
        pytest.skip(REFUSED)
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, nan_tail=nan_tail)
    got = _run(q, k, v, table, lens, q_start, kv_order, splits)
    want = _reference(q, k.nan_to_num(), v.nan_to_num(), table, lens, group)
    _check(got, want, lens)


@pytest.mark.parametrize("splits", [1, 5])
@pytest.mark.parametrize("head_dim,h_kv,group,page,kv_order,nan_tail", CASES)
def test_v_ones_is_exactly_one(head_dim, h_kv, group, page, kv_order, nan_tail, splits):
    if _refused_by_this_compiler(head_dim, group):
        pytest.skip(REFUSED)
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, v_ones=True,
                                          nan_tail=nan_tail)
    got = _run(q, k, v, table, lens, q_start, kv_order, splits)
    assert torch.all(got[: len(LENS)] == 1.0)


def test_gfx950_refuses_pages_of_half_blocks():
    if not _gfx950():
        pytest.skip("gfx950's limit")
    with pytest.raises(ValueError, match="whole number of 32-key blocks"):
        pdp.walk_kernel("t_port_p16", page=16)
