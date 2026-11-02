"""hk.ops.paged_decode_portable -- one decode source for RDNA and gfx950 -- against float64.

The same cases on every target: TRITON_ATTN's layouts, pages of 16 and hybrid models' pages with
NaN past the end, head_dim 128 and 256, split and unsplit, and V all ones giving exactly 1.0.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_paged_decode import _case, _check, _flash, _reference  # noqa: E402

from hk.ops import paged_decode as pd  # noqa: E402
from hk.ops import paged_decode_portable as pdp  # noqa: E402

pytestmark = pytest.mark.gpu

CASES = [
    # head_dim, h_kv, group, page, kv_order, nan_tail
    (128, 2, 8, 16, "planes", False),       # MiniCPM5-2B
    (128, 2, 6, 2064, "heads", True),
    (256, 4, 6, 784, "heads", True),        # Qwen3.8-27B
    (256, 2, 6, 16, "planes", False),
]
LENS = [1, 15, 16, 33, 1000, 2063, 4100]


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
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, nan_tail=nan_tail)
    got = _run(q, k, v, table, lens, q_start, kv_order, splits)
    want = _reference(q, k.nan_to_num(), v.nan_to_num(), table, lens, group)
    _check(got, want, lens)


@pytest.mark.parametrize("splits", [1, 5])
@pytest.mark.parametrize("head_dim,h_kv,group,page,kv_order,nan_tail", CASES)
def test_v_ones_is_exactly_one(head_dim, h_kv, group, page, kv_order, nan_tail, splits):
    q, k, v, table, lens, q_start = _case(LENS, head_dim=head_dim, h_kv=h_kv, group=group,
                                          page=page, kv_order=kv_order, v_ones=True,
                                          nan_tail=nan_tail)
    got = _run(q, k, v, table, lens, q_start, kv_order, splits)
    assert torch.all(got[: len(LENS)] == 1.0)
