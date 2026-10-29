"""hk.ops.sparse_mla traced for gfx950 (no hipcc, no GPU): both variants trace; K and V share one
st_16x32 allocation; the KV rows come by the DMA gather; the LDS fits two workgroups a CU."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.codegen import scaffold  # noqa: E402
from hk.ops import sparse_mla  # noqa: E402


@pytest.mark.parametrize("mxfp8", [False, True])
def test_the_kernel_traces(mxfp8):
    k = sparse_mla.kernel(f"smla_t_{int(mxfp8)}", scale=512 ** -0.5, mxfp8=mxfp8, arch="gfx950")
    src = scaffold.bare(k.trace())
    assert "kittens::st<bf16, 32, 512, kittens::ducks::st_shape::st_16x32> (&Ks)[2]" in src
    assert "group_load_async" in src and "gather" in src
    assert ("store_elems" in src) == mxfp8


def test_two_workgroups_fit_a_cu():
    k = sparse_mla.kernel("smla_t_lds", scale=512 ** -0.5, mxfp8=True, arch="gfx950")
    assert k.trace().lds_bytes <= 160000 // 2
