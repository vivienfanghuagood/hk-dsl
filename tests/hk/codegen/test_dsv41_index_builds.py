"""hk.ops.dsv41_index builds for gfx950 without spilling, in the variants vLLM's index layers take
(hk.integration.vllm_dsv41_index): a dense index source, the candidate source, a consumer, and
one whose logits / top-k stay with vLLM. A spill in an hk kernel corrupts silently (the build
refuses it); this holds the gate here, with hipcc and no GPU."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402
from hk.codegen import scaffold  # noqa: E402
from hk.ops import dsv41_index as X  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402

VARIANTS = {
    "dense": dict(RATIO=2, HASKV=1, MC=8, SEL=1, WIDTH=4096),
    "dense_wide": dict(RATIO=2, HASKV=1, MC=8, SEL=1, WIDTH=12288),   # the widest row it takes
    "source": dict(RATIO=1, HASKV=1, MC=8, SEL=1, CAND=1, WIDTH=8192),
    "consumer": dict(RATIO=1, HASKV=0, MC=1, SEL=1, WIDTH=8192),
    "no_select": dict(RATIO=2, HASKV=1, MC=4, SEL=0),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_it_builds_without_spilling(name):
    k = X.index_kernel(0.0884)
    k = hk.kernel(k.fn, arch="gfx950", warps=k.warps, grid=k.grid_fn, name=k.name, resident=True)
    t = k.trace(**VARIANTS[name])
    b = hc.build(k.with_defines(scaffold.device(t)), "gfx950", name=k.name, kind="co")
    (r,) = b.kernels
    assert not r.spills
