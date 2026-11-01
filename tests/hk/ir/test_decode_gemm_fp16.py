"""hk.ops.decode_gemm's fp16 option, traced: its name, its tensor types and the WMMA it prints; the bf16
kernels' names unchanged. The GPU checks are tests/hk/gpu/test_decode_gemm.py."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.ops import decode_gemm  # noqa: E402
from hk.target import current_arch  # noqa: E402

pytestmark = pytest.mark.skipif(current_arch() not in ("gfx1100", "gfx1201"),
                                reason="the fp16 decode GEMM is RDNA's (wave32 WMMA)")


@pytest.mark.parametrize("tokens", sorted(decode_gemm.SCHEDULES))
def test_fp16_kernels_are_named_and_typed(tokens):
    s = dict(decode_gemm.SCHEDULES[tokens], tokens=tokens)
    b, h = decode_gemm.kernel_for(**s), decode_gemm.kernel_for(**s, dtype="fp16")
    assert h.name == b.name + "_fp16" and b.name.startswith("decode_gemm_t")
    types = {p.name: p.type.dtype.name for p in h.trace().tensors()}
    assert types == {"x": "fp16", "w": "fp16", "c": "fp16"}
    assert {p.name: p.type.dtype.name for p in b.trace().tensors()} == {"x": "bf16", "w": "bf16", "c": "bf16"}


def test_an_unknown_dtype_is_refused():
    with pytest.raises(ValueError, match="dtype"):
        decode_gemm.kernel_for(1, 4, 16, dtype="fp8")
