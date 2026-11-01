"""hk.ops.moe's 4-bit form traced: its name, its parameters (W as codes, scales, zero codes) and the
load it prints; the bf16 kernel's parameters as they were. The GPU checks are
tests/hk/gpu/test_moe_w4.py."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")

from hk.ops import moe  # noqa: E402
from hk.target import current_arch  # noqa: E402


@pytest.mark.skipif(current_arch() not in ("gfx1100", "gfx1201"), reason="hk.load_w4(zp=) is RDNA's")
def test_the_w4_kernel_name_parameters_and_load():
    k = moe.kernel_for(16, 8, False, 8, 8, False, "fp16", 32)
    assert k.name == "moe_gemm_m16_k8_r0_s8_w8_fp16_w4g32"
    ir = k.trace()
    assert [p.name for p in ir.tensors()] == ["a", "qw", "s", "zp", "c", "sorted_ids", "expert_ids",
                                              "ntp", "topk_w"]
    assert "kittens::load_w4_zp<32, true>(" in k.source()


def test_the_bf16_kernel_is_as_it_was():
    k = moe.kernel_for(16, 2, True, 4, 16)
    assert k.name == "moe_gemm_m16_k2_r1_s4_w16"
    assert [p.name for p in k.trace().tensors()] == ["a", "w", "c", "sorted_ids", "expert_ids", "ntp",
                                                     "topk_w"]


def test_the_unroll_option_names_the_kernel():
    assert moe.kernel_for(16, 8, False, 4, 8, False, "fp16", 32, 2).name.endswith("_w4g32_u2")
