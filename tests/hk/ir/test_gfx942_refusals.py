"""gfx942 (MI300X / MI325X) is compile-only, and an hk.ops entry point with no kernel for it
refuses at the call with one message, instead of failing in hipcc on a missing include/cdna3
member (what matmul, attention, paged attention, the decode GEMM, MoE and W4A16 did). Drop-in
SDPA falls back to torch there, on attention's reason. No hipcc, no GPU: every refusal comes
before a build."""

from __future__ import annotations

import pytest

pytest.importorskip("hk._mlir")
torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import templates as T  # noqa: E402
from hk.ops import _arch, attn, decode_gemm, moe, paged, paged_prefill, w4a16  # noqa: E402

B = torch.bfloat16

REFUSED = [
    ("hk.ops.matmul", lambda: hk.ops.matmul(torch.zeros(256, 512, dtype=B),
                                            torch.zeros(256, 512, dtype=B))),
    ("hk.attention", lambda: hk.ops.attention(*[torch.zeros(1, 8, 256, 128, dtype=B)] * 3)),
    ("hk.ops.decode_gemm.matmul", lambda: decode_gemm.matmul(torch.zeros(1, 512, dtype=B),
                                                             torch.zeros(256, 512, dtype=B))),
    ("hk.ops.moe.moe_gemm", lambda: moe.moe_gemm(None, None, None, None, None, None,
                                                 top_k=1, block_m=16)),
    ("hk.ops.w4a16.matmul", lambda: w4a16.matmul(None, None, None, None, 128)),
    ("hk.ops.paged", lambda: paged.split_kernel(128, 4)),
    ("hk.ops.paged", lambda: paged.fused_kernel(128, 4)),
    ("hk.ops.paged", lambda: paged.merge_kernel(128, 4)),
    ("hk.ops.paged", lambda: paged.KERNELS[128]),
    ("hk.ops.paged_prefill", lambda: paged_prefill.kernel_for(128, 4)),
    ("hk.templates.gemm", lambda: T.gemm.plan(256, 256, 512)),
]


@pytest.mark.arch("gfx942")
@pytest.mark.parametrize("op,call", REFUSED, ids=[f"{op}-{i}" for i, (op, _) in enumerate(REFUSED)])
def test_an_op_without_a_gfx942_kernel_refuses_at_the_call(op, call):
    with pytest.raises(NotImplementedError) as e:
        call()
    msg = str(e.value)
    assert msg.startswith(f"{op} has no kernel for gfx942; it runs on "
                          "gfx1100, gfx1201, gfx950."), msg
    assert "compile-only" in msg and "docs/supported-hardware.md" in msg


@pytest.mark.arch("gfx942")
def test_the_strip_ops_refuse_on_gfx942_with_what_to_use():
    with pytest.raises(NotImplementedError, match="on gfx942 use hk.ops.rope"):
        hk.ops.rotary(torch.zeros(16, 8, 128, dtype=B), torch.zeros(64, 128))
    for name in ("rowwise", "elementwise"):
        fn = (lambda x: x * T.row_mean(x * x).rsqrt()) if name == "rowwise" else (lambda x: x * 2)
        with pytest.raises(NotImplementedError,
                           match="register strips, which gfx942 does not have "
                                 r"\(gfx1100, gfx1201 and gfx950 do\)"):
            getattr(T, name)(fn).plan(T.meta((64, 1024), hk.bf16))


@pytest.mark.arch("gfx942")
def test_attentions_reason_is_the_arch_so_drop_in_sdpa_falls_back():
    # hk.ops.sdpa asks attn._why whether a call fits and falls back to torch when not
    q = torch.randn(1, 8, 256, 128, dtype=B)
    assert not attn.supported(q, q, q)
    assert attn._why(q, q, q, False).startswith("hk.attention has no kernel for gfx942")


def test_an_explicit_arch_is_the_one_checked():
    # templates.gemm.plan takes arch=: the refusal follows it, not the process's arch
    with pytest.raises(NotImplementedError, match="no kernel for gfx942"):
        T.gemm.plan(256, 256, 512, arch="gfx942")
    for a in _arch.RDNA_AND_GFX950:
        assert _arch.why_not("hk.ops.matmul", arch=a) == ""
