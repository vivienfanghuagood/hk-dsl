"""The MFMA hazard gate's padding, run: gfx950's paged decode at two pages a
step without its QK fence. Built as the compiler leaves it, LLVM before
llvm/llvm-project#225989 reads the scores too early on one path and the
decode computes some of them wrong; HK_MFMA_GATE=fix pads that path, and the
padded decode answers what the fenced one does, bit for bit.
The build-side cases are tests/hk/codegen/test_mfma_hazard_gate.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import paged, paged_cdna  # noqa: E402
from hk.runtime import cocheck, mfmahaz  # noqa: E402

import test_paged_fused as T  # noqa: E402

pytestmark = pytest.mark.gpu


def test_the_padded_unfenced_decode_is_the_fenced_one(tmp_path, monkeypatch):
    if not paged._native_cdna():
        pytest.skip("gfx950's walk")
    if mfmahaz.compiler_fixed(cocheck._tool("llc")) is True:
        pytest.skip("this compiler pads every path (llvm/llvm-project#225989)")
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path))
    case = T._case([64, 96, 128, 160, 192, 4096, 8191], h_kv=1, group=8)
    fenced = T._decode(*case, 1, walk=paged_cdna.kernel_for(128, 1, "fused", ppstep=2))
    monkeypatch.setenv("HK_MFMA_GATE", "fix")
    padded = T._decode(*case, 1, walk=paged_cdna.kernel_for(128, 1, "fused", ppstep=2,
                                                            fence_qk=False))
    assert torch.equal(padded, fenced)


@pytest.mark.parametrize("b,h,h_kv,n", [(1, 8, 8, 2048), (2, 32, 8, 4096)])
def test_the_padded_dense_attention_is_bitwise_the_unpadded_one(tmp_path, monkeypatch, b, h, h_kv, n):
    """gfx950's dense attention under ROCm 7.2.3 reads S before a 32x32x16
    MFMA's window closes on one path (tests/hk/codegen/test_mfma_hazard_gate.py).
    Padded, it answers what the unpadded build does: on today's MI355X the
    early read has not shown, and the padding changes nothing but timing."""
    if not paged._native_cdna():
        pytest.skip("gfx950's kernel")
    from hk.ops import attn_cdna  # noqa: PLC0415

    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path))
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, h, n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k, v = (torch.randn(b, h_kv, n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(2))
    from hk.runtime import compile as hc  # noqa: PLC0415

    outs, logs = {}, {}
    for mode in ("0", "fix"):
        monkeypatch.setenv("HK_MFMA_GATE", mode)
        kern = attn_cdna.attn_kernel(f"mfmahaz_dense_{mode}", head_dim=128)
        outs[mode] = torch.empty_like(q)
        kern(q, k, v, outs[mode])
        logs[mode] = kern.build().log_path.read_text()
    torch.cuda.synchronize()
    assert torch.equal(outs["0"], outs["fix"])
    if "roc-7.2.3" in hc._hipcc_version():
        # the hit this test is for: 2 places, the softmax's max after the QK MFMAs
        assert "HK_MFMA_GATE=fix padded 2 MFMA hazard(s)" in logs["fix"]


@pytest.mark.parametrize("h,h_kv,n,d", [(8, 2, 256, 128), (16, 4, 512, 64)])
def test_hipkittens_causal_attention_padded_is_bitwise_unpadded(tmp_path, monkeypatch, h, h_kv, n, d):
    """HipKittens' attend_ker, causal, as hk builds it (hk.ops.hipkittens): ROCm
    10's LLVM leaves the softmax's max 3-6 wait states short of a 32x32x16 MFMA
    at the join after its exec branch (6 places at head_dim 128, 3 at 64). The
    default pads them; the output does not move."""
    from hk.ops import hipkittens as hkk  # noqa: PLC0415
    from hk.runtime import compile as hc  # noqa: PLC0415
    from hk.runtime.module import load  # noqa: PLC0415

    if hkk.gqa_unsupported(1, n, h, h_kv, d) or not paged._native_cdna():
        pytest.skip("HipKittens' CDNA4 GQA forward")
    monkeypatch.setenv("HK_CACHE_DIR", str(tmp_path))
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, n, h, d, device="cuda", dtype=torch.bfloat16, generator=g)
    k, v = (torch.randn(1, n, h_kv, d, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(2))
    outs, logs = {}, {}
    for mode in ("0", "fix"):
        monkeypatch.setenv("HK_MFMA_GATE", mode)
        try:
            b = hkk.gqa_build(h, h_kv, n, d, True, extra_flags=(f"-DHK_GATE_ARM_{mode}",))
        except hc.CompileError:
            pytest.skip(f"HipKittens' GQA does not build under {hkk.hipcc_version()}")
        mod = load(b.so_path, b.module_name)
        o = torch.empty_like(q)
        lse = torch.empty((1, h, 1, n), device="cuda", dtype=torch.float32)
        hkk._launch(mod, q.device, q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
                    lse.data_ptr(), 1)
        outs[mode], logs[mode] = o, b.log_path.read_text()
    torch.cuda.synchronize()
    assert torch.equal(outs["0"], outs["fix"])
    clang = hc._hipcc_version()
    if mfmahaz.compiler_fixed(cocheck._tool("llc")) is False and "clang version 23" in clang:
        assert "HK_MFMA_GATE=fix padded" in logs["fix"]
