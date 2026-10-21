"""The cross-architecture fixes of docs/dx/api-consistency.md: the same
call answers the same way, or says which arch does what, on gfx1100 and
gfx950. No GPU: the arch is monkeypatched, as tests/hk/ir/test_cdna_paged.py
does, and the checks run before any launch."""

from __future__ import annotations

import pytest

from hk.ops import paged


def _arch(monkeypatch, arch):
    from hk import target
    from hk.ops import attn, gemm

    for mod in (target, attn, gemm):     # attn and gemm import it by name
        monkeypatch.setattr(mod, "current_arch", lambda: arch)


def test_paged_kernels_get_routes_like_getitem(monkeypatch):
    _arch(monkeypatch, "gfx950")
    assert paged.KERNELS.get(128).name == paged.KERNELS[128].name == "paged_decode_d128_cdna"
    assert paged.KERNELS.get(96) is None
    _arch(monkeypatch, "gfx1100")
    assert paged.KERNELS.get(128).name == "paged_decode_d128"


def test_the_gfx950_merge_says_why(monkeypatch):
    _arch(monkeypatch, "gfx950")
    with pytest.raises(ValueError, match="gfx950's merge reads 8 splits at a time"):
        paged.merge_kernel(128, 12)


def test_matmul_refuses_k0_on_both(monkeypatch):
    torch = pytest.importorskip("torch")
    import hk

    for arch in ("gfx1100", "gfx950"):
        _arch(monkeypatch, arch)
        a = torch.empty(32, 0, dtype=torch.bfloat16)
        b = torch.empty(128, 0, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="K=0"):
            hk.ops.matmul(a, b)


def test_attention_length_message_names_the_arch(monkeypatch):
    torch = pytest.importorskip("torch")
    from hk.ops import attn

    _arch(monkeypatch, "gfx1100")
    q = torch.empty(1, 4, 64, 128, dtype=torch.bfloat16)
    why = attn._why(q, q, q, False)
    assert "gfx1100" in why and "gfx950's takes any length" in why
    _arch(monkeypatch, "gfx950")
    assert attn._why(q, q, q, False) == ""


def test_ops_exports_paged():
    import hk

    assert "paged" in hk.ops.__all__ and hk.ops.paged is paged
