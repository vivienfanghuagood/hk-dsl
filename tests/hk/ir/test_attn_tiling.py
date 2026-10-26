"""hk.ops.attn's per-arch tilings: which knobs a call gets, by arch, compiler and length."""

from __future__ import annotations

import pytest

from hk.ops import attn


@pytest.fixture
def on(monkeypatch):
    def set_(arch, clang):
        monkeypatch.setattr(attn, "current_arch", lambda: arch)
        monkeypatch.setattr(attn, "_CLANG", [clang])
    return set_


def test_gfx1100_keeps_its_own_tiling(on):
    on("gfx1100", 22)
    assert attn._tiling(128, True, 4096) == {} and attn._tiling(128, False, 4096) == {}


def test_gfx1201_takes_its_tiling_and_the_compiler_picks_the_causal_kv_block(on):
    on("gfx1201", 22)
    assert attn._tiling(128, True, 4096)["kv_block"] == 64
    on("gfx1201", 23)
    assert attn._tiling(128, True, 4096).get("kv_block", attn.KV_BLOCK) == 32
    assert attn._tiling(128, False, 4096) == dict(warps=16, heads_first=True)


def test_a_short_q_keeps_gfx1100s_tiling_so_what_attention_accepts_is_unchanged(on):
    on("gfx1201", 23)
    assert attn._tiling(128, True, 255) == {} and attn._tiling(128, True, 256)
    assert attn._tiling(64, True, 4096) == {}


def test_no_hipcc_is_the_base_tiling(on):
    on("gfx1201", None)
    assert attn._tiling(128, True, 4096)["kv_block"] == 64


def test_warp_skip_is_refused_without_causal():
    with pytest.raises(ValueError, match="causal"):
        attn.attn_kernel("t", causal=False, warp_skip=True)
