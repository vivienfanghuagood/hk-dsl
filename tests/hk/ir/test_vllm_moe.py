"""The MoE patch, without a GPU or vLLM: what it takes and what it hands back."""

import pytest

from hk.integration import _common, vllm_moe
from hk.ops import moe


def test_only_unscaled_unbiased_gemms_are_plain():
    assert vllm_moe._plain(None, None, None, None, False, False, False, False)
    for scaled in ((1, None, None, None), (None, 1, None, None),
                   (None, None, 1, None), (None, None, None, [128, 128])):
        assert not vllm_moe._plain(*scaled, False, False, False, False)
    for i in range(4):
        flags = [False] * 4
        flags[i] = True
        assert not vllm_moe._plain(None, None, None, None, *flags)


def test_the_patch_declares_itself_graph_safe():
    assert vllm_moe.PATCH.graph == "safe"
    assert vllm_moe.STATS is vllm_moe.PATCH.stats
    assert set(vllm_moe.PATCH.paths) == {"hk", "fallback"}


def test_apply_without_vllm_reports_rather_than_raises():
    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply()")
    assert "not importable" in vllm_moe.apply()
    assert vllm_moe.status() == "not patched"


@pytest.mark.parametrize("block_m,k,want", [
    (16, 2048, dict(slices=4, warps=16)),
    (16, 768, dict(slices=2, warps=8)),     # Qwen3-30B-A3B's w2 depth
    (32, 2048, dict(slices=2, warps=8)),
    (64, 1024, dict(slices=1, warps=4)),
    (128, 2048, None),                      # no schedule: vLLM's kernel
])
def test_schedules_stay_inside_the_measured_register_limits(block_m, k, want):
    assert moe.schedule_for(block_m, k) == want


def test_the_plugin_opts_in_by_environment(monkeypatch):
    from hk.integration import _vllm_plugin

    seen = []
    monkeypatch.setenv("HK_PATCH_VLLM", "1")
    monkeypatch.setenv("HK_VLLM_MOE", "1")
    monkeypatch.setattr(vllm_moe, "apply", lambda: seen.append("moe") or "moe: test")
    _vllm_plugin.register()
    assert seen == ["moe"]
