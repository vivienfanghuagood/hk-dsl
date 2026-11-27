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


def test_the_plugin_applies_it_when_asked(monkeypatch):
    from hk.integration import _vllm_plugin, contract

    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setattr(contract, "check_each", lambda routes: {})   # no vLLM here
    monkeypatch.setenv("HK_PATCH_VLLM", "1")
    monkeypatch.setenv("HK_VLLM_MOE", "1")
    monkeypatch.setattr(vllm_moe, "apply", lambda: seen.append("moe") or "moe: test")
    _vllm_plugin.register()
    assert seen == ["moe"]


def test_a_binding_left_after_revert_is_vllms_config_not_a_crash(monkeypatch):
    """revert() then a call through a binding it could not reach (taken by name into an object
    rebind does not walk, as vLLM's triton_moe holds it): vLLM's config, no 'NoneType is not
    callable' (hk 0.9.14). While patched the same binding gets hk's 16-row blocks."""
    import importlib.machinery
    import sys
    import types

    def stock_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
        return {"BLOCK_SIZE_M": 64, "M": M}

    def kernel(*a, **k):
        return "vllm"

    mods = {}
    for name in ("vllm", "vllm.platforms", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.fused_moe"):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mods[name] = m
    mods["vllm.platforms"].current_platform = types.SimpleNamespace(is_rocm=lambda: True)
    fm = mods["vllm.model_executor.layers.fused_moe.fused_moe"]
    fm.try_get_optimal_moe_config, fm.invoke_fused_moe_triton_kernel = stock_config, kernel
    mods["vllm.model_executor.layers.fused_moe"].fused_moe = fm
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setattr(vllm_moe, "_ORIG", None)
    monkeypatch.setattr(vllm_moe, "_ORIG_CONFIG", None)
    monkeypatch.setattr(vllm_moe, "block16_max", lambda: 8)
    monkeypatch.setattr(moe, "schedule_for", lambda block_m, k: {"slices": 4})

    assert "rebound" in vllm_moe.apply()
    held = types.SimpleNamespace(cfg=fm.try_get_optimal_moe_config)   # out of rebind's reach
    assert held.cfg((8, 512, 2048), (8, 2048, 256), 2, None, 4)["BLOCK_SIZE_M"] == 16
    assert "moe -> vllm" in vllm_moe.revert()
    assert fm.try_get_optimal_moe_config is stock_config
    assert held.cfg((8, 512, 2048), (8, 2048, 256), 2, None, 4) == {"BLOCK_SIZE_M": 64, "M": 4}
    assert "rebound" in vllm_moe.apply()                    # and it applies again cleanly
    assert fm.try_get_optimal_moe_config((8, 512, 2048), (8, 2048, 256), 2, None, 4)["BLOCK_SIZE_M"] == 16
    vllm_moe.revert()


def test_reverting_moe_under_moe_wna16_leaves_a_working_chain(monkeypatch):
    """Both routes on wrap the one vLLM config function (wna16's wrapper around moe's). Reverting
    moe first cannot rebind the inner binding: wna16's wrapper still calls moe's, which must then
    be vLLM's config, not a call into None (lane A's crash, hk 0.9.14)."""
    import importlib.machinery
    import sys
    import types

    from hk.integration import vllm_moe_wna16

    def stock_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
        return {"BLOCK_SIZE_M": 64}

    mods = {}
    for name in ("vllm", "vllm.platforms", "vllm.model_executor", "vllm.model_executor.layers",
                 "vllm.model_executor.layers.fused_moe", "vllm.model_executor.layers.fused_moe.fused_moe"):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        mods[name] = m
    mods["vllm.platforms"].current_platform = types.SimpleNamespace(is_rocm=lambda: True)
    fm = mods["vllm.model_executor.layers.fused_moe.fused_moe"]
    fm.try_get_optimal_moe_config = stock_config
    fm.invoke_fused_moe_triton_kernel = lambda *a, **k: "vllm"
    fm.invoke_fused_moe_wna16_triton_kernel = lambda *a, **k: "vllm"
    mods["vllm.model_executor.layers.fused_moe"].fused_moe = fm
    for name, m in mods.items():
        monkeypatch.setitem(sys.modules, name, m)
    for mod in (vllm_moe, vllm_moe_wna16):
        monkeypatch.setattr(mod, "_ORIG", None)
        monkeypatch.setattr(mod, "_ORIG_CONFIG", None)
    assert "rebound" in vllm_moe.apply()
    assert "rebound" in vllm_moe_wna16.apply()             # wraps moe's wrapper
    assert "moe -> vllm" in vllm_moe.revert()
    assert fm.try_get_optimal_moe_config((8, 512, 2048), (8, 2048, 256), 2, None, 4) == {"BLOCK_SIZE_M": 64}
    assert "moe_wna16 -> vllm" in vllm_moe_wna16.revert()
    assert fm.try_get_optimal_moe_config is not None
    assert fm.try_get_optimal_moe_config((8, 512, 2048), (8, 2048, 256), 2, None, 4) == {"BLOCK_SIZE_M": 64}


def _engine_sig(*added):
    """invoke_fused_moe_triton_kernel's recorded facts: vLLM up to 0.33 (ends at B_bias), plus
    the trailing parameters a later vLLM added, each defaulting to False."""
    from hk.integration import contract

    sig = contract.signature(vllm_moe.invoke_fused_moe_triton_kernel)
    upto = [p[0] for p in sig].index("B_bias") + 1
    return sig[:upto] + [[name, True, "False"] for name in added]


@pytest.mark.parametrize("added", [(), ("A_is_sorted", "C_is_sorted")])
def test_the_stand_in_matches_vllm_before_and_after_the_sorted_flags(added):
    """vLLM 0.34 added A_is_sorted / C_is_sorted. The stand-in takes them, and the contract still
    holds on a vLLM that predates them (0.33 and older), so one hk serves both."""
    from hk.integration import contract

    assert contract.same_signature(_engine_sig(*added), vllm_moe.invoke_fused_moe_triton_kernel,
                                   "invoke_fused_moe_triton_kernel",
                                   newer=vllm_moe.NEWER_PARAMS) == []


def test_a_parameter_hk_does_not_know_still_breaks_the_contract():
    """Only the declared newer parameters may be missing from the engine; one hk has never seen
    turns the route off rather than going into the wrong slot."""
    from hk.integration import contract

    for sig in (_engine_sig("A_is_sorted", "C_is_sorted", "D_is_new"), _engine_sig("C_is_sorted")):
        assert contract.same_signature(sig, vllm_moe.invoke_fused_moe_triton_kernel,
                                       "invoke_fused_moe_triton_kernel",
                                       newer=vllm_moe.NEWER_PARAMS) != []
    # without the declaration the older engine is a mismatch, as before
    assert contract.same_signature(_engine_sig(), vllm_moe.invoke_fused_moe_triton_kernel,
                                   "invoke_fused_moe_triton_kernel") != []


@pytest.mark.parametrize("flags", [dict(), dict(A_is_sorted=True), dict(C_is_sorted=True)])
def test_a_sorted_layout_goes_to_vllm_and_reaches_it(monkeypatch, flags):
    """hk's GEMM reads A and writes C in token order: a call in either sorted layout falls back,
    with the flag forwarded; at their defaults the flags are not forwarded at all, so a vLLM
    without them (0.33) is called as before."""
    seen = {}

    def orig(*args, **kwargs):
        seen["n"], seen["kw"] = len(args), kwargs

    launched = []
    monkeypatch.setattr(vllm_moe, "_ORIG", orig)
    monkeypatch.setattr(vllm_moe, "block16_max", lambda: 64)
    monkeypatch.setattr(moe, "takes", lambda A, B, bm: True)
    monkeypatch.setattr(moe, "_launch", lambda *a, **k: launched.append(1))

    class T:                        # just what the stand-in reads off A and C
        is_cuda = True
        shape = (4, 2, 64)

        def dim(self):
            return 3

    args = (T(), None, T(), None, None, None, None, None, None, False, 2, {"BLOCK_SIZE_M": 16},
            None, False, False, False, False, False)
    vllm_moe.invoke_fused_moe_triton_kernel(*args, **flags)
    if flags:
        assert launched == [] and seen == {"n": 20, "kw": flags}
    else:
        assert launched == [1] and seen == {}
