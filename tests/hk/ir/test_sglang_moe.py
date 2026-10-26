"""SGLang's MoE patch, without a GPU or SGLang: what the GEMM stand-in hands
back, the 16-row config it asks for, and what the plugin turns on."""

import importlib.machinery
import sys
import types

import pytest

from hk.integration import _common, sglang_moe

OLMOE_W1, OLMOE_W2 = (64, 2048, 2048), (64, 2048, 1024)


def test_only_unscaled_unbiased_unfused_gemms_are_plain():
    nones = [None] * 6
    assert sglang_moe._plain(*nones, *[False] * 13)
    for i in range(6):
        args = list(nones)
        args[i] = 1
        assert not sglang_moe._plain(*args, *[False] * 13)
    for i in range(13):
        flags = [False] * 13
        flags[i] = True
        assert not sglang_moe._plain(*nones, *flags)


def test_the_patch_declares_itself_graph_safe():
    assert sglang_moe.PATCH.graph == "safe"
    assert sglang_moe.STATS is sglang_moe.PATCH.stats
    assert set(sglang_moe.PATCH.paths) == {"hk", "fallback"}


def test_apply_without_sglang_reports_rather_than_raises():
    if _common.importable("sglang"):
        pytest.skip("sglang is installed here; the gpu tier covers apply()")
    assert "not importable" in sglang_moe.apply()
    assert "not importable" in sglang_moe.apply_topk()
    assert sglang_moe.status() == "not patched"


@pytest.fixture
def orig_config(monkeypatch):
    calls = []

    def orig(w1_shape, w2_shape, top_k, dtype, M, is_marlin=False, block_shape=None,
             per_channel_quant=False, return_down_config=False):
        calls.append(M)
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}
        return (config, ({"BLOCK_SIZE_M": 64}, 64)) if return_down_config else config

    monkeypatch.setattr(sglang_moe, "_ORIG_CONFIG", orig)
    return calls


@pytest.mark.parametrize("m", [1, 16, 256])
def test_decode_sized_batches_sort_into_16_row_blocks(orig_config, m):
    config, (down, max_block_m) = sglang_moe.try_get_optimal_moe_config(
        OLMOE_W1, OLMOE_W2, 8, None, m, return_down_config=True)
    assert config["BLOCK_SIZE_M"] == 16 and config["BLOCK_SIZE_N"] == 64
    assert down["BLOCK_SIZE_M"] == 16      # the down GEMM reads the same sort
    assert max_block_m == 64
    assert sglang_moe.try_get_optimal_moe_config(
        OLMOE_W1, OLMOE_W2, 8, None, m)["BLOCK_SIZE_M"] == 16


@pytest.mark.parametrize("how", [dict(M=257), dict(dtype="fp8_w8a8"),
                                 dict(block_shape=[128, 128]), dict(is_marlin=True)])
def test_everything_else_keeps_sglangs_config(orig_config, how):
    args = dict(w1_shape=OLMOE_W1, w2_shape=OLMOE_W2, top_k=8, dtype=None, M=16)
    args.update(how)
    assert sglang_moe.try_get_optimal_moe_config(**args)["BLOCK_SIZE_M"] == 64


def test_a_missing_down_config_stays_missing(monkeypatch):
    monkeypatch.setattr(sglang_moe, "_ORIG_CONFIG",
                        lambda *a, **k: ({"BLOCK_SIZE_M": 64}, (None, None)))
    config, (down, _) = sglang_moe.try_get_optimal_moe_config(
        OLMOE_W1, OLMOE_W2, 8, None, 4, return_down_config=True)
    assert config["BLOCK_SIZE_M"] == 16 and down is None


def test_a_call_hk_does_not_take_goes_to_sglangs_kernel(monkeypatch):
    seen = []
    monkeypatch.setattr(sglang_moe, "_ORIG", lambda *a, **k: seen.append(k) or "triton")
    cpu = types.SimpleNamespace(is_cuda=False)
    args = [cpu, None, None, None, None, None, None, None, None, None, None, None,
            True, 8, {"BLOCK_SIZE_M": 16}, None, False, False, False, False, False]
    before = sglang_moe.STATS.get("fallback", 0)
    assert sglang_moe.invoke_fused_moe_kernel(*args, router_topk=8) == "triton"
    assert sglang_moe.STATS["fallback"] == before + 1
    assert seen[0]["router_topk"] == 8 and seen[0]["filter_expert"] is True


def _fake(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def fake_sglang(monkeypatch):
    def topk_softmax(*a):
        return "sgl_kernel"

    def invoke(*a, **k):
        return "triton"

    def config(*a, **k):
        return {"BLOCK_SIZE_M": 64}

    for name in ("sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.srt",
                 "sglang.srt.layers", "sglang.srt.layers.moe",
                 "sglang.srt.layers.moe.moe_runner",
                 "sglang.srt.layers.moe.moe_runner.triton_utils"):
        _fake(monkeypatch, name)
    _fake(monkeypatch, "sglang.kernels.ops.moe", topk_softmax=topk_softmax,
          invoke_fused_moe_kernel=invoke)
    _fake(monkeypatch, "sglang.kernels.ops.moe.fused_moe_triton_kernels",
          invoke_fused_moe_kernel=invoke)
    _fake(monkeypatch, "sglang.srt.layers.moe.topk", topk_softmax=topk_softmax)
    _fake(monkeypatch, "sglang.srt.layers.moe.moe_runner.triton")
    _fake(monkeypatch, "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
          invoke_fused_moe_kernel=invoke, try_get_optimal_moe_config=config)
    _fake(monkeypatch, "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config",
          try_get_optimal_moe_config=config)
    yield types.SimpleNamespace(topk=topk_softmax, invoke=invoke, config=config)
    sglang_moe.revert()


def test_apply_rebinds_every_binding_and_revert_puts_them_back(fake_sglang):
    out = sglang_moe.apply()
    assert "topk_softmax -> torch (2 bindings rebound)" in out
    assert "(3 kernel and 2 config bindings rebound" in out
    fused = sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"]
    assert fused.invoke_fused_moe_kernel is sglang_moe._GUARDED
    assert fused.try_get_optimal_moe_config is sglang_moe.try_get_optimal_moe_config
    assert sys.modules["sglang.srt.layers.moe.topk"].topk_softmax is sglang_moe.topk_softmax
    assert "moe: already patched" in sglang_moe.apply()

    back = sglang_moe.revert()
    assert "moe -> sglang (3 rebound)" in back and "topk_softmax -> sgl_kernel" in back
    assert fused.invoke_fused_moe_kernel is fake_sglang.invoke
    assert sys.modules["sglang.srt.layers.moe.topk"].topk_softmax is fake_sglang.topk


def test_apply_topk_leaves_the_gemms_alone(fake_sglang):
    assert sglang_moe.apply_topk().startswith("topk_softmax -> torch")
    fused = sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"]
    assert fused.invoke_fused_moe_kernel is fake_sglang.invoke


@pytest.mark.parametrize("value,want", [("1", "apply"), ("topk", "apply_topk")])
def test_the_plugin_opts_in_by_environment(monkeypatch, capsys, value, want):
    from hk.integration import _sglang_plugin, contract

    seen = []
    monkeypatch.delenv("HK_DISABLE", raising=False)
    monkeypatch.setenv("HK_SGLANG_LINEAR", "0")
    monkeypatch.setattr(contract, "check", lambda routes: [])   # no SGLang here
    monkeypatch.setenv("HK_SGLANG_MOE", value)
    monkeypatch.setattr(sglang_moe, "apply", lambda: seen.append("apply") or "moe: test")
    monkeypatch.setattr(sglang_moe, "apply_topk",
                        lambda: seen.append("apply_topk") or "topk: test")
    _sglang_plugin.register()
    assert seen == [want]
    assert "[hk]" in capsys.readouterr().out
