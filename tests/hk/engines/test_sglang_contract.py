"""SGLang's side of every hk patch: the names they rebind, with the
signatures the stand-ins forward to, the call sites a rebind by identity
reaches, and the plugin group SGLang loads. Skipped without SGLang.

SGLang asks the GPU for its arch while importing its kernels, so these need
one visible, though they launch nothing.

    HIP_VISIBLE_DEVICES=3 python3 -m pytest tests/hk/engines/test_sglang_contract.py -q
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import inspect
from pathlib import Path

import pytest

sglang = pytest.importorskip("sglang")

from _contract import (  # noqa: E402
    assert_rebind_reaches_every_call_site,
    assert_same_parameters,
    names,
)

from hk.integration import sglang as hks  # noqa: E402
from hk.integration import sglang_moe  # noqa: E402


@pytest.fixture(scope="module")
def homes():
    try:
        return sglang_moe._homes()
    except RuntimeError as e:
        pytest.skip(f"SGLang's kernels did not import ({e}); they need a GPU visible")


def test_report_the_version_under_test(record_property):
    record_property("sglang", sglang.__version__)
    assert sglang.__version__


def test_unquantized_linear_is_where_the_patch_looks_and_takes_x_bias(homes):
    for home in hks._LINEAR_HOMES:
        cls = getattr(importlib.import_module(home), "UnquantizedLinearMethod", None)
        if cls is not None:
            break
    else:
        pytest.fail(f"no UnquantizedLinearMethod in {hks._LINEAR_HOMES}")
    # The replacement forwards (self, layer, x, bias, *args, **kwargs).
    assert names(cls.apply) == ["self", "layer", "x", "bias"]


def test_topk_softmax_takes_the_standins_parameters_and_is_rebindable(homes):
    topk, _, _ = homes
    assert_same_parameters(topk.topk_softmax, sglang_moe.topk_softmax, "topk_softmax")
    sites = assert_rebind_reaches_every_call_site("topk_softmax", "sglang", topk.topk_softmax)
    assert "sglang.srt.layers.moe.topk" in sites


def test_the_moe_kernel_and_config_take_the_standins_parameters(homes):
    _, kernels, config = homes
    assert_same_parameters(kernels.invoke_fused_moe_kernel,
                           sglang_moe.invoke_fused_moe_kernel, "invoke_fused_moe_kernel")
    assert_same_parameters(config.try_get_optimal_moe_config,
                           sglang_moe.try_get_optimal_moe_config, "try_get_optimal_moe_config")


def test_the_moe_call_sites_are_rebindable_after_what_apply_imports(homes):
    _, kernels, config = homes
    sites = assert_rebind_reaches_every_call_site(
        "invoke_fused_moe_kernel", "sglang", kernels.invoke_fused_moe_kernel)
    assert "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe" in sites
    assert_rebind_reaches_every_call_site(
        "try_get_optimal_moe_config", "sglang", config.try_get_optimal_moe_config)


def test_the_config_returns_what_the_standin_unpacks(homes, monkeypatch):
    # With return_down_config the stand-in unpacks (config, (down, max_block_m)).
    # The tuned-file lookup reads a running server's context; an override
    # config is the same return path without one.
    from sglang.srt.layers.moe.moe_runner import triton_utils

    _, _, config = homes
    fixed = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
    monkeypatch.setattr(triton_utils, "get_config", lambda: dict(fixed))
    w1, w2 = (64, 2048, 2048), (64, 2048, 1024)          # OLMoE's experts
    assert config.try_get_optimal_moe_config(w1, w2, 8, None, 16) == fixed
    cfg, (down, max_block_m) = config.try_get_optimal_moe_config(
        w1, w2, 8, None, 16, return_down_config=True)
    assert cfg == fixed and (down is None or isinstance(down, dict))
    assert max_block_m is None or isinstance(max_block_m, int)


def test_sglang_loads_hks_plugin_group_in_its_scheduler():
    from sglang.srt import plugins

    from hk.integration import _sglang_plugin

    assert plugins.GENERAL_PLUGINS_GROUP == "sglang.srt.plugins"
    sched = Path(importlib.util.find_spec("sglang.srt.managers.scheduler").origin)
    assert "load_plugins" in sched.read_text(), "the scheduler no longer loads plugins"
    assert not inspect.signature(_sglang_plugin.register).parameters


def test_the_installed_hk_registers_the_plugin():
    try:
        md.distribution("hk-dsl")
    except md.PackageNotFoundError:
        pytest.skip("hk-dsl is not installed as a distribution here")
    eps = {e.name: e.value for e in md.entry_points(group="sglang.srt.plugins")}
    assert eps.get("hk") == "hk.integration._sglang_plugin:register"
