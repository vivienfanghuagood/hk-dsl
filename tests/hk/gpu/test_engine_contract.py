"""The engine contract of hk's DeepSeek-V4.1 routes: every vLLM / AITER entry point they patch or
call still exists with the parameters they use, and a patched engine reaches hk and gives stock
results when it falls back (docs/archive/design/serving-readiness.md, A9).

The routes patch internals: RocmDotScaledMxfp8LinearKernel's two methods (vllm_mxfp8) and
aiter.fused_moe.fused_moe with its signature (vllm_moe_a8w4). They were measured on one vLLM
nightly. A nightly that renames, moves or re-signs any of these must fail here, loudly, not
silently route every call back to the vendor -- or, as a bare-signature wrapper once did, change
the stock path too. Needs vLLM's ROCm build and aiter: the vLLM image.
"""

from __future__ import annotations

import inspect

import pytest

from hk.target import current_arch

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
pytest.importorskip("aiter")
pytestmark = pytest.mark.gpu

if current_arch() != "gfx950":
    pytest.skip("the DeepSeek-V4.1 routes are gfx950's", allow_module_level=True)

from hk.integration import vllm_moe_a8w4 as PM  # noqa: E402
from hk.integration import vllm_mxfp8 as PX  # noqa: E402

from hk.integration import switches  # noqa: E402
from hk.integration.contract import MEASURED_VLLM  # noqa: E402


def _params(fn):
    return inspect.signature(fn).parameters


def test_the_vllm_under_test_is_reported():
    print(f"vLLM {vllm.__version__} (routes measured on {MEASURED_VLLM})")


# -- MXFP8 linear (K2) ------------------------------------------------------------------------


def test_vllm_still_has_the_mxfp8_linear_kernel_class_and_its_two_methods():
    cls = PX._cls()
    assert list(_params(cls.process_weights_after_loading))[:2] == ["self", "layer"]
    assert list(_params(cls.apply_weights))[:4] == ["self", "layer", "x", "bias"]


def test_vllm_still_has_the_quant_and_gemm_the_fallback_calls():
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    assert len(_params(rocm_mxfp8_block32_gemm)) >= 5            # (x, xs, w, ws, out_dtype)
    assert "x" in _params(mxfp8_e4m3_quantize)


def test_hk_quant_is_still_vllms_quant_bit_for_bit():
    """mxfp8.quant replaces vLLM's quant on hk's path (K2 c35); the route is bit-identical only
    while the two agree."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    from hk.ops import mxfp8

    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(256, 1152, device="cuda", generator=g)
         * torch.exp2(torch.randint(-12, 12, (256, 1), device="cuda", generator=g).float())).bfloat16()
    q, s = mxfp8_e4m3_quantize(x)
    hq, hp = mxfp8.quant(x)
    assert torch.equal(q.view(torch.uint8), hq.view(torch.uint8))
    assert torch.equal(hp, mxfp8.pack_scales(s.view(torch.uint8)))


# -- MoE a8w4 (K1) ----------------------------------------------------------------------------


def test_aiter_fused_moe_still_takes_every_parameter_the_route_reads():
    import aiter.fused_moe as afm

    have = set(_params(afm.fused_moe))
    missing = [p for p in PM.AITER_PARAMS if p not in have]
    assert not missing, f"aiter.fused_moe.fused_moe lost {missing}"
    assert list(_params(afm.fused_moe))[:5] == ["hidden_states", "w1", "w2", "topk_weight",
                                                 "topk_ids"]


def test_aiter_still_has_the_sort_and_quant_the_route_calls():
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort

    assert "block_size" in _params(moe_sorting)
    assert len(_params(fused_dynamic_mxfp8_quant_moe_sort)) >= 6


def test_the_patched_fused_moe_keeps_aiters_signature_and_reverts():
    import aiter.fused_moe as afm

    want = list(_params(afm.fused_moe))
    PM.apply_patch()
    try:
        assert list(_params(afm.fused_moe)) == want
    finally:
        PM.revert()
    assert list(_params(afm.fused_moe)) == want


# -- the DeepSeek-V4.1 index launch -----------------------------------------------------------


def test_mono_decode_still_has_the_select_the_index_route_replaces():
    import importlib

    from hk.integration import vllm_dsv41_index as PI
    from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as RI

    m = importlib.import_module(PI.MODULE)
    assert list(_params(m._select)) == list(_params(PI._select))
    assert "device" in _params(m._mono_runner)
    lay = RI.rocm_paged_mxfp4_cache_layout(32, 128, 64)
    assert {"n_per_tile", "d_per_tile", "scale_lanes"} <= set(dir(lay))


def test_the_index_route_patches_mono_decode_and_reverts():
    import importlib

    from hk.integration import vllm_dsv41_index as PI

    m = importlib.import_module(PI.MODULE)
    orig = m._select
    try:
        assert PI.apply_patch().startswith("DeepSeek-V4.1 index layers")
        assert m._select.__wrapped__ is PI._select
        assert list(_params(m._select)) == list(_params(orig))
    finally:
        PI.revert()
    assert m._select is orig


# -- the plugin -------------------------------------------------------------------------------


def test_vllms_plugin_loader_finds_hks_entry_point_when_installed():
    from importlib.metadata import entry_points

    eps = [e for e in entry_points(group="vllm.general_plugins") if e.value.startswith("hk.")]
    if not eps:
        pytest.skip("hk is on the path, not installed: no entry point to find")
    assert eps[0].load().__name__ == "register"


# -- the runtime check and the default-on plugin ----------------------------------------------
#
# hk is on by default: the plugin checks the contract of every route it applies
# (hk.integration.contract) and patches nothing if one is broken.

VLLM_ROUTES = [r.name for r in switches.routes("vllm")]


def test_the_runtime_contract_holds_for_every_vllm_route():
    from hk.integration import contract

    assert contract.check(VLLM_ROUTES) == []


@pytest.fixture
def plugin_env(monkeypatch):
    for e in ["HK_DISABLE", "HK_PATCH_VLLM", "HK_VLLM_USER",
              *[r.env for r in switches.routes("vllm")]]:
        monkeypatch.delenv(e, raising=False)
    from hk.integration import vllm_dsv41_index, vllm_int4, vllm_sparse_mla

    yield monkeypatch
    for mod in (PM, PX, vllm_int4, vllm_dsv41_index, vllm_sparse_mla):
        mod.revert()


def _patched():
    import aiter.fused_moe as afm

    cls = PX._cls()
    return {"vllm_moe_a8w4": afm.fused_moe is PM.fused_moe,
            "vllm_mxfp8": cls.apply_weights is PX.apply_weights}


def test_with_nothing_set_the_plugin_routes_k1_and_k2(plugin_env, capsys):
    from hk.integration import _vllm_plugin

    _vllm_plugin.register()
    assert _patched() == {"vllm_moe_a8w4": True, "vllm_mxfp8": True}
    out = capsys.readouterr().out
    assert "vllm_mxfp8 (default on (gfx950))" in out and "vllm_moe_a8w4" in out


def test_with_hk_disable_the_plugin_patches_nothing(plugin_env, capsys):
    from hk.integration import _vllm_plugin

    plugin_env.setenv("HK_DISABLE", "1")
    plugin_env.setenv("HK_VLLM_MXFP8", "1")
    _vllm_plugin.register()
    assert _patched() == {"vllm_moe_a8w4": False, "vllm_mxfp8": False}
    assert "stock" in capsys.readouterr().out


def test_an_aiter_that_lost_a_parameter_leaves_the_engine_stock(plugin_env, capsys):
    """A simulated version mismatch: aiter's fused_moe without gate_mode. The plugin must not
    patch K1, nor K2 beside it, and must say so in one line."""
    import aiter.fused_moe as afm

    from hk.integration import _vllm_plugin

    real = afm.fused_moe

    def fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, activation=None,
                  quant_type=None):
        return real(hidden_states, w1, w2, topk_weight, topk_ids)

    plugin_env.setattr(afm, "fused_moe", fused_moe)
    _vllm_plugin.register()
    assert afm.fused_moe is fused_moe
    assert _patched() == {"vllm_moe_a8w4": False, "vllm_mxfp8": False}
    err = [x for x in capsys.readouterr().err.splitlines() if x.startswith("[hk] WARNING")]
    assert len(err) == 1 and "engine contract FAILED" in err[0] and "gate_mode" in err[0]
