"""vLLM's side of every hk patch: the names they rebind, with the signatures
the stand-ins forward to, and the call sites a rebind by identity reaches.

A patch here monkeypatches engine internals by name. A release that renames
a function, reorders its parameters or moves the call that reaches it does
not break the patch loudly: the rebind finds nothing, or the stand-in
forwards arguments into the wrong slots, or a backend hook becomes dead code.
These tests say which, against the vLLM installed. Skipped without vLLM.

    python3 -m pytest tests/hk/engines/test_vllm_contract.py -q
"""

from __future__ import annotations

import importlib.metadata as md
import inspect

import pytest

vllm = pytest.importorskip("vllm")

from _contract import (  # noqa: E402
    assert_rebind_reaches_every_call_site,
    assert_same_parameters,
    names,
)

from hk.integration import vllm_backend, vllm_gemm, vllm_moe  # noqa: E402


def test_report_the_version_under_test(record_property):
    record_property("vllm", vllm.__version__)
    assert vllm.__version__


# ---------------------------------------------------------------- linear layers

def test_dispatch_unquantized_gemm_takes_nothing_it_needs():
    # The stand-in ignores its arguments: a linear-backend name, which selects
    # among CUDA backends. vLLM must be able to call it with none.
    from vllm.model_executor.layers import utils as vu

    for p in inspect.signature(vu.dispatch_unquantized_gemm).parameters.values():
        assert p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD), (
            f"dispatch_unquantized_gemm now requires `{p.name}`")


def test_the_gemm_it_hands_back_is_called_as_the_standins_is():
    from vllm.model_executor.layers import utils as vu

    assert names(vu.dispatch_unquantized_gemm()) == names(vllm_gemm._gemm) == [
        "layer", "x", "weight", "bias"]


def test_every_linear_reaches_the_dispatcher_through_a_rebindable_name():
    from vllm.model_executor.layers import linear, vocab_parallel_embedding  # noqa: F401
    from vllm.model_executor.layers import utils as vu

    sites = assert_rebind_reaches_every_call_site(
        "dispatch_unquantized_gemm", "vllm", vu.dispatch_unquantized_gemm)
    # UnquantizedLinearMethod (every linear) and the LM head.
    assert {"vllm.model_executor.layers.linear",
            "vllm.model_executor.layers.vocab_parallel_embedding"} <= sites


def test_the_fallback_op_takes_x_weight_bias():
    import torch
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        pytest.skip("rocm_unquantized_gemm is ROCm's")
    import vllm.model_executor.layers.utils  # noqa: F401 -- registers the op

    schema = torch.ops.vllm.rocm_unquantized_gemm.default._schema
    assert [a.name for a in schema.arguments] == ["x", "weight", "bias"]


# ---------------------------------------------------------------- MoE

def test_the_moe_kernel_and_config_take_the_standins_parameters():
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    assert_same_parameters(fm.invoke_fused_moe_triton_kernel,
                           vllm_moe.invoke_fused_moe_triton_kernel,
                           "invoke_fused_moe_triton_kernel")
    assert_same_parameters(fm.try_get_optimal_moe_config,
                           vllm_moe.try_get_optimal_moe_config,
                           "try_get_optimal_moe_config")


def test_the_moe_call_sites_are_rebindable_after_what_apply_imports():
    from vllm.model_executor.layers.fused_moe import fused_moe as fm

    assert_rebind_reaches_every_call_site(
        "invoke_fused_moe_triton_kernel", "vllm", fm.invoke_fused_moe_triton_kernel)
    assert_rebind_reaches_every_call_site(
        "try_get_optimal_moe_config", "vllm", fm.try_get_optimal_moe_config)


# ---------------------------------------------------------------- AWQ

def test_awq_methods_and_ops_take_what_the_wrapper_passes():
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization import auto_awq

    cls = auto_awq.AutoAWQLinearMethod
    assert names(cls.process_weights_after_loading) == ["self", "layer"]
    assert names(cls.apply) == ["self", "layer", "x", "bias"]
    # _vllm_awq: awq_gemm(x, qweight, scales, qzeros, 8), awq_dequantize(q, s, z, 0, 0, 0)
    assert len(names(ops.awq_gemm)) == 5
    assert len(names(ops.awq_dequantize)) == 6
    src = inspect.getsource(cls.create_weights)
    for p in ("qweight", "qzeros", "scales"):
        assert f'"{p}"' in src or f"'{p}'" in src, f"AWQ layers no longer have {p}"
    assert "group_size" in inspect.signature(auto_awq.AutoAWQConfig.__init__).parameters


# ---------------------------------------------------------------- attention backend

@pytest.fixture(scope="module")
def backend():
    note = vllm_backend.register()
    assert "registered over TRITON_ATTN" in note, note
    return vllm_backend.HKAttentionBackend, vllm_backend.HKAttentionImpl


def test_the_backend_is_complete(backend):
    be, impl = backend
    assert not inspect.isabstract(be) and not inspect.isabstract(impl)


#: Backend attributes hk sets that vLLM reads off the base class. A hook vLLM
#: renamed is dead code here.
READ_BY_VLLM = {"get_name", "get_impl_cls", "get_builder_cls", "supported_dtypes",
                "get_supported_kernel_block_sizes", "get_supported_head_sizes"}
#: How vLLM learns the head-major cache layout: from 0.31,
#: supported_kv_cache_layouts; before, get_kv_cache_shape. The one this vLLM
#: reads has to be on its base class, or the layout is a wrong answer.
LAYOUT = ("supported_kv_cache_layouts", "get_kv_cache_shape")
#: Read by vLLM up to 0.30 and kept for it; 0.31 always hands forward an
#: output buffer and asks the metadata builder about cascade attention.
LEGACY = {"accept_output_buffer", "use_cascade_attention"}
#: hk's own, called by hk.
OWN = {"validate_head_size"}


def test_every_hook_the_backend_overrides_is_accounted_for(backend):
    from vllm.v1.attention.backend import AttentionBackend

    be, _ = backend
    ours = {n for n in vars(be) if not n.startswith("_")}
    known = READ_BY_VLLM | set(LAYOUT) | LEGACY | OWN
    assert ours <= known, f"classify {sorted(ours - known)} in this test"
    gone = sorted(n for n in READ_BY_VLLM if not hasattr(AttentionBackend, n))
    assert not gone, f"vLLM's AttentionBackend has no {gone}"
    assert any(hasattr(AttentionBackend, n) for n in LAYOUT), (
        f"vLLM's AttentionBackend has neither layout hook {LAYOUT}")


def test_vllm_leaves_the_cache_write_to_forward(backend):
    # forward writes K/V itself (reshape_and_cache_flash); a backend that
    # says otherwise gets a second write from vLLM.
    be, _ = backend
    assert getattr(be, "forward_includes_kv_cache_update", True) is True


def test_the_impl_takes_vllms_arguments_in_vllms_order(backend):
    from vllm.v1.attention.backend import AttentionImpl

    _, impl = backend
    base = names(AttentionImpl.__init__, drop_self=True)
    assert names(impl.__init__, drop_self=True)[:len(base)] == base
    assert names(impl.forward) == names(AttentionImpl.forward)


def test_the_builder_and_metadata_have_what_forward_reads(backend):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionMetadata,
        RocmAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    be, _ = backend
    builder = be.get_builder_cls()
    assert names(builder.build) == names(RocmAttentionMetadataBuilder.build)
    assert hasattr(RocmAttentionMetadataBuilder, "_init_reorder_batch_threshold")
    assert type(builder._cudagraph_support).__name__ == "AttentionCGSupport"
    assert "decode_threshold" in inspect.signature(split_decodes_and_prefills).parameters
    assert "query_start_loc_cpu" in CommonAttentionMetadata.__dataclass_fields__
    read = {"num_actual_tokens", "slot_mapping", "max_query_len", "block_table",
            "seq_lens", "query_start_loc", "max_seq_len"}
    assert read <= set(RocmAttentionMetadata.__dataclass_fields__)


def test_the_cache_is_head_major_in_the_hook_this_vllm_reads(backend):
    from vllm.v1.attention.backend import AttentionBackend

    be, _ = backend
    if hasattr(AttentionBackend, "supported_kv_cache_layouts"):
        assert [x.name for x in be.supported_kv_cache_layouts()] == ["LBHNC"]
    else:
        assert be.get_kv_cache_shape(10, 32, 8, 128) == (2, 10, 8, 32, 128)


def test_the_cache_write_takes_eight_arguments_in_this_order():
    from vllm import _custom_ops as ops

    assert names(ops.reshape_and_cache_flash)[:8] == [
        "key", "value", "key_cache", "value_cache", "slot_mapping",
        "kv_cache_dtype", "k_scale", "v_scale"]


def test_tritons_decode_takes_every_argument_hk_passes(monkeypatch):
    # triton_decode returns False -- and the decode quietly takes hk's walk --
    # when vLLM's unified_attention lacks an argument it passes.
    import torch

    monkeypatch.setattr(vllm_backend, "_TRITON", [])
    got = vllm_backend._triton_decode()
    assert got is not None, "vllm.v1.attention.ops.triton_unified_attention moved"
    fn, accepted = got
    seen = {}
    monkeypatch.setattr(vllm_backend, "_TRITON", [(lambda **kw: seen.update(kw), accepted)])
    monkeypatch.setattr(vllm_backend, "_SEGM", {})
    n, h, hkv, d = 2, 4, 2, 128
    q = torch.zeros(n, h, d)
    cache = torch.zeros(3, hkv, 32, d)
    one = torch.ones(())
    assert vllm_backend.triton_decode(
        q, cache, cache, torch.zeros(n, 3, dtype=torch.int32),
        torch.ones(n, dtype=torch.int32), torch.arange(n + 1, dtype=torch.int32),
        torch.zeros_like(q), 0.1, 64, one, one), (
        f"unified_attention no longer takes some of what triton_decode passes; "
        f"it takes {sorted(accepted)}")
    assert set(seen) <= accepted


# ---------------------------------------------------------------- SDPA, plugin

def test_the_vit_path_still_calls_torch_sdpa():
    import torch
    from vllm.v1.attention.ops import vit_attn_wrappers as w

    assert hasattr(w, "apply_sdpa") and w.F is torch.nn.functional


def test_vllm_loads_general_plugins_from_hks_group_with_no_arguments():
    from vllm import plugins

    from hk.integration import _vllm_plugin

    assert plugins.DEFAULT_PLUGINS_GROUP == "vllm.general_plugins"
    assert "func()" in inspect.getsource(plugins.load_general_plugins)
    assert not inspect.signature(_vllm_plugin.register).parameters


def test_the_installed_hk_registers_the_plugin():
    try:
        md.distribution("hk-dsl")
    except md.PackageNotFoundError:
        pytest.skip("hk-dsl is not installed as a distribution here")
    eps = {e.name: e.value for e in md.entry_points(group="vllm.general_plugins")}
    assert eps.get("hk_attention") == "hk.integration._vllm_plugin:register"
