"""The engine contract, checked when a plugin starts: every engine entry point an enabled route
patches or falls back to still exists, with the parameters the route uses
(docs/archive/design/serving-readiness.md, A9).

Routes patch engine internals by name, and they were measured against one engine build. If a
release renames, moves or re-signs one of those names, the patch does not fail loudly. Either it
quietly routes nothing, or its stand-in passes arguments into the wrong slots. A bare-signature
wrapper once did the second, and it changed vLLM's stock MoE path too. So a plugin checks each
route's contract before it patches anything (`check_each`).

**Per route.** A route whose contract is broken, or whose apply raises, is turned off, with one
warning line per route on stderr (`refuse_route`); the engine's other routes are applied as
usual (`apply_routes`). Each route patches its own entry points and falls back by itself for
the shapes and arches it does not take, so a broken one says nothing about the others. Until
2026-11 it was all or nothing per engine: one broken contract left every route off, and that
happened three times on engine updates for a reason in one route.

The checks only introspect the engine: imports, attributes and `inspect.signature`. They launch
nothing. tests/hk/gpu/test_engine_contract.py and tests/hk/engines hold the same contract in CI,
and add the call-site and numeric checks that are too slow for engine startup.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Callable, Dict, Iterable, List, Optional

#: The vLLM the DeepSeek-V4.1 routes were measured on. Another version is not a
#: failure by itself; a broken contract is.
MEASURED_VLLM = "0.32.1rc1.dev36+ge68efc685"


def _params(fn: Callable, drop_self: bool = False) -> List[inspect.Parameter]:
    ps = [p for p in inspect.signature(fn).parameters.values()
          if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return ps[1:] if drop_self and ps and ps[0].name == "self" else ps


def names(fn: Callable, drop_self: bool = False) -> List[str]:
    return [p.name for p in _params(fn, drop_self)]


def same_parameters(engine_fn: Callable, standin: Callable, what: str) -> List[str]:
    """A stand-in that forwards positionally has to take the engine's parameters in the engine's
    order, with the engine's defaults."""
    e, s = _params(engine_fn), _params(standin)
    if [p.name for p in e] != [p.name for p in s]:
        return [f"{what} now takes {[p.name for p in e]}, the stand-in {[p.name for p in s]}"]
    return [f"{what}: `{pe.name}` defaults to {pe.default!r}, the stand-in's to {ps.default!r}"
            for pe, ps in zip(e, s)
            if pe.default is not pe.empty and (ps.default is ps.empty or ps.default != pe.default)]


def _expect(ok: bool, problem: str) -> List[str]:
    return [] if ok else [problem]


def signature(fn: Callable, drop_self: bool = False) -> List[list]:
    """`fn`'s parameters as facts: [name, has a default, repr of the default], in order -- what
    `same_parameters` compares, in a form a fixture can hold."""
    return [[p.name, p.default is not p.empty, repr(p.default) if p.default is not p.empty else None]
            for p in _params(fn, drop_self)]


def same_signature(engine_sig: List[list], standin: Callable, what: str) -> List[str]:
    """`same_parameters`, with the engine's side as recorded facts (`signature`)."""
    s = _params(standin)
    if [p[0] for p in engine_sig] != [p.name for p in s]:
        return [f"{what} now takes {[p[0] for p in engine_sig]}, the stand-in {[p.name for p in s]}"]
    return [f"{what}: `{name}` defaults to {rep}, the stand-in's to {ps.default!r}"
            for (name, has, rep), ps in zip(engine_sig, s)
            if has and (ps.default is ps.empty or repr(ps.default) != rep)]


def _has_names(src: str, params) -> Dict[str, bool]:
    return {p: f'"{p}"' in src or f"'{p}'" in src for p in params}


# Each route's contract is two halves. gather_<route>() reads the engine -- imports, attributes,
# signatures, source -- and returns the facts the check judges, as plain JSON (None where the
# route does not apply here). judge_<route>(facts) turns them into problems, reading only hk's own
# stand-ins. check() runs both, so a plugin's startup is what it was. `python3 -m
# hk.integration.contract --record` writes the installed engine's facts to a file; the production
# images' are kept in tests/hk/ir/engine_facts/ and judged there without the images.


# -- vLLM -------------------------------------------------------------------------------------


def _gfx950() -> bool:
    from . import switches  # noqa: PLC0415

    return switches.arch() == "gfx950"


def gather_vllm_mxfp8() -> Optional[dict]:
    if not _gfx950():
        return None                 # gfx950's scaled MFMA: the route reports it and stays out
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (  # noqa: PLC0415
        rocm_mxfp8_block32_gemm)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: PLC0415
        mxfp8_e4m3_quantize)

    from . import vllm_mxfp8 as PX  # noqa: PLC0415

    cls = PX._cls()
    return {"cls": cls.__name__, "pwal": names(cls.process_weights_after_loading),
            "apply_weights": names(cls.apply_weights),
            "gemm_nargs": len(_params(rocm_mxfp8_block32_gemm)),
            "quantize": names(mxfp8_e4m3_quantize)}


def judge_vllm_mxfp8(f: dict) -> List[str]:
    return (_expect(f["pwal"][:2] == ["self", "layer"],
                    f"{f['cls']}.process_weights_after_loading takes {f['pwal']}")
            + _expect(f["apply_weights"][:4] == ["self", "layer", "x", "bias"],
                      f"{f['cls']}.apply_weights takes {f['apply_weights']}")
            + _expect(f["gemm_nargs"] >= 5,
                      "rocm_mxfp8_block32_gemm takes fewer than (x, xs, w, ws, out_dtype)")
            + _expect("x" in f["quantize"], "mxfp8_e4m3_quantize has no `x`"))


def gather_vllm_moe_a8w4() -> Optional[dict]:
    from . import _common  # noqa: PLC0415

    if not _gfx950() or not _common.importable("aiter"):
        return None                 # nothing to patch: the route reports it and stays out
    import aiter.fused_moe as afm  # noqa: PLC0415
    from aiter.fused_moe import moe_sorting  # noqa: PLC0415
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: PLC0415

    fn = getattr(afm.fused_moe, "__wrapped__", afm.fused_moe)
    return {"fused_moe_params": list(inspect.signature(fn).parameters), "fused_moe": names(fn),
            "moe_sorting": names(moe_sorting),
            "quant_sort_nargs": len(_params(fused_dynamic_mxfp8_quant_moe_sort))}


def judge_vllm_moe_a8w4(f: dict) -> List[str]:
    from . import vllm_moe_a8w4 as PM  # noqa: PLC0415

    lost = [p for p in PM.AITER_PARAMS if p not in set(f["fused_moe_params"])]
    return (_expect(not lost, f"aiter.fused_moe.fused_moe lost {lost}")
            + _expect(f["fused_moe"][:5] == ["hidden_states", "w1", "w2", "topk_weight", "topk_ids"],
                      f"aiter.fused_moe.fused_moe begins {f['fused_moe'][:5]}")
            + _expect("block_size" in f["moe_sorting"], "aiter's moe_sorting has no block_size")
            + _expect(f["quant_sort_nargs"] >= 6,
                      "aiter's fused_dynamic_mxfp8_quant_moe_sort takes fewer than 6 arguments"))


def gather_vllm_sparse_mla() -> Optional[dict]:
    if not _gfx950():
        return None                 # gfx950's DMA gather: the route reports it and stays out
    import importlib  # noqa: PLC0415

    from . import vllm_sparse_mla as PS  # noqa: PLC0415

    have = PS._ORIG or PS._entry_points()
    return {"entry_points": {k: list(inspect.signature(fn).parameters) for k, fn in have.items()},
            "has_ragged": hasattr(importlib.import_module(PS._OPS_MOD),
                                  "build_ragged_indices_from_dense")}


def judge_vllm_sparse_mla(f: dict) -> List[str]:
    from . import vllm_sparse_mla as PS  # noqa: PLC0415

    out = [f"vLLM's {k} entry point takes {got}, not {PS.WANT[k]}"
           for k, got in f["entry_points"].items() if got != PS.WANT[k]]
    return out + _expect(f["has_ragged"], "vLLM has no build_ragged_indices_from_dense")


def gather_vllm_int4() -> dict:
    from vllm import _custom_ops as ops  # noqa: PLC0415
    from vllm.model_executor.layers.quantization import auto_awq  # noqa: PLC0415

    cls = auto_awq.AutoAWQLinearMethod
    return {"pwal": names(cls.process_weights_after_loading), "apply": names(cls.apply),
            "awq_gemm_nargs": len(names(ops.awq_gemm)),
            "awq_dequantize_nargs": len(names(ops.awq_dequantize)),
            "create_weights_has": _has_names(inspect.getsource(cls.create_weights),
                                             ("qweight", "qzeros", "scales"))}


def judge_vllm_int4(f: dict) -> List[str]:
    return (_expect(f["pwal"] == ["self", "layer"],
                    f"AutoAWQLinearMethod.process_weights_after_loading takes {f['pwal']}")
            + _expect(f["apply"] == ["self", "layer", "x", "bias"],
                      f"AutoAWQLinearMethod.apply takes {f['apply']}")
            + _expect(f["awq_gemm_nargs"] == 5, "awq_gemm no longer takes 5 arguments")
            + _expect(f["awq_dequantize_nargs"] == 6, "awq_dequantize no longer takes 6 arguments")
            + [f"AWQ layers no longer have {p}" for p, ok in f["create_weights_has"].items()
               if not ok])


def gather_vllm_gptq() -> Optional[dict]:
    """Only what the stand-ins read: the two AutoGPTQLinearMethod entry points they wrap and
    forward positionally. gptq_gemm_rdna3 is called by name, whatever this build's signature
    (vllm_gptq.plan_call), and a layer whose kernel or weights lack something the route needs is
    left on vLLM's path and counted (vllm_gptq.REFUSED) -- never the whole engine."""
    from vllm import _custom_ops as ops  # noqa: PLC0415
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    if not hasattr(ops, "gptq_gemm_rdna3"):
        return None                 # no RDNA3 GPTQ kernel in this vLLM: nothing to route
    cls = auto_gptq.AutoGPTQLinearMethod
    return {"pwal": names(cls.process_weights_after_loading), "apply": names(cls.apply)}


def judge_vllm_gptq(f: dict) -> List[str]:
    return (_expect(f["pwal"] == ["self", "layer"],
                    f"AutoGPTQLinearMethod.process_weights_after_loading takes {f['pwal']}")
            + _expect(f["apply"] == ["self", "layer", "x", "bias"],
                      f"AutoGPTQLinearMethod.apply takes {f['apply']}"))


def gather_vllm_quark() -> Optional[dict]:
    """The QuarkW4A16Int4 entry points the stand-ins wrap and forward positionally, and the names
    it gives its kernel (the route reads the kernel's tensors by them). None where this vLLM has
    no Quark W4A16 scheme: nothing to route."""
    try:
        from vllm.model_executor.layers.quantization.quark.schemes import (  # noqa: PLC0415
            quark_w4a16_int4 as qs)
    except ImportError:
        return None
    cls = qs.QuarkW4A16Int4
    return {"pwal": names(cls.process_weights_after_loading), "apply": names(cls.apply_weights),
            "class_has": {a: a in inspect.getsource(cls)
                          for a in ("self.is_symmetric", "self.group_size", "self.kernel",
                                    "self.pack_reorder")}}


def judge_vllm_quark(f: dict) -> List[str]:
    return (_expect(f["pwal"] == ["self", "layer"],
                    f"QuarkW4A16Int4.process_weights_after_loading takes {f['pwal']}")
            + _expect(f["apply"] == ["self", "layer", "x", "bias"],
                      f"QuarkW4A16Int4.apply_weights takes {f['apply']}")
            + [f"QuarkW4A16Int4 no longer sets {p}" for p, ok in f["class_has"].items() if not ok])


def gather_vllm_gemm() -> dict:
    from vllm.model_executor.layers import utils as vu  # noqa: PLC0415
    from vllm.v1.worker import gpu_worker  # noqa: PLC0415

    required = [p.name for p in inspect.signature(vu.dispatch_unquantized_gemm).parameters.values()
                if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return {"dispatch_required": required,
            "dispatched": None if required else names(vu.dispatch_unquantized_gemm()),
            "worker_load_model": callable(getattr(gpu_worker.Worker, "load_model", None))}


def judge_vllm_gemm(f: dict) -> List[str]:
    from . import vllm_gemm  # noqa: PLC0415

    out = [f"dispatch_unquantized_gemm now requires `{p}`" for p in f["dispatch_required"]]
    if not out:
        got = f["dispatched"]
        out += _expect(got == names(vllm_gemm._gemm) == ["layer", "x", "weight", "bias"],
                       f"the GEMM dispatch_unquantized_gemm hands back takes {got}")
    out += _expect(f["worker_load_model"],
                   "vLLM's GPU worker has no load_model (the tuning at load wraps it)")
    return out


def gather_vllm_moe() -> dict:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm  # noqa: PLC0415

    return {"invoke_fused_moe_triton_kernel": signature(fm.invoke_fused_moe_triton_kernel),
            "try_get_optimal_moe_config": signature(fm.try_get_optimal_moe_config)}


def judge_vllm_moe(f: dict) -> List[str]:
    from . import vllm_moe  # noqa: PLC0415

    return (same_signature(f["invoke_fused_moe_triton_kernel"],
                           vllm_moe.invoke_fused_moe_triton_kernel, "invoke_fused_moe_triton_kernel")
            + same_signature(f["try_get_optimal_moe_config"], vllm_moe.try_get_optimal_moe_config,
                             "try_get_optimal_moe_config"))


def gather_vllm_moe_wna16() -> dict:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm  # noqa: PLC0415

    return {"invoke_fused_moe_wna16_triton_kernel": signature(fm.invoke_fused_moe_wna16_triton_kernel),
            "try_get_optimal_moe_config": signature(fm.try_get_optimal_moe_config)}


def judge_vllm_moe_wna16(f: dict) -> List[str]:
    from . import vllm_moe_wna16  # noqa: PLC0415

    return (same_signature(f["invoke_fused_moe_wna16_triton_kernel"],
                           vllm_moe_wna16.invoke_fused_moe_wna16_triton_kernel,
                           "invoke_fused_moe_wna16_triton_kernel")
            + same_signature(f["try_get_optimal_moe_config"],
                             vllm_moe_wna16.try_get_optimal_moe_config, "try_get_optimal_moe_config"))


def gather_vllm_sdpa() -> dict:
    import torch  # noqa: PLC0415
    from vllm.v1.attention.ops import vit_attn_wrappers as w  # noqa: PLC0415

    return {"calls_torch_sdpa": hasattr(w, "apply_sdpa")
            and getattr(w, "F", None) is torch.nn.functional}


def judge_vllm_sdpa(f: dict) -> List[str]:
    return _expect(f["calls_torch_sdpa"],
                   "vLLM's ViT attention no longer calls F.scaled_dot_product_attention "
                   "(vit_attn_wrappers.apply_sdpa)")


def gather_vllm_norm() -> dict:
    from vllm.model_executor.layers import layernorm as L  # noqa: PLC0415

    from . import vllm_norm  # noqa: PLC0415

    cls = getattr(L, "GemmaRMSNorm", None)
    if cls is None:
        return {"has_gemma_rms_norm": False}
    fn = vllm_norm._ORIG[1] if vllm_norm._ORIG else cls.forward_cuda
    return {"has_gemma_rms_norm": True, "forward_cuda": names(fn), "init": names(cls.__init__)}


def judge_vllm_norm(f: dict) -> List[str]:
    if not f["has_gemma_rms_norm"]:
        return ["no GemmaRMSNorm in vllm.model_executor.layers.layernorm"]
    return (_expect(f["forward_cuda"] == ["self", "x", "residual"],
                    f"GemmaRMSNorm.forward_cuda takes {f['forward_cuda']}, not (self, x, residual)")
            + _expect(f["init"][:3] == ["self", "hidden_size", "eps"],
                      f"GemmaRMSNorm.__init__ takes {f['init']}: weight / variance_epsilon may have moved"))


def gather_vllm_prefill() -> Optional[dict]:
    from . import switches  # noqa: PLC0415

    from . import vllm_prefill  # noqa: PLC0415

    if switches.arch() not in vllm_prefill.ARCHES:
        return None                 # RDNA's kernels: the route reports it and stays out
    from vllm.v1.attention.ops import prefix_prefill  # noqa: PLC0415
    from vllm.v1.attention.ops import triton_unified_attention as tua  # noqa: PLC0415

    fn = vllm_prefill._ORIG or prefix_prefill.context_attention_fwd
    u = vllm_prefill._ORIG_U or tua.unified_attention
    return {"context_attention_fwd": names(fn), "unified_attention": names(u)}


def judge_vllm_prefill(f: dict) -> List[str]:
    from . import vllm_prefill  # noqa: PLC0415

    need = {"q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k", "softmax_scale",
            "causal", "block_table"}
    lost = sorted(need - set(f["unified_attention"]))
    # the stand-ins take vLLM's own signature and read what they need by name; a parameter a
    # newer vLLM adds is refused per call unless it is at its default -- a contract failure here
    # would turn the route off for every call, not only the calls that pass it
    p_lost = sorted(vllm_prefill._P_NEED - set(f["context_attention_fwd"]))
    return (_expect(not p_lost, f"context_attention_fwd lost {p_lost}")
            + _expect(not lost, f"unified_attention lost {lost}"))


def gather_vllm_backend() -> dict:
    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl  # noqa: F401,PLC0415
    from vllm.v1.attention.backends.registry import (  # noqa: PLC0415
        AttentionBackendEnum, register_backend)

    return {"register_backend": callable(register_backend),
            "backend_slots": list(AttentionBackendEnum.__members__)}


def judge_vllm_backend(f: dict) -> List[str]:
    from . import switches, vllm_backend  # noqa: PLC0415

    slot = (switches.value("vllm_backend") or os.environ.get("HK_VLLM_BACKEND_SLOT")
            or vllm_backend.SLOT)
    return (_expect(f["register_backend"], "vLLM has no register_backend")
            + _expect(slot in f["backend_slots"], f"vLLM's AttentionBackendEnum has no {slot}"))


def gather_vllm_dsv41_index() -> Optional[dict]:
    if not _gfx950():
        return None
    import ast  # noqa: PLC0415

    from . import vllm_dsv41_index as PI  # noqa: PLC0415

    path = PI.source_path()
    if path is None:
        return None                 # no DeepSeek-V4.1 mono path: the route reports it and stays out
    # Read, not imported: importing the model package costs every process seconds, and vLLM
    # imports it later anyway (the route patches it then, and checks the signature again).
    with open(path) as fh:
        tree = ast.parse(fh.read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    sel = fns.get(PI.TARGET)
    lay = path.replace(os.path.join("models", "deepseek_v41", "amd", "mono_decode.py"),
                       os.path.join("v1", "attention", "ops", "rocm_paged_mxfp4_indexer.py"))
    has_layout = False
    if os.path.isfile(lay):
        with open(lay) as fh:
            has_layout = "def rocm_paged_mxfp4_cache_layout" in fh.read()
    return {"args": [a.arg for a in sel.args.args] if sel is not None else None,
            "has_mono_runner": "_mono_runner" in fns, "has_layout": has_layout}


def judge_vllm_dsv41_index(f: dict) -> List[str]:
    from . import vllm_dsv41_index as PI  # noqa: PLC0415

    return (_expect(f["args"] == names(PI._select), f"{PI.MODULE}.{PI.TARGET} takes {f['args']}, "
                                                    f"the stand-in {names(PI._select)}")
            + _expect(f["has_mono_runner"], f"{PI.MODULE} has no _mono_runner")
            + _expect(f["has_layout"], "vLLM has no rocm_paged_mxfp4_cache_layout"))


# -- SGLang -----------------------------------------------------------------------------------


def gather_sglang_linear() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang as hks  # noqa: PLC0415

    for home in hks._LINEAR_HOMES:
        try:
            cls = getattr(importlib.import_module(home), "UnquantizedLinearMethod", None)
        except ImportError:
            continue
        if cls is not None:
            return {"apply": names(cls.apply)}
    return {"apply": None}


def judge_sglang_linear(f: dict) -> List[str]:
    from . import sglang as hks  # noqa: PLC0415

    if f["apply"] is None:
        return [f"no UnquantizedLinearMethod in {hks._LINEAR_HOMES}"]
    return _expect(f["apply"] == ["self", "layer", "x", "bias"],
                   f"SGLang's UnquantizedLinearMethod.apply takes {f['apply']}")


def gather_sglang_norm() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang as hks  # noqa: PLC0415

    L = importlib.import_module(hks._NORM_HOME)
    cls = getattr(L, "GemmaRMSNorm", None)
    return {"has_gemma_rms_norm": cls is not None,
            "forward_hip": names(cls.forward_hip) if cls is not None else None,
            "says_aiter_and_vllm_norms": hasattr(L, "_use_aiter") and hasattr(L, "_has_vllm_rms_norm")}


def judge_sglang_norm(f: dict) -> List[str]:
    from . import sglang as hks  # noqa: PLC0415

    if not f["has_gemma_rms_norm"]:
        return [f"no GemmaRMSNorm in {hks._NORM_HOME}"]
    want = ["self", "x", "residual", "post_residual_addition"]
    return (_expect(f["forward_hip"] == want,
                    f"GemmaRMSNorm.forward_hip takes {f['forward_hip']}, not {want}")
            + _expect(f["says_aiter_and_vllm_norms"],
                      f"{hks._NORM_HOME} no longer says whether aiter / vLLM norms are in use"))


def gather_sglang_moe() -> dict:
    from . import sglang_moe  # noqa: PLC0415

    topk, kernels, config = sglang_moe._homes()
    return {"topk_softmax": signature(topk.topk_softmax),
            "invoke_fused_moe_kernel": signature(kernels.invoke_fused_moe_kernel),
            "try_get_optimal_moe_config": signature(config.try_get_optimal_moe_config)}


def judge_sglang_moe(f: dict) -> List[str]:
    from . import sglang_moe  # noqa: PLC0415

    return (same_signature(f["topk_softmax"], sglang_moe.topk_softmax, "topk_softmax")
            + same_signature(f["invoke_fused_moe_kernel"], sglang_moe.invoke_fused_moe_kernel,
                             "invoke_fused_moe_kernel")
            + same_signature(f["try_get_optimal_moe_config"],
                             sglang_moe.try_get_optimal_moe_config, "try_get_optimal_moe_config"))


# -- SGLang-Diffusion -------------------------------------------------------------------------


def gather_sgld_attn() -> dict:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = sys.modules.get(sd.TARGET)
    if mod is None:
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(sd.TARGET)
    return {"calls_sdpa": "scaled_dot_product_attention" in inspect.getsource(mod)}


def judge_sgld_attn(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return _expect(f["calls_sdpa"], f"{sd.TARGET} no longer calls scaled_dot_product_attention")


def gather_sgld_linear() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    cls = getattr(importlib.import_module(sd.LINEAR_HOME), "UnquantizedLinearMethod")
    return {"apply": names(cls.apply)}


def judge_sgld_linear(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return _expect(f["apply"][:4] == ["self", "layer", "x", "bias"],
                   f"{sd.LINEAR_HOME}.UnquantizedLinearMethod.apply takes {f['apply']}")


def gather_sgld_convrot() -> Optional[dict]:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    try:
        mod = importlib.import_module(sd.CONVROT_HOME)
    except ImportError:
        return None                 # an SGLang without convrot_int8: the route says so itself
    cls = getattr(mod, "ConvRotInt8Config", None)
    return {"hooks": cls is not None and all(hasattr(cls, a) for a in ("resolve_backend", "_select"))
            and hasattr(mod, "BACKENDS") and hasattr(mod, "BACKEND_GROUP_SIZES")}


def judge_sgld_convrot(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return _expect(f["hooks"],
                   f"{sd.CONVROT_HOME} no longer has the ConvRotInt8Config hooks hk extends")


def gather_sgld_vaeconv() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    cls = getattr(importlib.import_module(sd.VAE_HOME), "WanCausalConv3d")
    import torch  # noqa: PLC0415

    return {"conv3d": issubclass(cls, torch.nn.Conv3d),
            "conv_forward": names(cls._conv_forward),
            # the time padding is WanCausalConv3d's own, the conv's is 0 (checked per call too)
            "pads_itself": "self.padding = (0, 0, 0)" in inspect.getsource(cls.__init__),
            # the test that puts the VAE in channels-last, which the route answers no
            "cl3d_switch": callable(getattr(importlib.import_module(sd.VAE_HOME),
                                            "_conv3d_weight_is_channels_last_3d", None))}


def judge_sgld_vaeconv(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return (_expect(f["conv3d"], f"{sd.VAE_HOME}.WanCausalConv3d is no longer an nn.Conv3d")
            + _expect(f["conv_forward"][:4] == ["self", "input", "weight", "bias"],
                      f"nn.Conv3d._conv_forward takes {f['conv_forward']}")
            + _expect(f["pads_itself"],
                      f"{sd.VAE_HOME}.WanCausalConv3d no longer pads time itself")
            + _expect(f.get("cl3d_switch", False),
                      f"{sd.VAE_HOME} has no _conv3d_weight_is_channels_last_3d"))


def gather_sgld_rope() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = importlib.import_module(sd.QWEN21_HOME)
    fn = mod.apply_rope
    src = inspect.getsource(fn)
    # SGLang since 2026-11 keeps the eager math in _eager_rope, which apply_rope returns off CUDA
    # (ROCm); its fused CUDA kernel is the same multiply.
    eager = getattr(mod, "_eager_rope", None)
    if eager is not None and "return _eager_rope(x, rope)" in src:
        src += inspect.getsource(eager)
    return {"apply_rope": names(fn),
            # the computation rotary_pairs reproduces
            "complex_multiply": "view_as_complex(x.float()" in src and "rope[None, :, None]" in src}


def judge_sgld_rope(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return (_expect(f["apply_rope"] == ["x", "rope"], f"{sd.QWEN21_HOME}.apply_rope takes {f['apply_rope']}")
            + _expect(f["complex_multiply"],
                      f"{sd.QWEN21_HOME}.apply_rope no longer multiplies view_as_complex(x.float()) by rope"))


#: apply_qk_norm_rope's fallback off CUDA, as SGLang has written it: through apply_qk_norm (whose
#: own fallback off CUDA is norm(x)) up to 0.5.22, norm(x) directly from 0.5.23.
NORM_THEN_ROPE = ("apply_rope(apply_qk_norm(x, norm), rope)", "apply_rope(norm(x), rope)")


def norm_then_rope(src: str) -> bool:
    """Whether apply_qk_norm_rope's source falls back to the chain rmsnorm_rotary_pairs computes:
    the norm, then apply_rope."""
    return any(form in src for form in NORM_THEN_ROPE)


def gather_sgld_qknorm() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = importlib.import_module(sd.QWEN21_HOME)
    fn = mod.apply_qk_norm_rope
    from sglang.srt.layers.layernorm import RMSNorm  # noqa: PLC0415

    native = inspect.getsource(RMSNorm.forward_native)
    return {"apply_qk_norm_rope": names(fn),
            # the chain rmsnorm_rotary_pairs computes: norm, then apply_rope
            "norm_then_rope": norm_then_rope(inspect.getsource(fn)),
            "rmsnorm_native": all(s in native for s in (
                ".pow(2).mean(dim=-1, keepdim=True)", "torch.rsqrt(variance + self.variance_epsilon)",
                "self.weight * x.to(orig_dtype)"))}


def judge_sgld_qknorm(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return (_expect(f["apply_qk_norm_rope"] == ["x", "norm", "rope"],
                    f"{sd.QWEN21_HOME}.apply_qk_norm_rope takes {f['apply_qk_norm_rope']}")
            + _expect(f["norm_then_rope"],
                      f"{sd.QWEN21_HOME}.apply_qk_norm_rope no longer falls back to norm then rope")
            + _expect(f["rmsnorm_native"],
                      "sglang.srt.layers.layernorm.RMSNorm.forward_native computes differently"))


def gather_sgld_ditnorm() -> dict:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = importlib.import_module(sd.DITNORM_HOME)
    res, plain = mod._ScaleResidualNormScaleShift, mod._NormScaleShift
    src = inspect.getsource(res.forward_native) + inspect.getsource(plain.forward_native)
    return {"res_forward_native": names(res.forward_native),
            "forward_native": names(plain.forward_native),
            # the chain the kernel computes: gated residual, the norm, then the scale and shift
            "chain": all(t in src for t in ("residual + x * gate", "self.norm(", "fuse_scale_shift_kernel("))}


def judge_sgld_ditnorm(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    return (_expect(f["res_forward_native"][:6] == ["self", "residual", "x", "gate", "shift", "scale"],
                    f"{sd.DITNORM_HOME}._ScaleResidualNormScaleShift.forward_native takes "
                    f"{f['res_forward_native']}")
            + _expect(f["forward_native"][:4] == ["self", "x", "shift", "scale"],
                      f"{sd.DITNORM_HOME}._NormScaleShift.forward_native takes {f['forward_native']}")
            + _expect(f["chain"], f"{sd.DITNORM_HOME}'s scale-residual-norm chain changed"))


def _module_or_none(home: str):
    """The engine module `home`, or None where this engine has no such module (nothing to route).
    A module that is there but fails to import is a broken contract, not an absent one."""
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(home)
    except ModuleNotFoundError as e:
        if e.name and (home == e.name or home.startswith(e.name + ".")):
            return None
        raise


def _flat(src: str) -> str:
    """Source without whitespace or trailing commas: the same code however it is wrapped."""
    return "".join(src.split()).replace(",)", ")").replace(",]", "]")


def gather_sgld_h3vae() -> Optional[dict]:
    """MiniMax H3's BaseConv3d: the padding the route reproduces for a one-frame input (spatial in
    pad_mode, kt - 1 causal zero frames in front) and the conv3d it then runs. None where this
    SGLang has no H3 VAE."""
    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = _module_or_none(sd.H3VAE_HOME)
    if mod is None:
        return None
    import torch  # noqa: PLC0415

    cls = mod.BaseConv3d
    init, tpad = _flat(inspect.getsource(cls.__init__)), _flat(inspect.getsource(cls._apply_temporal_padding))
    pad, fwd = _flat(inspect.getsource(cls._apply_padding)), _flat(inspect.getsource(cls.forward))
    return {"conv3d": issubclass(cls, torch.nn.Conv3d),
            "forward": names(cls.forward),
            "sets": all(f"self.{a}=" in init for a in ("pad_mode", "pad_mode_t", "causal")),
            # one frame, zeros: kt - 1 zero frames in front
            "one_frame_causal_zeros": "(0,0,0,0,self.kernel_size[0]-1,0)" in tpad
                                      and 'mode="constant"' in tpad,
            "spatial_pad": "(self.padding[2],self.padding[2],self.padding[1],self.padding[1],0,0)" in pad
                           and "mode=self.pad_mode" in pad,
            "conv": "F.conv3d(" in fwd and "stride=self.stride" in fwd and "padding=0" in fwd
                    and "dilation=self.dilation" in fwd and "groups" not in fwd}


def judge_sgld_h3vae(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    where = f"{sd.H3VAE_HOME}.BaseConv3d"
    return (_expect(f["conv3d"], f"{where} is no longer an nn.Conv3d")
            + _expect(f["forward"] in (["self", "x"], ["self", "x", "pre_padded"]),
                      f"{where}.forward takes {f['forward']}")
            + _expect(f["sets"], f"{where} no longer sets pad_mode / pad_mode_t / causal")
            + _expect(f["one_frame_causal_zeros"],
                      f"{where} no longer pads one frame with kt - 1 zero frames in front")
            + _expect(f["spatial_pad"], f"{where} pads space differently")
            + _expect(f["conv"], f"{where}.forward no longer runs F.conv3d(stride, padding=0, dilation)"))


def gather_sgld_patchembed() -> Optional[dict]:
    """Qwen3-VL's vision patch embedding: a conv3d with kernel == stride over each
    (C, T, P, P) patch, which the route runs as its GEMM. None where this SGLang has no
    Qwen3-VL."""
    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = _module_or_none(sd.PATCHEMBED_HOME)
    cls = getattr(mod, "Qwen3VLVisionPatchEmbed", None) if mod is not None else None
    if cls is None:
        return None
    init, fwd = _flat(inspect.getsource(cls.__init__)), _flat(inspect.getsource(cls.forward))
    return {"forward": names(cls.forward),
            "sets": all(f"self.{a}=" in init for a in ("patch_size", "temporal_patch_size",
                                                       "in_channels", "embed_dim", "proj")),
            "kernel_is_stride": "stride=kernel_size" in init,
            # each patch is (C, T, P, P), the weight's order after its out dim
            "patch_order": "self.in_channels,self.temporal_patch_size,self.patch_size,self.patch_size" in fwd
                           and ".view(-1,self.embed_dim)" in fwd}


def judge_sgld_patchembed(f: dict) -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    where = f"{sd.PATCHEMBED_HOME}.Qwen3VLVisionPatchEmbed"
    return (_expect(f["forward"] == ["self", "hidden_states"], f"{where}.forward takes {f['forward']}")
            + _expect(f["sets"], f"{where} no longer sets patch_size / temporal_patch_size / "
                                 f"in_channels / embed_dim / proj")
            + _expect(f["kernel_is_stride"], f"{where}'s proj no longer has stride == kernel_size")
            + _expect(f["patch_order"], f"{where}.forward no longer views patches as (C, T, P, P)"))


#: route -> (gather, judge). gather returns the route's facts, or None where it does not apply here.
SPLIT: Dict[str, tuple] = {
    r: (globals()[f"gather_{r}"], globals()[f"judge_{r}"]) for r in (
        "vllm_mxfp8", "vllm_moe_a8w4", "vllm_int4", "vllm_gptq", "vllm_quark", "vllm_gemm", "vllm_moe",
        "vllm_moe_wna16", "vllm_sdpa", "vllm_backend", "vllm_dsv41_index", "vllm_sparse_mla",
        "vllm_prefill", "vllm_norm", "sglang_linear", "sglang_norm", "sglang_moe", "sgld_attn", "sgld_linear",
        "sgld_convrot", "sgld_vaeconv", "sgld_rope", "sgld_qknorm", "sgld_ditnorm", "sgld_h3vae",
        "sgld_patchembed")}


def _checker(name: str) -> Callable[[], Optional[List[str]]]:
    def run() -> Optional[List[str]]:
        facts = gather(name)
        return None if facts is None else judge(name, facts)
    run.__name__ = f"check_{name}"
    return run


#: route -> its check on the installed engines: the route's problems ([] holds), or None where the
#: route does not apply here. judge(gather()) of SPLIT's pair.
CHECKS: Dict[str, Callable[[], Optional[List[str]]]] = {r: _checker(r) for r in SPLIT}


def gather(name: str):
    """One route's facts: a dict, None where the route does not apply here, or {"_error": ...}
    when reading the engine raised (a moved module is a broken contract)."""
    try:
        return SPLIT[name][0]()
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def judge(name: str, facts) -> List[str]:
    """One route's problems from its facts ([] holds; None facts: the route does not apply)."""
    if facts is None:
        return []
    if "_error" in facts:
        return [facts["_error"]]
    try:
        return SPLIT[name][1](facts) or []
    except Exception as e:  # noqa: BLE001
        return [f"{type(e).__name__}: {e}"]


def check_each(routes: Iterable[str]) -> Dict[str, List[str]]:
    """Each route among `routes` whose contract is broken, with its problems; {} if all hold."""
    broken: Dict[str, List[str]] = {}
    for name in routes:
        if name not in CHECKS:
            continue
        try:
            got = CHECKS[name]()    # judge(gather()), unless a test or a caller replaced it
        except Exception as e:  # noqa: BLE001 -- a moved module is a broken contract
            got = [f"{type(e).__name__}: {e}"]
        if got:
            broken[name] = list(got)
    return broken


def check(routes: Iterable[str]) -> List[str]:
    """Every broken contract among `routes`, as "route: problem" lines; [] if all hold."""
    return [f"{name}: {p}" for name, got in check_each(routes).items() for p in got]


def engine_version(engine: str) -> str:
    mod = {"vllm": "vllm", "sglang": "sglang", "sglang_diffusion": "sglang"}.get(engine, engine)
    try:
        import importlib.metadata as md  # noqa: PLC0415

        return f"{mod} {md.version(mod)}"
    except Exception:  # noqa: BLE001
        return f"{mod} (version unknown)"


#: What refuse() / refuse_route() say went wrong.
BROKEN = "engine contract FAILED"
RAISED = "the route failed to apply"


def refuse_route(engine: str, route: str, problems: List[str], what: str = BROKEN) -> str:
    """The warning line for one route left off; the engine's other routes are unaffected. Also
    returned."""
    from . import switches  # noqa: PLC0415

    measured = f"; routes measured on vllm {MEASURED_VLLM}" if engine == "vllm" else ""
    r = switches.ROUTES.get(route)
    off = (f"{r.env}=0" if r is not None
           else f"{route}=" if route.startswith("HK_") else "HK_DISABLE=1")
    line = (f"[hk] WARNING: {engine}: route {route} OFF -- {what} "
            f"({engine_version(engine)}{measured}): " + "; ".join(problems)
            + f". hk's other routes in this engine are unaffected. Use the engine version hk "
              f"was measured on, or set {off} to silence this. (pid {os.getpid()})")
    print(line, file=sys.stderr, flush=True)
    return line


def apply_routes(engine: str, routes: Iterable[str],
                 appliers: Iterable[tuple]) -> tuple:
    """Check each route's contract and apply the ones that hold, in `appliers`' order.

    `appliers` is (route, apply) pairs; apply() patches and returns its one-line report, or
    (report, revert). A route with a broken contract is not applied, and one whose apply raises
    is reported and skipped (its apply is what must not leave a half patch behind); either way
    `refuse_route` says so and the others go on. Returns (reports, applied route names,
    {route left off: why})."""
    routes = list(routes)
    off = {name: (problems, BROKEN) for name, problems in check_each(routes).items()}
    for name, (problems, what) in off.items():
        refuse_route(engine, name, problems, what)
    reports, applied = [], []
    for name, how in appliers:
        if name not in routes or name in off:
            continue
        try:
            got = how()
        except Exception as e:  # noqa: BLE001 -- one route's failure leaves the others on
            off[name] = ([f"raised {type(e).__name__}: {e}"], RAISED)
            refuse_route(engine, name, off[name][0], RAISED)
            continue
        reports.append(got[0] if isinstance(got, tuple) else got)
        applied.append(name)
    return reports, applied, {n: p for n, (p, _) in off.items()}


def refuse(engine: str, problems: List[str], what: str = BROKEN) -> str:
    """The one warning line for a whole engine left stock (a user's own modules failed to
    load, say). Also returned."""
    measured = f"; routes measured on vllm {MEASURED_VLLM}" if engine == "vllm" else ""
    line = (f"[hk] WARNING: {engine}: {what} -- hk patched NOTHING in this engine, it runs "
            f"stock ({engine_version(engine)}{measured}): " + "; ".join(problems)
            + f". Use the engine version hk was measured on, or set HK_DISABLE=1 to silence "
              f"this. (pid {os.getpid()})")
    print(line, file=sys.stderr, flush=True)
    return line


#: The roles a recorded image has (`--role`), in the support matrix's order.
ROLES = ("production", "measured", "development")


def stack() -> dict:
    """What the engines run on, as the support matrix shows it: the interpreter, torch and the HIP
    it was built with, Triton's distribution (`triton`, `triton-rocm` or `pytorch-triton-rocm`),
    AITER and SGLang's Radeon package, each where installed."""
    import importlib.metadata as md  # noqa: PLC0415
    import platform  # noqa: PLC0415

    out = {"python": platform.python_version()}
    for d in ("torch", "triton", "triton-rocm", "pytorch-triton-rocm", "aiter",
              "sglang-radeon-rdna3"):
        try:
            out[d] = md.version(d)
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch  # noqa: PLC0415

        out["hip"] = torch.version.hip
    except Exception:  # noqa: BLE001
        pass
    return out


def record(routes: Optional[Iterable[str]] = None) -> dict:
    """The installed engines' facts for `routes` (default: every route), as a fixture for
    tests/hk/ir/test_engine_facts.py (tests/hk/ir/engine_facts/): what the engine is and runs on,
    and each route's gathered facts."""
    from . import switches  # noqa: PLC0415

    return {"engines": {e: engine_version(e) for e in ("vllm", "sglang")},
            "arch": switches.arch(), "stack": stack(),
            "facts": {r: gather(r) for r in (routes or CHECKS)}}


def main(argv=None) -> int:
    """python3 -m hk.integration.contract --record OUT.json [--image DIGEST] [--deployed-on ARCH]
    [--role production|measured|development]"""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--record", required=True, help="write the facts here")
    ap.add_argument("--image", default="", help="the image's digest (sha256:...), for the record")
    ap.add_argument("--deployed-on", default="", help="the arch this image serves on")
    ap.add_argument("--role", choices=ROLES, default="production",
                    help="production: an image we serve with; measured: the engine version a "
                         "route was measured on; development: neither, the newest we have")
    a = ap.parse_args(argv)
    rec = record()
    rec.update(image=a.image, deployed_on=a.deployed_on or rec["arch"], role=a.role)
    with open(a.record, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    bad = {r: judge(r, f) for r, f in rec["facts"].items() if judge(r, f)}
    print(f"recorded {len(rec['facts'])} routes to {a.record}; broken here: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
