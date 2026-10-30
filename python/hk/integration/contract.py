"""The engine contract, checked when a plugin starts: every engine entry point an enabled route
patches or falls back to still exists, with the parameters the route uses
(docs/design/serving-readiness.md, A9).

Routes patch engine internals by name, and they were measured against one engine build. If a
release renames, moves or re-signs one of those names, the patch does not fail loudly. Either it
quietly routes nothing, or its stand-in passes arguments into the wrong slots. A bare-signature
wrapper once did the second, and it changed vLLM's stock MoE path too. So a plugin calls `check()`
before it patches anything. If any route it is about to apply has a broken contract, it patches
**nothing** in that engine, prints one warning line and leaves the engine stock (`refuse()`).

**All or nothing, per engine.** The routes in one engine patch different entry points, but a
broken contract on any of them means this engine is not the build they were measured on. A
half-patched engine was never measured as a whole, and nobody chose it. Each route's apply
already falls back for shapes and arches it does not take.

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


# -- vLLM -------------------------------------------------------------------------------------


def _gfx950() -> bool:
    from . import switches  # noqa: PLC0415

    return switches.arch() == "gfx950"


def _vllm_mxfp8() -> Optional[List[str]]:
    if not _gfx950():
        return None                 # gfx950's scaled MFMA: the route reports it and stays out
    from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (  # noqa: PLC0415
        rocm_mxfp8_block32_gemm)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: PLC0415
        mxfp8_e4m3_quantize)

    from . import vllm_mxfp8 as PX  # noqa: PLC0415

    cls = PX._cls()
    return (_expect(names(cls.process_weights_after_loading)[:2] == ["self", "layer"],
                    f"{cls.__name__}.process_weights_after_loading takes "
                    f"{names(cls.process_weights_after_loading)}")
            + _expect(names(cls.apply_weights)[:4] == ["self", "layer", "x", "bias"],
                      f"{cls.__name__}.apply_weights takes {names(cls.apply_weights)}")
            + _expect(len(_params(rocm_mxfp8_block32_gemm)) >= 5,
                      "rocm_mxfp8_block32_gemm takes fewer than (x, xs, w, ws, out_dtype)")
            + _expect("x" in names(mxfp8_e4m3_quantize), "mxfp8_e4m3_quantize has no `x`"))


def _vllm_moe_a8w4() -> Optional[List[str]]:
    from . import _common  # noqa: PLC0415

    if not _gfx950() or not _common.importable("aiter"):
        return None                 # nothing to patch: the route reports it and stays out
    import aiter.fused_moe as afm  # noqa: PLC0415
    from aiter.fused_moe import moe_sorting  # noqa: PLC0415
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: PLC0415

    from . import vllm_moe_a8w4 as PM  # noqa: PLC0415

    fn = getattr(afm.fused_moe, "__wrapped__", afm.fused_moe)
    have = set(inspect.signature(fn).parameters)
    lost = [p for p in PM.AITER_PARAMS if p not in have]
    return (_expect(not lost, f"aiter.fused_moe.fused_moe lost {lost}")
            + _expect(names(fn)[:5] == ["hidden_states", "w1", "w2", "topk_weight", "topk_ids"],
                      f"aiter.fused_moe.fused_moe begins {names(fn)[:5]}")
            + _expect("block_size" in names(moe_sorting), "aiter's moe_sorting has no block_size")
            + _expect(len(_params(fused_dynamic_mxfp8_quant_moe_sort)) >= 6,
                      "aiter's fused_dynamic_mxfp8_quant_moe_sort takes fewer than 6 arguments"))


def _vllm_sparse_mla() -> Optional[List[str]]:
    if not _gfx950():
        return None                 # gfx950's DMA gather: the route reports it and stays out
    from . import vllm_sparse_mla as PS  # noqa: PLC0415

    return PS.problems()


def _vllm_int4() -> List[str]:
    from vllm import _custom_ops as ops  # noqa: PLC0415
    from vllm.model_executor.layers.quantization import auto_awq  # noqa: PLC0415

    cls = auto_awq.AutoAWQLinearMethod
    src = inspect.getsource(cls.create_weights)
    return (_expect(names(cls.process_weights_after_loading) == ["self", "layer"],
                    f"AutoAWQLinearMethod.process_weights_after_loading takes "
                    f"{names(cls.process_weights_after_loading)}")
            + _expect(names(cls.apply) == ["self", "layer", "x", "bias"],
                      f"AutoAWQLinearMethod.apply takes {names(cls.apply)}")
            + _expect(len(names(ops.awq_gemm)) == 5, "awq_gemm no longer takes 5 arguments")
            + _expect(len(names(ops.awq_dequantize)) == 6,
                      "awq_dequantize no longer takes 6 arguments")
            + [f"AWQ layers no longer have {p}" for p in ("qweight", "qzeros", "scales")
               if f'"{p}"' not in src and f"'{p}'" not in src])


def _vllm_gemm() -> List[str]:
    from vllm.model_executor.layers import utils as vu  # noqa: PLC0415
    from vllm.v1.worker import gpu_worker  # noqa: PLC0415

    from . import vllm_gemm  # noqa: PLC0415

    out = [f"dispatch_unquantized_gemm now requires `{p.name}`"
           for p in inspect.signature(vu.dispatch_unquantized_gemm).parameters.values()
           if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    if not out:
        got = names(vu.dispatch_unquantized_gemm())
        out += _expect(got == names(vllm_gemm._gemm) == ["layer", "x", "weight", "bias"],
                       f"the GEMM dispatch_unquantized_gemm hands back takes {got}")
    out += _expect(callable(getattr(gpu_worker.Worker, "load_model", None)),
                   "vLLM's GPU worker has no load_model (the tuning at load wraps it)")
    return out


def _vllm_moe() -> List[str]:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm  # noqa: PLC0415

    from . import vllm_moe  # noqa: PLC0415

    return (same_parameters(fm.invoke_fused_moe_triton_kernel,
                            vllm_moe.invoke_fused_moe_triton_kernel,
                            "invoke_fused_moe_triton_kernel")
            + same_parameters(fm.try_get_optimal_moe_config, vllm_moe.try_get_optimal_moe_config,
                              "try_get_optimal_moe_config"))


def _vllm_sdpa() -> List[str]:
    import torch  # noqa: PLC0415
    from vllm.v1.attention.ops import vit_attn_wrappers as w  # noqa: PLC0415

    return _expect(hasattr(w, "apply_sdpa") and getattr(w, "F", None) is torch.nn.functional,
                   "vLLM's ViT attention no longer calls F.scaled_dot_product_attention "
                   "(vit_attn_wrappers.apply_sdpa)")


def _vllm_prefill() -> Optional[List[str]]:
    from . import switches  # noqa: PLC0415

    if switches.arch() != "gfx1100":
        return None                 # gfx1100's kernel: the route reports it and stays out
    from vllm.v1.attention.ops import prefix_prefill  # noqa: PLC0415

    from . import vllm_prefill  # noqa: PLC0415

    from vllm.v1.attention.ops import triton_unified_attention as tua  # noqa: PLC0415

    fn = vllm_prefill._ORIG or prefix_prefill.context_attention_fwd
    u = vllm_prefill._ORIG_U or tua.unified_attention
    need = {"q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k", "softmax_scale",
            "causal", "block_table"}
    lost = sorted(need - set(names(u)))
    return (same_parameters(fn, vllm_prefill.context_attention_fwd, "context_attention_fwd")
            + _expect(not lost, f"unified_attention lost {lost}"))


def _vllm_backend() -> List[str]:
    import os  # noqa: PLC0415

    from vllm.v1.attention.backend import AttentionBackend, AttentionImpl  # noqa: F401,PLC0415
    from vllm.v1.attention.backends.registry import (  # noqa: PLC0415
        AttentionBackendEnum, register_backend)

    from . import switches, vllm_backend  # noqa: PLC0415

    slot = (switches.value("vllm_backend") or os.environ.get("HK_VLLM_BACKEND_SLOT")
            or vllm_backend.SLOT)
    return (_expect(callable(register_backend), "vLLM has no register_backend")
            + _expect(getattr(AttentionBackendEnum, slot, None) is not None,
                      f"vLLM's AttentionBackendEnum has no {slot}"))


def _vllm_dsv41_index() -> Optional[List[str]]:
    if not _gfx950():
        return None
    import ast  # noqa: PLC0415

    from . import vllm_dsv41_index as PI  # noqa: PLC0415

    path = PI.source_path()
    if path is None:
        return None                 # no DeepSeek-V4.1 mono path: the route reports it and stays out
    # Read, not imported: importing the model package costs every process seconds, and vLLM
    # imports it later anyway (the route patches it then, and checks the signature again).
    with open(path) as f:
        tree = ast.parse(f.read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    sel = fns.get(PI.TARGET)
    args = [a.arg for a in sel.args.args] if sel is not None else None
    lay = path.replace(os.path.join("models", "deepseek_v41", "amd", "mono_decode.py"),
                       os.path.join("v1", "attention", "ops", "rocm_paged_mxfp4_indexer.py"))
    has_layout = False
    if os.path.isfile(lay):
        with open(lay) as f:
            has_layout = "def rocm_paged_mxfp4_cache_layout" in f.read()
    return (_expect(args == names(PI._select), f"{PI.MODULE}.{PI.TARGET} takes {args}, the "
                                                 f"stand-in {names(PI._select)}")
            + _expect("_mono_runner" in fns, f"{PI.MODULE} has no _mono_runner")
            + _expect(has_layout, "vLLM has no rocm_paged_mxfp4_cache_layout"))


# -- SGLang -----------------------------------------------------------------------------------


def _sglang_linear() -> List[str]:
    import importlib  # noqa: PLC0415

    from . import sglang as hks  # noqa: PLC0415

    for home in hks._LINEAR_HOMES:
        try:
            cls = getattr(importlib.import_module(home), "UnquantizedLinearMethod", None)
        except ImportError:
            continue
        if cls is not None:
            return _expect(names(cls.apply) == ["self", "layer", "x", "bias"],
                           f"SGLang's UnquantizedLinearMethod.apply takes {names(cls.apply)}")
    return [f"no UnquantizedLinearMethod in {hks._LINEAR_HOMES}"]


def _sglang_moe() -> List[str]:
    from . import sglang_moe  # noqa: PLC0415

    topk, kernels, config = sglang_moe._homes()
    return (same_parameters(topk.topk_softmax, sglang_moe.topk_softmax, "topk_softmax")
            + same_parameters(kernels.invoke_fused_moe_kernel, sglang_moe.invoke_fused_moe_kernel,
                              "invoke_fused_moe_kernel")
            + same_parameters(config.try_get_optimal_moe_config,
                              sglang_moe.try_get_optimal_moe_config, "try_get_optimal_moe_config"))


# -- SGLang-Diffusion -------------------------------------------------------------------------


def _sgld_attn() -> List[str]:
    from . import sglang_diffusion as sd  # noqa: PLC0415

    mod = sys.modules.get(sd.TARGET)
    if mod is None:
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(sd.TARGET)
    return _expect("scaled_dot_product_attention" in inspect.getsource(mod),
                   f"{sd.TARGET} no longer calls scaled_dot_product_attention")


def _sgld_linear() -> List[str]:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    cls = getattr(importlib.import_module(sd.LINEAR_HOME), "UnquantizedLinearMethod")
    return _expect(names(cls.apply)[:4] == ["self", "layer", "x", "bias"],
                   f"{sd.LINEAR_HOME}.UnquantizedLinearMethod.apply takes {names(cls.apply)}")


def _sgld_convrot() -> List[str]:
    import importlib  # noqa: PLC0415

    from . import sglang_diffusion as sd  # noqa: PLC0415

    try:
        mod = importlib.import_module(sd.CONVROT_HOME)
    except ImportError:
        return []                   # an SGLang without convrot_int8: the route says so itself
    cls = getattr(mod, "ConvRotInt8Config", None)
    return _expect(cls is not None and all(hasattr(cls, a) for a in ("resolve_backend", "_select"))
                   and hasattr(mod, "BACKENDS") and hasattr(mod, "BACKEND_GROUP_SIZES"),
                   f"{sd.CONVROT_HOME} no longer has the ConvRotInt8Config hooks hk extends")


#: route -> its contract: a list of problems ([] holds), or None where it does not apply here.
CHECKS: Dict[str, Callable[[], Optional[List[str]]]] = {
    "vllm_mxfp8": _vllm_mxfp8, "vllm_moe_a8w4": _vllm_moe_a8w4, "vllm_int4": _vllm_int4,
    "vllm_gemm": _vllm_gemm, "vllm_moe": _vllm_moe, "vllm_sdpa": _vllm_sdpa,
    "vllm_backend": _vllm_backend, "vllm_dsv41_index": _vllm_dsv41_index,
    "vllm_sparse_mla": _vllm_sparse_mla, "vllm_prefill": _vllm_prefill,
    "sglang_linear": _sglang_linear, "sglang_moe": _sglang_moe,
    "sgld_attn": _sgld_attn, "sgld_linear": _sgld_linear, "sgld_convrot": _sgld_convrot,
}


def check(routes: Iterable[str]) -> List[str]:
    """Every broken contract among `routes`, as "route: problem" lines; [] if all hold."""
    problems = []
    for name in routes:
        fn = CHECKS.get(name)
        if fn is None:
            continue
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001 -- a moved module is a broken contract
            got = [f"{type(e).__name__}: {e}"]
        problems += [f"{name}: {p}" for p in (got or [])]
    return problems


def engine_version(engine: str) -> str:
    mod = {"vllm": "vllm", "sglang": "sglang", "sglang_diffusion": "sglang"}.get(engine, engine)
    try:
        import importlib.metadata as md  # noqa: PLC0415

        return f"{mod} {md.version(mod)}"
    except Exception:  # noqa: BLE001
        return f"{mod} (version unknown)"


#: What refuse() says went wrong.
BROKEN = "engine contract FAILED"
RAISED = "a route failed to apply"


def refuse(engine: str, problems: List[str], what: str = BROKEN) -> str:
    """The one warning line a plugin prints when it leaves an engine stock. Also returned."""
    import os  # noqa: PLC0415

    measured = f"; routes measured on vllm {MEASURED_VLLM}" if engine == "vllm" else ""
    line = (f"[hk] WARNING: {engine}: {what} -- hk patched NOTHING in this engine, it runs "
            f"stock ({engine_version(engine)}{measured}): " + "; ".join(problems)
            + f". Use the engine version hk was measured on, or set HK_DISABLE=1 to silence "
              f"this. (pid {os.getpid()})")
    print(line, file=sys.stderr, flush=True)
    return line
