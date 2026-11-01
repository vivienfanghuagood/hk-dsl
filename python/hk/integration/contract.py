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
    from vllm import _custom_ops as ops  # noqa: PLC0415
    from vllm.model_executor.layers.quantization import auto_gptq  # noqa: PLC0415

    if not hasattr(ops, "gptq_gemm_rdna3"):
        return None                 # no RDNA3 GPTQ kernel in this vLLM: nothing to route
    from vllm.model_executor.kernels.linear.mixed_precision import rdna3_w4a16  # noqa: PLC0415

    cls = auto_gptq.AutoGPTQLinearMethod
    kern = rdna3_w4a16.RDNA3W4A16LinearKernel
    return {"pwal": names(cls.process_weights_after_loading), "apply": names(cls.apply),
            "gptq_gemm_rdna3_nargs": len(names(ops.gptq_gemm_rdna3)),
            "kernel_has_get_weight_params": hasattr(kern, "_get_weight_params"),
            "create_weights_has": _has_names(inspect.getsource(cls.create_weights),
                                             ("qweight", "qzeros", "scales"))}


def judge_vllm_gptq(f: dict) -> List[str]:
    return (_expect(f["pwal"] == ["self", "layer"],
                    f"AutoGPTQLinearMethod.process_weights_after_loading takes {f['pwal']}")
            + _expect(f["apply"] == ["self", "layer", "x", "bias"],
                      f"AutoGPTQLinearMethod.apply takes {f['apply']}")
            + _expect(f["gptq_gemm_rdna3_nargs"] == 5, "gptq_gemm_rdna3 no longer takes 5 arguments")
            + _expect(f["kernel_has_get_weight_params"],
                      "RDNA3W4A16LinearKernel has no _get_weight_params")
            + [f"GPTQ layers no longer have {p}" for p, ok in f["create_weights_has"].items()
               if not ok])


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


def gather_vllm_sdpa() -> dict:
    import torch  # noqa: PLC0415
    from vllm.v1.attention.ops import vit_attn_wrappers as w  # noqa: PLC0415

    return {"calls_torch_sdpa": hasattr(w, "apply_sdpa")
            and getattr(w, "F", None) is torch.nn.functional}


def judge_vllm_sdpa(f: dict) -> List[str]:
    return _expect(f["calls_torch_sdpa"],
                   "vLLM's ViT attention no longer calls F.scaled_dot_product_attention "
                   "(vit_attn_wrappers.apply_sdpa)")


def gather_vllm_prefill() -> Optional[dict]:
    from . import switches  # noqa: PLC0415

    if switches.arch() != "gfx1100":
        return None                 # gfx1100's kernel: the route reports it and stays out
    from vllm.v1.attention.ops import prefix_prefill  # noqa: PLC0415
    from vllm.v1.attention.ops import triton_unified_attention as tua  # noqa: PLC0415

    from . import vllm_prefill  # noqa: PLC0415

    fn = vllm_prefill._ORIG or prefix_prefill.context_attention_fwd
    u = vllm_prefill._ORIG_U or tua.unified_attention
    return {"context_attention_fwd": signature(fn), "unified_attention": names(u)}


def judge_vllm_prefill(f: dict) -> List[str]:
    from . import vllm_prefill  # noqa: PLC0415

    need = {"q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k", "softmax_scale",
            "causal", "block_table"}
    lost = sorted(need - set(f["unified_attention"]))
    return (same_signature(f["context_attention_fwd"], vllm_prefill.context_attention_fwd,
                           "context_attention_fwd")
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


#: route -> (gather, judge). gather returns the route's facts, or None where it does not apply here.
CHECKS: Dict[str, tuple] = {
    r: (globals()[f"gather_{r}"], globals()[f"judge_{r}"]) for r in (
        "vllm_mxfp8", "vllm_moe_a8w4", "vllm_int4", "vllm_gptq", "vllm_gemm", "vllm_moe",
        "vllm_sdpa", "vllm_backend", "vllm_dsv41_index", "vllm_sparse_mla", "vllm_prefill",
        "sglang_linear", "sglang_moe", "sgld_attn", "sgld_linear", "sgld_convrot")}


def gather(name: str):
    """One route's facts: a dict, None where the route does not apply here, or {"_error": ...}
    when reading the engine raised (a moved module is a broken contract)."""
    try:
        return CHECKS[name][0]()
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def judge(name: str, facts) -> List[str]:
    """One route's problems from its facts ([] holds; None facts: the route does not apply)."""
    if facts is None:
        return []
    if "_error" in facts:
        return [facts["_error"]]
    try:
        return CHECKS[name][1](facts) or []
    except Exception as e:  # noqa: BLE001
        return [f"{type(e).__name__}: {e}"]


def check(routes: Iterable[str]) -> List[str]:
    """Every broken contract among `routes`, as "route: problem" lines; [] if all hold."""
    problems = []
    for name in routes:
        if name not in CHECKS:
            continue
        problems += [f"{name}: {p}" for p in judge(name, gather(name))]
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


def record(routes: Optional[Iterable[str]] = None) -> dict:
    """The installed engines' facts for `routes` (default: every route), as a fixture for
    tests/hk/ir/test_engine_facts.py (tests/hk/ir/engine_facts/): what the engine is, and each
    route's gathered facts."""
    from . import switches  # noqa: PLC0415

    return {"engines": {e: engine_version(e) for e in ("vllm", "sglang")},
            "arch": switches.arch(),
            "facts": {r: gather(r) for r in (routes or CHECKS)}}


def main(argv=None) -> int:
    """python3 -m hk.integration.contract --record OUT.json [--image DIGEST] [--deployed-on ARCH]"""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--record", required=True, help="write the facts here")
    ap.add_argument("--image", default="", help="the image's digest (sha256:...), for the record")
    ap.add_argument("--deployed-on", default="", help="the arch this image serves on")
    a = ap.parse_args(argv)
    rec = record()
    rec.update(image=a.image, deployed_on=a.deployed_on or rec["arch"])
    with open(a.record, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    bad = {r: judge(r, f) for r, f in rec["facts"].items() if judge(r, f)}
    print(f"recorded {len(rec['facts'])} routes to {a.record}; broken here: {bad or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
