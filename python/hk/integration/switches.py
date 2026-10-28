"""Which of hk's engine routes are on. hk is **on by default**; off is for comparisons.

A route is one patch an engine plugin can make: vLLM's MXFP8 linear layers on hk's GEMM, SGLang's
MoE on hk's, and so on. A route is on by default when it was validated with real weights on that
GPU, with outputs bit-identical to the engine's or with error within the engine's own noise
(docs/integration.md#on-by-default has the evidence). Lossy routes (ConvRot INT8) and routes
without that evidence stay off until you turn them on.

**Precedence.** The first rule that applies decides:

1. `HK_DISABLE=1` turns every route off, in every engine. Nothing else overrides it.
2. An engine's own switch set to off turns off all of that engine's routes: `HK_PATCH_VLLM=0`,
   `HK_PATCH_SGLANG=0` or `HK_SGLD=0`.
3. A route's own switch decides that route. `HK_VLLM_MXFP8=0` turns it off. `=1` turns it on,
   including a default-off route or an arch it was not validated on. Some routes take a value
   instead: `HK_SGLANG_MOE=topk`, or `HK_VLLM_BACKEND=<slot>`.
4. The legacy engine values still mean what they meant. `HK_SGLD=attn,linear,convrot` turns on
   exactly the SGLang-Diffusion routes it names, and `HK_SGLD=1` attention and linear.
   `HK_PATCH_VLLM=1` turns on the vLLM encoder SDPA patch, which it always applied.
5. Otherwise the route's default applies. It is on only on a GPU the route was validated on.

An empty value counts as unset. Off is `0`, `off`, `no` or `false`. On is `1`, `on`, `yes`,
`true`, or a route-specific value.

    python3 -m hk.integration --status      # every route: on or off, and which rule decided
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

OFF = ("0", "off", "no", "false")
ON = ("1", "on", "yes", "true", "all")

#: The switch that turns everything off.
DISABLE = "HK_DISABLE"

#: Each engine's own switch: off turns all its routes off (rule 2).
ENGINES: Dict[str, str] = {
    "vllm": "HK_PATCH_VLLM",
    "sglang": "HK_PATCH_SGLANG",
    "sglang_diffusion": "HK_SGLD",
    "ops": "",
}


@dataclass(frozen=True)
class Route:
    name: str
    engine: str
    env: str
    #: On by default (on `arches`), or off until asked for.
    default: bool
    #: The GPUs the route was validated on. Its default applies only there.
    arches: Tuple[str, ...]
    what: str
    #: Why the default is what it is: the evidence, or what is missing.
    evidence: str
    #: For an SGLang-Diffusion part, its name in HK_SGLD's list.
    part: str = ""


_R = Route
ROUTES: Dict[str, Route] = {r.name: r for r in (
    _R("vllm_mxfp8", "vllm", "HK_VLLM_MXFP8", True, ("gfx950",),
       "K2: block-scaled MXFP8 linear layers (DeepSeek-V4.1) at M >= 2048 (indexer wq_b "
       "2825, engram wkv 1024)",
       "bit-identical to vLLM's kernel; real-weight GSM8K / ARC within stock's noise "
       "(serving-readiness B1)"),
    _R("vllm_moe_a8w4", "vllm", "HK_VLLM_MOE_A8W4", True, ("gfx950",),
       "K1: AITER a8w4 MoE (DeepSeek-V4.1) at M >= 2560",
       "error vs fp32 at or below AITER's; real-weight GSM8K / ARC within stock's noise (B1)"),
    _R("vllm_dsv41_index", "vllm", "HK_VLLM_DSV41_INDEX", True, ("gfx950",),
       "DeepSeek-V4.1's index layers under --monokernel: mono_decode._select's chain (compressor, "
       "indexer K / Q, logits, top-k) as one launch at decode steps of <= 8 rows",
       "caches and q byte-identical to vLLM's chain in vLLM, top-k sets equal at M=1; TPOT 3.3-5.0% "
       "below mono's at B1/4/8; top-1 / KL vs stock and GSM8K-200 within mono's band"),
    _R("vllm_int4", "vllm", "HK_VLLM_INT4", True, ("gfx1100", "gfx950", "gfx1201"),
       "AWQ 4-bit linear layers on hk's decode GEMM",
       "greedy tokens identical to stock's, Qwen3-8B/14B-AWQ (history 22, 42); gfx1201: "
       "Qwen3-8B-AWQ tokens identical, counts equal round for round, decode 2.5-4.5x (history 50)"),
    _R("vllm_gemm", "vllm", "HK_VLLM_GEMM", True, ("gfx1100", "gfx1201"),
       "unquantized linear layers by the measured route table, plus tuning at load",
       "gfx1100: greedy tokens identical to stock's, or a bf16 near-tie (history 35). "
       "gfx1201: greedy tokens identical to stock's, routing counts equal round for round, "
       "Qwen3-1.7B and Qwen3.5-9B in the R9700 vLLM image; prefill 1.12-1.16x (history 50). "
       "gfx950: measured to change nothing against aiter (history 42), so it stays off there"),
    _R("vllm_moe", "vllm", "HK_VLLM_MOE", True, ("gfx1100", "gfx1201"),
       "bf16 MoE GEMMs (Triton's fused_moe kernel)",
       "gfx1100: OLMoE greedy tokens identical to stock's (history 21). gfx1201: up to 2 "
       "tokens a step only (vllm_moe.BLOCK16_MAX_TOKENS_AT); OLMoE tokens identical, counts "
       "equal round for round, decode 1.10-1.12x at 1 request and 1.06-1.07x at 2, a tie at 4 "
       "(history 52). gfx950 with aiter: it routes nothing"),
    _R("vllm_sdpa", "vllm", "HK_VLLM_SDPA", True, ("gfx1100",),
       "the vision encoder's F.scaled_dot_product_attention (with --mm-encoder-attn-backend "
       "TORCH_SDPA)",
       "gfx1100: InternVL2-2B within the spread of vLLM's own backends (history 05). "
       "Not run on gfx950"),
    _R("vllm_backend", "vllm", "HK_VLLM_BACKEND", False, (),
       "the decoder's paged attention, registered over TRITON_ATTN",
       "off: it replaces ROCm's default decoder attention for every model. It raises, rather "
       "than falls back, on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and other "
       "head sizes. Validated on Qwen3-8B only"),
    _R("sglang_linear", "sglang", "HK_SGLANG_LINEAR", True, ("gfx1100",),
       "unquantized linear layers by the measured route table",
       "Qwen3-8B greedy output matches stock's up to a bf16 near-tie; error vs fp64 equal to "
       "F.linear's (history 24, 32)"),
    _R("sglang_moe", "sglang", "HK_SGLANG_MOE", True, ("gfx1100",),
       "the softmax top-k that fails on gfx1100, and the bf16 MoE GEMMs (=topk: the top-k only)",
       "OLMoE greedy tokens identical to stock's (history 25)"),
    _R("sgld_attn", "sglang_diffusion", "HK_SGLD_ATTN", True, ("gfx1100",),
       "the DiT's attention (torch_sdpa backend)",
       "Qwen-Image 2.1: 42.1 dB from stock and closer to an fp32 reference than stock "
       "(history 46)", part="attn"),
    _R("sgld_linear", "sglang_diffusion", "HK_SGLD_LINEAR", True, ("gfx1100",),
       "the DiT's projections at M >= 1024 where hk measured faster",
       "Qwen-Image 2.1 with attention: 40.0 dB from stock and closer to fp32 than stock "
       "(history 46)", part="linear"),
    _R("sgld_convrot", "sglang_diffusion", "HK_SGLD_CONVROT", False, (),
       "ConvRot W8A8 linear for --quantization convrot_int8",
       "off: lossy by design (PSNR 30.4 dB from bf16, visible drift); a model-level choice",
       part="convrot"),
    _R("sdpa_k3", "ops", "HK_SDPA_K3", True, ("gfx1100",),
       "K3: hk.ops.sdpa's non-causal head_dim-128 calls on hk.ops.attn_k3",
       "per-call error 2.95e-5 vs the current path's 4.50e-5 at Qwen-Image's shape (#24)"),
)}


@dataclass(frozen=True)
class Decision:
    on: bool
    #: Which rule decided, for --status and the plugins' report lines.
    why: str
    #: The explicit value, when one was given (`topk`, a slot name), else "".
    value: str = ""


def _get(env: Mapping[str, str], name: str) -> str:
    return env.get(name, "").strip() if name else ""


def is_off(v: str) -> bool:
    return v.strip().lower() in OFF


def disabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether HK_DISABLE turns everything off."""
    v = _get(os.environ if env is None else env, DISABLE)
    return bool(v) and not is_off(v)


_ARCH: List[Optional[str]] = []


def arch() -> Optional[str]:
    """The GPU's arch, or None without one: a route's default never assumes a GPU.
    hk.runtime.arch.detect(): HK_ARCH, then the KFD topology in sysfs (no HIP context)."""
    if not _ARCH:
        try:
            from ..runtime.arch import detect  # noqa: PLC0415
            from ..target import _ALIASES  # noqa: PLC0415

            a = detect()
            _ARCH.append(_ALIASES.get(a, a) if a else None)
        except Exception:  # noqa: BLE001 -- no GPU stack: nothing is validated here
            _ARCH.append(None)
    return _ARCH[0]


def decide(name: str, env: Optional[Mapping[str, str]] = None,
           gpu: Optional[str] = "detect") -> Decision:
    """On or off for one route, and why (the module docstring's precedence)."""
    env = os.environ if env is None else env
    r = ROUTES[name]
    if disabled(env):
        return Decision(False, f"{DISABLE}={_get(env, DISABLE)}")
    master = ENGINES.get(r.engine, "")
    mv = _get(env, master)
    if mv and is_off(mv):
        return Decision(False, f"{master}={mv}")
    v = _get(env, r.env)
    if v:
        if is_off(v):
            return Decision(False, f"{r.env}={v}")
        return Decision(True, f"{r.env}={v}", "" if v.lower() in ON else v)
    if r.part and mv:
        # HK_SGLD=1 always meant attention and linear; a list names its parts.
        named = ({"attn", "linear"} if mv.lower() in ON
                 else {p.strip().lower() for p in mv.split(",") if p.strip()})
        return Decision(r.part in named, f"{master}={mv}")
    if name == "vllm_sdpa" and mv:
        return Decision(True, f"{master}={mv} (it always applied the encoder SDPA)")
    if not r.default:
        return Decision(False, "default off")
    here = arch() if gpu == "detect" else gpu
    if here in r.arches:
        return Decision(True, f"default on ({here})")
    return Decision(False, f"default off on {here or 'no GPU'} (validated on "
                           f"{', '.join(r.arches)})")


def enabled(name: str, env: Optional[Mapping[str, str]] = None) -> bool:
    return decide(name, env).on


def value(name: str, env: Optional[Mapping[str, str]] = None) -> str:
    """A route's explicit value (`topk`, a slot name), or "" for plain on."""
    return decide(name, env).value


def routes(engine: str) -> List[Route]:
    return [r for r in ROUTES.values() if r.engine == engine]


def on(engine: str, env: Optional[Mapping[str, str]] = None) -> Dict[str, Decision]:
    """Every route of `engine` that is on: name -> its decision."""
    out = {}
    for r in routes(engine):
        d = decide(r.name, env)
        if d.on:
            out[r.name] = d
    return out


def table(env: Optional[Mapping[str, str]] = None) -> str:
    """Every route, on or off and why: what `python3 -m hk.integration --status` prints."""
    rows = [(r.name, r.env, "on" if d.on else "off", d.why)
            for r in ROUTES.values() for d in (decide(r.name, env),)]
    w = [max(len(x[i]) for x in rows) for i in range(3)]
    return "\n".join(f"{a:{w[0]}}  {b:{w[1]}}  {c:{w[2]}}  {e}" for a, b, c, e in rows)
