"""Which of hk's engine routes are on. hk is **on by default**; off is for comparisons.

A route is one patch an engine plugin can make: vLLM's MXFP8 linear layers on hk's GEMM, SGLang's
MoE on hk's, and so on. A route is on by default on a GPU where it meets the default-on rule
(docs/testing.md#the-default-on-rule): a time-weighted gain of 1.15x or more over the engine's best
at a real model's mix of shapes, taking only the shapes where hk is faster beyond noise, its
correctness suite passing, and one end-to-end smoke per engine with equal quality (docs/integration.md#on-by-default-and-turning-it-off has each route's evidence). Lossy
routes (ConvRot INT8) and routes without that evidence stay off until you turn them on.

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
5. Otherwise the route's default applies. It is on only on a GPU where the route met the rule.

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
       "block-scaled MXFP8 linear layers (DeepSeek-V4.1) at M >= 2048 (indexer wq_b 2825, "
       "engram wkv 1024)",
       "bit-identical to vLLM's kernel; real-weight GSM8K / ARC within stock's noise "
       "(serving-readiness B1)"),
    _R("vllm_moe_a8w4", "vllm", "HK_VLLM_MOE_A8W4", True, ("gfx950",),
       "AITER's a8w4 MoE, MXFP4 weights (DeepSeek-V4.1), at M >= 2560",
       "error vs fp32 at or below AITER's; real-weight GSM8K / ARC within stock's noise (B1)"),
    _R("vllm_dsv41_index", "vllm", "HK_VLLM_DSV41_INDEX", True, ("gfx950",),
       "DeepSeek-V4.1's index layers under --monokernel: mono_decode._select's chain (compressor, "
       "indexer K / Q, logits, top-k) as one launch at decode steps of <= 8 rows",
       "caches and q byte-identical to vLLM's chain in vLLM, top-k sets equal at M=1; real-weight "
       "GSM8K-1319 within stock's noise"),
    _R("vllm_sparse_mla", "vllm", "HK_VLLM_SPARSE_MLA", True, ("gfx950",),
       "DeepSeek-V4.1's sparse MLA prefill (T >= 512 queries a call) and its MXFP8 output rows; "
       "it runs where vLLM's own sparse MLA does (VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA unset)",
       "real weights, TP2: GSM8K-1319 equal to stock's. Its cost is at decode: it needs vLLM's "
       "own sparse MLA, whose decode kernel replaces AITER's, so decode steps can be slightly "
       "slower. Inactive while VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1; HK_VLLM_SPARSE_MLA=0 "
       "turns it off"),
    _R("vllm_int4", "vllm", "HK_VLLM_INT4", True, ("gfx1100", "gfx950", "gfx1201"),
       "AWQ 4-bit linear layers on hk's decode GEMM",
       "greedy tokens identical to stock's, Qwen3-8B/14B-AWQ (history 22, 42); gfx1201: "
       "Qwen3-8B-AWQ tokens identical, counts equal round for round (history 50)"),
    _R("vllm_gptq", "vllm", "HK_VLLM_GPTQ", True, ("gfx1100", "gfx1201"),
       "4-bit symmetric GPTQ linear layers on RDNA3W4A16LinearKernel, on hk's decode GEMM at "
       "1-64 tokens; on gfx1201's RDNAHybridW4A16LinearKernel, the fp16 batches vLLM runs on "
       "Triton on hk's 4-bit forms (decode to 64 tokens, the large-batch form above: "
       "prefill), reading vLLM's tensors in place; "
       "act-order checkpoints refused",
       "gfx1100: Qwen3.8-27B-GPTQ-Int4 in the Radeon TheRock vLLM image against stock-fair "
       "(TRITON_ATTN, TunableOp with a 256 MB rotating buffer), 3 starts an arm: GSM8K-200 equal, "
       "TPOT ahead at 8-16 concurrent requests and level at 1 (RDNA scorecard); gfx1201: "
       "Qwen3.8-27B-GPTQ-Int4 on one R9700 in the R9700 vLLM image (TRITON_ATTN), 3 alternating "
       "starts an arm: GSM8K-200 equal, TPOT ahead at 4 and 8 concurrent requests (disjoint), "
       "level at 1 and for an 8k prompt"),
    _R("vllm_quark", "vllm", "HK_VLLM_QUARK", True, ("gfx1100", "gfx1201"),
       "AMD Quark W4A16 (int4, symmetric) linear layers: on gfx1201's RDNAHybridW4A16LinearKernel "
       "the fp16 batches vLLM runs on Triton on hk's 4-bit forms in place (decode to 64 tokens, "
       "the large-batch form above: prefill); on gfx1100's RDNA3W4A16LinearKernel 1-64 tokens on "
       "hk's decode GEMM (a copy of Quark's codes in hk's layout)",
       "amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16, one process, engines rebuilt per arm, 4 alternating "
       "rounds, GSM8K-200 equal (0 discordant) on both GPUs. gfx1201 (one R9700, R9700 vLLM image, "
       "TRITON_ATTN): TPOT at 8 concurrent requests 1.64x, TTFT at 512 / 2048 tokens 1.16x / 1.17x "
       "(disjoint), level at 1 and 4; kernel 1.9-3.6x vLLM at 6-32 tokens, 1.13-1.30x at 128-4096, "
       "1.05-1.20x at 64. gfx1100 (one W7900D, MiniCPM-V vLLM image): TPOT 1.13x / 1.18x / 1.25x "
       "at 1 / 4 / 8 concurrent requests (disjoint), TTFT level (prefill stays vLLM's there)"),
    _R("vllm_gemm", "vllm", "HK_VLLM_GEMM", True, ("gfx1100", "gfx1201"),
       "unquantized linear layers by the measured route table, plus tuning at load",
       "gfx1100: greedy tokens identical to stock's, or a bf16 near-tie (history 35). "
       "gfx1201: greedy tokens identical to stock's, routing counts equal round for round, "
       "Qwen3-1.7B and Qwen3.5-9B in the R9700 vLLM image (history 50). "
       "gfx950: measured to change nothing against aiter (history 42), so it stays off there"),
    _R("vllm_moe", "vllm", "HK_VLLM_MOE", True, ("gfx1100", "gfx1201"),
       "bf16 MoE GEMMs (Triton's fused_moe kernel)",
       "gfx1100: OLMoE greedy tokens identical to stock's (history 21). gfx1201: up to 2 "
       "tokens a step only (vllm_moe.BLOCK16_MAX_TOKENS_AT); OLMoE tokens identical, counts "
       "equal round for round (history 52). gfx950 with aiter: it routes nothing"),
    _R("vllm_moe_wna16", "vllm", "HK_VLLM_MOE_WNA16", True, ("gfx1100", "gfx1201"),
       "4-bit MoE GEMMs with zero points (AWQ through vLLM's moe_wna16) at up to 32 tokens a step, "
       "on vLLM's own tensors, for the validated expert shapes (VALIDATED)",
       "gfx1100: Qwen3.6-35B-A3B-AWQ (group 32) in the Radeon TheRock vLLM image against "
       "stock-fair' (TRITON_ATTN, TunableOp, vLLM's MoE config chosen per batch by in-vLLM time), "
       "3 alternating starts an arm: GSM8K-200 equal, TPOT ahead at 1, 8 and 16 concurrent "
       "requests and at 8k (disjoint), TTFT level. gfx1201: the same checkpoint on one R9700 in "
       "the R9700 vLLM image against stock-fair (TRITON_ATTN, a searched MoE config), 3 "
       "alternating starts an arm"),
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
    _R("vllm_prefill", "vllm", "HK_VLLM_PREFILL", True, ("gfx1100", "gfx1201"),
       "prefill attention in place on vLLM's own cache: ROCM_ATTN's prefix-prefill, and "
       "TRITON_ATTN's unified_attention on batches with prefill in them, and its decode-only "
       "batches at head size 256 (HK_VLLM_DECODE; =1 adds 128), and with HK_VLLM_ROCM_DECODE=1 "
       "ROCM_ATTN's decode-only batches at head size 128 (gfx1100); cache write and block size "
       "stay vLLM's",
       "end-to-end gates passed best vs best (16k TTFT on a d128 model; 8k on a d256 hybrid "
       "model at TP2, also against ROCM_ATTN), quality within stock's spread. "
       "gfx1100 and gfx1201 (decode-only batches: gfx1100), bf16 (prefill calls: bf16 or fp16), "
       "head size 64 / 128 "
       "(and 256 on TRITON_ATTN), pages of 16 or 32 "
       "(on TRITON_ATTN any multiple of 16: hybrid models' 544, 784); any other call falls back to "
       "vLLM's kernel"),
    _R("sglang_linear", "sglang", "HK_SGLANG_LINEAR", True, ("gfx1100",),
       "unquantized linear layers by the measured route table",
       "Qwen3-8B greedy output matches stock's up to a bf16 near-tie; error vs fp64 equal to "
       "F.linear's (history 24, 32)"),
    _R("sglang_norm", "sglang", "HK_SGLANG_NORM", True, ("gfx1100",),
       "Gemma-style RMSNorm (GemmaRMSNorm) where SGLang would run its native torch path (no "
       "aiter, no vLLM): hk's weighted norm kernels, the stored weight with weight offset 1",
       "within half a unit of bf16 of a float64 reference, as SGLang's Triton Gemma kernel; the "
       "residual bit-exact; Qwen3.8-27B GSM8K equal to stock's"),
    _R("sglang_moe", "sglang", "HK_SGLANG_MOE", True, ("gfx1100",),
       "the softmax top-k that fails on gfx1100, and the bf16 MoE GEMMs (=topk: the top-k only)",
       "OLMoE greedy tokens identical to stock's (history 25)"),
    _R("sgld_attn", "sglang_diffusion", "HK_SGLD_ATTN", True, ("gfx1100", "gfx1201"),
       "the DiT's attention (torch_sdpa backend)",
       "Qwen-Image 2.1: 42.1 dB from stock and closer to an fp32 reference than stock "
       "(history 46); gfx1201 (hk.ops.sdpa): 1.82-1.90x torch's SDPA at Wan2.1 / Qwen-Image "
       "shapes, error vs fp32 equal to it; Wan2.1 832x480x81 denoise 228.7 -> 142.4 s (1.61x)",
       part="attn"),
    _R("sgld_linear", "sglang_diffusion", "HK_SGLD_LINEAR", True, ("gfx1100",),
       "the DiT's projections at M >= 1024 where hk measured faster, with or without a bias "
       "(a bias in the GEMM's store)",
       "Qwen-Image 2.1 with attention: 40.0 dB from stock and closer to fp32 than stock "
       "(history 46); Wan2.1 832x480x81's biased projections 1.22-1.26x F.linear, same error vs "
       "fp32: 221.4 -> 211.4 s a video (1.047x, same process), inside stock's run-to-run spread",
       part="linear"),
    _R("sgld_convrot", "sglang_diffusion", "HK_SGLD_CONVROT", False, (),
       "ConvRot W8A8 linear for --quantization convrot_int8",
       "off: lossy by design (PSNR 30.4 dB from bf16, visible drift); a model-level choice",
       part="convrot"),
    _R("sgld_vaeconv", "sglang_diffusion", "HK_SGLD_VAECONV", True, ("gfx1100", "gfx1201"),
       "Wan's VAE conv3d (WanCausalConv3d) as one conv2d over its stacked time taps (or calls of 2 "
       "output frames), the engine's own convs: MIOpen runs the full-resolution 6-frame conv3d on its "
       "naive reference kernel and the rest as an explicit im2col; and the VAE kept in NCDHW "
       "(SGLang's channels-last path costs MIOpen a layout copy every conv)",
       "the same conv per output frame (bit-identical where MIOpen picks a real solver for both); "
       "Wan2.1-T2V-1.3B 832x480x81: VAE decode 969 s -> 37.8 s (a generation 2.77x), 36.2 dB from "
       "stock and closer to an fp32 VAE decode than stock (36.9 dB vs 36.1); NCDHW: 211.9 -> 201.4 s a "
       "video with every hk route on (1.052x, same process), inside stock's run-to-run spread; one conv2d: "
       "convs 1.75-2.7x the 2-frame calls at the same error, 197.6 -> 187.3 s a video (1.055x). gfx1201: "
       "MIOpen runs every channels-last VAE conv on its naive kernel, a 9-frame decode 2,680 s -> 4.1 s, "
       "closer to an fp32 VAE decode than stock (38.60 dB vs 38.53)",
       part="vaeconv"),
    _R("sgld_rope", "sglang_diffusion", "HK_SGLD_ROPE", True, ("gfx1100",),
       "Qwen-Image 2.1's complex RoPE on hk.ops.rotary_pairs (bf16, head_dim 128, one sequence)",
       "bit-identical to the engine's on every call (rotary_pairs reproduces torch's complex "
       "multiply); Qwen-Image 2.1 1024^2 30 steps: 28.88 -> 28.15 s a request (1.026x, same "
       "process, ABBA x3)", part="rope"),
    _R("sgld_qknorm", "sglang_diffusion", "HK_SGLD_QKNORM", True, ("gfx1100",),
       "Qwen-Image 2.1's q / k RMSNorm and complex RoPE as one hk.ops.fused.rmsnorm_rotary_pairs "
       "launch (bf16, head_dim 128, one sequence)",
       "error vs fp64 equal to the engine's (91 of 17M elements differ by a bf16 unit, the mean's "
       "sum order); Qwen-Image 2.1 1024^2 30 steps: 27.36 -> 26.22 s a request (1.044x, same "
       "process, ABBA x3); 37.9 dB from stock, where torch's own norm with only its sum order "
       "changed is 36.5 dB from stock", part="qknorm"),
    _R("sdpa_k3", "ops", "HK_SDPA_K3", True, ("gfx1100",),
       "hk.ops.sdpa's non-causal head_dim-128 calls on its newer kernel (hk.ops.attn_k3)",
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
