"""Where a step of the engine spends its device time, and what hk has for each part.

    python -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096
    python -m hk.integration headroom --model Qwen/Qwen3-8B --batch 16 --context 4096 \\
        --arms stock,hk+gemm --json headroom.json
    python -m hk.integration headroom --trace decode=t.json.gz --batch 16 --arch gfx1100

Two phases, each profiled in the engine itself: a **decode** step at the full
batch, once every request is past its prefill, and the **prefill** of the
batch (each request generating one token, as `ab` times prefill). For vLLM
the steps are driven one at a time (`LLMEngine.step`) and counted where the
scheduler hands them out, so a window of steps is exactly that many steps.
SGLang profiles in its scheduler's process: one session around the batch's
prefill, and its profiler's first `--steps` decode steps, each step counted by
its one embedding lookup in the trace. Device time is attributed to
kernels from the profiler's chrome trace, never from `key_averages()`: on
ROCm 7.2's torch `self_device_time_total` is 0 for every kernel.

Each kernel is put in a family (linear, attention, norm, ...) by its name and
marked as hk's or the engine's; each family is then set against what hk has:
the `hk.ops` that compute it, the patch and `ab` arm that route it, and for
the linear layers the routing table's path for every one of the model's
(N, K) at this step's M.

**What is ranked.** A family's share of the step is not what can be won back.
Its device time per step is the **upper bound**: if the family took zero
time the step would be at most that much shorter, and less where the host,
not the GPU, holds the step up. The rank is that bound times a **ceiling**,
the fraction of it hk could credibly take, from

  * a measurement in this run -- another arm (`--arms`), each family's time
    in it against the first arm's; each arm is its own process, so only a
    gap past the noise means anything;
  * the routing table's per-call measurements at this step's M, for the
    linear layers (the tuner timed the engine's op and hk's, interleaved in
    one process);
  * the record, where neither measured it (a ratio from a chapter, at the
    shapes that chapter measured, cited);

or none: a family nothing has measured ranks by its upper bound, flagged.
A ceiling from a ratio r is 1 - 1/r, and 0 where hk's kernel loses.

**CUDA graphs.** By default the engine runs eager, where every kernel is a
launch of its own, and a launch that reaches an idle GPU 5-16 us late can
cost ~10 us more (docs/history/32), which inflates small kernels: the step's
idle time is reported beside the kernels. `--graphs` profiles the engine as
it ships, CUDA graphs and torch.compile. On ROCm 7.2 with torch 2.13 the
profiler records the kernels a graph replays: vLLM 0.31's decode step is one
graph launch with every kernel of the eager step in it (docs/history/36). A
trace whose graph launches leave most of the step without device time did
not see the replays; that phase is reported as blind, not ranked.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PKG_ROOT = str(Path(__file__).resolve().parents[2])

PHASES = ("decode", "prefill")

# ------------------------------------------------------------------ traces


def read_trace(path) -> List[dict]:
    """The events of a chrome trace (`.json` or `.json.gz`)."""
    p = Path(path)
    raw = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
    d = json.loads(raw)
    return d["traceEvents"] if isinstance(d, dict) else d


#: Trace categories that are device work, in the traces torch writes.
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def device_events(events: Iterable[dict]) -> List[Tuple[str, float, float]]:
    """(name, ts_us, dur_us) of every device event, in time order."""
    out = [(e.get("name", "?"), float(e.get("ts", 0.0)), float(e.get("dur", 0.0)))
           for e in events if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS]
    return sorted(out, key=lambda x: x[1])


def graph_launches(events: Iterable[dict]) -> int:
    """Host calls that replay a CUDA/HIP graph."""
    return sum(1 for e in events if e.get("ph") == "X"
               and re.search(r"(cuda|hip)GraphLaunch", e.get("name", "")))


def busy_us(dev: Sequence[Tuple[str, float, float]]) -> float:
    """Device time with overlapping events counted once."""
    total, end = 0.0, float("-inf")
    for _, ts, dur in dev:
        if ts >= end:
            total += dur
            end = ts + dur
        elif ts + dur > end:
            total += ts + dur - end
            end = ts + dur
    return total


def summarize_trace(events: List[dict]) -> dict:
    """What a phase keeps of its trace: per kernel name, total us and calls;
    the device's busy time; the span from first to last device event; and
    the graph launches."""
    dev = device_events(events)
    per: Dict[str, List[float]] = {}
    for name, _, dur in dev:
        ent = per.setdefault(name, [0.0, 0])
        ent[0] += dur
        ent[1] += 1
    span = (dev[-1][1] + dev[-1][2] - dev[0][1]) if dev else 0.0
    return {"kernels": {k: [round(v[0], 3), v[1]] for k, v in per.items()},
            "device_us": round(sum(v[0] for v in per.values()), 3),
            "busy_us": round(busy_us(dev), 3), "span_us": round(span, 3),
            "graph_launches": graph_launches(events)}


# ------------------------------------------------------------------ families

#: (family, pattern), first match wins, on the kernel's name lowercased.
#: Written from vLLM 0.31's and SGLang 0.5.18's ROCm kernels, hk's own,
#: Inductor's Triton names and CDNA's libraries; a kernel no pattern names
#: is "other", listed by name.
FAMILIES: List[Tuple[str, str]] = [
    ("copy", r"^(memcpy|memset)|\bmemcpy|\bmemset|rocclr_(copy|fill)"),
    ("kv_write", r"reshape_and_cache|concat_and_cache|kv_cache|cache_kernel|store_kvcache"
                 r"|set_kv_buffer|kv_scatter"),
    ("attention", r"attn|attention|flash|fmha|paged|reduce_segments|_fwd_kernel"
                  r"|_fwd_grouped_kernel|decode_grouped|extend_|mha_|ll4mi|unified_attention"),
    ("moe", r"moe|expert|topk_softmax|topkgating|grouped_topk"),
    ("linear", r"^cijk_|gemm|gemv|matmul|wvsplitk|llmm1|skinny|hipblaslt|rocblas|_mm_"
               r"|w4a16|awq|gptq|splitk|addmm|\bmm\b"),
    ("norm", r"rms_?norm|layer_?norm|rsqrt|_norm_"),
    ("activation", r"silu|gelu|act_and_mul|swiglu|swish"),
    ("rotary", r"rotary|rope|cos_sin"),
    ("sampling", r"topk|top_k|top_p|argmax|argsort|softmax|sampl|sort|cumsum|multinomial"
                 r"|exponential|radix|penalt|logprob|log_softmax|max_reduce|reduce_kernel"),
    ("embedding", r"embedding|indexselect|index_select|vocab"),
    ("elementwise", r"elementwise|fill|copy|cat_|cat\b|triton_poi|unrolled|index|scatter"
                    r"|gather|where|arange|clamp|compare"),
]
_FAMILY_RE = [(f, re.compile(p)) for f, p in FAMILIES]

#: hk's kernels are `void <name>_kernel(const globals g)`, or on CDNA
#: `hk_flat::<name>_kernel(...)` (a noalias kernel's flat arguments).
_HK_SIG = re.compile(r"_kernel\s*\(\s*(const\s+)?globals|_kernel7globals|hk_flat::|_ZN7hk_flat")
#: Kernels of libraries a routing table of hk's may send calls to: FlyDSL's
#: GEMM is `wmma_gemm_kernel_<n>` on RDNA3.
_ROUTED = {"flydsl": re.compile(r"^wmma_gemm_kernel_\d|flydsl")}


def family(name: str) -> str:
    low = name.lower()
    if _HK_SIG.search(name):
        # hk's names say what they are; route them by the same patterns, with
        # hk's own words for the families the patterns do not spell.
        if re.search(r"strip|rms|norm", low):
            return "norm"
        if re.search(r"decode_gemm|gemv|gemm|w4|moe", low):
            return "moe" if "moe" in low else "linear"
    for fam, rx in _FAMILY_RE:
        if rx.search(low):
            return fam
    return "other"


def provider(name: str) -> str:
    """"hk" for a kernel hk generated; the library's name for one hk's routes
    may choose (FlyDSL's GEMM); "engine" for anything else."""
    if _HK_SIG.search(name):
        return "hk"
    for lib, rx in _ROUTED.items():
        if rx.search(name.lower()):
            return lib
    return "engine"


#: What hk has for each family: the ops that compute it, and per engine the
#: `ab` arm and switch that route it (None: no patch). `note` is why not.
COVERAGE: Dict[str, dict] = {
    "linear": {"ops": ["hk.ops.matmul", "hk.ops.decode_gemm"],
               "patch": {"vllm": ("gemm", "HK_VLLM_GEMM=1"),
                         "sglang": ("linear", "HK_SGLANG_LINEAR=1")}},
    "attention": {"ops": ["hk.ops.attention", "hk.ops.paged_prefill", "hk.ops.paged"],
                  "patch": {"vllm": ("hk", "HK_VLLM_BACKEND=1")}},
    "moe": {"ops": ["hk.ops.moe"],
            "patch": {"vllm": ("moe", "HK_VLLM_MOE=1"), "sglang": ("moe", "HK_SGLANG_MOE=1")},
            "note": "vLLM's patch rebinds TritonExperts' kernel; with aiter on (CK's "
                    "kernel_moe_gemm, aiter's moe_sorting) it routes nothing"},
    "norm": {"ops": ["hk.ops.rms_norm", "hk.ops.fused_add_rms_norm"], "patch": {},
             "note": "not patched: an opaque op would cost Inductor its fusions "
                     "(docs/integration.md#what-is-deliberately-not-patched)"},
    "activation": {"ops": ["hk.ops.silu_mul"],
                   "patch": {"vllm": ("user", "HK_VLLM_USER=examples/11_vllm_integration.py")},
                   "note": "the example's kernel loses to Inductor's (chapter 23)"},
    "rotary": {"ops": ["hk.ops.rope"], "patch": {}, "note": "no patch"},
    "kv_write": {"ops": [], "patch": {},
                 "note": "hk's backend calls vLLM's reshape_and_cache_flash; "
                         "examples/09_kv_scatter.py is a kernel, not a patch"},
    "sampling": {"ops": [], "patch": {}, "note": "nothing in hk"},
    "embedding": {"ops": [], "patch": {}, "note": "examples/08_embedding.py; no patch"},
    "elementwise": {"ops": ["hk.ops.add", "hk.ops.mul"], "patch": {},
                    "note": "framework glue: copies, casts, index arithmetic"},
    "copy": {"ops": [], "patch": {}, "note": "host<->device copies and fills"},
    "other": {"ops": [], "patch": {}, "note": "no pattern names these kernels"},
}

#: Ratios the record has for hk's kernel against one engine kernel:
#: (arch, family, phase or None for both, the engine kernel's name pattern,
#: low, high, where). Used only where nothing in the run measured the
#: family, and only if the engine's largest kernel in it matches: a ratio
#: against TRITON_ATTN's attention says nothing about another backend's. The
#: low end sets the ceiling.
RECORD: List[Tuple[str, str, Optional[str], str, float, float, str]] = [
    ("gfx1100", "attention", "prefill", r"unified_attention", 3.6, 7.9,
     "hk's paged prefill against TRITON_ATTN's unified_attention, per layer, "
     "vLLM 0.16 (chapter 17)"),
    ("gfx1100", "attention", "decode", r"unified_attention", 0.86, 1.12,
     "hk's paged decode against TRITON_ATTN's unified_attention on hk's cache at 16-64 "
     "requests, vLLM 0.31 with triton-rocm 3.7.1 (chapter 38); 1.03-1.30x under Triton 3.8.0"),
    ("gfx1100", "moe", None, r"fused_moe", 1.0, 2.77,
     "hk's MoE GEMM against fused_moe, 1-256 tokens (chapter 21)"),
    ("gfx1100", "norm", None, r"^triton_", 1.02, 1.12,
     "strips against the best tuned Triton or FlyDSL RMSNorm, standalone (chapter 27); "
     "inside vLLM's compiled graph not re-measured"),
    ("gfx1100", "activation", None, r"^triton_", 0.62, 0.93,
     "examples/11's SiLU-and-mul against Inductor's (chapter 23)"),
]


def record_for(arch: str, fam: str, phase: str, top_kernel: str):
    """The RECORD entry for this family's largest engine kernel, or None."""
    for a, f, ph, pat, lo, hi, where in RECORD:
        if a == arch and f == fam and ph in (None, phase) and re.search(pat, top_kernel):
            return lo, hi, where
    return None


def ceiling_of(ratio: Optional[float]) -> Optional[float]:
    """The fraction of a family's time a kernel `ratio` times faster saves."""
    if ratio is None:
        return None
    return max(0.0, 1.0 - 1.0 / ratio) if ratio > 0 else 0.0


# ------------------------------------------------------------------ linear routes


#: Per engine: the routing table's kind, and the variable that overrides it.
TABLES = {"vllm": ("vllm_linear", "HK_VLLM_ROUTES"),
          "sglang": ("sglang_linear", "HK_SGLANG_ROUTES")}


def linear_routes(model_cfg: dict, m: int, lm_rows: int, *, arch: str, engine_name="vllm",
                  table=None, route_for=None) -> List[dict]:
    """For each of the model's linear shapes at `m` rows (the LM head at
    `lm_rows`): the path the engine's table routes it to, and what the tuner
    measured per call there, at the nearest M it measured.

    `table` is a RouteTable (default: the shipped one with this machine's
    records); `route_for(m, n, k, bias=)` the patch's own decision, which also
    asks whether the kernel takes the call (default: the table's lookup)."""
    from ..autotune import routes as _routes  # noqa: PLC0415
    from . import engine  # noqa: PLC0415

    kind, env = TABLES[engine_name]
    if table is None:
        table = engine.shipped_table(kind, arch, env)
    raw = {}
    if table is not None and table.source and Path(table.source).is_file():
        raw = json.loads(Path(table.source).read_text()).get("shapes", {})
    if table is not None:
        user = _routes.load_user(kind, arch) or {}
        raw.update(user.get("shapes", {}))
    biased = _routes.biased(model_cfg)
    out = []
    for name, n, k, calls in _routes.linear_shapes(model_cfg):
        rows = lm_rows if name == "lm_head" else m
        bias = name in biased
        if table is None:
            path = "fallback"
        elif route_for is not None:
            path = route_for(rows, n, k, bias=bias)
        else:
            path = "fallback" if bias else table.lookup(rows, n, k)
        ent = {"name": name, "n": n, "k": k, "m": rows, "calls": calls, "path": path,
               "bias": bias}
        pts = (raw.get(f"{n}x{k}") or {}).get("points") or []
        if pts:
            near = min(pts, key=lambda p: abs(p[0] - rows) / max(rows, 1))
            at, t = near[0], near[1]
            ent["measured_m"] = at
            ent["engine_ms"] = t.get("engine")
            ent["path_ms"] = t.get(path if path != "fallback" else "engine", t.get("engine"))
            best = min((v, p) for p, v in t.items() if isinstance(v, (int, float))
                       and p not in ("noise",))
            ent["best"] = best[1]
        out.append(ent)
    return out


def linear_ratio(routes: List[dict]) -> Optional[Tuple[float, str]]:
    """engine / routed time over the model's linear calls in one step, from
    the tuner's per-call measurements; None where a shape was not measured."""
    if not routes or any(r.get("engine_ms") is None for r in routes):
        return None
    eng = sum(r["engine_ms"] * r["calls"] for r in routes)
    hk = sum(r["path_ms"] * r["calls"] for r in routes)
    paths = sorted({f"{r['name']} {r['path']}@{r['m']}" for r in routes})
    return eng / hk, "route table, per call at this M: " + ", ".join(paths)


# ------------------------------------------------------------------ rooflines

#: Per arch: (bf16 matrix FLOP/s, DRAM bytes/s, where they are from). An arch
#: without an entry gets no roofline, rather than a guessed one.
PEAKS: Dict[str, Tuple[float, float, str]] = {
    "gfx1100": (122.6e12, 864e9, "the W7900D's bf16 WMMA peak and GDDR6, 384 bits at "
                                 "18 Gbps (tools/hk-bench/_rounds.py)"),
    "gfx1201": (197e12, 640e9, "the R9700 measured: back-to-back bf16 WMMAs on every CU, "
                               "197.1 TFLOP/s (the datasheet's 191 is at the boost clock); "
                               "GDDR6 at 20 Gbps x 256 bits (a copy past the 64 MB "
                               "Infinity Cache moves 0.62 TB/s)"),
    "gfx950": (2.48e15, 6.6e12, "the MI355X measured (tools/cdna-probes/peaks.hip): a "
                                "register-only v_mfma_f32_32x32x16_bf16 loop on every CU, "
                                "2471-2484 TFLOP/s sustained for 0.1 s at 2381-2396 MHz "
                                "(the datasheet's 2516.6 at 2400); a read of 1 GiB "
                                "buffers rotated past the 256 MB Infinity Cache, "
                                "6.54-6.60 TB/s (a copy moves 5.7; the datasheet's HBM3E "
                                "is 8.0)"),
}


def floors(cfg: dict, phase: dict, ph: str, *, batch: int, context: int,
           arch: str) -> Dict[str, Tuple[float, str]]:
    """{family: (the step's shortest possible time for it in ms, how)} for the
    families whose work follows from the model and the step: the linear
    layers (each call max(FLOPs / peak, bytes / bandwidth)) and attention
    (a decode step streams every request's KV cache; a prefill step's causal
    attention is 2 h n^2 d FLOPs a layer and request). Empty where the arch
    has no PEAKS entry."""
    if arch not in PEAKS or not cfg:
        return {}
    from ..autotune import routes as _routes  # noqa: PLC0415

    peak, bw, _ = PEAKS[arch]
    text = cfg.get("text_config", cfg)
    heads = text["num_attention_heads"]
    kv = text.get("num_key_value_heads") or heads
    hd = text.get("head_dim") or text["hidden_size"] // heads
    layers = text["num_hidden_layers"]
    toks = int(statistics.median(phase.get("tokens_per_step") or [batch]))
    reqs = int(statistics.median(phase.get("requests_per_step") or [batch]))
    # An MoE model's MLP is its experts, the moe family's: only its dense
    # layers (DeepSeek's first few, Qwen3-MoE's mlp_only_layers) are linear.
    # The experts get no floor: the bytes a step reads depend on how many
    # experts its tokens pick, which the config does not say.
    moe = text.get("num_experts") or text.get("num_local_experts") or \
        text.get("n_routed_experts")
    dense = (text.get("first_k_dense_replace") or len(text.get("mlp_only_layers") or ())
             if moe else layers)
    lin = 0.0
    for name, n, k, calls in _routes.linear_shapes(cfg):
        if name in ("gate_up", "down"):
            calls = calls * dense // layers
        m = reqs if name == "lm_head" else toks
        lin += calls * max(2 * m * n * k / peak, (n * k + m * k + m * n) * 2 / bw)
    out = {"linear": (lin * 1e3, f"the model's linear calls at M={toks} (LM head {reqs}), "
                                 f"each at max(FLOPs / peak, bytes / bandwidth)"
                                 + (f"; the MLP in {dense} dense layer(s) of {layers}, "
                                    f"the rest is experts" if moe else ""))}
    if ph == "decode":
        kv_bytes = reqs * context * layers * 2 * kv * hd * 2
        out["attention"] = (kv_bytes / bw * 1e3,
                            f"{reqs} requests' KV cache, {context} tokens, read once: "
                            f"{kv_bytes / 1e9:.2f} GB at the DRAM's rate")
    else:
        n = toks / max(reqs, 1)
        flops = reqs * layers * 2 * heads * n * n * hd
        out["attention"] = (flops / peak * 1e3,
                            f"{reqs} causal prompts of {n:.0f} tokens: "
                            f"{flops / 1e12:.2f} TFLOP at the bf16 peak")
    return out


# ------------------------------------------------------------------ attribution


def attribute(phase: dict) -> Dict[str, dict]:
    """{family: {us, calls, hk_us, kernels: [(name, us, calls)]}} per step;
    `hk_us` is the time in kernels hk put there: its own, and those of the
    libraries its routes chose."""
    steps = max(phase.get("steps") or 1, 1)
    fams: Dict[str, dict] = {}
    for name, (us, calls) in phase["kernels"].items():
        f = fams.setdefault(family(name), {"us": 0.0, "calls": 0.0, "hk_us": 0.0,
                                           "kernels": []})
        f["us"] += us / steps
        f["calls"] += calls / steps
        if provider(name) != "engine":
            f["hk_us"] += us / steps
        f["kernels"].append((name, us / steps, calls / steps))
    for f in fams.values():
        f["kernels"].sort(key=lambda x: -x[1])
    return fams


def _ms(us: float) -> float:
    return round(us / 1e3, 4)


def rank(phases: Dict[str, dict], *, engine_name: str, arch: str,
         routes: Optional[Dict[str, List[dict]]] = None,
         compare: Optional[Dict[str, Dict[str, dict]]] = None,
         patched: Sequence[str] = (),
         roof: Optional[Dict[str, Dict[str, Tuple[float, str]]]] = None) -> List[dict]:
    """One entry per (phase, family), ranked within each phase by upper
    bound x ceiling. `compare` is {arm: phases} measured beside these;
    `patched` the patch names active in the run the phases come from;
    `roof` {phase: floors()}."""
    entries = []
    for ph, data in phases.items():
        if not data or not data.get("kernels"):
            continue
        step_ms = data.get("step_ms")
        fams = attribute(data)
        dev = sum(f["us"] for f in fams.values())
        for fam, f in fams.items():
            cov = COVERAGE.get(fam, COVERAGE["other"])
            arm = (cov.get("patch") or {}).get(engine_name)
            ent = {
                "phase": ph, "family": fam,
                "ms_per_step": _ms(f["us"]), "calls_per_step": round(f["calls"], 2),
                "hk_ms_per_step": _ms(f["hk_us"]),
                "share_of_device": round(f["us"] / dev, 4) if dev else None,
                "share_of_step": (round(f["us"] / 1e3 / step_ms, 4) if step_ms else None),
                "upper_bound_ms": _ms(f["us"]),
                "upper_bound": "if this family took zero time the step would save at "
                               "most this; less where the host holds the step up",
                "hk": {"ops": cov["ops"],
                       "arm": arm[0] if arm else None, "switch": arm[1] if arm else None,
                       "on_in_this_run": bool(arm) and arm[0] in patched,
                       "runs_hk_kernels": f["hk_us"] > 0,
                       "note": cov.get("note", "")},
                "ratio": None, "ratio_source": None, "ceiling": None,
                "ceiling_source": "none: nothing measured; ranked by its upper bound",
                "kernels": [[n[:160], _ms(us), round(c, 2)] for n, us, c in f["kernels"][:6]],
            }
            # Every ratio there is for hk's kernel against the engine's here,
            # best evidence first: this step measured in another arm, the
            # tuner's per-call times at this M, the record.
            ratios = []
            for other, ophases in (compare or {}).items():
                of = attribute(ophases[ph]).get(fam) if ophases.get(ph) else None
                if of is None:
                    continue
                ent.setdefault("arms", {})[other] = {
                    "ms_per_step": _ms(of["us"]), "hk_ms_per_step": _ms(of["hk_us"])}
                # Only where the other arm ran hk's kernels for this family is
                # the ratio hk's against the engine's.
                if of["us"] > 0 and of["hk_us"] > 0 and not f["hk_us"]:
                    ratios.append((f["us"] / of["us"], f"measured: arm {other}",
                                   f"this step in arm {other} ({of['hk_us'] / of['us']:.0%} "
                                   f"hk's kernels) against this arm, each its own process"))
            if fam == "linear" and routes and routes.get(ph):
                ent["routes"] = routes[ph]
                lr = linear_ratio(routes[ph])
                if lr and not f["hk_us"]:
                    ratios.append((lr[0], "measured: route table", lr[1]))
            top = next((n for n, _, _ in f["kernels"] if provider(n) == "engine"), "")
            rec = record_for(arch, fam, ph, top)
            if rec and not f["hk_us"]:
                ratios.append((rec[0], "the record (low end), not measured here",
                               f"the record, {rec[0]:g}-{rec[1]:g}x: {rec[2]}"))
            ent["ratios"] = [{"ratio": round(r, 4), "kind": k, "source": s}
                             for r, k, s in ratios]
            if ratios:
                r, kind, src = ratios[0]
                ent["ratio"], ent["ceiling_source"], ent["ratio_source"] = round(r, 4), kind, src
            if f["hk_us"] and ent["ratio"] is None:
                ent["ceiling_source"] = ("none: hk's kernels already run "
                                         f"{f['hk_us'] / f['us']:.0%} of it")
            ent["ceiling"] = (round(ceiling_of(ent["ratio"]), 4)
                              if ent["ratio"] is not None else None)
            fl = (roof or {}).get(ph, {}).get(fam)
            if fl and f["us"] > 0:
                floor_ms, how = fl
                of_peak = floor_ms / (f["us"] / 1e3)
                ent["roofline"] = {"floor_ms": round(floor_ms, 4), "of_peak": round(of_peak, 4),
                                   "what": how}
                if ent["ceiling"] is None:
                    # No ratio: what the card's peak leaves, which no kernel
                    # reaches -- a ceiling that is itself an upper bound.
                    ent["ceiling"] = round(max(0.0, 1 - of_peak), 4)
                    ent["ceiling_source"] = (f"roofline, at most: the family runs at "
                                             f"{of_peak:.0%} of the card's peak ({how})")
            c = ent["ceiling"] if ent["ceiling"] is not None else 1.0
            ent["score_ms"] = round(ent["upper_bound_ms"] * c, 4)
            entries.append(ent)
        idle = data.get("idle_ms")
        if idle and step_ms:
            entries.append({
                "phase": ph, "family": "(GPU idle)", "ms_per_step": round(idle, 4),
                "calls_per_step": 0, "hk_ms_per_step": 0.0, "share_of_device": None,
                "share_of_step": round(idle / step_ms, 4), "upper_bound_ms": round(idle, 4),
                "upper_bound": "the step's wall time the device was not busy: host work "
                               "and launches; no kernel wins it back",
                "hk": {"ops": [], "arm": None, "switch": None, "on_in_this_run": False,
                       "runs_hk_kernels": False,
                       "note": "CUDA graphs, fewer launches or a host-side change"},
                "ratio": None, "ratio_source": None, "ceiling": None,
                "ceiling_source": "none: not a kernel", "score_ms": round(idle, 4),
                "kernels": []})
    order = {p: i for i, p in enumerate(PHASES)}
    entries.sort(key=lambda e: (order.get(e["phase"], 9), -e["score_ms"]))
    for ph in phases:
        for i, e in enumerate(x for x in entries if x["phase"] == ph):
            e["rank"] = i + 1
    return entries


# ------------------------------------------------------------------ report


def _pct(x: Optional[float]) -> str:
    return f"{100 * x:5.1f}%" if x is not None else "    - "


def report(res: dict, top: int = 12) -> List[str]:
    """The human-readable form of `res` (what main writes with --json)."""
    meta, lines = res["meta"], []
    lines.append(f"{meta.get('model')}, {meta.get('engine')} {meta.get('engine_version', '')}, "
                 f"{meta.get('arch')}, arm {meta.get('arm', 'stock')}, {meta.get('mode')}")
    for ph in PHASES:
        d = res["phases"].get(ph)
        if not d:
            continue
        if d.get("blind"):
            lines.append(f"\n{ph}: {d['blind']}")
            continue
        busy = d.get("busy_ms")
        lines.append(
            f"\n{ph}: {d.get('what', '')}; {d.get('step_ms') or 0:.2f} ms a step "
            f"({d.get('step_note', '')}), device busy {busy or 0:.2f} ms"
            + (f" ({busy / d['step_ms']:.0%})" if busy and d.get("step_ms") else "")
            + (f"; {d['graph_launches']} graph launches in the window, their kernels "
               f"recorded" if d.get("graph_launches") else ""))
        lines.append(f"  {'#':>2} {'family':<12}{'step':>7}{'ms/step':>9}{'hk':>5}"
                     f"{'bound*':>10}{'ratio':>8}{'ceiling':>9}{'score':>10}  "
                     f"hk has / where the ratio is from")
        for e in [x for x in res["ranking"] if x["phase"] == ph][:top]:
            hk = e["hk"]
            has = ", ".join(hk["ops"]) or "-"
            if hk.get("arm"):
                has += f"; arm {hk['arm']} ({'on' if hk['on_in_this_run'] else 'off'})"
            ratio = "-" if e["ratio"] is None else f"{e['ratio']:.2f}x"
            ceil = "?" if e["ceiling"] is None else f"{e['ceiling']:.2f}"
            mine = "yes" if hk["runs_hk_kernels"] else ""
            lines.append(
                f"  {e['rank']:>2} {e['family']:<12}{_pct(e['share_of_step']):>7}"
                f"{e['ms_per_step']:>9.3f}{mine:>5}{e['upper_bound_ms']:>10.3f}"
                f"{ratio:>8}{ceil:>9}{e['score_ms']:>10.3f}  {has[:90]}")
            if e.get("ratio_source"):
                lines.append(f"  {'':<40}ratio: {e['ratio_source'][:150]}")
            if e.get("roofline"):
                r = e["roofline"]
                lines.append(f"  {'':<40}roofline: {r['floor_ms']:.3f} ms at the card's peak; "
                             f"this family runs at {r['of_peak']:.0%} of it")
            if e["kernels"]:
                n, ms, c = e["kernels"][0]
                lines.append(f"  {'':<40}top kernel {ms:.3f} ms x{c:g}: {n[:90]}")
    lines.append("\n* bound: the upper bound -- what the step would save at most if the family "
                 "took zero time. score = bound x ceiling; a ceiling of '?' means nothing "
                 "measured it, so the score is the bound itself.")
    if res.get("request"):
        r = res["request"]
        lines.append(f"\nper request ({r['what']}): " + ", ".join(
            f"{f} {s:.1%}" for f, s in r["shares"][:8]))
    return lines


def request_view(phases: Dict[str, dict], tokens: int) -> Optional[dict]:
    """Shares of device time over one request batch: its prefill and
    `tokens - 1` decode steps -- what one profile of a whole generate shows."""
    pre, dec = phases.get("prefill"), phases.get("decode")
    if not pre or not dec or not pre.get("kernels") or not dec.get("kernels"):
        return None
    tot: Dict[str, float] = {}
    for f, v in attribute(pre).items():
        tot[f"{f} (prefill)"] = v["us"] * max(pre.get("steps") or 1, 1)
    for f, v in attribute(dec).items():
        tot[f"{f} (decode)"] = v["us"] * (tokens - 1)
    s = sum(tot.values())
    return {"what": f"prefill + {tokens - 1} decode steps", "tokens": tokens,
            "shares": sorted(((k, round(v / s, 4)) for k, v in tot.items()),
                             key=lambda x: -x[1])}


def _config(model: str) -> dict:
    """The model's config.json where it can be found without a network, or {}."""
    from ..autotune import routes as _routes  # noqa: PLC0415

    try:
        return _routes.model_config(model) if model else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def analyze(child: dict, *, compare: Optional[Dict[str, dict]] = None,
            tokens: int = 0) -> dict:
    """The whole result from one child's phases (and the other arms')."""
    meta = child["meta"]
    phases = child["phases"]
    patched = meta.get("patches", [])
    res = {"meta": meta, "phases": {}, "ranking": []}
    for ph, d in phases.items():
        res["phases"][ph] = {k: v for k, v in d.items() if k != "kernels"}
    visible = {ph: d for ph, d in phases.items() if d and not d.get("blind")}
    cfg = child.get("model_cfg") or _config(meta.get("model", ""))
    roof = {}
    for ph, d in visible.items():
        try:
            roof[ph] = floors(cfg, d, ph, batch=meta.get("batch", 1),
                              context=meta.get("context", 1), arch=meta.get("arch", ""))
        except (KeyError, TypeError, ZeroDivisionError):  # a config without the fields
            roof[ph] = {}
    routes = child.get("routes") or routes_for(child, cfg)
    res["ranking"] = rank(visible, engine_name=meta.get("engine", "vllm"),
                          arch=meta.get("arch", ""), routes=routes,
                          compare={a: c["phases"] for a, c in (compare or {}).items()},
                          patched=patched, roof=roof)
    res["peaks"] = PEAKS.get(meta.get("arch", ""))
    res["routes"] = routes
    if tokens:
        res["request"] = request_view(visible, tokens)
    return res


# ------------------------------------------------------------------ collection


def _phase(summary: dict, steps: int, wall_ms: Sequence[float], what: str = "",
           note: str = "") -> dict:
    step_ms = statistics.median(wall_ms) if wall_ms else None
    busy = summary["busy_us"] / 1e3 / max(steps, 1)
    d = {**summary, "steps": steps, "what": what,
         "step_ms": round(step_ms, 4) if step_ms else None,
         "step_note": note, "wall_ms": [round(w, 4) for w in wall_ms],
         "busy_ms": round(busy, 4),
         "idle_ms": round(max(0.0, step_ms - busy), 4) if step_ms else None}
    if wall_ms and len(wall_ms) > 1:
        d["step_spread"] = round(statistics.median(abs(w / step_ms - 1) for w in wall_ms), 4)
    return d


def _blind(d: dict, graphs: bool) -> Optional[str]:
    """Why a phase's trace cannot be ranked, if it cannot: graph replays
    whose kernels the profiler did not record."""
    if graphs and d.get("graph_launches") and d.get("step_ms"):
        if d["busy_ms"] < 0.5 * d["step_ms"]:
            return (f"{d['graph_launches']} graph launches in the window and device time "
                    f"for {d['busy_ms'] / d['step_ms']:.0%} of the step: the profiler did "
                    f"not see the replayed kernels. Profile eager (drop --graphs).")
    return None


def _profiler():
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])


def _export(prof, keep: str, name: str) -> List[dict]:
    if keep:
        os.makedirs(keep, exist_ok=True)
    path = os.path.join(keep or tempfile.gettempdir(), f"hk_headroom_{name}.json")
    prof.export_chrome_trace(path)
    events = read_trace(path)
    if not keep:
        os.unlink(path)
    return events


class _Steps:
    """Counts the steps vLLM's scheduler hands out, and their tokens."""

    def __init__(self, core):
        self.log: List[Tuple[int, int]] = []
        sched = core.scheduler
        orig = sched.schedule

        def schedule(*a, **kw):
            so = orig(*a, **kw)
            n = int(getattr(so, "total_num_scheduled_tokens", 0) or 0)
            if n:
                self.log.append((n, len(getattr(so, "num_scheduled_tokens", {}) or {})))
            return so

        sched.schedule = schedule

    def mark(self) -> int:
        return len(self.log)

    def since(self, i: int) -> List[Tuple[int, int]]:
        return self.log[i:]


def collect_vllm(args, arm: str) -> dict:
    """Both phases in vLLM, in this process, with `arm`'s patches."""
    from .ab import apply_arm  # noqa: PLC0415

    sel, patches, want = apply_arm(arm, args.routes)
    import torch  # noqa: PLC0415
    import vllm  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    B, C, K, W, R = args.batch, args.context, args.steps, args.warm, args.windows
    tokens = W + (R + 1) * K + 16
    defaults = os.environ.get("HK_VLLM_DEFAULTS", "0") == "1"
    knobs = {} if defaults else dict(enable_prefix_caching=False,
                                     enable_chunked_prefill=False)
    llm = LLM(model=args.model, dtype=args.dtype, block_size=32,
              max_model_len=C + tokens + 64, max_num_seqs=B,
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              attention_backend=sel, enforce_eager=not args.graphs, **knobs)
    eng = llm.llm_engine
    core = getattr(getattr(eng, "engine_core", None), "engine_core", None)
    if core is None or not hasattr(core, "scheduler"):
        raise RuntimeError("vLLM's engine core is not in this process; headroom needs "
                           "VLLM_ENABLE_V1_MULTIPROCESSING=0")
    steps = _Steps(core)
    tok = llm.get_tokenizer()
    salt = iter(range(1 << 30))

    def prompts(n):
        s = next(salt)
        return [{"prompt_token_ids": tok.encode(f"{s}.{i} " + "the quick brown fox " *
                                                (C // 4))[:C]} for i in range(n)]

    def params(n):
        return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    def add(ps, n):
        ids = []
        for p in ps:
            rid = f"hr{next(salt)}"
            eng.add_request(rid, p, params(n))
            ids.append(rid)
        return ids

    def prefill_once(prof=None):
        add(prompts(B), 1)
        torch.cuda.synchronize()
        i, t0 = steps.mark(), time.perf_counter()
        while eng.has_unfinished_requests():
            eng.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3, steps.since(i)

    active = set(want) | ({"hk"} if arm.partition("@")[0].split("+")[0] == "hk" else set())
    out = {"meta": {"engine": "vllm", "engine_version": vllm.__version__,
                    "model": args.model, "batch": B, "context": C, "arm": arm,
                    "patches": sorted(active),
                    "device": torch.cuda.get_device_name(),
                    "arch": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
                    "torch": torch.__version__,
                    "mode": ("graphs: CUDA graphs and torch.compile, as vLLM ships"
                             if args.graphs else
                             "eager: enforce_eager, every kernel a launch the profiler sees"),
                    "prefix_caching_and_chunked_prefill": "on" if defaults else "off"},
           "phases": {}}

    if "prefill" in args.phases:
        prefill_once()                                    # compile, autotune, warm
        walls, log = [], []
        for _ in range(R):
            w, log = prefill_once()
            walls.append(w / max(len(log), 1))
        with _profiler() as prof:
            w, plog = prefill_once()
        d = _phase(summarize_trace(_export(prof, args.keep_trace, f"{arm}_prefill")),
                   len(plog), walls)
        toks = [n for n, _ in plog]
        reqs = [r for _, r in plog]
        d["what"] = (f"{B} requests of {C} tokens, one output token each: {len(plog)} "
                     f"step(s) of {min(toks)}-{max(toks)} tokens, "
                     f"{min(reqs)}-{max(reqs)} requests a step")
        d["step_note"] = f"median of {R} unprofiled prefills, per step"
        d["tokens_per_step"] = toks
        d["requests_per_step"] = reqs
        out["phases"]["prefill"] = d

    if "decode" in args.phases:
        ids = add(prompts(B), tokens)
        seen = {}
        while len([r for r in ids if seen.get(r, 0) >= 1]) < B:
            for o in eng.step():
                seen[o.request_id] = len(o.outputs[0].token_ids)
        for _ in range(W):
            eng.step()
        walls = []
        for _ in range(R):
            torch.cuda.synchronize()
            i, t0 = steps.mark(), time.perf_counter()
            for _ in range(K):
                eng.step()
            torch.cuda.synchronize()
            n = len(steps.since(i))
            walls.append((time.perf_counter() - t0) * 1e3 / max(n, 1))
        torch.cuda.synchronize()
        i = steps.mark()
        with _profiler() as prof:
            for _ in range(K):
                eng.step()
            torch.cuda.synchronize()
        dlog = steps.since(i)
        d = _phase(summarize_trace(_export(prof, args.keep_trace, f"{arm}_decode")),
                   len(dlog), walls)
        d["what"] = (f"{len(dlog)} decode steps at {B} requests of {C} context, "
                     f"{sorted({n for n, _ in dlog})} tokens a step")
        d["step_note"] = f"median of {R} unprofiled windows of {K} steps"
        d["tokens_per_step"] = sorted({n for n, _ in dlog})
        out["phases"]["decode"] = d
        eng.abort_request(ids)
        while eng.has_unfinished_requests():
            eng.step()

    for ph, d in out["phases"].items():
        d["blind"] = _blind(d, args.graphs)
    out["patch_stats"] = {n: dict(p.stats) for n, p in patches.items()}
    out["model_cfg"] = _config(args.model)
    route_for = None
    try:
        from . import vllm_gemm  # noqa: PLC0415

        route_for = vllm_gemm.route_for
    except Exception:  # noqa: BLE001 -- no table here; the routes say fallback
        pass
    out["routes"] = routes_for(out, out["model_cfg"], route_for)
    return out


def routes_for(out: dict, cfg: dict, route_for=None) -> Dict[str, List[dict]]:
    """The linear routes at each phase's M, by the engine's table; with
    `route_for`, as the linear patch would take them in this process."""
    if not cfg or not out["meta"].get("arch"):
        return {}
    res = {}
    for ph, d in out["phases"].items():
        if not d:
            continue
        toks = d.get("tokens_per_step") or [out["meta"].get("batch", 1)]
        m = int(statistics.median(toks))
        lm = int(statistics.median(d.get("requests_per_step") or [out["meta"].get("batch", 1)]))
        try:
            res[ph] = linear_routes(cfg, m, lm, arch=out["meta"]["arch"],
                                    engine_name=out["meta"].get("engine", "vllm"),
                                    route_for=route_for)
        except Exception as e:  # noqa: BLE001 -- a report without routes, saying why
            res[ph] = [{"error": f"{type(e).__name__}: {e}"}]
    return res


def collect_sglang(args, arm: str) -> dict:
    """Both phases in SGLang, through its profiler's by-stage steps."""
    import glob  # noqa: PLC0415

    import sglang  # noqa: PLC0415
    import torch  # noqa: PLC0415

    B, C, K = args.batch, args.context, args.steps
    switches = {"linear": "HK_SGLANG_LINEAR", "gemm": "HK_SGLANG_LINEAR", "moe": "HK_SGLANG_MOE"}
    parts = set(arm.split("+")) - {"stock"}
    if parts - set(switches):
        raise ValueError(f"SGLang's arms are stock, linear (or gemm) and moe, joined with "
                         f"+; not {sorted(parts - set(switches))}")
    # Read by the plugin in SGLang's processes. Every switch explicitly: hk is on by default,
    # so an unset switch is not the stock arm.
    for env in set(switches.values()):
        os.environ[env] = "1" if any(switches[p] == env for p in parts) else "0"
    if args.keep_trace:
        os.makedirs(args.keep_trace, exist_ok=True)
    # The radix cache off, as `ab` turns vLLM's prefix caching off: a prompt
    # seen before would be a cache hit, and its prefill would not be timed.
    # The overlap scheduler off as well: it schedules a step while the GPU
    # runs the one before, so a profile's start and stop land inside steps.
    eng = sglang.Engine(model_path=args.model, dtype=args.dtype,
                        max_running_requests=B, context_length=C + K * 4 + 64,
                        disable_cuda_graph=not args.graphs, attention_backend="triton",
                        disable_radix_cache=True, disable_overlap_schedule=True,
                        mem_fraction_static=float(os.environ.get("HK_GPU_UTIL", "0.80")))
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.model)
    salt = iter(range(1 << 30))

    def prompts():
        s = next(salt)
        return [tok.encode(f"{s}.{i} " + "the quick brown fox " * (C // 4))[:C]
                for i in range(B)]

    n = 2 * K + 8

    def timed(tokens):
        t0 = time.perf_counter()
        eng.generate(input_ids=prompts(), sampling_params={
            "temperature": 0.0, "max_new_tokens": tokens, "ignore_eos": True})
        return (time.perf_counter() - t0) * 1e3

    timed(n)                                                        # warm
    walls = {"prefill": [timed(1) for _ in range(args.windows)],
             "long": [timed(n) for _ in range(args.windows)]}
    # GPU activity alone, one stage a session: with the CPU's as well, SGLang
    # 0.5.18's scheduler segfaulted in the profiler's op recording when its
    # by-stage profile moved from the prefill steps to the decode steps.
    dirs = {ph: tempfile.mkdtemp(prefix=f"hk_headroom_sgl_{ph}_", dir=args.keep_trace or None)
            for ph in ("prefill", "decode")}
    if "prefill" in args.phases:
        eng.start_profile(output_dir=dirs["prefill"], activities=["GPU"])
        timed(1)
        eng.stop_profile()
    if "decode" in args.phases:
        eng.start_profile(output_dir=dirs["decode"], activities=["GPU"], num_steps=K,
                          profile_by_stage=True, profile_stages=["DECODE"])
        timed(n)
    for _ in range(60):                    # the scheduler writes the traces
        if all(glob.glob(os.path.join(dirs[ph], "*.trace.json*")) for ph in args.phases):
            break
        time.sleep(1)
    time.sleep(2)
    out = {"meta": {"engine": "sglang", "engine_version": sglang.__version__,
                    "model": args.model, "batch": B, "context": C, "arm": arm,
                    "patches": sorted("linear" if p == "gemm" else p for p in parts),
                    "device": torch.cuda.get_device_name(),
                    "arch": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
                    "mode": ("graphs: CUDA graphs, as SGLang ships" if args.graphs else
                             "eager: disable_cuda_graph"),
                    "prefix_caching_and_chunked_prefill": "radix cache off, chunked prefill "
                                                          "as SGLang ships"},
           "phases": {}}
    pre = statistics.median(walls["prefill"])
    cfg = _config(args.model)
    layers = (cfg.get("text_config", cfg) or {}).get("num_hidden_layers")
    for ph in args.phases:
        files = sorted(glob.glob(os.path.join(dirs[ph], "*.trace.json*")), key=os.path.getmtime)
        if not files:
            continue
        s = summarize_trace(read_trace(files[-1]))
        steps = layer_steps(s["kernels"], layers) or (K if ph == "decode" else 1)
        if ph == "prefill":
            wall = [w / steps for w in walls["prefill"]]
            what, note = (f"{B} requests of {C} tokens, one output token each: {steps:g} "
                          f"step(s)", f"median of {args.windows} unprofiled prefills, per step")
        else:
            wall = [(w - pre) / (n - 1) for w in walls["long"]]
            what, note = (f"{steps:g} decode steps at {B} requests of {C} context",
                          f"(T({n}) - T(1)) / {n - 1}, median of {args.windows}")
        d = _phase(s, steps, wall, what, note)
        if ph == "prefill":
            d["tokens_per_step"] = [round(B * C / steps)]
            d["requests_per_step"] = [max(1, round(B / steps))]
        d["blind"] = _blind(d, args.graphs)
        out["phases"][ph] = d
    eng.shutdown()
    out["model_cfg"] = cfg
    return out


def layer_steps(kernels: Dict[str, list], layers: Optional[int]) -> Optional[float]:
    """How many forward steps a trace holds, from the attention kernel called
    most often: once a layer and step. None without one, or without the
    model's layer count. A window that cut a step leaves a fraction."""
    if not layers:
        return None
    calls = [c for name, (_, c) in kernels.items() if family(name) == "attention"]
    return round(max(calls) / layers, 2) if calls else None


# ------------------------------------------------------------------ main


def _child_args(args) -> List[str]:
    return ["--model", args.model, "--batch", str(args.batch), "--context", str(args.context),
            "--engine", args.engine, "--dtype", args.dtype, "--steps", str(args.steps),
            "--warm", str(args.warm), "--windows", str(args.windows),
            "--phases", ",".join(args.phases), "--routes", args.routes,
            "--keep-trace", args.keep_trace] + (["--graphs"] if args.graphs else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hk.integration headroom",
                                 description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[1:]))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--arms", default="stock",
                    help="comma-separated, as `ab` spells them (stock, hk, gemm, hk+gemm, "
                         "user, ...): the first is ranked, the others measured beside it")
    ap.add_argument("--routes", default="", help="NAME=routes.json,... for gemm@NAME arms")
    ap.add_argument("--phases", default="decode,prefill")
    ap.add_argument("--steps", type=int, default=8, help="decode steps per window")
    ap.add_argument("--warm", type=int, default=4, help="decode steps before the windows")
    ap.add_argument("--windows", type=int, default=3,
                    help="unprofiled windows the step's wall time is the median of")
    ap.add_argument("--graphs", action="store_true",
                    help="CUDA graphs and torch.compile, as the engine ships")
    ap.add_argument("--tokens", type=int, default=0,
                    help="also give shares over a request of this many output tokens")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", default="", help="write the result here")
    ap.add_argument("--keep-trace", default="", help="keep the chrome traces in this directory")
    ap.add_argument("--trace", action="append", default=[],
                    help="PHASE=PATH: rank a chrome trace from elsewhere (one step "
                         "per --steps, unless PATH has its own) instead of running")
    ap.add_argument("--arch", default="", help="with --trace: the arch it was taken on")
    ap.add_argument("--from", dest="from_", default="",
                    help="rank an earlier --json result again, without running")
    ap.add_argument("--arm", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.phases = [p for p in args.phases.split(",") if p]
    if not set(args.phases) <= set(PHASES):
        ap.error(f"phases are {PHASES}")

    if args.arm:                                             # child: one engine
        res = (collect_vllm if args.engine == "vllm" else collect_sglang)(args, args.arm)
        Path(args.out).write_text(json.dumps(res))
        return 0
    if args.from_:
        return _finish(args, rerank(Path(args.from_), args.tokens))

    if args.trace:
        child = {"meta": {"engine": args.engine, "model": args.model, "arch": args.arch,
                          "arm": "(trace)", "mode": "a trace from elsewhere",
                          "batch": args.batch, "context": args.context},
                 "phases": {}}
        for spec in args.trace:
            ph, _, path = spec.partition("=")
            d = summarize_trace(read_trace(path))
            child["phases"][ph] = {**d, "steps": args.steps, "what": f"trace {path}",
                                   "step_ms": None}
        return _finish(args, analyze(child, tokens=args.tokens))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_PKG_ROOT, env.get("PYTHONPATH", "")) if p)
    arms = [a for a in args.arms.split(",") if a]
    got, failed = {}, {}
    for arm in arms:
        f = os.path.join(tempfile.gettempdir(), f"hk_headroom_{os.getpid()}_{arm}.json")
        print(f"--- {arm}", flush=True)
        r = subprocess.run([sys.executable, "-m", "hk.integration.headroom", "--arm", arm,
                            "--out", f] + _child_args(args), env=env)
        if r.returncode:
            failed[arm] = r.returncode
            print(f"--- {arm} FAILED, exit {r.returncode}", flush=True)
            continue
        got[arm] = json.loads(Path(f).read_text())
    if arms[0] not in got:
        print(f"{arms[0]} failed; nothing to rank")
        return 1
    res = analyze(got[arms[0]], compare={a: got[a] for a in arms[1:] if a in got},
                  tokens=args.tokens)
    res["patch_stats"] = {a: g.get("patch_stats", {}) for a, g in got.items()}
    res["arms"] = {a: {ph: {k: d.get(k) for k in ("step_ms", "busy_ms", "steps", "step_spread")}
                        for ph, d in g["phases"].items()} for a, g in got.items()}
    res["failed"] = failed
    res["raw"] = got
    return _finish(args, res)


def rerank(path: Path, tokens: int = 0) -> dict:
    """`--from`: the result of an earlier run, ranked again from its raw
    per-kernel totals (after a change to the families or the record, say)."""
    old = json.loads(path.read_text())
    got = old["raw"]
    arms = list(got)
    res = analyze(got[arms[0]], compare={a: got[a] for a in arms[1:]},
                  tokens=tokens or (old.get("request") or {}).get("tokens", 0))
    for k in ("patch_stats", "arms", "failed", "raw"):
        if k in old:
            res[k] = old[k]
    return res


def _finish(args, res: dict) -> int:
    for line in report(res, args.top):
        print(line)
    for a, ph in (res.get("arms") or {}).items():
        print(f"arm {a}: " + "; ".join(f"{p} {d['step_ms']} ms a step" for p, d in ph.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
