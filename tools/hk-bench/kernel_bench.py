#!/usr/bin/env python3
"""hk's GEMM and MoE kernels against the best alternative on the GPU, at a real model's shapes.

    python3 tools/hk-bench/kernel_bench.py --op linear --model openbmb/MiniCPM5-2B
    python3 tools/hk-bench/kernel_bench.py --op moe --model Qwen/Qwen3.6-35B-A3B --tp 2
    python3 tools/hk-bench/kernel_bench.py --op moe-w4 --model Qwen/Qwen3.6-35B-A3B-AWQ

What a row compares, and how (README.md in this directory says more):

* **The shapes** are the model's own, read from its config.json (a path, the local HF cache or the
  hub) and sharded for `--tp` the way vLLM shards them.
* **The competitors** are every alternative that runs here, each set up as well as its users
  would set it up:
  - linear: hipBLASLt (torch's default); hipBLASLt through TunableOp, tuned in this process on
    exactly these shapes with a 256 MB rotating buffer (every M, M = 1 included); vLLM's
    `rocm_unquantized_gemm` (its skinny GEMMs at small M); Inductor's `max-autotune` (Triton
    templates and aten, autotuned before timing); FlyDSL's GEMM and AITER where they run;
  - MoE: vLLM's Triton `fused_moe` / `moe_wna16` with the config vLLM picks and with a config
    searched in this process at each token count (as vLLM's `benchmark_moe` does); FlyDSL and
    AITER where they run.
  An alternative that does not run here is listed as n/a with the reason.
* **The timing** is one process, every arm interleaved round by round, the fastest round kept.
  Each arm's calls are captured once in a CUDA graph and its replays timed, so no arm pays host
  cost. Every call reads weights of its own: each shape's copies span `--rotate-mb` (576 MB by
  default, six times a W7900D's Infinity Cache), so nothing is read from the cache.
  - Calls of one shape back to back favour a tuned hipBLASLt kernel, which ran 2-5 us faster that
    way than inside a model's step (hk-dsl #189). So these ratios are conservative for hk.
* **Correctness** is checked before timing: every arm against an fp32 reference. An arm that
  disagrees is reported as "wrong", never timed.

Output: a markdown table per (arch, op) -- shape, M, hk, the best competitor and the ratio, then
every arm -- and with `--out`, a JSON with each arm's version and configuration, the environment
and every round.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))

LINEAR_MS = (1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096)
#: Relative error against the reference (max |error| / max |reference|): above LOOSE an arm's
#: time is marked in the table (its accuracy is looser than one rounding of a 16-bit output with
#: fp32 accumulation allows), above WRONG it is not timed at all.
LOOSE, WRONG = 2e-2, 0.1
#: The most calls one MoE graph replays (one routing each).
MOE_CALLS_MAX = 64
MOE_MS = (1, 2, 4, 8, 16, 32, 64, 256, 1024)


# ------------------------------------------------------------------------------- timing


def graph_us(arms: Dict[str, List[Callable]], rounds: int) -> Dict[str, Dict[str, float]]:
    """{arm: {"us": fastest round's us a call, "spread": slowest/fastest - 1}} for arms whose
    calls (one per weight copy or routing) are each captured as one CUDA graph. Captured on a
    side stream, warmed there first: a library with per-stream state (hipBLASLt's handle and
    workspace) makes it on first use, which a capturing stream does not permit."""
    import torch

    graphs = {}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    for name, calls in arms.items():
        with torch.cuda.stream(side):
            calls[0]()
        side.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=side):
            for c in calls:
                c()
        graphs[name] = g
    torch.cuda.synchronize()
    took: Dict[str, List[float]] = {n: [] for n in arms}
    names = list(arms)
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            graphs[n].replay()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            graphs[n].replay()
            e.record()
            torch.cuda.synchronize()
            took[n].append(s.elapsed_time(e) * 1e3 / len(arms[n]))
    del graphs
    return {n: {"us": min(v), "spread": max(v) / min(v) - 1} for n, v in took.items()}


def rel_err(got, want) -> float:
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------- environment


def versions() -> Dict[str, str]:
    import torch

    out = {"python": platform.python_version(), "torch": torch.__version__,
           "hip": str(torch.version.hip), "gpu": torch.cuda.get_device_name(0),
           "arch": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]}
    for mod in ("vllm", "triton", "flydsl", "aiter", "hk"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "installed"))
        except Exception as e:  # noqa: BLE001
            out[mod] = f"n/a ({type(e).__name__}: {str(e)[:80]})"
    return out


# ------------------------------------------------------------------------------- linear


def linear_shapes(model: str, tp: int) -> List[Tuple[str, int, int]]:
    """(name, N, K) of a model's dense linear layers, sharded for `tp`. In a MoE model the dense
    MLP is the shared expert (its routed experts are `--op moe`)."""
    from hk.autotune import routes

    cfg = routes.model_config(model)
    text = dict(cfg.get("text_config", cfg))
    moe = not text.get("intermediate_size")
    if moe:
        text["intermediate_size"] = text.get("shared_expert_intermediate_size") or 0
    out = []
    for name, n, k, _ in routes.linear_shapes(text, tp):
        if moe and name in ("gate_up", "down"):
            if not text["intermediate_size"]:
                continue
            name = "shared_" + name
        out.append((name, n, k))
    return out


class TunableOp:
    """PyTorch's TunableOp, switched on and off in this process, its results tuned here."""

    def __init__(self, rotate_mb: int):
        import torch

        self.t = torch.cuda.tunable
        self.t.enable(False)
        self.rotate_mb = rotate_mb
        self.tuned: set = set()

    def tune(self, x, w):
        import torch.nn.functional as F

        key = (x.shape[0], w.shape[0], w.shape[1])
        if key in self.tuned:
            return
        self.t.enable(True)
        self.t.tuning_enable(True)
        self.t.set_rotating_buffer_size(self.rotate_mb)
        F.linear(x, w)
        self.t.tuning_enable(False)
        self.t.enable(False)
        self.tuned.add(key)

    def results(self) -> int:
        return len(self.t.get_results())


def _linear_fn():
    """A fresh function object for each torch.compile, so that each shape is compiled and
    autotuned on its own rather than recompiled until dynamo gives up and runs it eagerly."""
    import torch.nn.functional as F

    def linear(a, b):
        return F.linear(a, b)

    return linear


def model_label(model: str) -> str:
    """'org/name' for an HF cache path (.../models--org--name/snapshots/...), else as given."""
    for part in Path(model).parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/")
    return model


def linear_arms(m: int, n: int, k: int, ws, x, out, to: TunableOp, only: set, notes: dict):
    """{arm: (call of one weight copy, how it is set up)} for y = x @ W.T, W (n, k)."""
    import torch
    import torch.nn.functional as F

    from hk.autotune import routes

    arms = {}
    dop = routes._decode_op(m, n, k)
    if dop is not None:
        arms["hk decode_gemm"] = (lambda w: dop(x, w, out), "hk.ops.decode_gemm, its schedule for M")
    op = routes._gemm_op(m, n, k)
    if op is not None:
        arms["hk gemm"] = (lambda w: op(x, w, out), "hk.ops.gemm, its tiling for M")
    arms["hipBLASLt"] = (lambda w: F.linear(x, w), "torch F.linear, TunableOp off")

    def tuned(w):
        to.t.enable(True)
        y = F.linear(x, w)
        to.t.enable(False)
        return y

    to.tune(x, ws[0])
    arms["hipBLASLt+TunableOp"] = (tuned, f"TunableOp tuned here, {to.rotate_mb} MB rotating buffer")
    if "vllm" in only:
        try:
            import vllm.model_executor.layers.utils  # noqa: F401 -- registers the op

            eng = torch.ops.vllm.rocm_unquantized_gemm

            def vl(w):
                to.t.enable(True)
                y = eng(x, w, None)
                to.t.enable(False)
                return y

            arms["vLLM"] = (vl, "torch.ops.vllm.rocm_unquantized_gemm (wvSplitK/LLMM1 at small M, "
                                "else hipBLASLt+TunableOp)")
        except Exception as e:  # noqa: BLE001
            notes["vLLM"] = f"n/a: {type(e).__name__}: {str(e)[:100]}"
    if "inductor" in only:
        try:
            # one compiled function per shape, each autotuned for its own (M, N, K)
            fn = torch.compile(_linear_fn(), mode="max-autotune-no-cudagraphs", dynamic=False)
            fn(x, ws[0])
            arms["Inductor max-autotune"] = (lambda w: fn(x, w),
                                             "torch.compile mode=max-autotune-no-cudagraphs, "
                                             "autotuned before timing")
        except Exception as e:  # noqa: BLE001
            notes["Inductor max-autotune"] = f"n/a: {type(e).__name__}: {str(e)[:100]}"
    if "flydsl" in only:
        try:
            from hk.integration import flydsl_gemm

            if not flydsl_gemm.available():
                notes["FlyDSL"] = f"n/a: {flydsl_gemm.status()}"
            elif flydsl_gemm.variant(m, n, k) is None:
                notes.setdefault("FlyDSL", "n/a at some shapes: no tile takes them")
            else:
                arms["FlyDSL"] = (lambda w: flydsl_gemm.run(x, w, out),
                                  f"FlyDSL GEMM, tile {flydsl_gemm.variant(m, n, k)}")
        except Exception as e:  # noqa: BLE001
            notes["FlyDSL"] = f"n/a: {type(e).__name__}: {str(e)[:100]}"
    if "aiter" in only:
        try:
            from aiter.tuned_gemm import tgemm  # noqa: PLC0415

            arms["AITER"] = (lambda w: tgemm.mm(x, w), "aiter.tuned_gemm.tgemm.mm")
        except Exception as e:  # noqa: BLE001
            notes["AITER"] = (f"n/a: {type(e).__name__}: {str(e)[:100]} (AITER's GEMM and MoE "
                              f"kernels target CDNA: gfx942, gfx950)")
    return arms


def run_linear(a, to: TunableOp, only: set, notes: dict) -> List[dict]:
    import torch

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    for name, n, k in linear_shapes(a.model, a.tp):
        copies = max(2, math.ceil(a.rotate_mb * 2**20 / (n * k * 2)))
        copies = min(copies, max(2, (8 << 30) // (n * k * 2)))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * k ** -0.5
              for _ in range(copies)]
        for m in a.ms:
            if name == "lm_head" and m > 256:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            arms = linear_arms(m, n, k, ws, x, out, to, only, notes)
            ref = x.float() @ ws[0].float().t()
            row = {"shape": name, "n": n, "k": k, "m": m, "arms": {}, "setup": {}}
            timed, errs = {}, {}
            for arm, (call, how) in arms.items():
                try:
                    y = call(ws[0])
                    y = out if y is None else y
                    torch.cuda.synchronize()
                    err = rel_err(y, ref)
                except Exception as e:  # noqa: BLE001
                    row["arms"][arm] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                    continue
                row["setup"][arm] = how
                errs[arm] = err
                if not err < WRONG:
                    row["arms"][arm] = {"wrong": round(err, 4)}
                    continue
                timed[arm] = [lambda w=w, call=call: call(w) for w in ws]
            for arm, t in graph_us(timed, a.rounds).items():
                row["arms"][arm] = {**{k2: round(v, 3) for k2, v in t.items()}, "err": round(errs.get(arm, 0.0), 4)}
            rows.append(summarise(row))
            print(line(row), flush=True)
        del ws
        torch.cuda.empty_cache()
    return rows


def call_by_name(fn, **kw):
    """`fn` with the keyword arguments its signature has, positionally in its order: vLLM's
    custom-op wrappers change their parameter lists between versions."""
    import inspect

    names = list(inspect.signature(fn).parameters)
    return fn(*[kw[n] for n in names if n in kw])


def gptq_checkpoint(n, k, group, dev, g):
    """A random symmetric GPTQ v1 checkpoint (qweight [K/8, N], qzeros, scales [K/group, N])
    and the fp32 W [N, K] it means."""
    import torch

    q = torch.randint(0, 16, (k, n), device=dev, dtype=torch.int32, generator=g)
    qweight = torch.zeros(k // 8, n, device=dev, dtype=torch.int32)
    for t in range(8):
        qweight |= (q[t::8] & 0xF) << (4 * t)
    qzeros = torch.full((k // group, n // 8), 0x77777777, device=dev, dtype=torch.int32)
    scales = (torch.rand(k // group, n, device=dev, generator=g) * 0.02 + 0.005).half()
    w = ((q.float() - 8) * scales.float().repeat_interleave(group, 0)).t().contiguous()
    return qweight, qzeros, scales, w



def awq_checkpoint(n, k, group, dev, g):
    """A random AutoAWQ GEMM checkpoint (qweight [K, N/8] and qzeros [K/group, N/8], both packed
    along N in AWQ's order; scales [K/group, N]) and the fp32 W [N, K] it means."""
    import torch

    from hk.ops.w4a16 import AWQ_ORDER

    def pack(v):                        # [R, N] codes -> int32 [R, N/8], AWQ's nibble order
        words = torch.zeros(v.shape[0], v.shape[1] // 8, device=dev, dtype=torch.int32)
        for c, o in enumerate(AWQ_ORDER):
            words |= (v[:, c::8] & 0xF) << (4 * o)
        return words

    q = torch.randint(0, 16, (k, n), device=dev, dtype=torch.int32, generator=g)
    z = torch.randint(0, 16, (k // group, n), device=dev, dtype=torch.int32, generator=g)
    scales = (torch.rand(k // group, n, device=dev, generator=g) * 0.02 + 0.005).half()
    w = ((q - z.repeat_interleave(group, 0)).float()
         * scales.float().repeat_interleave(group, 0)).t().contiguous()
    return pack(q), pack(z), scales, w

#: The checkpoint's modules behind each shape `linear_shapes` names, for `quantized`.
MODULES = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
           "o": ("self_attn.o_proj",), "qkvz": ("linear_attn.in_proj_qkvz",),
           "ba": ("linear_attn.in_proj_ba", "linear_attn.in_proj_a", "linear_attn.in_proj_b"),
           "out": ("linear_attn.out_proj",), "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
           "down": ("mlp.down_proj",)}


def quantized(qc: dict, name: str) -> bool:
    """Whether the checkpoint quantizes shape `name`: GPTQModel's `dynamic` rules starting "-:"
    leave the modules they match unquantized (a Qwen3.8 checkpoint keeps in_proj_ba in 16 bits)."""
    import re

    mods = [f"model.layers.0.{m}" for m in MODULES.get(name, (name,))]
    return not any(rule.startswith("-:") and re.fullmatch(rule[2:], m)
                   for rule in (qc.get("dynamic") or {}) for m in mods)


def run_linear_w4(a, only: set, notes: dict) -> List[dict]:
    """4-bit linear layers, fp16 activations. A GPTQ checkpoint (symmetric): hk's w4a16 decode
    GEMM against vLLM's RDNA3 GPTQ kernel and its exllama kernel. An AWQ one (zero points): hk's
    against what vLLM's AutoAWQLinearMethod runs, its dequantize + matmul (the branch its compiled
    graph takes at every batch, docs/history/22) and its Triton awq_gemm (below 256 tokens, eager)."""
    import torch

    from hk.autotune import routes
    from hk.ops import w4a16 as W

    cfg = routes.model_config(a.model)
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}
    group = int(qc.get("group_size") or 128)
    awq = qc.get("quant_method") == "awq"
    if awq:
        os.environ.setdefault("VLLM_USE_TRITON_AWQ", "1")     # ROCm's AWQ is the Triton kernel
    try:
        from vllm import _custom_ops as ops
    except Exception as e:  # noqa: BLE001
        ops = None
        notes["vLLM"] = f"n/a: {type(e).__name__}: {str(e)[:100]}"
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    rows = []
    skipped = []
    for name, n, k in linear_shapes(a.model, a.tp):
        if name == "lm_head" or k % group or n % 8:
            continue
        if not quantized(qc, name):
            skipped.append(name)
            continue
        copies = max(2, math.ceil(a.rotate_mb * 2**20 / (n * k // 2)))
        if awq:
            ck = [awq_checkpoint(n, k, group, dev, g) for _ in range(copies)]
            hkw = [W.repack_awq(qw, qz, sc, group, dtype=torch.float16) for qw, qz, sc, _ in ck]
        else:
            ck = [gptq_checkpoint(n, k, group, dev, g) for _ in range(copies)]
            # The symmetric form, as vllm_gptq loads it: every zero point 8, scales in 16-row blocks.
            hkw = [W.repack_gptq(qw, qz, sc, group, dtype=torch.float16, symmetric=True)
                   for qw, qz, sc, _ in ck]
        vl = []
        if ops is not None and not awq:
            empty = torch.empty(0, device=dev, dtype=torch.int32)        # no act-order g_idx
            for qw, qz, sc, _ in ck:
                w = qw.clone()
                call_by_name(ops.gptq_shuffle, q_weight=w, q_perm=empty, bit=4)
                vl.append((w, qz, sc))
        for m in a.ms:
            x = (torch.randn(m, k, device=dev, generator=g) * 0.5).half()
            ref = x.float() @ ck[0][3].t()
            arms = {}
            if W.plan(m, n, k, group, torch.float16, not awq) is not None:
                arms["hk w4a16"] = ([lambda i=i, x=x: W.matmul(x, *hkw[i], group)
                                     for i in range(copies)],
                                    "hk.ops.w4a16.matmul, " + (
                                        "zero points (repack_awq, as vllm_int4 loads it)" if awq else
                                        "symmetric form (repack_gptq(symmetric=True), as vllm_gptq "
                                        "loads it)"))
            if awq and ops is not None:
                arms["vLLM AWQ dequantize + matmul"] = (
                    [lambda i=i, x=x: torch.matmul(x, ops.awq_dequantize(ck[i][0], ck[i][2], ck[i][1],
                                                                         0, 0, 0))
                     for i in range(copies)],
                    "vllm._custom_ops.awq_dequantize + torch.matmul: what AutoAWQLinearMethod's "
                    "compiled graph runs (VLLM_USE_TRITON_AWQ=1)")
                arms["vLLM awq_gemm (Triton)"] = (
                    [lambda i=i, x=x: ops.awq_gemm(x, ck[i][0], ck[i][2], ck[i][1], 8)
                     for i in range(copies)],
                    "vllm._custom_ops.awq_gemm, Triton (VLLM_USE_TRITON_AWQ=1): AutoAWQLinearMethod "
                    "below 256 tokens, eager")
            if ops is not None and not awq and hasattr(ops, "gptq_gemm_rdna3"):
                arms["vLLM gptq_gemm_rdna3"] = (
                    [lambda i=i, x=x: call_by_name(
                        ops.gptq_gemm_rdna3, a=x, b_q_weight=vl[i][0], b_qzeros=vl[i][1],
                        b_scales=vl[i][2], b_g_idx=empty, use_v2_format=False, use_exllama=False)
                     for i in range(copies)],
                    "vllm._custom_ops.gptq_gemm_rdna3 (vLLM's RDNA3 GPTQ kernel, what its GPTQ "
                    "layers run on gfx11)")
            if ops is not None and not awq and hasattr(ops, "gptq_gemm"):
                arms["vLLM gptq_gemm (exllama)"] = (
                    [lambda i=i, x=x: call_by_name(
                        ops.gptq_gemm, a=x, b_q_weight=vl[i][0], b_gptq_qzeros=vl[i][1],
                        b_gptq_scales=vl[i][2], b_g_idx=empty, use_exllama=True,
                        use_v2_format=False, bit=4)
                     for i in range(copies)],
                    "vllm._custom_ops.gptq_gemm, exllama kernel")
            row = {"shape": name, "n": n, "k": k, "m": m, "arms": {}, "setup": {}}
            timed, errs = {}, {}
            for arm, (calls, how) in arms.items():
                try:
                    y = calls[0]()
                    torch.cuda.synchronize()
                    err = rel_err(y, ref)
                except Exception as e:  # noqa: BLE001
                    row["arms"][arm] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                    continue
                row["setup"][arm] = how
                errs[arm] = err
                if not err < WRONG:
                    row["arms"][arm] = {"wrong": round(err, 4)}
                    continue
                timed[arm] = calls
            for arm, t in graph_us(timed, a.rounds).items():
                row["arms"][arm] = {**{k2: round(v, 3) for k2, v in t.items()}, "err": round(errs.get(arm, 0.0), 4)}
            rows.append(summarise(row))
            print(line(row), flush=True)
        del ck, hkw, vl
        torch.cuda.empty_cache()
    if skipped:
        notes["skipped"] = f"{', '.join(skipped)}: left unquantized by the checkpoint"
    return rows


# ------------------------------------------------------------------------------- MoE


def moe_shape(model: str, tp: int) -> Tuple[int, int, int, int]:
    """(experts, top_k, hidden, intermediate per rank) of a MoE model's routed experts."""
    from hk.autotune import routes

    cfg = routes.model_config(model)
    cfg = cfg.get("text_config", cfg)
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg["num_local_experts"]
    top_k = cfg.get("num_experts_per_tok") or cfg["moe_topk"]
    inter = cfg.get("moe_intermediate_size") or cfg["intermediate_size"]
    return experts, top_k, cfg["hidden_size"], inter // tp


#: The config search, per token count: vLLM's benchmark_moe space, cut to what fits a W7900D's
#: LDS and registers in seconds a point.
SEARCH = [dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=gm,
               num_warps=nw, num_stages=2, SPLIT_K=1)
          for bm in (16, 32, 64) for bn in (32, 64, 128) for bk in (32, 64, 128)
          for gm in (1, 8) for nw in (4, 8)]


def run_moe(a, w4: bool, only: set, notes: dict) -> List[dict]:
    import torch
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    from vllm.triton_utils import tl

    from hk.ops import moe

    experts, top_k, hidden, inter = moe_shape(a.model, a.tp)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    dt = torch.float16 if w4 else torch.bfloat16
    group = 32
    layer_bytes = experts * 3 * inter * hidden * (0.5 if w4 else 2)
    copies = max(2, math.ceil(a.rotate_mb * 2**20 / layer_bytes))

    def weights():
        if not w4:
            return (torch.randn(experts, 2 * inter, hidden, device=dev, dtype=dt, generator=g)
                    * hidden ** -0.5,
                    torch.randn(experts, hidden, inter, device=dev, dtype=dt, generator=g)
                    * inter ** -0.5)
        out = []
        for n, k in ((2 * inter, hidden), (hidden, inter)):
            q = torch.randint(0, 256, (experts, n, k // 2), device=dev, dtype=torch.uint8,
                              generator=g)
            s = (torch.rand(experts, n, k // group, device=dev, generator=g) * 0.02 + 0.001).to(dt)
            z = torch.randint(0, 256, (experts, n // 2, k // group), device=dev,
                              dtype=torch.uint8, generator=g)
            out.append((q, s, z))
        return tuple(out)

    layers = [weights() for _ in range(copies)]
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size as align)
    except ImportError:
        align = moe.moe_align
    rows = []
    for m in a.ms:
        x = torch.randn(m, hidden, device=dev, dtype=dt, generator=g)
        h = torch.randn(m * top_k, inter, device=dev, dtype=dt, generator=g)
        # Each call its own routing, and enough calls that the experts they read cover the
        # rotation: at a few tokens a call reads only top_k experts, and two layer copies' worth
        # of calls would be served from the Infinity Cache.
        touched = min(experts, m * top_k) * layer_bytes / experts
        calls = min(MOE_CALLS_MAX, max(copies, math.ceil(a.rotate_mb * 2**20 / touched)))
        routings = []
        for _ in range(calls):
            scores = torch.rand(m, experts, device=dev, generator=g)
            tw, ti = scores.topk(top_k, dim=1)
            routings.append((tw.softmax(dim=1).float(), ti.to(torch.int32)))
        c1 = torch.empty(m, top_k, 2 * inter, device=dev, dtype=dt)
        c2 = torch.empty(m, top_k, hidden, device=dev, dtype=dt)

        def sorted_for(config, naive_ok=True):
            """vLLM's assignment for `config` (its naive one-pair-a-block one where vLLM would use
            it and `naive_ok`), one per routing."""
            out = []
            for tw, ti in routings:
                if naive_ok:
                    out.append((tw, *fm._prepare_expert_assignment(
                        ti, config, m, top_k, experts, None, ignore_invalid_experts=True)))
                else:
                    out.append((tw, *align(ti, config["BLOCK_SIZE_M"], experts)))
            return out

        def triton_layer(config, L, s, o1, o2):
            tw, sid, eid, ntp = s
            if not w4:
                w1, w2 = L
                fm.invoke_fused_moe_triton_kernel(x, w1, o1, None, None, None, sid, eid, ntp,
                                                  False, top_k, config, tl.bfloat16, False, False,
                                                  False, False, False, None)
                fm.invoke_fused_moe_triton_kernel(h, w2, o2, None, None, tw, sid, eid, ntp,
                                                  True, 1, config, tl.bfloat16, False, False,
                                                  False, False, False, None)
            else:
                (q1, s1, z1), (q2, s2, z2) = L
                fm.invoke_fused_moe_wna16_triton_kernel(x, q1, o1, s1, z1, None, sid, eid, ntp,
                                                        False, top_k, config, tl.float16, False,
                                                        True, [0, group])
                fm.invoke_fused_moe_wna16_triton_kernel(h, q2, o2, s2, z2, tw, sid, eid, ntp,
                                                        True, 1, config, tl.float16, False,
                                                        True, [0, group])

        def hk_layer(L, s, o1, o2):
            tw, sid, eid, ntp = s
            if not w4:
                w1, w2 = L
                moe.moe_gemm(x, w1, o1, sid, eid, ntp, top_k=top_k, block_m=16)
                moe.moe_gemm(h, w2, o2, sid, eid, ntp, top_k=1, block_m=16, topk_weights=tw)
                return
            from hk.integration import vllm_moe_wna16 as route

            for a_, (q, s_, z), o, tk, wts in ((x, L[0], o1, top_k, None), (h, L[1], o2, 1, tw)):
                e_, n_, kb = q.shape
                sl, wp = route.schedule(a_.shape[1])
                moe.moe_gemm_w4(a_, q.view(torch.int32).reshape(e_ * n_, kb // 4),
                                s_.reshape(e_ * n_, -1), z.view(torch.int8).reshape(e_ * n_ // 2, -1),
                                o, sid, eid, ntp, top_k=tk, block_m=16, group=group, warps=wp,
                                slices=sl, topk_weights=wts)

        if w4:
            vcfg = fm.try_get_optimal_moe_config((experts, 2 * inter, hidden // 2),
                                                 (experts, hidden, inter // 2), top_k,
                                                 "int4_w4a16", m, block_shape=[0, group])
        else:
            vcfg = fm.try_get_optimal_moe_config((experts, 2 * inter, hidden),
                                                 (experts, hidden, inter), top_k, None, m)
        row = {"shape": f"E{experts} top{top_k} {hidden}x{inter}", "m": m, "arms": {}, "setup": {}}
        # {arm: (layer(L, s, o1, o2), its assignments, how it is set up)}
        cands = {"vLLM Triton (vLLM's config)": (
                     lambda L, s, o1, o2, c=vcfg: triton_layer(c, L, s, o1, o2),
                     sorted_for(vcfg, naive_ok=not w4), f"{'moe_wna16' if w4 else 'fused_moe'} "
                     f"Triton kernel, config {vcfg}"),
                 "hk": (hk_layer, sorted_for(dict(vcfg, BLOCK_SIZE_M=16), naive_ok=not w4),
                        "hk.ops.moe." + ("moe_gemm_w4" if w4 else "moe_gemm") + ", BLOCK_M 16, "
                        "the schedule its vLLM route picks")}

        def calls_of(fn, sets, o1, o2):
            return [lambda L=layers[i % copies], s=sets[i]: fn(L, s, o1, o2) for i in range(calls)]

        if "search" in only:
            best, best_cfg, refused = float("inf"), None, 0
            o1, o2 = torch.empty_like(c1), torch.empty_like(c2)
            # vLLM's own config is one of the candidates, so the search never does worse than it
            # the keys this vLLM's kernel takes are the ones its own config has (SPLIT_K or not)
            space = [{k2: v for k2, v in c.items() if k2 in vcfg or k2 != "SPLIT_K"} for c in SEARCH]
            for cfg in [dict(vcfg)] + [c for c in space if c != vcfg]:
                try:
                    st = sorted_for(cfg, naive_ok=not w4)
                    fn = lambda L, s, o1, o2, c=cfg: triton_layer(c, L, s, o1, o2)  # noqa: E731
                    t = graph_us({"c": calls_of(fn, st, o1, o2)}, 3)["c"]["us"]
                except Exception:  # noqa: BLE001 -- a config the compiler or the LDS refuses
                    refused += 1
                    continue
                if t < best:
                    best, best_cfg = t, cfg
            if best_cfg is not None:
                cands["vLLM Triton (searched config)"] = (
                    lambda L, s, o1, o2, c=best_cfg: triton_layer(c, L, s, o1, o2),
                    sorted_for(best_cfg, naive_ok=not w4),
                    f"best of vLLM's config and {len(SEARCH)} searched here ({refused} refused "
                    f"by the compiler or the LDS): {best_cfg}")
            else:
                notes.setdefault("vLLM Triton (searched config)",
                                 f"n/a at M={m}: every config was refused")
        for arm, why in (("FlyDSL", "no RDNA MoE kernel in FlyDSL"),
                         ("AITER", "aiter's fused_moe kernels are CDNA-only")):
            if arm.lower() in only:
                notes.setdefault(arm, f"n/a: {why}")
        ref = None
        timed, errs = {}, {}
        for arm, (fn, sets, how) in cands.items():
            o1, o2 = torch.zeros_like(c1), torch.zeros_like(c2)
            try:
                fn(layers[0], sets[0], o1, o2)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                row["arms"][arm] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
                continue
            row["setup"][arm] = how
            if ref is None:      # vLLM's own kernel on its own config is the reference
                ref = (o1, o2)
            else:
                err = max(rel_err(o1, ref[0]), rel_err(o2, ref[1]))
                errs[arm] = err
                if not err < WRONG:
                    row["arms"][arm] = {"wrong": round(err, 4)}
                    continue
            timed[arm] = calls_of(fn, sets, o1, o2)
        for arm, t in graph_us(timed, a.rounds).items():
            row["arms"][arm] = {**{k2: round(v, 3) for k2, v in t.items()}, "err": round(errs.get(arm, 0.0), 4)}
        rows.append(summarise(row))
        print(line(row), flush=True)
    return rows


# ------------------------------------------------------------------------------- report


def summarise(row: dict) -> dict:
    t = {a: v["us"] for a, v in row["arms"].items() if "us" in v}
    hk = {a: v for a, v in t.items() if a.startswith("hk")}
    comp = {a: v for a, v in t.items() if not a.startswith("hk")}
    if hk:
        row["hk"] = min(hk, key=hk.get)
    if comp:
        row["best"] = min(comp, key=comp.get)
    if hk and comp:
        row["ratio"] = round(comp[row["best"]] / hk[row["hk"]], 3)
    return row


def line(row: dict) -> str:
    t = {a: v.get("us") for a, v in row["arms"].items()}
    hk, best = row.get("hk"), row.get("best")
    head = f"{row['shape']:>12} M={row['m']:<5}"
    if hk and best:
        return (f"{head} hk {t[hk]:9.2f} us ({hk})  best {t[best]:9.2f} us ({best})  "
                f"{row['ratio']:.2f}x")
    return f"{head} {row['arms']}"


def table(rows: List[dict], arms: List[str]) -> str:
    out = ["| shape | M | hk, us | best competitor, us | hk vs best | "
           + " | ".join(arms) + " |",
           "|---|---:|---:|---|---:|" + "---:|" * len(arms)]
    for r in rows:
        t = r["arms"]

        def cell(a):
            v = t.get(a)
            if v is None:
                return "--"
            if "us" in v:
                return f"{v['us']:.2f}" + (" †" if v.get("err", 0) > LOOSE else "")
            return "wrong" if "wrong" in v else "error"

        hk = f"{t[r['hk']]['us']:.2f}" if r.get("hk") else "--"
        best = f"{t[r['best']]['us']:.2f} ({r['best']})" if r.get("best") else "--"
        ratio = f"**{r['ratio']:.2f}x**" if r.get("ratio") else "--"
        out.append(f"| {r['shape']} | {r['m']} | {hk} | {best} | {ratio} | "
                   + " | ".join(cell(a) for a in arms) + " |")
    return "\n".join(out)


def render(d: dict) -> str:
    """The markdown for one run's JSON: the table, then how each arm was set up."""
    rows, env = d["rows"], d["env"]
    arms = sorted({x for r in rows for x in r["arms"]}, key=lambda s: (not s.startswith("hk"), s))
    md = (f"### {d['op']} at {model_label(d['model'])} (TP{d['tp']}) on {env['gpu']} "
          f"({env['arch']})\n\n"
          f"torch {env['torch']}, HIP {env['hip']}, vLLM {env['vllm']}, Triton {env['triton']}, "
          f"FlyDSL {env['flydsl']}, AITER {env['aiter']}, hk {env['hk']}. Same process, "
          f"{d['rounds']} interleaved rounds, the fastest kept; graph-timed, weights rotated over "
          f"{d['rotate_mb']} MB a shape.\n\n" + table(rows, arms) + "\n")
    if d.get("tunableop"):
        t = d["tunableop"]
        md += (f"\nTunableOp: {t['tuned']} of {t['tuned']} (M, N, K) tuned here (M = 1 included), "
               f"{t['results']} results, {t['rotate_mb']} MB rotating buffer.\n")
    if any(v.get("err", 0) > LOOSE for r in rows for v in r["arms"].values()):
        md += (f"\n† relative error above {LOOSE:g} against the reference (looser than one rounding "
               f"of a 16-bit output with fp32 accumulation); still below {WRONG:g}, so timed. The "
               f"JSON has each arm's error.\n")
    md += "\n**Setup of each arm.**\n"
    for arm in arms:
        how = {}
        for r in rows:
            if arm in r.get("setup", {}):
                how.setdefault(r["setup"][arm], []).append(f"{r['shape']} M={r['m']}")
        if len(how) == 1:
            md += f"- {arm}: {next(iter(how))}\n"
        elif how:
            md += f"- {arm}, by row:\n" + "".join(f"  - {', '.join(w)}: {h}\n"
                                                   for h, w in how.items())
    if d.get("notes"):
        md += "\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(d["notes"].items())) + "\n"
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--op", choices=("linear", "linear-w4", "moe", "moe-w4"))
    ap.add_argument("--model", help="HF name, a snapshot dir, or a config.json")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--m", default="", help="comma-separated Ms / token counts")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--rotate-mb", type=int, default=576,
                    help="weight bytes of each shape's copies (> 2x the Infinity Cache)")
    ap.add_argument("--competitors", default="vllm,inductor,flydsl,aiter,search",
                    help="which optional competitors to run (search: the MoE config search)")
    ap.add_argument("--out", help="JSON with every arm, its setup and the environment")
    ap.add_argument("--md", help="write the markdown table here too")
    ap.add_argument("--render", nargs="+", metavar="JSON",
                    help="no GPU: print the markdown of runs saved with --out")
    a = ap.parse_args()
    if a.render:
        print("\n".join(render(json.loads(Path(p).read_text())) for p in a.render))
        return 0
    if not (a.op and a.model):
        ap.error("--op and --model are required (or --render)")
    a.ms = [int(v) for v in a.m.split(",")] if a.m else list(LINEAR_MS if a.op.startswith("linear")
                                                              else MOE_MS)
    only = set(a.competitors.split(","))
    if "inductor" in only:
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 4096
        torch._dynamo.config.accumulated_cache_size_limit = 1 << 16
    env = versions()
    notes: dict = {}
    t0 = time.time()
    to = None
    if a.op == "linear":
        to = TunableOp(256)
        rows = run_linear(a, to, only, notes)
    elif a.op == "linear-w4":
        rows = run_linear_w4(a, only, notes)
    else:
        rows = run_moe(a, a.op == "moe-w4", only, notes)
    d = dict(op=a.op, model=a.model, tp=a.tp, env=env, notes=notes, rounds=a.rounds,
             rotate_mb=a.rotate_mb, seconds=round(time.time() - t0), rows=rows,
             tunableop=(dict(tuned=len(to.tuned), results=to.results(), rotate_mb=to.rotate_mb)
                        if to is not None else None))
    md = render(d)
    print("\n" + md)
    if a.md:
        Path(a.md).write_text(md)
    if a.out:
        Path(a.out).write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
