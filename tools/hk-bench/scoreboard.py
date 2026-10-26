#!/usr/bin/env python3
"""hk against Triton and FlyDSL, kernel by kernel, on this GPU -- each DSL
out of the box and tuned, and every arm saying which.

Every row is one operation at shapes an 8B model runs, every column one way
to compute it, all in one process:

  hk            hk.ops on its default schedule (GEMM), or its planner (norms)
  hk_tuned      hk.ops after hk's own tuner searched the shape (GEMM; --tune)
  torch         torch eager: hipBLASLt for GEMM, aotriton (AOT-compiled Triton)
                for attention, ATen for the rest
  inductor      torch.compile of the torch expression -- the Triton a framework
                generates, and what vLLM runs for its norms in graph mode
  triton        a Triton kernel as a Triton user first writes it (triton_ref.py)
  triton_tuned  the same kernel autotuned over gfx1100's space (triton_ref.py)
  flydsl        FlyDSL's own dispatcher, no measurement: its tile heuristic and
                per-tile options, or its default block size
  flydsl_tuned  FlyDSL's autotuner searching (FLYDSL_AUTOTUNE=1)
  aiter         AITER's own entry point for the op, as a vLLM or SGLang on ROCm
                calls it: its FlyDSL flash attention on gfx1201, tuned_gemm,
                rms_norm; what it dispatched to is recorded

Every arm records how it was tuned -- what was searched, how many candidates,
what won -- because a comparison against an opponent who did not try is not a
comparison.

Correctness first: every arm's output against an fp32 torch reference; an
arm that is wrong is reported and not timed, an arm with no kernel here says
why. Then GPU time, from the torch profiler, from the kernels' own records:
each arm in a profiler session of its own, so nothing has to be attributed by
name, and every launch path -- pybind, Triton, FlyDSL's module launch -- is
measured the same way. Rounds interleave the arms, rotating the order.

Not every round is believed (_rounds.py): a round whose session lost kernel
records, one faster than the card can go at the case's footprint (bytes
inside the 96 MB Infinity Cache, or past it to GDDR6), and a best round no
other round comes near are rejected and listed. An arm's time is the median
of the rounds it kept (`--stat best`: the fastest), its noise their median
absolute deviation; --json keeps every round, its kernel record count and
why any was rejected. `--probe-ceilings` measures what the cache's ceiling
stands on.

Two verdicts per row, hk's best against the best other: out of the box (hk,
torch, inductor, triton, flydsl, aiter) and tuned (everything). A gap inside the
band -- the larger of 2% and the two arms' noise -- is a tie.

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/scoreboard.py [--only gemm,norm,...] [--tune] [--json f]

`--arms` runs a subset; the performance gate (check_regression.py) runs the
out-of-the-box arms only, because the tuned ones search on every run.

`--cases PATH` (or $HK_SCOREBOARD_CASES) adds the rows a file of your own
defines, `scoreboard_cases(tune)`, as a group named by its NAME (_cases.py);
`python -m hk.integration new-op` writes such a file. With `--cases` and no
`--only`, those groups are what runs.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))

import torch                                       # noqa: E402
import torch.nn.functional as F                    # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import _cases                                      # noqa: E402
import _rounds                                     # noqa: E402
import hk                                          # noqa: E402
import triton_ref                                  # noqa: E402
from _timing import burn_in                        # noqa: E402

BF16 = torch.bfloat16
ARMS = ("hk", "hk_tuned", "torch", "inductor", "triton", "triton_tuned",
        "flydsl", "flydsl_tuned", "aiter")
OUT_OF_THE_BOX = ("torch", "inductor", "triton", "flydsl", "aiter")
LEVELS = {"oob": (("hk",), OUT_OF_THE_BOX),
          "tuned": (("hk", "hk_tuned"), tuple(a for a in ARMS if not a.startswith("hk")))}
BAND = 0.02


def _arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def rel(got, want):
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max()).item()


def _why(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:90]}" if str(e) else type(e).__name__


# ---------------------------------------------------------------- arms
#
# A case is (name, work, unit, reference, {arm: (run, out, tuning) | "reason"}):
# `run` launches, `out()` returns what to check, `work` is bytes or FLOPs and
# `tuning` is a dict, or a callable returning one once the arm has run (an
# autotuner knows what it picked only after its first call).

def _triton_arms(arms, plain, tuned, call_plain, call_tuned, out_plain, out_tuned):
    arms["triton"] = (call_plain, out_plain, lambda: triton_ref.tuning(plain))
    arms["triton_tuned"] = (call_tuned, out_tuned, lambda: triton_ref.tuning(tuned))


def _flydsl_chosen(tuner):
    """The config FlyDSL's autotuner last resolved -- for an arm's first call,
    the winner of the search it just ran.

    Read off `resolve_config` rather than out of `tuner.cache`: the cache also
    holds entries loaded from earlier processes' disk caches, and a key prefix
    of (rows, cols) matched a stale one of those ahead of this search's."""
    if not hasattr(tuner, "_sb_last"):
        resolve = tuner.resolve_config

        def recorded(*a, **kw):
            tuner._sb_last = resolve(*a, **kw)
            return tuner._sb_last

        tuner._sb_last = None
        tuner.resolve_config = recorded
    return lambda: str(tuner._sb_last) if tuner._sb_last is not None else None


def _flydsl_gemm_tuned(fg, a, b, out, m, n, k):
    """FlyDSL's GEMM autotuner: one forced search, then its winner's launcher.

    `rdna3_gemm_autotuned` serves whatever launcher `rdna3_gemm_dispatch` last
    stored for the signature, and the dispatcher stores pick_tile's on every
    call. With the `flydsl` arm in the process the tuned arm would never
    search, and would time pick_tile's kernel under the tuner's name. So the
    stored launcher is dropped before the search, and the one the search
    leaves -- its last dispatch is the winner -- is kept and called directly,
    which is what the autotuned entry does on a hit."""
    sig = fg._signature(m, n, k, "bf16", "bf16", "rn", a.stride(0), b.stride(0),
                        out.stride(0))
    chosen = _flydsl_chosen(fg._gemm_tuner)
    held = {}

    def run():
        launch = held.get("launch")
        if launch is not None:
            return launch(out, a, b, torch.cuda.current_stream(), 0)
        fg._resolved.pop(sig, None)
        fg.rdna3_gemm_autotuned(out, a, b)
        held["launch"] = fg._resolved[sig]

    def info():
        return {"level": "tuned", "space": "FLYDSL_AUTOTUNE=1 over feasible_tiles",
                "bench": "FlyDSL's graph replay, best of rounds",
                "candidates": len(fg.feasible_tiles(m, n, k)), "chosen": chosen()}

    return run, lambda: out, info


def _hk_gemm_arms(arms, a, b, outs, m, n, k, tune):
    from hk.autotune import load_records
    from hk.ops import gemm as hk_gemm

    default = hk_gemm.TUNER._kernel_for(hk_gemm.DEFAULT)
    arms["hk"] = (lambda: default(a, b, outs["hk"]), lambda: outs["hk"],
                  {"level": "default", "space": "hk's default schedule", "candidates": 1,
                   "chosen": hk_gemm._sched_name(hk_gemm.DEFAULT)})
    key = f"{hk_gemm._bucket(m)}x{n}x{k}"
    rec = load_records(hk_gemm.TUNER.name).get(key)
    if tune and not rec:
        print(f"  tuning hk {key} ...", flush=True)
        hk_gemm.TUNER.tune(key=key, rounds=3)
        rec = load_records(hk_gemm.TUNER.name, reload=True).get(key)
    if not rec:
        arms["hk_tuned"] = f"no hk record for {key}; run with --tune"
        return
    sched = hk_gemm.schedule_for(m, n, k)
    arms["hk_tuned"] = (lambda: hk.ops.matmul(a, b, out=outs["hk_tuned"]),
                        lambda: outs["hk_tuned"],
                        {"level": "tuned", "space": f"hk's gemm space, {len(hk_gemm.SPACE)} points",
                         "candidates": rec.get("measured"), "chosen": hk_gemm._sched_name(sched),
                         "gain_over_default": rec.get("gain"),
                         "adopted": sched != hk_gemm.DEFAULT})


def gemm_case(m, n, k, tune):
    a = torch.randn(m, k, device="cuda", dtype=BF16)
    b = torch.randn(n, k, device="cuda", dtype=BF16) / k ** 0.5
    ref = a.float() @ b.float().T
    outs = {x: torch.empty(m, n, device="cuda", dtype=BF16) for x in ARMS}
    arms = {
        "torch": (lambda: torch.mm(a, b.T, out=outs["torch"]), lambda: outs["torch"],
                  {"level": "library", "space": "hipBLASLt's heuristic", "candidates": None}),
        "inductor": "the same hipBLASLt call as torch (Inductor does not "
                    "template a bf16 GEMM here)",
    }
    _hk_gemm_arms(arms, a, b, outs, m, n, k, tune)
    _triton_arms(arms, triton_ref.matmul_abt, triton_ref.matmul_abt_tuned,
                 lambda: triton_ref.matmul_abt(a, b, out=outs["triton"]),
                 lambda: triton_ref.matmul_abt_tuned(a, b, out=outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    try:
        if _arch().startswith("gfx12"):
            _flydsl_gemm_gfx12(arms, a, b, outs["flydsl"], m, n, k)
        else:
            _flydsl_gemm_gfx11(arms, a, b, outs, m, n, k)
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = arms["flydsl_tuned"] = _why(e)
    arms["aiter"] = _aiter_gemm(a, b, outs["aiter"], m, n, k)
    return f"gemm {m}x{n}x{k}", 2 * m * n * k, "flop", ref, arms, None


def _flydsl_gemm_gfx11(arms, a, b, outs, m, n, k):
    from kernels.gemm import rdna3_f16_gemm_autotune as fg
    tile = fg.pick_tile(m, n, k)
    arms["flydsl"] = (
        lambda: fg.rdna3_gemm_dispatch(outs["flydsl"], a, b, m, n, k),
        lambda: outs["flydsl"],
        {"level": "default", "space": "pick_tile's heuristic, no measurement",
         "candidates": 1, "chosen": f"tile {tile} {fg.tile_opts(tile)}"})
    arms["flydsl_tuned"] = _flydsl_gemm_tuned(fg, a, b, outs["flydsl_tuned"], m, n, k)


def _flydsl_gemm_gfx12(arms, a, b, out, m, n, k):
    """FlyDSL's RDNA4 GEMM (kernels/gemm/rdna_f16_gemm.py): one kernel per
    shape, its module's default geometry, and nothing to search -- FlyDSL's
    RDNA autotuner and pick_tile are rdna3_f16_gemm's, gfx11 only."""
    from kernels.gemm.rdna_f16_gemm import create_wmma_gemm_module
    launch, bm, bn, bk = create_wmma_gemm_module(m, n, k, "bf16", "bf16")
    s = torch.cuda.current_stream()
    arms["flydsl"] = (lambda: launch(out, a, b, s), lambda: out,
                      {"level": "default", "space": "create_wmma_gemm_module's defaults, "
                       "no heuristic or measurement", "candidates": 1,
                       "chosen": f"tile {bm}x{bn}x{bk}, 2x2 waves of 4x4 WMMAs"})
    arms["flydsl_tuned"] = ("FlyDSL has no tuner for its gfx12 GEMM (rdna3_f16_gemm_"
                            "autotune and its pick_tile are gfx11's)")


def _aiter_gemm(a, b, out, m, n, k):
    """AITER's bf16 GEMM entry (tgemm.mm, what its linear layers call): a tuned
    CSV row for (M, N, K, CUs) if there is one, else its default backend."""
    try:
        from aiter.tuned_gemm import get_GEMM_A16W16_config, tgemm
        cfg = get_GEMM_A16W16_config(M=m, N=n, K=k, bias=False, dtype=str(a.dtype),
                                     otype=str(a.dtype), scaleAB=False, bpreshuffle=False)
    except Exception as e:                         # noqa: BLE001
        return _why(e)
    lib = cfg.get("libtype")
    space = ("AITER's tuned GEMM CSV row" if "kernelName" in cfg else
             "no row in AITER's tuned GEMM CSV for this shape and CU count: its default, "
             + ("F.linear (hipBLASLt)" if lib == "torch" else lib))
    return (lambda: tgemm.mm(a, b, out=out), lambda: out,
            {"level": "library", "space": space, "candidates": None,
             "chosen": f"{lib} solution {cfg.get('solidx')} {cfg.get('kernelName') or ''}".strip()})


def attn_case(n, causal, tune, h=32, hkv=8, d=128):
    q = torch.randn(1, h, n, d, device="cuda", dtype=BF16)
    k = torch.randn(1, hkv, n, d, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         is_causal=causal, enable_gqa=True)
    out = {}
    arms = {
        "hk": (lambda: out.__setitem__("hk", hk.ops.attention(q, k, v, causal=causal)),
               lambda: out["hk"],
               {"level": "shipped", "space": "hk's attention records", "candidates": None}),
        "torch": (lambda: out.__setitem__("torch", F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=True)), lambda: out["torch"],
                  {"level": "library", "space": "aotriton", "candidates": None}),
        "inductor": "torch.compile lowers SDPA to the same aotriton call",
        "triton": "aotriton is Triton, compiled ahead of time (the torch "
                  "column); vLLM's own Triton attention is in paged_ab.py",
        "flydsl": _NO_FLYDSL_ATTN.get(_arch()[:5], _NO_FLYDSL_ATTN["gfx11"]),
        "aiter": _aiter_attn(q, k, v, causal, out),
    }
    flops = 4 * h * n * n * d // (2 if causal else 1)
    return (f"attn n{n} {'causal' if causal else 'full'} gqa{h // hkv}",
            flops, "flop", ref, arms, None)


#: FlyDSL's dense attention (flydsl_flash_attn_func -> flash_attn_generic) on
#: RDNA: not called -- on gfx12 it does not fail, it aborts the process.
_NO_FLYDSL_ATTN = {
    "gfx11": "flash_attn_generic stages through raw_ptr_buffer_load_lds; "
             "gfx11 has no global->LDS direct load (flydsl_probe.py)",
    "gfx12": "flash_attn_generic stages through raw_ptr_buffer_load_lds; gfx12 has no "
             "global->LDS direct load, and LLVM aborts the process on it ('Do not know how "
             "to expand this operator's operand', tools/rdna4-probes/fly_probe.py)",
}


def _aiter_attn(q, k, v, causal, out):
    """AITER's FlyDSL flash attention for gfx1201 (aiter.ops.flydsl.fmha_kernels,
    a FlyDSL kernel AITER ships): BSHD, one head count for q, k and v. So the
    GQA heads are expanded and the layout made BSHD once, before and outside
    the timing -- the call times its kernel, not those copies."""
    try:
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_func
    except Exception as e:                         # noqa: BLE001
        return _why(e)
    g = q.shape[1] // k.shape[1]
    qb = q.transpose(1, 2).contiguous()
    kb, vb = (x.repeat_interleave(g, dim=1).transpose(1, 2).contiguous() for x in (k, v))
    return (lambda: out.__setitem__("aiter", flydsl_flash_attn_func(qb, kb, vb, causal=causal)),
            lambda: out["aiter"].transpose(1, 2),
            {"level": "library", "space": "flydsl_flash_attn_func's one kernel "
             "(128-row tile, waves_per_eu 2)", "candidates": 1,
             "chosen": f"K/V expanded to {q.shape[1]} heads, BSHD (no GQA in the kernel)"})


def _flydsl_compiled(build, *args):
    import flydsl.compiler as flyc
    return flyc.compile(build, *args)


_HK_PLANNER = "no search: hk's planner picks the schedule, as in the hk column"

#: Static shapes for Inductor (the llm-* groups): every case is its own graph,
#: compiled for its shape. The expressions are lambdas of one code object per
#: op, so without this Dynamo marks the row axis dynamic at the second shape
#: and, past its recompile limit, runs eager under Inductor's name.
_STATIC = []


def _compile(fn):
    if not _STATIC:
        return torch.compile(fn)
    return torch.compile(fn, dynamic=False)


def rmsnorm_case(m, tune, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = _compile(lambda x, w: F.rms_norm(x, (n,), w, 1e-6))
    res = {}
    arms = {
        "hk": (lambda: hk.ops.rms_norm(x, w, out=outs["hk"]), lambda: outs["hk"],
               {"level": "planner", "space": "hk's norm planner", "candidates": None}),
        "hk_tuned": _HK_PLANNER,
        "torch": (lambda: res.__setitem__("torch", F.rms_norm(x, (n,), w, 1e-6)),
                  lambda: res["torch"], {"level": "library", "candidates": None}),
        "inductor": (lambda: res.__setitem__("inductor", comp(x, w)),
                     lambda: res["inductor"], {"level": "default mode", "candidates": None}),
    }
    _triton_arms(arms, triton_ref.rms_norm, triton_ref.rms_norm_tuned,
                 lambda: triton_ref.rms_norm(x, w, out=outs["triton"]),
                 lambda: triton_ref.rms_norm_tuned(x, w, out=outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    try:
        from kernels.norm.rmsnorm_kernel import build_rmsnorm_module
        s = torch.cuda.current_stream()
        f = _flydsl_compiled(build_rmsnorm_module(n, "bf16"), x, w, outs["flydsl"], m, s)
        arms["flydsl"] = (lambda: f(x, w, outs["flydsl"], m, s), lambda: outs["flydsl"],
                          {"level": "default", "space": "build_rmsnorm_module's block size",
                           "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    try:
        from kernels.norm import rmsnorm_autotune as fr
        chosen = _flydsl_chosen(fr._rmsnorm_tuner)
        arms["flydsl_tuned"] = (
            lambda: fr.rmsnorm_autotuned(x, w, outs["flydsl_tuned"], m),
            lambda: outs["flydsl_tuned"],
            lambda: {"level": "tuned", "space": "BLOCK_THREADS x waves_per_eu",
                     "bench": "FlyDSL's do_bench: warm batches behind a GPU sleep, median",
                     "candidates": len(fr._SEARCH_CONFIGS), "chosen": chosen()})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl_tuned"] = _why(e)
    arms["aiter"] = _aiter_rms(x, w, res)
    return f"rms_norm {m}x{n}", 2 * m * n * 2 + 2 * n, "byte", ref, arms, None


def _aiter_rms(x, w, res):
    """aiter.rms_norm: its HIP module for bf16 at these widths, else opus."""
    try:
        import aiter
    except Exception as e:                         # noqa: BLE001
        return _why(e)
    return (lambda: res.__setitem__("aiter", aiter.rms_norm(x, w, 1e-6)),
            lambda: res["aiter"], {"level": "library", "space": "aiter.rms_norm",
                                   "candidates": None})


def fused_add_case(m, tune, n=4096):
    x0 = torch.randn(m, n, device="cuda", dtype=BF16)
    r0 = torch.randn_like(x0)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    z = x0.float() + r0.float()
    ref = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    # In place, so each arm owns a copy: repeated calls drift the values, not
    # the timing, and the check runs on the first call from fresh copies.
    bufs = {a: (x0.clone(), r0.clone()) for a in ARMS}

    def eager():
        x, r = bufs["torch"]
        r.add_(x)
        x.copy_(F.rms_norm(r, (n,), w, 1e-6))

    comp = _compile(lambda x, r, w: (F.rms_norm(x + r, (n,), w, 1e-6), x + r))
    got = {}

    def inductor():
        x, r = bufs["inductor"]
        got["inductor"] = comp(x, r, w)[0]

    arms = {
        "hk": (lambda: hk.ops.fused_add_rms_norm(*bufs["hk"], w), lambda: bufs["hk"][0],
               {"level": "planner", "space": "hk's norm planner", "candidates": None}),
        "hk_tuned": _HK_PLANNER,
        "torch": (eager, lambda: bufs["torch"][0], {"level": "library", "candidates": None}),
        "inductor": (inductor, lambda: got["inductor"],
                     {"level": "default mode", "candidates": None}),
    }
    _triton_arms(arms, triton_ref.fused_add_rms_norm, triton_ref.fused_add_rms_norm_tuned,
                 lambda: triton_ref.fused_add_rms_norm(*bufs["triton"], w),
                 lambda: triton_ref.fused_add_rms_norm_tuned(*bufs["triton_tuned"], w),
                 lambda: bufs["triton"][0], lambda: bufs["triton_tuned"][0])
    try:
        from kernels.norm.rmsnorm_kernel import build_fused_add_rmsnorm_module
        s = torch.cuda.current_stream()
        xf, rf = bufs["flydsl"]
        y, ro = torch.empty_like(xf), torch.empty_like(rf)
        f = _flydsl_compiled(build_fused_add_rmsnorm_module(n, "bf16"), xf, rf,
                             w, y, ro, m, s)
        arms["flydsl"] = (lambda: f(xf, rf, w, y, ro, m, s), lambda: y,
                          {"level": "default", "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    arms["flydsl_tuned"] = "FlyDSL has no autotuned fused add + RMSNorm"
    try:
        import aiter
        xa, ra = bufs["aiter"]
        ya, rao = torch.empty_like(xa), torch.empty_like(ra)
        arms["aiter"] = (lambda: aiter.rmsnorm2d_fwd_with_add(ya, xa, ra, rao, w, 1e-6),
                         lambda: ya, {"level": "library", "space": "aiter.rmsnorm2d_fwd_with_add",
                                      "candidates": None})
    except Exception as e:                         # noqa: BLE001
        arms["aiter"] = _why(e)
    # Moves x and r in and out: twice the bytes it touches.
    return (f"fused_add_rms_norm {m}x{n}", 2 * m * n * 4 + 2 * n, "byte", ref, arms,
            2 * m * n * 2 + 2 * n)


def softmax_case(m, tune, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    ref = torch.softmax(x.float(), dim=-1)
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = _compile(lambda x: torch.softmax(x, dim=-1))
    res = {}
    arms = {
        "hk": (lambda: hk.ops.softmax(x, out=outs["hk"]), lambda: outs["hk"],
               {"level": "planner", "space": "hk's row planner", "candidates": None}),
        "hk_tuned": _HK_PLANNER,
        "torch": (lambda: torch.softmax(x, dim=-1, out=outs["torch"]),
                  lambda: outs["torch"], {"level": "library", "candidates": None}),
        "inductor": (lambda: res.__setitem__("inductor", comp(x)),
                     lambda: res["inductor"], {"level": "default mode", "candidates": None}),
    }
    _triton_arms(arms, triton_ref.softmax, triton_ref.softmax_tuned,
                 lambda: triton_ref.softmax(x, out=outs["triton"]),
                 lambda: triton_ref.softmax_tuned(x, out=outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    try:
        from kernels.norm.softmax_kernel import build_softmax_module
        f = build_softmax_module(m, n, "bf16")
        s = torch.cuda.current_stream()
        arms["flydsl"] = (lambda: f(x, outs["flydsl"], m, stream=s), lambda: outs["flydsl"],
                          {"level": "default", "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    try:
        from kernels.norm import softmax_autotune as fs
        chosen = _flydsl_chosen(fs._softmax_tuner)
        arms["flydsl_tuned"] = (
            lambda: fs.softmax_autotuned(x, outs["flydsl_tuned"]),
            lambda: outs["flydsl_tuned"],
            lambda: {"level": "tuned", "space": "softmax_autotune's candidate geometries",
                     "bench": "FlyDSL's do_bench: warm batches behind a GPU sleep, median",
                     "candidates": len(fs.candidate_configs(m_in=m, N=n, dtype_str="bf16")),
                     "chosen": chosen()})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl_tuned"] = _why(e)
    arms["aiter"] = "AITER has no softmax op (only fused into its attention and MoE kernels)"
    return f"softmax {m}x{n}", 2 * m * n * 2, "byte", ref, arms, None


def fp8_case(m, n, k, tune):
    """An fp8 linear layer, W8A8 with a scale a token and a scale a channel:
    bf16 (A @ B^T) * scale_a * scale_b. gfx1201 (an fp8 WMMA) only."""
    f8 = torch.float8_e4m3fn
    a = (torch.randn(m, k, device="cuda") * 0.5).to(f8)
    b = (torch.randn(n, k, device="cuda") * 0.5).to(f8)
    sa = torch.rand(m, 1, device="cuda") + 0.5
    sb = torch.rand(n, device="cuda") + 0.5
    ref = (a.float() @ b.float().T) * sa * sb
    outs = {x: torch.empty(m, n, device="cuda", dtype=BF16) for x in ARMS}
    res = {}
    arms = {
        "torch": (lambda: res.__setitem__("torch", torch._scaled_mm(
            a, b.T, scale_a=sa, scale_b=sb.view(1, n), out_dtype=BF16)),
                  lambda: res["torch"],
                  {"level": "library", "space": "hipBLASLt's heuristic (rowwise scales)",
                   "candidates": None}),
        "inductor": "the same hipBLASLt call as torch",
        "triton": "triton_ref has no fp8 GEMM (AITER's Triton one is the aiter column)",
        "hk_tuned": "no hk fp8 tuning records yet",
    }
    try:
        from hk.ops.gemm import _fp8_block_m, scaled_mm
        arms["hk"] = (lambda: scaled_mm(a, b, sa, sb, out=outs["hk"]), lambda: outs["hk"],
                      {"level": "default", "space": "FP8_DEFAULT by M", "candidates": 1,
                       "chosen": f"block_m {_fp8_block_m(m)}"})
    except Exception as e:                         # noqa: BLE001
        arms["hk"] = _why(e)
    try:
        from kernels.gemm.rdna_fp8_preshuffle_gemm import compile_fp8_gemm, preshuffle_b_fp8
        launch = compile_fp8_gemm(M=m, N=n, K=k)
        # B preshuffled once, as a weight is at load; A is read raw.
        bs = preshuffle_b_fp8(b.T.contiguous()).view(torch.float32).contiguous()
        av = a.view(torch.float32)
        sa1, sb1 = sa.view(m).contiguous(), sb.contiguous()
        s = torch.cuda.current_stream()
        arms["flydsl"] = (lambda: launch(outs["flydsl"], av, bs, sa1, sb1, s), lambda: outs["flydsl"],
                          {"level": "default", "space": "compile_fp8_gemm's defaults "
                           "(tuned for 32x8192x6144), B preshuffled outside the timing",
                           "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    arms["flydsl_tuned"] = "FlyDSL has no tuner for its gfx12 fp8 GEMM"
    try:
        import aiter
        arms["aiter"] = (lambda: res.__setitem__("aiter", aiter.gemm_a8w8(
            a, b, sa, sb.view(1, n), None, BF16)), lambda: res["aiter"],
                         {"level": "library", "space": "aiter.gemm_a8w8 (on gfx1201 its Triton "
                          "kernel: the CK one has no gfx12 build)", "candidates": None})
    except Exception as e:                         # noqa: BLE001
        arms["aiter"] = _why(e)
    # "flop8": screened against the fp8 WMMA peak, not bf16's (_rounds).
    return f"fp8 {m}x{n}x{k}", 2 * m * n * k, "flop8", ref, arms, None


def int8_case(m, n, k, tune):
    """An int8 GEMM, int32 C: hk.ops.gemm's kernel on the iu8 WMMA against
    torch._int_mm. AITER's gemm_a8w8 (scales of one, bf16 C) beside them."""
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    ref = a.float() @ b.float().T
    outs = {x: torch.empty(m, n, device="cuda", dtype=torch.int32) for x in ARMS}
    res = {}
    arms = {
        "torch": (lambda: torch._int_mm(a, b.T, out=outs["torch"]), lambda: outs["torch"],
                  {"level": "library", "space": "torch._int_mm (hipBLASLt)", "candidates": None}),
        "inductor": "the same torch._int_mm call",
        "triton": "triton_ref has no int8 GEMM",
        "hk_tuned": "no hk int8 tuning records",
        "flydsl": "FlyDSL's int8 GEMM (kernels/gemm/rdna3_int8_gemm.py) is gfx11's; on gfx12 "
                  "it has the int4 one (the int4 rows)",
        "flydsl_tuned": "no kernel",
    }
    try:
        from hk.ops import gemm as G
        from hk.types import i8
        kern = G.gemm_kernel("scoreboard_i8", op_dtype=i8, **G.DEFAULT)
        arms["hk"] = (lambda: kern(a, b, outs["hk"]), lambda: outs["hk"],
                      {"level": "default", "space": "hk.ops.gemm DEFAULT, op_dtype=i8",
                       "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["hk"] = _why(e)
    try:
        import aiter
        one_a, one_b = torch.ones(m, 1, device="cuda"), torch.ones(1, n, device="cuda")
        arms["aiter"] = (lambda: res.__setitem__("aiter", aiter.gemm_a8w8(
            a, b, one_a, one_b, None, BF16)), lambda: res["aiter"],
                         {"level": "library", "space": "aiter.gemm_a8w8, scales of one, bf16 C "
                          "(its Triton kernel on gfx1201)", "candidates": None})
    except Exception as e:                         # noqa: BLE001
        arms["aiter"] = _why(e)
    return f"int8 {m}x{n}x{k}", 2 * m * n * k, "flop8", ref, arms, None


def int4_case(m, n, k, tune):
    """FlyDSL's signed-int4 GEMM on gfx12's iu4 WMMA (16x16x32), exact int32 C:
    what FlyDSL has for RDNA4 that nothing else here does."""
    import numpy as np
    la = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    lb = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    ref = (la.float() @ lb.float().T).cuda()

    def pack(t):                                   # 8 nibbles an int32, little-endian
        nib = (t.numpy().astype(np.int64) & 0xF).reshape(t.shape[0], -1, 8)
        words = (nib << (np.arange(8) * 4)).sum(-1).astype(np.uint32).view(np.int32)
        return torch.from_numpy(words).cuda().contiguous()

    pa, pb = pack(la), pack(lb)
    out = torch.empty(m, n, device="cuda", dtype=torch.int32)
    none = "no int4 GEMM"
    arms = {a: none for a in ARMS}
    arms["hk"] = "hk has no int4 (iu4) WMMA kernel yet"
    arms["torch"] = "torch has no int4 x int4 GEMM"
    try:
        from kernels.gemm.rdna4_int4_gemm import choose_int4_tile, create_wmma_int4_gemm_module
        launch = create_wmma_int4_gemm_module(m, n, k)
        s = torch.cuda.current_stream()
        arms["flydsl"] = (lambda: launch(pa, pb, out, stream=s), lambda: out,
                          {"level": "default", "space": "choose_int4_tile",
                           "candidates": 1, "chosen": str(choose_int4_tile(m, n, k))})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    return f"int4 {m}x{n}x{k}", 2 * m * n * k, "iop4", ref, arms, None


# ---------------------------------------------------------------- the LLM rows
#
# Every row and pointwise op of a decoder layer at the shapes LLMs run it:
# rows (tokens) x hidden for the norms, softmax, quantize, the elementwise ops
# and RoPE (hidden = heads x 128), rows x intermediate for silu_and_mul. The
# llm-* groups; --rows, --hidden and --inter pick the shapes.

LLM_ROWS = (1, 32, 2048, 8192)
LLM_HIDDEN = (4096, 5120, 8192)
#: Llama-3-8B, Qwen3-14B and Llama-3-70B's intermediate sizes.
LLM_INTER = (14336, 17408, 28672)
_SHAPES = {"rows": LLM_ROWS, "hidden": LLM_HIDDEN, "inter": LLM_INTER}

_NO_FLYDSL = "FlyDSL has no standalone {} kernel (kernels/: only fused into norms or caches)"


def _hk_arm(run, out, what="hk.ops"):
    return (run, out, {"level": "planner", "space": f"{what}'s planner", "candidates": None})


def _torch_arm(run, out):
    return (run, out, {"level": "library", "candidates": None})


def _inductor_arm(run, out):
    return (run, out, {"level": "default mode", "candidates": None})


def _flydsl_tuned_arm(tuner, run, out, space, candidates):
    chosen = _flydsl_chosen(tuner)
    return (run, out, lambda: {"level": "tuned", "space": space,
                               "bench": "FlyDSL's do_bench: warm batches behind a GPU "
                                        "sleep, median",
                               "candidates": candidates, "chosen": chosen()})


def llm_rms_case(kind, m, n):
    """rms_norm (weighted), rmsnorm (no weight) or layernorm (unweighted)."""
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    ones, zeros = torch.ones_like(w), torch.zeros_like(w)
    xf = x.float()
    if kind == "layernorm":
        ref = F.layer_norm(xf, (n,), eps=1e-5)
    else:
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        if kind == "rms_norm":
            ref = ref * w.float()
    outs = {a: torch.empty_like(x) for a in ARMS}
    res = {}
    if kind == "rms_norm":
        hk_run = lambda: hk.ops.rms_norm(x, w, out=outs["hk"])          # noqa: E731
        expr = lambda x: F.rms_norm(x, (n,), w, 1e-6)                   # noqa: E731
        tri, tri_t = (lambda: triton_ref.rms_norm(x, w, out=outs["triton"]),
                      lambda: triton_ref.rms_norm_tuned(x, w, out=outs["triton_tuned"]))
        fns = (triton_ref.rms_norm, triton_ref.rms_norm_tuned)
    elif kind == "rmsnorm":
        hk_run = lambda: hk.ops.rmsnorm(x, out=outs["hk"])              # noqa: E731
        expr = lambda x: F.rms_norm(x, (n,), None, 1e-6)                # noqa: E731
        tri, tri_t = (lambda: triton_ref.rmsnorm(x, out=outs["triton"]),
                      lambda: triton_ref.rmsnorm_tuned(x, out=outs["triton_tuned"]))
        fns = (triton_ref.rmsnorm, triton_ref.rmsnorm_tuned)
    else:
        hk_run = lambda: hk.ops.layernorm(x, out=outs["hk"])            # noqa: E731
        expr = lambda x: F.layer_norm(x, (n,), eps=1e-5)                # noqa: E731
        tri, tri_t = (lambda: triton_ref.layernorm(x, out=outs["triton"]),
                      lambda: triton_ref.layernorm_tuned(x, out=outs["triton_tuned"]))
        fns = (triton_ref.layernorm, triton_ref.layernorm_tuned)
    comp = _compile(expr)
    arms = {
        "hk": _hk_arm(hk_run, lambda: outs["hk"]),
        "hk_tuned": _HK_PLANNER,
        "torch": _torch_arm(lambda: res.__setitem__("torch", expr(x)), lambda: res["torch"]),
        "inductor": _inductor_arm(lambda: res.__setitem__("inductor", comp(x)),
                                  lambda: res["inductor"]),
    }
    _triton_arms(arms, *fns, tri, tri_t, lambda: outs["triton"],
                 lambda: outs["triton_tuned"])
    s = torch.cuda.current_stream()
    try:
        if kind == "layernorm":
            import flydsl.compiler as flyc
            import flydsl.expr as fx
            from kernels.norm.layernorm_kernel import build_layernorm_module
            ptrs = [t.data_ptr() for t in (x, ones, zeros, outs["flydsl"])]
            f = _flydsl_compiled(build_layernorm_module(n, "bf16", eps=1e-5),
                                 *(flyc.from_c_void_p(fx.BFloat16, p) for p in ptrs), m, s)
            arms["flydsl"] = (lambda: f(*ptrs, m, s), lambda: outs["flydsl"],
                              {"level": "default", "space": "build_layernorm_module's "
                               "block size, gamma 1 and beta 0", "candidates": 1})
        else:
            from kernels.norm.rmsnorm_kernel import build_rmsnorm_module
            g = w if kind == "rms_norm" else ones
            f = _flydsl_compiled(build_rmsnorm_module(n, "bf16"), x, g, outs["flydsl"], m, s)
            arms["flydsl"] = (lambda: f(x, g, outs["flydsl"], m, s), lambda: outs["flydsl"],
                              {"level": "default", "space": "build_rmsnorm_module's block "
                               "size" + ("" if kind == "rms_norm" else ", a weight of ones"),
                               "candidates": 1})
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = _why(e)
    if kind == "layernorm":
        arms["flydsl_tuned"] = "FlyDSL has no autotuned LayerNorm"
    else:
        try:
            from kernels.norm import rmsnorm_autotune as fr
            g = w if kind == "rms_norm" else ones
            arms["flydsl_tuned"] = _flydsl_tuned_arm(
                fr._rmsnorm_tuner, lambda: fr.rmsnorm_autotuned(x, g, outs["flydsl_tuned"], m),
                lambda: outs["flydsl_tuned"], "BLOCK_THREADS x waves_per_eu",
                len(fr._SEARCH_CONFIGS))
        except Exception as e:                     # noqa: BLE001
            arms["flydsl_tuned"] = _why(e)
    arms["aiter"] = (_aiter_rms(x, w if kind == "rms_norm" else ones, res)
                     if kind != "layernorm" else "aiter's layernorm is not in this row")
    work = 2 * m * n * 2 + (2 * n if kind == "rms_norm" else 0)
    return f"{kind} {m}x{n}", work, "byte", ref, arms, None


def llm_softmax_case(m, n):
    name, work, unit, ref, arms, fp = softmax_case(m, False, n)
    return name, work, unit, ref, arms, fp


def llm_fused_add_case(m, n):
    return fused_add_case(m, False, n)


def llm_quant_case(m, n):
    """Per-row symmetric int8: scale = absmax / 127, q = round(x / scale)."""
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    ref = (xf * (127.0 / amax)).round().clamp(-128, 127)
    q = {a: torch.empty(m, n, device="cuda", dtype=torch.int8) for a in ARMS}
    sc = {a: torch.empty(m, device="cuda", dtype=torch.float32) for a in ARMS}
    res = {}

    def eager(x):
        a = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        return (x.float() * (127.0 / a)).round().clamp(-128, 127).to(torch.int8), a / 127.0

    comp = _compile(eager)

    # Checked as q in units of one step, so one rounding flip is 1/127.
    def check(qq):
        return lambda: qq.float()

    arms = {
        "hk": _hk_arm(lambda: hk.ops.quantize(x, out=q["hk"], scales=sc["hk"]), check(q["hk"])),
        "hk_tuned": _HK_PLANNER,
        "torch": _torch_arm(lambda: res.__setitem__("torch", eager(x)),
                            lambda: res["torch"][0].float()),
        "inductor": _inductor_arm(lambda: res.__setitem__("inductor", comp(x)),
                                  lambda: res["inductor"][0].float()),
        "flydsl": _NO_FLYDSL.format("int8 quantize"),
        "flydsl_tuned": _NO_FLYDSL.format("int8 quantize"),
    }
    _triton_arms(arms, triton_ref.quantize, triton_ref.quantize_tuned,
                 lambda: triton_ref.quantize(x, q["triton"], sc["triton"]),
                 lambda: triton_ref.quantize_tuned(x, q["triton_tuned"], sc["triton_tuned"]),
                 check(q["triton"]), check(q["triton_tuned"]))
    return f"quantize {m}x{n}", m * n * 3 + 4 * m, "byte", ref, arms, None


def llm_elementwise_case(op, m, n):
    a = torch.randn(m, n, device="cuda", dtype=BF16)
    b = torch.randn_like(a)
    if op == "add":
        expr = lambda a, b: a + b                       # noqa: E731
    elif op == "mul":
        expr = lambda a, b: a * b                       # noqa: E731
    else:
        a = a.clamp(-8, 8)
        expr = lambda a, b: torch.exp(a)                # noqa: E731
    ref = expr(a.float(), b.float())
    outs = {x: torch.empty_like(a) for x in ARMS}
    comp = _compile(expr)
    res = {}
    hk_fn = getattr(hk.ops, op)
    hk_run = ((lambda: hk_fn(a, out=outs["hk"])) if op == "exp"
              else (lambda: hk_fn(a, b, out=outs["hk"])))
    tfn = {"add": torch.add, "mul": torch.mul}.get(op)
    torch_run = ((lambda: torch.exp(a, out=outs["torch"])) if op == "exp"
                 else (lambda: tfn(a, b, out=outs["torch"])))
    arms = {
        "hk": _hk_arm(hk_run, lambda: outs["hk"]),
        "hk_tuned": _HK_PLANNER,
        "torch": _torch_arm(torch_run, lambda: outs["torch"]),
        "inductor": _inductor_arm(lambda: res.__setitem__("inductor", comp(a, b)),
                                  lambda: res["inductor"]),
        "flydsl": _NO_FLYDSL.format(op),
        "flydsl_tuned": _NO_FLYDSL.format(op),
    }
    _triton_arms(arms, triton_ref.elementwise, triton_ref.elementwise_tuned,
                 lambda: triton_ref.elementwise(op, a, b, outs["triton"]),
                 lambda: triton_ref.elementwise_tuned(op, a, b, outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    work = m * n * 2 * (2 if op == "exp" else 3)
    return f"{op} {m}x{n}", work, "byte", ref, arms, None


def llm_silu_mul_case(m, i):
    """vLLM's silu_and_mul: the gate/up GEMM's (m, 2i) output to (m, i)."""
    x = torch.randn(m, 2 * i, device="cuda", dtype=BF16)
    xf = x.float()
    ref = F.silu(xf[:, :i]) * xf[:, i:]
    outs = {a: torch.empty(m, i, device="cuda", dtype=BF16) for a in ARMS}

    def expr(x):
        return F.silu(x[:, :i]) * x[:, i:]

    comp = _compile(expr)
    res = {}
    gate, up = x[:, :i], x[:, i:]
    hk_note = "on x's halves in place"
    try:
        hk.ops.silu_mul(gate, up, out=outs["hk"])
    except ValueError:
        # A wrapper that wants contiguous operands: the halves copied once,
        # outside the timing. The kernel moves the same bytes either way.
        gate, up = gate.contiguous(), up.contiguous()
        hk_note = "on contiguous copies of x's halves (the wrapper refused the views)"
    arms = {
        "hk": (lambda: hk.ops.silu_mul(gate, up, out=outs["hk"]), lambda: outs["hk"],
               {"level": "planner", "space": "hk.ops's planner", "candidates": None,
                "note": hk_note}),
        "hk_tuned": _HK_PLANNER,
        "torch": _torch_arm(lambda: res.__setitem__("torch", expr(x)), lambda: res["torch"]),
        "inductor": _inductor_arm(lambda: res.__setitem__("inductor", comp(x)),
                                  lambda: res["inductor"]),
        "flydsl": _NO_FLYDSL.format("silu_and_mul"),
        "flydsl_tuned": _NO_FLYDSL.format("silu_and_mul"),
    }
    _triton_arms(arms, triton_ref.silu_and_mul, triton_ref.silu_and_mul_tuned,
                 lambda: triton_ref.silu_and_mul(x, outs["triton"]),
                 lambda: triton_ref.silu_and_mul_tuned(x, outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    return f"silu_mul {m}x{i}", m * i * 6, "byte", ref, arms, None


def llm_rope_case(t, hidden, d=128):
    """NeoX RoPE on (t, heads, d): each token's angles, cos and sin (t, d/2)
    in fp32 -- vLLM's rotary_embedding after its cos_sin_cache lookup."""
    h = hidden // d
    x = torch.randn(t, h, d, device="cuda", dtype=BF16)
    pos = torch.arange(t, device="cuda", dtype=torch.float32)[:, None]
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda", dtype=torch.float32) / d))
    cos, sin = torch.cos(pos * inv), torch.sin(pos * inv)
    half = d // 2

    def expr(x, cos, sin):
        x1, x2 = x[..., :half].float(), x[..., half:].float()
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).to(x.dtype)

    ref = expr(x.float(), cos, sin)
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = _compile(expr)
    res = {}
    # hk takes vLLM's cos_sin_cache layout, one (t, d) fp32 row [cos | sin]
    # a token: the same bytes as cos and sin apart.
    cs = torch.cat((cos, sin), -1).contiguous()
    arms = {
        "hk": _hk_arm(lambda: hk.ops.rotary(x, cs, out=outs["hk"]), lambda: outs["hk"]),
        "hk_tuned": _HK_PLANNER,
        "torch": _torch_arm(lambda: res.__setitem__("torch", expr(x, cos, sin)),
                            lambda: res["torch"]),
        "inductor": _inductor_arm(lambda: res.__setitem__("inductor", comp(x, cos, sin)),
                                  lambda: res["inductor"]),
        "flydsl": _NO_FLYDSL.format("RoPE"),
        "flydsl_tuned": _NO_FLYDSL.format("RoPE"),
    }
    _triton_arms(arms, triton_ref.rope, triton_ref.rope_tuned,
                 lambda: triton_ref.rope(x, cos, sin, outs["triton"]),
                 lambda: triton_ref.rope_tuned(x, cos, sin, outs["triton_tuned"]),
                 lambda: outs["triton"], lambda: outs["triton_tuned"])
    return f"rope {t}x{h}x{d}", 2 * t * h * d * 2 + 2 * t * half * 4, "byte", ref, arms, None


def _llm(fn, *dims):
    """Each case of `fn` over the shapes, built one at a time as the run
    reaches it: a case holds its arms' buffers and compiled kernels."""
    import itertools

    def gen(_tune):
        for shape in itertools.product(*(_SHAPES[d] for d in dims)):
            yield fn(*shape)
    return gen


LLM_GROUPS = {
    "llm-norm": [(lambda m, n, k=k: llm_rms_case(k, m, n), ("rows", "hidden"))
                 for k in ("rms_norm", "rmsnorm", "layernorm")]
                + [(llm_fused_add_case, ("rows", "hidden")),
                   (llm_softmax_case, ("rows", "hidden"))],
    "llm-quant": [(llm_quant_case, ("rows", "hidden"))],
    "llm-pointwise": [(lambda m, n, o=o: llm_elementwise_case(o, m, n), ("rows", "hidden"))
                      for o in ("add", "mul", "exp")]
                     + [(llm_silu_mul_case, ("rows", "inter")),
                        (llm_rope_case, ("rows", "hidden"))],
}


def _llm_group(parts):
    def gen(tune):
        for fn, dims in parts:
            yield from _llm(fn, *dims)(tune)
    return gen


CASES = {
    "gemm": lambda t: [gemm_case(4096, 4096, 4096, t), gemm_case(2048, 6144, 4096, t),
                       gemm_case(2048, 4096, 4096, t), gemm_case(2048, 24576, 4096, t),
                       gemm_case(2048, 4096, 12288, t)],
    "attn": lambda t: [attn_case(4096, True, t), attn_case(4096, False, t)],
    "norm": lambda t: [rmsnorm_case(m, t) for m in (1, 32, 2048, 8192)]
                      + [fused_add_case(m, t) for m in (1, 32, 2048, 8192)],
    "softmax": lambda t: [softmax_case(m, t) for m in (2048, 8192)],
    # gfx1201 only (fp8, iu8 and iu4 WMMA); not in the default groups.
    "fp8": lambda t: [fp8_case(m, n, 4096, t) for m in (32, 128, 2048) for n in (4096, 14336)],
    "int": lambda t: [int8_case(m, 4096, 4096, t) for m in (2048, 4096)]
                     + [int4_case(m, 4096, 4096, t) for m in (2048, 4096)],
    **{g: _llm_group(parts) for g, parts in LLM_GROUPS.items()},
}


# ---------------------------------------------------------------- timing

def _kernels(prof):
    return [e for e in prof.events() if str(e.device_type).endswith("CUDA")
            and not e.name.startswith(("Memcpy", "Memset"))]


#: Each round's calls wait behind a GPU spin this long per call (at least
#: QUEUE_MIN_US), so the host has launched them all before the first runs.
#: Inside a profiler session a launch costs the host more than outside one:
#: behind 40 us a call, hk's one-row rounds still varied by 1.5x.
QUEUE_US_PER_CALL = 1000
QUEUE_MIN_US = 10000


def _spin(us):
    """torch's spin kernel for `us` microseconds (on gfx1100 it counts
    100 MHz), on hk.tune.measure's clock: measured after a warm-up call, since
    timing the first one -- which loads the kernel's module -- reads a clock
    16-100x too slow and makes every spin that much too short."""
    from hk.tune.measure import spin_hz  # noqa: PLC0415
    torch.cuda._sleep(int(us * 1e-6 * spin_hz(torch)))


def _is_spin(e):
    return "spin_kernel" in e.name


def _arm_buffers(arms):
    """The per-arm buffer dicts a case's runs read at call time -- `outs`,
    `bufs`, `q`, `sc`: dicts keyed by arm name, found in the runs' closures."""
    found, seen = [], set()

    def visit(fn, depth):
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, dict) and id(v) not in seen and len(set(v) & set(ARMS)) > 1:
                seen.add(id(v))
                found.append(v)
            elif callable(v) and depth < 2:
                visit(v, depth + 1)

    for spec in arms.values():
        if not isinstance(spec, str):
            visit(spec[0], 0)
    return found


def buffer_rotation(arms, timed):
    """rotate(r): before round r, each timed arm that owns a buffer takes the
    one r arms along, so over the rounds every arm runs on the same buffers.

    A one-row kernel's time moves 0.1 us (3.78-3.95) with where its output
    sits, whichever kernel writes it; one buffer per arm for the whole run
    turned that into a verdict (docs/history/43)."""
    plan = []
    for d in _arm_buffers(arms):
        owners = [a for a in timed if a in d]
        if len(owners) > 1:
            plan.append((d, owners, dict(d)))

    def rotate(r):
        for d, owners, orig in plan:
            for i, a in enumerate(owners):
                d[a] = orig[owners[(i + r) % len(owners)]]
    return rotate if plan else None


def gpu_rounds(fns, rounds, calls, queue=True, rotate=None):
    """{arm: ([per-call GPU ms, one per round], [kernel records, per round])}.

    Each arm runs in a profiler session of its own, and its time is the sum
    of the kernel records in it -- the hardware's timestamps. On ROCm the
    profiler's range -> kernel correlation is unreliable for anything
    launched outside ATen (pybind, Triton, FlyDSL's module launch): duplicate
    flow warnings, kernels credited to the wrong range, a GEMM at 347 TFLOP/s
    on a 123 TFLOP/s card. A session per arm needs no correlation at all.
    Rounds interleave the arms and rotate which goes first. The record count
    is kept because a session that lost records sums fewer calls than it is
    divided by (_rounds.py).

    With `queue`, a GPU spin goes first in each session, long enough for the
    host to launch every call behind it, so the calls run back to back
    whatever their launcher costs. Without it, a microsecond kernel's time
    was the launcher's: a call that reached the GPU a few microseconds after
    it went idle took about 10 us more, 10-20% of the calls of the arms that
    launch in 5-8 us (hk's pybind entry, torch, FlyDSL) and none of those
    that take 21-28 (Triton, Inductor), nor any queued or spaced 20 us
    apart (small_rows.py, docs/history/32).
    """
    per = {a: ([], []) for a in fns}
    order = list(fns)
    for r in range(rounds):
        if rotate:
            rotate(r)
        k = r % len(order)
        for a in order[k:] + order[:k]:
            fns[a]()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                if queue:
                    _spin(max(QUEUE_MIN_US, QUEUE_US_PER_CALL * calls))
                for _ in range(calls):
                    fns[a]()
                torch.cuda.synchronize()
            ks = [e for e in _kernels(p) if not _is_spin(e)]
            total = sum(e.device_time_total for e in ks)
            per[a][0].append(total / 1e3 / calls if total else float("inf"))
            per[a][1].append(len(ks))
    if rotate:
        rotate(0)
    return per


def gpu_times(fns, rounds, calls, stat=None, queue=True):
    """{arm: (per-call GPU ms, spread)}: `_rounds.STAT` (or `stat`) of the
    rounds `_rounds.screen` keeps, without a ceiling, which needs the work."""
    out = {}
    for a, (ts, recs) in gpu_rounds(fns, rounds, calls, queue).items():
        kept, _ = _rounds.screen(ts, recs)
        ms, spread, _ = _rounds.summarize(ts, kept, stat or _rounds.STAT)
        out[a] = (ms if ms is not None else float("inf"), spread)
    return out


def probe_ceilings(rounds, calls):
    """The fastest a plain copy and a two-input add move bytes on this card,
    by footprint, timed as the arms are: what _rounds.IC_BYTES_S stands on."""
    print(f"footprint MB   copy TB/s   add TB/s   (best kept round of {rounds} x {calls})")
    sizes = (8, 16, 32, 48, 64, 80, 96, 128, 192, 256, 512)
    if _rounds.IC_BYTES > 128 * 2**20:
        sizes = (16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 1024)
    for mb in sizes:
        fp = mb * 2**20
        x = torch.randn(fp // 4, device="cuda", dtype=BF16)          # copy: 2 tensors
        y = torch.empty_like(x)
        a, b = (torch.randn(fp // 6, device="cuda", dtype=BF16) for _ in range(2))
        c = torch.empty_like(a)                                       # add: 3 tensors
        # On gfx950 copy_ is a DMA memcpy, which leaves no kernel record:
        # a one-input elementwise kernel moves the same bytes. And an
        # in-place add there, a += b, three tensors' bytes over a footprint
        # of two: what the in-place fused add does. gfx1201's copy_ (ROCm
        # 7.2.3) is a DMA too, so the neg stands in wherever copy_ left none.
        wide = _rounds.IC_BYTES > 128 * 2**20
        dma = not gpu_rounds({"copy": lambda: y.copy_(x)}, 1, 1)["copy"][1][0]
        copy = (lambda: torch.neg(x, out=y)) if wide or dma else (lambda: y.copy_(x))
        fns = {"copy": copy, "add": lambda: torch.add(a, b, out=c)}
        if wide:
            p, q = (torch.randn(fp // 4, device="cuda", dtype=BF16) for _ in range(2))
            fns["add_"] = lambda: p.add_(q)
        res = {}
        for name, (ts, recs) in gpu_rounds(fns, rounds, calls).items():
            kept, _ = _rounds.screen(ts, recs)
            _, _, best = _rounds.summarize(ts, kept)
            moved = {"copy": 2 * x.numel() * 2, "add": 3 * a.numel() * 2,
                     "add_": 3 * fp // 4 * 2}[name]
            res[name] = moved / (best * 1e-3) / 1e12
        print(f"{mb:>12}   {res['copy']:>9.3f}   {res['add']:>8.3f}"
              + (f"   in place {res['add_']:>6.3f}" if wide else ""))
        del x, y, a, b, c
        torch.cuda.empty_cache()
    return 0


def verdicts(t):
    """Per level: hk's best arm against the best other, as a ratio of the
    other's time to hk's (above 1 is hk faster), with its noise band."""
    res = {}
    for level, (mine, others) in LEVELS.items():
        h = [(t[a][0], a) for a in mine if a in t]
        o = [(t[a][0], a) for a in others if a in t]
        if not h or not o:
            continue
        (th, ah), (to, ao) = min(h), min(o)
        ratio = to / th
        band = max(BAND, t[ah][1], t[ao][1])
        kind = "tie" if abs(ratio - 1) <= band else ("win" if ratio > 1 else "loss")
        res[level] = {"hk": ah, "other": ao, "ratio": round(ratio, 3),
                      "band": round(band, 3), "verdict": kind}
    return res


def _versions():
    v = {"device": torch.cuda.get_device_name(),
         "arch": torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
         "torch": torch.__version__, "triton": __import__("triton").__version__,
         "hk": hk.__version__}
    try:
        import flydsl
        v["flydsl"] = getattr(flydsl, "__version__", "?")
    except Exception as e:                         # noqa: BLE001
        v["flydsl"] = _why(e)
    return v


def _aiter_version():
    """AITER's, if a case imported it (importing it builds its JIT modules)."""
    if "aiter" not in sys.modules:
        return None
    from importlib import metadata
    try:
        return metadata.version("aiter")
    except metadata.PackageNotFoundError:
        return getattr(sys.modules["aiter"], "__version__", "source tree")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="",
                    help="comma-separated groups (default gemm,attn,norm,softmax, or "
                         "the --cases groups)")
    ap.add_argument("--cases", default="",
                    help="comma-separated .py files or directories defining "
                         "scoreboard_cases(tune) (_cases.py)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--tune", action="store_true",
                    help="run hk's GEMM tuner for shapes it has no record for")
    ap.add_argument("--json", default="", help="write every case, arm and verdict here")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated arms to run; the others are listed as "
                    "not selected (the tuned arms search, which takes minutes)")
    ap.add_argument("--stat", choices=_rounds.STATS, default=_rounds.STAT,
                    help="an arm's time over the rounds it kept (_rounds.py)")
    ap.add_argument("--probe-ceilings", action="store_true",
                    help="time a copy and an add by footprint, and stop")
    ap.add_argument("--rotate-buffers", action="store_true",
                    help="each round, pass the arms' output buffers one arm along "
                    "(buffer_rotation)")
    ap.add_argument("--no-queue", action="store_true",
                    help="launch each round's calls onto an idle GPU, as before "
                    "chapter 32 (gpu_rounds)")
    for dim, default in _SHAPES.items():
        ap.add_argument(f"--{dim}", default=",".join(map(str, default)),
                        help=f"the llm-* groups' {dim} (default %(default)s)")
    args = ap.parse_args()
    selected = set(args.arms.split(","))
    if not selected <= set(ARMS):
        ap.error(f"unknown arms {sorted(selected - set(ARMS))}; arms are {ARMS}")
    extra = _cases.load(_cases.paths(args.cases))
    clash = sorted(set(extra) & set(CASES))
    if clash:
        ap.error(f"--cases groups {clash} are the scoreboard's own; rename them (NAME)")
    CASES.update(extra)
    args.only = args.only or (",".join(extra) if extra else "gemm,attn,norm,softmax")
    unknown = [g for g in args.only.split(",") if g not in CASES]
    if unknown:
        ap.error(f"unknown groups {unknown}; have {sorted(CASES)}")
    for dim in _SHAPES:
        _SHAPES[dim] = tuple(int(v) for v in getattr(args, dim).split(","))
    if any(g in LLM_GROUPS for g in args.only.split(",")):
        _STATIC.append(True)
        for knob in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                     "accumulated_cache_size_limit"):
            if hasattr(torch._dynamo.config, knob):
                setattr(torch._dynamo.config, knob, 4096)
    meta = _versions()
    _rounds.use(meta["arch"])
    print(f"{meta['device']} {meta['arch']}, torch {meta['torch']}, triton "
          f"{meta['triton']}, flydsl {meta['flydsl']}, hk {meta['hk']}")
    burn_in()
    if args.probe_ceilings:
        return probe_ceilings(args.rounds, args.calls)

    rules = {"stat": args.stat, "queued": not args.no_queue,
             "rotated_buffers": args.rotate_buffers, "peak_flops": _rounds.PEAK_FLOPS,
             "hbm_bytes_s": _rounds.HBM_BYTES_S, "ic_bytes": _rounds.IC_BYTES,
             "ic_bytes_s": _rounds.IC_BYTES_S, "margin": _rounds.MARGIN,
             "gap_floor": _rounds.GAP_FLOOR, "gap_k": _rounds.GAP_K}
    report = {"meta": {**meta, "rounds": args.rounds, "calls": args.calls, "band": BAND,
                       "arms": [a for a in ARMS if a in selected], "screen": rules,
                       "date": time.strftime("%Y-%m-%d %H:%M")}, "cases": []}
    head = f"{'case':<28}" + "".join(f"{a:>13}" for a in ARMS) + "   oob / tuned"
    for group in args.only.split(","):
        print(f"\n[{group}] GPU ms per call ({args.stat} of the rounds kept), then rate "
              f"(TFLOP/s or TB/s)")
        print(head)
        for name, work, unit, ref, arms, footprint in CASES[group](args.tune):
            footprint = footprint or (work if unit == "byte" else None)
            fns, notes, tuning, errs = {}, {}, {}, {}
            # FlyDSL's autotuner searches on every call while FLYDSL_AUTOTUNE
            # is set, and serves its cached winner once it is not: set for the
            # checked first call, unset for the timed ones. Its default arms
            # do not go through the autotuner at all.
            os.environ["FLYDSL_AUTOTUNE"] = "1"
            for a in ARMS:
                spec = arms.get(a, "no kernel") if a in selected else "not selected (--arms)"
                if isinstance(spec, str):
                    notes[a] = spec
                    continue
                run, out, tune_info = spec
                try:
                    run()
                    torch.cuda.synchronize()
                    err = rel(out(), ref)
                    tuning[a] = tune_info() if callable(tune_info) else tune_info
                    errs[a] = err
                    if err > 3e-2:
                        notes[a] = f"WRONG: rel err {err:.2e}"
                        continue
                    fns[a] = run
                except Exception as e:             # noqa: BLE001
                    notes[a] = _why(e)
            os.environ.pop("FLYDSL_AUTOTUNE", None)
            floor = _rounds.floor_ms(work, unit, footprint)
            rotate = buffer_rotation(arms, list(fns)) if args.rotate_buffers else None
            t, raw = {}, (gpu_rounds(fns, args.rounds, args.calls, not args.no_queue, rotate)
                          if fns else {})
            for a, (ts, recs) in raw.items():
                kept, rejected = _rounds.screen(ts, recs, floor)
                ms, spread, best = _rounds.summarize(ts, kept, args.stat)
                raw[a] = (ts, recs, rejected, best)
                if ms is None:
                    notes[a] = f"every round rejected: {rejected[0]}"
                else:
                    t[a] = (ms, spread)
            v = verdicts(t)
            cells = "".join(f"{t[a][0]:>13.4f}" if a in t else f"{'-':>13}" for a in ARMS)
            vtxt = " / ".join(f"{v[l]['ratio']:.2f}x {v[l]['verdict']}" if l in v else "-"
                              for l in LEVELS)
            print(f"{name:<28}{cells}   {vtxt}")
            rates = "".join(f"{work / (t[a][0] * 1e-3) / 1e12:>13.2f}" if a in t
                            else f"{'':>13}" for a in ARMS)
            print(f"{'':<28}{rates}")
            for a, why in notes.items():
                if a in selected:
                    print(f"{'':<30}{a}: {why}")
            for a in ("hk_tuned", "triton_tuned", "flydsl_tuned", "aiter"):
                if a in tuning and tuning[a].get("chosen"):
                    print(f"{'':<30}{a} chose {tuning[a]['chosen'][:110]}")
            for a, (ts, _, rejected, _) in raw.items():
                for i, why in sorted(rejected.items()):
                    print(f"{'':<30}{a}: round {i + 1} of {len(ts)} rejected, "
                          f"{ts[i]:.5g} ms -- {why}")

            def arm(a):
                ent = {"ms": round(t[a][0], 6) if a in t else None,
                       "rate": (round(work / (t[a][0] * 1e-3) / 1e12, 3) if a in t else None),
                       "spread": round(t[a][1], 4) if a in t else None,
                       "rel_err": errs.get(a), "tuning": tuning.get(a), "note": notes.get(a)}
                if a in raw:
                    ts, recs, rejected, best = raw[a]
                    ent.update(best=round(best, 6) if best else None,
                               rounds=[round(x, 6) if x < float("inf") else None for x in ts],
                               records=recs,
                               rejected={str(i): why for i, why in sorted(rejected.items())})
                return ent

            report["cases"].append({
                "group": group, "case": name, "work": work, "unit": unit,
                "footprint": footprint, "arms": {a: arm(a) for a in ARMS}, "verdicts": v})
    from hk.runtime.compile import (built_fingerprint, dialect_version, hipcc_fingerprint,
                                    toolchain_fingerprint)
    report["meta"].update(aiter=_aiter_version(),
                          kernels=built_fingerprint(), hipcc=hipcc_fingerprint(),
                          toolchain=toolchain_fingerprint(), dialect=dialect_version())
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json}")
    loc()
    return 0


def _loc(src: str) -> int:
    n, in_doc = 0, False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(('"""', "'''")):
            if not (len(s) > 3 and s.endswith(('"""', "'''"))):
                in_doc = not in_doc
            continue
        if in_doc or not s or s.startswith("#"):
            continue
        n += 1
    return n


def loc():
    """Lines of code per kernel, docstrings and comments out: what writing
    the same operation took in each DSL."""
    from hk.ops import attn, gemm, norm

    def py(f):
        while not inspect.isfunction(f):           # Autotuner -> JITFunction
            f = f.fn
        return f

    rows = [
        ("gemm", gemm.gemm_kernel, py(triton_ref._matmul_abt),
         "kernels/gemm/rdna_f16_gemm.py" if _arch().startswith("gfx12")
         else "kernels/gemm/rdna3_f16_gemm.py"),
        ("attention", attn.attn_kernel, None,
         "kernels/attention/flash_attn_generic.py"),
        ("rms_norm + fused add", norm._strip_kernel,
         py(triton_ref._rmsnorm), "kernels/norm/rmsnorm_kernel.py"),
        ("softmax", norm._strip_kernel, py(triton_ref._softmax),
         "kernels/norm/softmax_kernel.py"),
    ]
    root = Path(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))
    print("\nlines of code per kernel (no docstrings, comments or blanks)")
    print(f"{'kernel':<24}{'hk':>8}{'triton':>8}{'flydsl':>8}")
    for name, h, t, f in rows:
        fl = root / f
        print(f"{name:<24}{_loc(inspect.getsource(h)):>8}"
              f"{(_loc(inspect.getsource(t)) if t else 0):>8}"
              f"{(_loc(fl.read_text()) if fl.exists() else 0):>8}")
    print("hk's norm rows are one strip kernel, rms/layer/softmax and the "
          "weighted and fused-add forms; FlyDSL's rows are whole files, helpers "
          "included.")


if __name__ == "__main__":
    raise SystemExit(main())
