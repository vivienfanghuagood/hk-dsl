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
torch, inductor, triton, flydsl) and tuned (everything). A gap inside the
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
        "flydsl", "flydsl_tuned")
OUT_OF_THE_BOX = ("torch", "inductor", "triton", "flydsl")
LEVELS = {"oob": (("hk",), OUT_OF_THE_BOX),
          "tuned": (("hk", "hk_tuned"), tuple(a for a in ARMS if not a.startswith("hk")))}
BAND = 0.02


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
        from kernels.gemm import rdna3_f16_gemm_autotune as fg
        tile = fg.pick_tile(m, n, k)
        arms["flydsl"] = (
            lambda: fg.rdna3_gemm_dispatch(outs["flydsl"], a, b, m, n, k),
            lambda: outs["flydsl"],
            {"level": "default", "space": "pick_tile's heuristic, no measurement",
             "candidates": 1, "chosen": f"tile {tile} {fg.tile_opts(tile)}"})
        arms["flydsl_tuned"] = _flydsl_gemm_tuned(fg, a, b, outs["flydsl_tuned"], m, n, k)
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = arms["flydsl_tuned"] = _why(e)
    return f"gemm {m}x{n}x{k}", 2 * m * n * k, "flop", ref, arms, None


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
        "flydsl": "flash_attn_generic stages through raw_ptr_buffer_load_lds; "
                  "gfx11 has no global->LDS direct load (flydsl_probe.py)",
    }
    flops = 4 * h * n * n * d // (2 if causal else 1)
    return (f"attn n{n} {'causal' if causal else 'full'} gqa{h // hkv}",
            flops, "flop", ref, arms, None)


def _flydsl_compiled(build, *args):
    import flydsl.compiler as flyc
    return flyc.compile(build, *args)


_HK_PLANNER = "no search: hk's planner picks the schedule, as in the hk column"


def rmsnorm_case(m, tune, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = torch.compile(lambda x, w: F.rms_norm(x, (n,), w, 1e-6))
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
    return f"rms_norm {m}x{n}", 2 * m * n * 2 + 2 * n, "byte", ref, arms, None


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

    comp = torch.compile(lambda x, r, w: (F.rms_norm(x + r, (n,), w, 1e-6), x + r))
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
    # Moves x and r in and out: twice the bytes it touches.
    return (f"fused_add_rms_norm {m}x{n}", 2 * m * n * 4 + 2 * n, "byte", ref, arms,
            2 * m * n * 2 + 2 * n)


def softmax_case(m, tune, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    ref = torch.softmax(x.float(), dim=-1)
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = torch.compile(lambda x: torch.softmax(x, dim=-1))
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
    return f"softmax {m}x{n}", 2 * m * n * 2, "byte", ref, arms, None


CASES = {
    "gemm": lambda t: [gemm_case(4096, 4096, 4096, t), gemm_case(2048, 6144, 4096, t),
                       gemm_case(2048, 4096, 4096, t), gemm_case(2048, 24576, 4096, t),
                       gemm_case(2048, 4096, 12288, t)],
    "attn": lambda t: [attn_case(4096, True, t), attn_case(4096, False, t)],
    "norm": lambda t: [rmsnorm_case(m, t) for m in (1, 32, 2048, 8192)]
                      + [fused_add_case(m, t) for m in (1, 32, 2048, 8192)],
    "softmax": lambda t: [softmax_case(m, t) for m in (2048, 8192)],
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
_SPIN = []


def _spin(us):
    """torch's spin kernel for `us` microseconds, its clock calibrated on
    first use (on gfx1100 it counts 100 MHz)."""
    if not _SPIN:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        s.record()
        torch.cuda._sleep(200_000)
        e.record()
        torch.cuda.synchronize()
        _SPIN.append(200_000 / (s.elapsed_time(e) * 1e3))
    torch.cuda._sleep(int(us * _SPIN[0]))


def _is_spin(e):
    return "spin_kernel" in e.name


def gpu_rounds(fns, rounds, calls, queue=True):
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
    for mb in (8, 16, 32, 48, 64, 80, 96, 128, 192, 256, 512):
        fp = mb * 2**20
        x = torch.randn(fp // 4, device="cuda", dtype=BF16)          # copy: 2 tensors
        y = torch.empty_like(x)
        a, b = (torch.randn(fp // 6, device="cuda", dtype=BF16) for _ in range(2))
        c = torch.empty_like(a)                                       # add: 3 tensors
        fns = {"copy": lambda: y.copy_(x), "add": lambda: torch.add(a, b, out=c)}
        res = {}
        for name, (ts, recs) in gpu_rounds(fns, rounds, calls).items():
            kept, _ = _rounds.screen(ts, recs)
            _, _, best = _rounds.summarize(ts, kept)
            moved = 2 * x.numel() * 2 if name == "copy" else 3 * a.numel() * 2
            res[name] = moved / (best * 1e-3) / 1e12
        print(f"{mb:>12}   {res['copy']:>9.3f}   {res['add']:>8.3f}")
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
    ap.add_argument("--no-queue", action="store_true",
                    help="launch each round's calls onto an idle GPU, as before "
                    "chapter 32 (gpu_rounds)")
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
    meta = _versions()
    print(f"{meta['device']} {meta['arch']}, torch {meta['torch']}, triton "
          f"{meta['triton']}, flydsl {meta['flydsl']}, hk {meta['hk']}")
    burn_in()
    if args.probe_ceilings:
        return probe_ceilings(args.rounds, args.calls)

    rules = {"stat": args.stat, "queued": not args.no_queue, "peak_flops": _rounds.PEAK_FLOPS,
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
            t, raw = {}, (gpu_rounds(fns, args.rounds, args.calls, not args.no_queue)
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
            for a in ("hk_tuned", "triton_tuned", "flydsl_tuned"):
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
    report["meta"].update(kernels=built_fingerprint(), hipcc=hipcc_fingerprint(),
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
         "kernels/gemm/rdna3_f16_gemm.py"),
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
