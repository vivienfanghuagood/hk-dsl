#!/usr/bin/env python3
"""hk against Triton and FlyDSL, kernel by kernel, on this GPU.

Every row is one operation at shapes an 8B model runs, every column one way
to compute it, all in one process:

  hk       hk.ops
  torch    torch eager: hipBLASLt for GEMM, aotriton (AOT-compiled Triton)
           for attention, ATen for the rest
  inductor torch.compile of the torch expression -- the Triton a framework
           generates, and what vLLM runs for its norms in graph mode
  triton   a Triton kernel as a Triton user writes it (triton_ref.py)
  flydsl   FlyDSL's own kernel for the operation (FLYDSL_ROOT/kernels)

Correctness first: every arm's output against an fp32 torch reference; an
arm that is wrong is reported and not timed, an arm with no kernel here says
why. Then GPU time, from the torch profiler: each arm's launches inside a
record_function, so a Python launcher's overhead is not charged to its
kernel and every launch path -- pybind, Triton, FlyDSL's module launch -- is
measured the same way. Arms are interleaved over rounds; the best round
counts.

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/scoreboard.py [--only gemm,norm,...]
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))

import torch                                       # noqa: E402
import torch.nn.functional as F                    # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

import hk                                          # noqa: E402
import triton_ref                                  # noqa: E402
from _timing import burn_in                        # noqa: E402

BF16 = torch.bfloat16
ARMS = ("hk", "torch", "inductor", "triton", "flydsl")


def rel(got, want):
    got, want = got.float(), want.float()
    return ((got - want).abs().max() / want.abs().max()).item()


# ---------------------------------------------------------------- arms
#
# A case is (name, work, reference, {arm: (run, out) | "reason"}): `run`
# launches, `out()` returns what to check, `work` is bytes or FLOPs.


def gemm_case(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=BF16)
    b = torch.randn(n, k, device="cuda", dtype=BF16) / k ** 0.5
    ref = a.float() @ b.float().T
    outs = {x: torch.empty(m, n, device="cuda", dtype=BF16) for x in ARMS}
    arms = {
        "hk": (lambda: hk.ops.matmul(a, b, out=outs["hk"]), lambda: outs["hk"]),
        "torch": (lambda: torch.mm(a, b.T, out=outs["torch"]),
                  lambda: outs["torch"]),
        "inductor": "the same hipBLASLt call as torch (Inductor does not "
                    "template a bf16 GEMM here)",
        "triton": (lambda: triton_ref.matmul_abt(a, b, out=outs["triton"]),
                   lambda: outs["triton"]),
    }
    try:
        from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module
        launch, *_ = create_wmma_gemm_module(m, n, k, "bf16", "bf16")
        s = torch.cuda.current_stream()
        arms["flydsl"] = (lambda: launch(outs["flydsl"], a, b, s),
                          lambda: outs["flydsl"])
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    return f"gemm {m}x{n}x{k}", 2 * m * n * k, "flop", ref, arms


def attn_case(n, causal, h=32, hkv=8, d=128):
    q = torch.randn(1, h, n, d, device="cuda", dtype=BF16)
    k = torch.randn(1, hkv, n, d, device="cuda", dtype=BF16)
    v = torch.randn_like(k)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                         is_causal=causal, enable_gqa=True)
    out = {}
    arms = {
        "hk": (lambda: out.__setitem__("hk", hk.ops.attention(q, k, v,
                                                              causal=causal)),
               lambda: out["hk"]),
        "torch": (lambda: out.__setitem__("torch", F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=True)), lambda: out["torch"]),
        "inductor": "torch.compile lowers SDPA to the same aotriton call",
        "triton": "aotriton is Triton, compiled ahead of time (the torch "
                  "column); vLLM's own Triton attention is in paged_ab.py",
        "flydsl": "flash_attn_generic stages through raw_ptr_buffer_load_lds; "
                  "gfx11 has no global->LDS direct load (flydsl_probe.py)",
    }
    flops = 4 * h * n * n * d // (2 if causal else 1)
    return (f"attn n{n} {'causal' if causal else 'full'} gqa{h // hkv}",
            flops, "flop", ref, arms)


def _flydsl_compiled(build, *args):
    import flydsl.compiler as flyc
    return flyc.compile(build, *args)


def rmsnorm_case(m, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = torch.compile(lambda x, w: F.rms_norm(x, (n,), w, 1e-6))
    res = {}
    arms = {
        "hk": (lambda: hk.ops.rms_norm(x, w, out=outs["hk"]), lambda: outs["hk"]),
        "torch": (lambda: res.__setitem__("torch", F.rms_norm(x, (n,), w, 1e-6)),
                  lambda: res["torch"]),
        "inductor": (lambda: res.__setitem__("inductor", comp(x, w)),
                     lambda: res["inductor"]),
        "triton": (lambda: triton_ref.rms_norm(x, w, out=outs["triton"]),
                   lambda: outs["triton"]),
    }
    try:
        from kernels.norm.rmsnorm_kernel import build_rmsnorm_module
        s = torch.cuda.current_stream()
        f = _flydsl_compiled(build_rmsnorm_module(n, "bf16"), x, w,
                             outs["flydsl"], m, s)
        arms["flydsl"] = (lambda: f(x, w, outs["flydsl"], m, s),
                          lambda: outs["flydsl"])
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    return f"rms_norm {m}x{n}", 2 * m * n * 2 + 2 * n, "byte", ref, arms


def fused_add_case(m, n=4096):
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
        "hk": (lambda: hk.ops.fused_add_rms_norm(*bufs["hk"], w),
               lambda: bufs["hk"][0]),
        "torch": (eager, lambda: bufs["torch"][0]),
        "inductor": (inductor, lambda: got["inductor"]),
        "triton": (lambda: triton_ref.fused_add_rms_norm(*bufs["triton"], w),
                   lambda: bufs["triton"][0]),
    }
    try:
        from kernels.norm.rmsnorm_kernel import build_fused_add_rmsnorm_module
        s = torch.cuda.current_stream()
        xf, rf = bufs["flydsl"]
        y, ro = torch.empty_like(xf), torch.empty_like(rf)
        f = _flydsl_compiled(build_fused_add_rmsnorm_module(n, "bf16"), xf, rf,
                             w, y, ro, m, s)
        arms["flydsl"] = (lambda: f(xf, rf, w, y, ro, m, s), lambda: y)
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    return f"fused_add_rms_norm {m}x{n}", 2 * m * n * 4 + 2 * n, "byte", ref, arms


def softmax_case(m, n=4096):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    ref = torch.softmax(x.float(), dim=-1)
    outs = {a: torch.empty_like(x) for a in ARMS}
    comp = torch.compile(lambda x: torch.softmax(x, dim=-1))
    res = {}
    arms = {
        "hk": (lambda: hk.ops.softmax(x, out=outs["hk"]), lambda: outs["hk"]),
        "torch": (lambda: torch.softmax(x, dim=-1, out=outs["torch"]),
                  lambda: outs["torch"]),
        "inductor": (lambda: res.__setitem__("inductor", comp(x)),
                     lambda: res["inductor"]),
        "triton": (lambda: triton_ref.softmax(x, out=outs["triton"]),
                   lambda: outs["triton"]),
    }
    try:
        from kernels.norm.softmax_kernel import build_softmax_module
        f = build_softmax_module(m, n, "bf16")
        s = torch.cuda.current_stream()
        arms["flydsl"] = (lambda: f(x, outs["flydsl"], m, stream=s),
                          lambda: outs["flydsl"])
    except Exception as e:                         # noqa: BLE001
        arms["flydsl"] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    return f"softmax {m}x{n}", 2 * m * n * 2, "byte", ref, arms


CASES = {
    "gemm": lambda: [gemm_case(4096, 4096, 4096), gemm_case(2048, 6144, 4096),
                     gemm_case(2048, 4096, 4096), gemm_case(2048, 24576, 4096),
                     gemm_case(2048, 4096, 12288)],
    "attn": lambda: [attn_case(4096, True), attn_case(4096, False)],
    "norm": lambda: [rmsnorm_case(m) for m in (1, 32, 2048, 8192)]
                    + [fused_add_case(m) for m in (1, 32, 2048, 8192)],
    "softmax": lambda: [softmax_case(m) for m in (2048, 8192)],
}


# ---------------------------------------------------------------- timing

def _kernels(prof):
    return [e for e in prof.events() if str(e.device_type).endswith("CUDA")
            and not e.name.startswith(("Memcpy", "Memset"))]


def gpu_times(fns, rounds, calls):
    """{arm: mean per-call GPU ms}: the durations of the arm's own kernels,
    over interleaved rounds.

    Attributed by kernel name, not by profiler range. On ROCm the range ->
    kernel correlation of anything launched outside ATen (pybind, Triton,
    FlyDSL's module launch) is unreliable -- duplicate flow warnings, kernels
    credited to the wrong range, a GEMM at 347 TFLOP/s on a 123 TFLOP/s card
    -- while the kernel records themselves carry the hardware's timestamps.
    So each arm is profiled alone first to learn its kernels' names; the
    arms' kernels do not share names, and a name two arms share is refused.
    """
    names = {}
    for a, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        names[a] = {e.name for e in _kernels(p)}
    owner = {}
    for a, ns in names.items():
        for nm in ns:
            if nm in owner:
                raise RuntimeError(f"kernel {nm} is launched by both {owner[nm]} "
                                   f"and {a}; cannot attribute it")
            owner[nm] = a
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(rounds):
            for fn in fns.values():
                for _ in range(calls):
                    fn()
                torch.cuda.synchronize()
    total = {a: 0.0 for a in fns}
    for e in _kernels(prof):
        a = owner.get(e.name)
        if a is not None:
            total[a] += e.device_time_total
    return {a: (total[a] / 1e3 / (rounds * calls) if total[a] else float("inf"))
            for a in fns}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="gemm,attn,norm,softmax")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    print(f"{torch.cuda.get_device_name()} "
          f"{torch.cuda.get_device_properties(0).gcnArchName}, torch "
          f"{torch.__version__}, triton {__import__('triton').__version__}")
    burn_in()

    head = f"{'case':<30}" + "".join(f"{a:>11}" for a in ARMS) + "   hk vs best other"
    for group in args.only.split(","):
        print(f"\n[{group}] GPU ms per call, then rate (TFLOP/s or TB/s)")
        print(head)
        for name, work, unit, ref, arms in CASES[group]():
            fns, notes = {}, {}
            for a in ARMS:
                spec = arms.get(a, "no kernel")
                if isinstance(spec, str):
                    notes[a] = spec
                    continue
                run, out = spec
                try:
                    run()
                    torch.cuda.synchronize()
                    err = rel(out(), ref)
                    if err > 3e-2:
                        notes[a] = f"WRONG: rel err {err:.2e}"
                        continue
                    fns[a] = run
                except Exception as e:             # noqa: BLE001
                    notes[a] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
            t = gpu_times(fns, args.rounds, args.calls) if fns else {}
            cells = "".join(f"{t[a]:>11.4f}" if a in t else f"{'-':>11}"
                            for a in ARMS)
            others = [t[a] for a in t if a != "hk"]
            verdict = (f"   {min(others) / t['hk']:.2f}x" if "hk" in t and others
                       else "")
            print(f"{name:<30}{cells}{verdict}")
            scale = 1e12 if unit == "flop" else 1e12
            rates = "".join(f"{work / (t[a] * 1e-3) / scale:>11.2f}" if a in t
                            else f"{'':>11}" for a in ARMS)
            print(f"{'':<30}{rates}")
            for a, why in notes.items():
                print(f"{'':<32}{a}: {why}")
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
        ("rms_norm + fused add", norm._rms_weighted_kernel,
         py(triton_ref._rmsnorm), "kernels/norm/rmsnorm_kernel.py"),
        ("softmax", norm._row_reduce_kernel, py(triton_ref._softmax),
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
    print("hk's softmax row is the whole rms/layer/softmax kernel; FlyDSL's "
          "rows are whole files, helpers included.")


if __name__ == "__main__":
    raise SystemExit(main())
