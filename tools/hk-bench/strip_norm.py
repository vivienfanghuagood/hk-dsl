#!/usr/bin/env python3
"""Row kernels on the strip layout against the shipped hk kernels, tuned
Triton and FlyDSL, at the scoreboard's shapes -- one process, the
scoreboard's timing (a profiler session per arm, rounds interleaved).

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/strip_norm.py [--ops rms,add,softmax] [--rows 1,32,2048,8192]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))

import torch                                       # noqa: E402
import torch.nn.functional as F                    # noqa: E402

import hk                                          # noqa: E402
import triton_ref                                  # noqa: E402
from hk.dsl.collective import cross_warp          # noqa: E402
from _timing import burn_in                        # noqa: E402
from scoreboard import gpu_times, rel              # noqa: E402

BF16 = torch.bfloat16


def strip_rms(n: int, warps: int, residual: bool = False, defines=None, tag=""):
    """One workgroup a row, each warp n/warps contiguous columns."""
    cw = n // warps
    S = hk.RS(hk.fp32, 1, cw)

    def core(x, w, o, r, r_out, EPS):
        row = hk.arch.block_idx.x
        c0 = hk.arch.warp_id() * cw if warps > 1 else 0
        g = w.at[0, c0].load(hk.row_vec(S))
        v = x.at[row, c0].load(S)
        if r is not None:
            v += r.at[row, c0].load(S)
            r_out.at[row, c0].store(v)
        sq = cross_warp((v * v).sum(axis=1), "add", warps)
        sq /= x.cols
        sq += EPS
        v *= sq.rsqrt()
        v *= g
        o.at[row, c0].store(v)

    if residual:
        def body(x: hk.Tensor[hk.bf16], r: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16],
                 o: hk.Tensor[hk.bf16], r_out: hk.Tensor[hk.bf16], *, EPS=1e-6):
            core(x, w, o, r, r_out, EPS)
    else:
        def body(x: hk.Tensor[hk.bf16], w: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
                 *, EPS=1e-6):
            core(x, w, o, None, None, EPS)
    name = f"strip_{'add' if residual else ''}rms_w{warps}{tag}"
    body.__name__ = name
    defines = dict(defines or {})
    if residual:
        defines["HK_BF16_ROUND"] = 2
    return hk.kernel(body, warps=warps, grid=lambda p: (p.x.rows, 1, 1), name=name,
                     defines=defines or None)


def strip_softmax(n: int, warps: int, defines=None, tag=""):
    cw = n // warps
    S = hk.RS(hk.fp32, 1, cw)

    def body(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        row = hk.arch.block_idx.x
        c0 = hk.arch.warp_id() * cw if warps > 1 else 0
        v = x.at[row, c0].load(S)
        m = cross_warp(v.max(axis=1), "max", warps)
        v -= m
        v.exp_()
        s = cross_warp(v.sum(axis=1), "add", warps)
        inv = hk.full(hk.col_vec(S), 1.0)
        inv /= s
        v *= inv
        o.at[row, c0].store(v)
    name = f"strip_softmax_w{warps}{tag}"
    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=lambda p: (p.x.rows, 1, 1), name=name,
                     defines=defines)


def rms_case(m, n, warps):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    outs = {}

    def arm(name, fn):
        outs[name] = torch.empty_like(x)
        return name, (lambda: fn(outs[name])), (lambda: outs[name])

    arms = [arm("hk", lambda o: hk.ops.rms_norm(x, w, out=o)),
            arm("triton_tuned", lambda o: triton_ref.rms_norm_tuned(x, w, out=o))]
    for wp in warps:
        k = strip_rms(n, wp)
        arms.append(arm(f"strip w{wp}", lambda o, k=k: k(x, w.view(1, n), o)))
    try:
        from kernels.norm import rmsnorm_autotune as fr
        arms.append(arm("flydsl_tuned", lambda o: fr.rmsnorm_autotuned(x, w, o, m)))
    except Exception as e:                         # noqa: BLE001
        print("flydsl_tuned:", e)
    return f"rms_norm {m}x{n}", ref, arms


def add_case(m, n, warps):
    x0 = torch.randn(m, n, device="cuda", dtype=BF16)
    r0 = torch.randn_like(x0)
    w = torch.rand(n, device="cuda", dtype=BF16) + 0.5
    z = x0.float() + r0.float()
    ref = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    bufs = {}

    def arm(name, fn):
        bufs[name] = (x0.clone(), r0.clone())
        return name, (lambda: fn(*bufs[name])), (lambda: bufs[name][0])

    arms = [arm("hk", lambda x, r: hk.ops.fused_add_rms_norm(x, r, w)),
            arm("triton_tuned", lambda x, r: triton_ref.fused_add_rms_norm_tuned(x, r, w))]
    for wp in warps:
        k = strip_rms(n, wp, residual=True)
        arms.append(arm(f"strip w{wp}", lambda x, r, k=k: k(x, r, w.view(1, n), x, r)))
    return f"fused_add {m}x{n}", ref, arms


def softmax_case(m, n, warps):
    x = torch.randn(m, n, device="cuda", dtype=BF16)
    ref = torch.softmax(x.float(), dim=-1)
    outs = {}

    def arm(name, fn):
        outs[name] = torch.empty_like(x)
        return name, (lambda: fn(outs[name])), (lambda: outs[name])

    arms = [arm("hk", lambda o: hk.ops.softmax(x, out=o)),
            arm("triton_tuned", lambda o: triton_ref.softmax_tuned(x, out=o))]
    for wp in warps:
        k = strip_softmax(n, wp)
        arms.append(arm(f"strip w{wp}", lambda o, k=k: k(x, o)))
    try:
        from kernels.norm import softmax_autotune as fs
        arms.append(arm("flydsl_tuned", lambda o: fs.softmax_autotuned(x, o)))
    except Exception as e:                         # noqa: BLE001
        print("flydsl_tuned:", e)
    return f"softmax {m}x{n}", ref, arms


CASES = {"rms": rms_case, "add": add_case, "softmax": softmax_case}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ops", default="rms,add,softmax")
    ap.add_argument("--rows", default="1,32,2048,8192")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--warps", default="4,8,16")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    warps = [int(w) for w in args.warps.split(",")]
    burn_in()
    for op in args.ops.split(","):
        for m in (int(r) for r in args.rows.split(",")):
            name, ref, arms = CASES[op](m, args.n, warps)
            fns = {}
            for a, run, out in arms:
                try:
                    run()
                    torch.cuda.synchronize()
                    err = rel(out(), ref)
                except Exception as e:             # noqa: BLE001
                    print(f"  {a}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                if err > 2e-2:
                    print(f"  {a}: WRONG, rel err {err:.3g}")
                    continue
                fns[a] = run
            calls = args.calls if m >= 2048 else 50
            rounds = args.rounds if m >= 2048 else 9
            t = gpu_times(fns, rounds, calls)
            best = min(v[0] for k, v in t.items() if not k.startswith(("hk", "strip")))
            print(f"{name}")
            for a, (ms, sp) in sorted(t.items(), key=lambda kv: kv[1][0]):
                print(f"  {a:16s} {ms * 1e3:9.2f} us  noise {sp * 100:4.1f}%  "
                      f"best-other/this {best / ms:5.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
