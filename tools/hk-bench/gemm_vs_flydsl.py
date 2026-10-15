#!/usr/bin/env python3
"""Where FlyDSL's GEMM lead over hk comes from, and which hk tiles close it.

FlyDSL's dispatcher (`rdna3_f16_gemm_autotune.pick_tile` + `tile_opts`) runs a
128x128x32 tile of four waves -- a 64x64 warp tile -- with four options on
top: a padded LDS layout, `sched_group_barrier` hints, `stagger` (each
workgroup starts its K loop at a different k-tile) and, at 256+ tiles, its
"persistent" path, which is one tile a workgroup again but rotates the K start
with a wrap that works for any K-tile count. At 2048x24576 it picks 256x256x32
of sixteen waves with an unpadded k-major ("kblock") layout.

This builds FlyDSL's kernel with each option taken away in turn, beside hk's
default, hk's tuned record, hk with FlyDSL's stagger (`gemm_kernel(stagger=)`)
and the hk tiles that are expressible but not in hk's tuning space, and times
them all in one process the way scoreboard.py
does -- each arm in a profiler session of its own, rounds rotated, best round.
Every arm is checked against fp32 first.

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/gemm_vs_flydsl.py [--shapes 4096x4096x4096,...]
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

from _timing import burn_in                        # noqa: E402
from scoreboard import _why, gpu_times, rel        # noqa: E402

SHAPES = ["4096x4096x4096", "2048x6144x4096", "2048x4096x4096",
          "2048x24576x4096", "2048x4096x12288"]

#: hk schedules, as overrides of hk.ops.gemm.DEFAULT. The 256x256 tiles are
#: FlyDSL's large tile in hk's schedule: sixteen warps of 64x64, which fits
#: (252 VGPRs, occupancy 5) where the four-warp 128x128 of 64x64 spills.
HK = {
    "hk default": {},
    "hk +stagger": dict(stagger=1),
    "hk s2 p1": dict(n_split=2, write_pos=1),
    "hk s2 p1 +stagger": dict(n_split=2, write_pos=1, stagger=1),
    "hk 256x256x32 l1 g8": dict(block_m=256, block_n=256, k_step=32, warps=16,
                                warp_rows=4, lds_buffers=1, wgm=8),
    "hk 256x256x32 l2 g8": dict(block_m=256, block_n=256, k_step=32, warps=16,
                                warp_rows=4, lds_buffers=2, wgm=8),
    "hk 256x256x32 l2 g16": dict(block_m=256, block_n=256, k_step=32, warps=16,
                                 warp_rows=4, lds_buffers=2, wgm=16),
    "hk 256x256x32 l2 g16 +stagger": dict(block_m=256, block_n=256, k_step=32, warps=16,
                                          warp_rows=4, lds_buffers=2, wgm=16, stagger=1),
}

T128 = (4, 4, 2, 2, 2)
T256 = (4, 4, 2, 4, 4)

#: FlyDSL builds: (tile, options). `persist` resolves to the dispatcher's own
#: rule (persistent_wgs) when True.
FLY = {
    "fly 128 dispatch opts": (T128, dict(lds_layout="pad", sched_hint=True, group_m=8,
                                         stagger=1, persist=True)),
    "fly 128 -sched": (T128, dict(lds_layout="pad", sched_hint=False, group_m=8,
                                  stagger=1, persist=True)),
    "fly 128 -stagger": (T128, dict(lds_layout="pad", sched_hint=True, group_m=8,
                                    stagger=0, persist=False)),
    "fly 128 -persist": (T128, dict(lds_layout="pad", sched_hint=True, group_m=8,
                                    stagger=1, persist=False)),
    "fly 128 bare": (T128, dict(lds_layout="pad", sched_hint=False, group_m=8,
                                stagger=0, persist=False)),
    "fly 256 dispatch opts": (T256, dict(lds_layout="kblock", sched_hint=False,
                                         group_m=16, stagger=1, persist=False)),
    "fly 256 -stagger": (T256, dict(lds_layout="kblock", sched_hint=False,
                                    group_m=16, stagger=0, persist=False)),
}


def hk_arms(a, b, out):
    from hk.ops import gemm as g
    m, k = a.shape
    n = b.shape[0]
    arms = {}
    for name, over in HK.items():
        s = {**g.DEFAULT, **over}
        if n % s["block_n"] or k % s["k_step"]:
            continue
        try:
            kern = g.TUNER._kernel_for(s)
            arms[name] = (lambda kern=kern, c=out[name]: kern(a, b, c), out[name])
        except Exception as e:                     # noqa: BLE001
            arms[name] = _why(e)
    return arms


def fly_arms(a, b, out):
    from kernels.gemm import rdna3_f16_gemm_autotune as fg
    from kernels.gemm.rdna3_f16_gemm import create_wmma_gemm_module
    m, k = a.shape
    n = b.shape[0]
    s = torch.cuda.current_stream()
    arms = {"fly dispatcher": (lambda c=out["fly dispatcher"]: fg.rdna3_gemm_dispatch(
        c, a, b, m, n, k), out["fly dispatcher"])}
    feasible = dict(fg.feasible_tiles(m, n, k))
    for name, (tile, o) in FLY.items():
        if tile not in feasible:
            continue
        o = dict(o)
        persist = fg.persistent_wgs(m, n, k, tile) if o.pop("persist") else 0
        rm, rn, rk, wm, wn = tile
        try:
            launch, *_ = create_wmma_gemm_module(
                m, n, k, in_dtype="bf16", out_dtype="bf16", reg_m=rm, reg_n=rn, reg_k=rk,
                waves_m=wm, waves_n=wn, persistent_wgs=persist, **o)
            arms[name] = (lambda launch=launch, c=out[name]: launch(c, a, b, s, 0), out[name])
        except Exception as e:                     # noqa: BLE001
            arms[name] = _why(e)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names to keep (default: all)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--calls", type=int, default=10)
    args = ap.parse_args()
    keep = set(args.arms.split(",")) if args.arms else None
    if keep:
        for d in (HK, FLY):
            for name in [n for n in d if n not in keep]:
                del d[name]
    burn_in()
    names = list(HK) + ["fly dispatcher"] + list(FLY)
    for shape in args.shapes.split(","):
        m, n, k = (int(x) for x in shape.split("x"))
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        ref = a.float() @ b.float().T
        out = {x: torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for x in names}
        arms = {**hk_arms(a, b, out), **fly_arms(a, b, out)}
        fns, notes = {}, {}
        for name, spec in arms.items():
            if isinstance(spec, str):
                notes[name] = spec
                continue
            run, o = spec
            try:
                run()
                torch.cuda.synchronize()
                err = rel(o, ref)
                if err > 3e-2:
                    notes[name] = f"WRONG: rel err {err:.2e}"
                    continue
                fns[name] = run
            except Exception as e:                 # noqa: BLE001
                notes[name] = _why(e)
        t = gpu_times(fns, args.rounds, args.calls)
        ctrl = t["hk default"][0]
        print(f"\n{shape}   ms, TFLOP/s, speed vs hk default, noise")
        for name in sorted(t, key=lambda x: t[x][0]):
            ms, sp = t[name]
            print(f"  {name:<24}{ms:>9.4f}{2 * m * n * k / ms / 1e9:>9.2f}"
                  f"{ctrl / ms:>9.3f}x{sp * 100:>7.2f}%")
        for name, why in notes.items():
            print(f"  {name:<24}{why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
