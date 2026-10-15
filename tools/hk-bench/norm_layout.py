#!/usr/bin/env python3
"""What hk's register tiles cost a memory-bound row kernel, and whether
reading and writing each row once instead of twice gets it back.

A 16-bit row tile holds each row twice, in lanes l and l + 16, and the packed
fp32 -> 16-bit store completes the same 32-byte run in both lanes; so the
bounded loads and stores every hk norm uses move each run twice. HipKittens'
HK_RT_HALF_SPLIT (global_to_register.cuh) has each half read its own 16
bytes and take the other 16 from its partner over v_permlanex16 (bit 1), and
write only its own 16 (bit 2).

Two sections, one process each, every arm checked before it is timed:

  copy    a 16x64 tile per warp through a bf16 and an fp32 register tile, on
          the 2D tensor and on the folded view the small-M norms use, against
          torch's elementwise multiply by one as the ceiling;
  norms   hk.ops.rms_norm and fused_add_rms_norm's own kernels and plans,
          built with each HK_RT_HALF_SPLIT, against Triton and FlyDSL;
  softmax the same for hk.ops.softmax.

    FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 \\
        python3 tools/hk-bench/norm_layout.py [--only copy,norms,softmax]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))

import hk  # noqa: E402
from hk.ops import norm  # noqa: E402

WARPS = 4
SPLITS = (0, 1, 2, 3)


def _grid(p):
    return (hk.cdiv(p.x.cols, 64 * WARPS), hk.cdiv(p.x.rows, 16), 1)


def copy_kernel(dtype, split):
    t = hk.RT(dtype, 16, 64)
    name = f"probe_copy_{dtype.name}_s{split}"

    def body(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16,
                            (hk.arch.block_idx.x * WARPS + hk.arch.warp_id()) * 64)
        o[idx].store(x[idx].load(t, bounded=True), bounded=True)
    body.__name__ = name
    return hk.kernel(body, warps=WARPS, grid=_grid, name=name,
                     defines={"HK_RT_HALF_SPLIT": split} if split else None)


COPIES = {f"{d.name} tile s{s}": copy_kernel(d, s) for d in (hk.bf16, hk.fp32) for s in SPLITS}


def norm_kernel(kind, warps, split):
    """norm's own weighted kernel, built with HK_RT_HALF_SPLIT=split."""
    return norm._rms_weighted_kernel(f"{kind}_bf16_w{warps}_s{split}", hk.bf16,
                                     warps, kind == "addrmsw", split=split)


def section_copy(torch, gpu_times):
    for m in (2048, 8192):
        x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
        views = {"2d": (x, (m, 4096)), "folded": (x, (m * 16, 256))}
        fns, outs = {}, {}
        for vname, (t, shape) in views.items():
            xv = t.view(shape)
            for k, kern in COPIES.items():
                name = f"{vname} {k}"
                outs[name] = torch.empty_like(xv)
                fns[name] = (lambda kern=kern, xv=xv, o=outs[name]: kern(xv, o))
        outs["torch mul 1"] = torch.empty_like(x)
        fns["torch mul 1"] = lambda: torch.mul(x, 1.0, out=outs["torch mul 1"])
        for k, f in fns.items():
            f()
            torch.cuda.synchronize()
            assert torch.equal(outs[k].view(m, 4096), x), k
        t = gpu_times(fns, rounds=5, calls=20)
        mb = 2 * m * 4096 * 2
        print(f"\ncopy {m}x4096 bf16, {mb / 1e6:.0f} MB moved   us, TB/s, noise")
        for k in sorted(t, key=lambda k: t[k][0]):
            ms, noise = t[k]
            print(f"  {k:<22}{ms * 1e3:9.2f}{mb / ms / 1e9:8.2f}  {noise * 100:4.1f}%")


#: softmax arms: (bounded accesses, HK_RT_HALF_SPLIT). The shipped kernel is
#: unbounded, which the split does not reach.
SOFTMAX = {"hk shipped": (False, 0), "hk bounded": (True, 0),
           "hk bounded s2": (True, 2), "hk bounded s3": (True, 3)}


def softmax_kernel(warps, bounded, split):
    return norm._row_reduce_kernel(f"softmax_bf16_w{warps}_b{int(bounded)}s{split}",
                                   hk.bf16, "softmax", warps, bounded=bounded, split=split)


def section_softmax(torch, gpu_times):
    import triton_ref  # noqa: PLC0415
    for rows in (2048, 8192):
        cols = 4096
        shipped, tpw, fold, exact = norm._resolve("softmax", torch.bfloat16, cols, rows)
        warps = int(shipped.name.rsplit("_w", 1)[1])
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        shape = (rows * fold, cols // fold)
        fns, outs = {}, {}
        for name, (bounded, s) in SOFTMAX.items():
            k = softmax_kernel(warps, bounded, s)
            o = outs[name] = torch.empty_like(x)
            fns[name] = (lambda k=k, o=o: k(x.view(shape), o.view(shape), TPW=tpw,
                                              FOLD=fold, EXACT=exact))
        outs["triton tuned"] = torch.empty_like(x)
        fns["triton tuned"] = lambda: triton_ref.softmax_tuned(x, out=outs["triton tuned"])
        try:
            from kernels.norm import softmax_autotune as fs  # noqa: PLC0415
            os.environ["FLYDSL_AUTOTUNE"] = "1"
            outs["flydsl tuned"] = torch.empty_like(x)
            fs.softmax_autotuned(x, outs["flydsl tuned"])
            os.environ.pop("FLYDSL_AUTOTUNE")
            fns["flydsl tuned"] = lambda: fs.softmax_autotuned(x, outs["flydsl tuned"])
        except Exception as e:                     # noqa: BLE001
            print(f"  flydsl: {type(e).__name__}: {e}")
        for name, f in fns.items():
            f()
            torch.cuda.synchronize()
            if name.startswith("hk ") and name != "hk shipped":
                assert torch.equal(outs[name], outs["hk shipped"]), f"softmax {rows}: {name}"
        t = gpu_times(fns, rounds=7, calls=20)
        base = t["hk shipped"][0]
        print(f"\nsoftmax {rows}x{cols}  plan w{warps} tpw{tpw} fold{fold}   us, vs shipped, noise")
        for k in sorted(t, key=lambda k: t[k][0]):
            ms, noise = t[k]
            print(f"  {k:<16}{ms * 1e3:9.2f}{base / ms:8.3f}x  {noise * 100:4.1f}%")


def section_norms(torch, gpu_times):
    import triton_ref  # noqa: PLC0415
    eps = 1e-6
    for kind, rows in (("rmsw", 1), ("rmsw", 32), ("rmsw", 2048), ("rmsw", 8192),
                       ("addrmsw", 1), ("addrmsw", 2048), ("addrmsw", 8192)):
        cols = 4096
        shipped, tpw, fold, exact = norm._resolve_weighted(kind, torch.bfloat16, cols, rows)
        warps = int(shipped.name.rsplit("_w", 1)[1])
        x0 = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
        r0 = torch.randn_like(x0)
        w = torch.rand(cols, device="cuda", dtype=torch.bfloat16) + 0.5
        shape = (rows * fold, cols // fold)
        wv = w.view(fold, cols // fold)
        fns, outs = {}, {}
        for s in SPLITS:
            k = norm_kernel(kind, warps, s)
            name = f"hk s{s}"
            if kind == "rmsw":
                o = torch.empty_like(x0)
                outs[name] = o
                fns[name] = (lambda k=k, o=o: k(x0.view(shape), wv, o.view(shape), TPW=tpw,
                                                  FOLD=fold, EXACT=exact, EPS=eps))
            else:
                xb, rb = x0.clone(), r0.clone()
                outs[name] = xb
                fns[name] = (lambda k=k, xb=xb, rb=rb: k(
                    xb.view(shape), rb.view(shape), wv, xb.view(shape), rb.view(shape),
                    TPW=tpw, FOLD=fold, EXACT=exact, EPS=eps))
        if kind == "rmsw":
            outs["triton tuned"] = torch.empty_like(x0)
            fns["triton tuned"] = lambda: triton_ref.rms_norm_tuned(x0, w, out=outs["triton tuned"])
        else:
            xt, rt = x0.clone(), r0.clone()
            outs["triton tuned"] = xt
            fns["triton tuned"] = lambda: triton_ref.fused_add_rms_norm_tuned(xt, rt, w)
        try:
            if kind == "rmsw":
                from kernels.norm import rmsnorm_autotune as fr  # noqa: PLC0415
                os.environ["FLYDSL_AUTOTUNE"] = "1"
                outs["flydsl tuned"] = torch.empty_like(x0)
                fr.rmsnorm_autotuned(x0, w, outs["flydsl tuned"], rows)
                os.environ.pop("FLYDSL_AUTOTUNE")
                fns["flydsl tuned"] = lambda: fr.rmsnorm_autotuned(x0, w, outs["flydsl tuned"], rows)
        except Exception as e:                     # noqa: BLE001
            print(f"  flydsl: {type(e).__name__}: {e}")
        # The check: one call each from fresh inputs, every split against s0
        # bit for bit -- the split moves bytes, it computes nothing.
        ref = None
        for name, f in fns.items():
            if kind == "addrmsw":
                outs[name].copy_(x0)
            f()
            torch.cuda.synchronize()
            if name == "hk s0":
                ref = outs[name].clone()
            elif name.startswith("hk "):
                assert torch.equal(outs[name], ref), f"{kind} {rows}: {name} differs from s0"
        t = gpu_times(fns, rounds=7, calls=20)
        base = t["hk s0"][0]
        label = "rms_norm" if kind == "rmsw" else "fused_add_rms_norm"
        print(f"\n{label} {rows}x{cols}  plan w{warps} tpw{tpw} fold{fold}   us, vs s0, noise")
        for k in sorted(t, key=lambda k: t[k][0]):
            ms, noise = t[k]
            print(f"  {k:<16}{ms * 1e3:9.2f}{base / ms:8.3f}x  {noise * 100:4.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="copy,norms,softmax")
    args = ap.parse_args()
    import torch  # noqa: PLC0415

    from _timing import burn_in  # noqa: PLC0415
    from scoreboard import gpu_times  # noqa: PLC0415

    burn_in()
    if "copy" in args.only:
        section_copy(torch, gpu_times)
    if "norms" in args.only:
        section_norms(torch, gpu_times)
    if "softmax" in args.only:
        section_softmax(torch, gpu_times)


if __name__ == "__main__":
    main()
