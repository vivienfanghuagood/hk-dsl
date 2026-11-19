"""The fp8 dgrad GEMM on gfx1201: dX = dY @ W as a training backward runs it, e5m2 gradient x e4m3 weight,
per-tensor scales, bf16 out -- hk.ops.scaled_mm (mixed pair) and its schedules against torch._scaled_mm
(hipBLASLt), stock and TunableOp-tuned, one process, interleaved.

    HK_ARCH=gfx1201 python tools/rdna4-probes/fp8_dgrad.py [--sweep]

Shapes: a 1.28B Llama-style model (hidden 2048, SwiGLU FFN 8192, so fc1 is 2048 -> 16384) at 4096 and 8192
tokens. dgrad's (M, N, K) is (tokens, in, out) of each linear: qkv (2048, 6144), o (2048, 2048), fc1
(2048, 16384), fc2 (8192, 8192). Operands are TN, as Transformer Engine hands them over (its fp8 weight
transpose is the (N, K) B). Each arm is checked against fp32 on the dequantized operands first.
"""
import argparse
import itertools
import statistics

import torch

import hk
from hk.ops import gemm as G

SHAPES = [(t, n, k) for t in (4096, 8192) for n, k in ((2048, 6144), (2048, 2048), (2048, 16384),
                                                       (8192, 8192))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also time a grid of hk schedules")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(0)
    pts = {}
    if args.sweep:
        for bm, bn, ks, ns in itertools.product((128, 256), (128, 256), (64, 128), (2, 4)):
            if bm == 256 and bn == 256:
                continue
            p = dict(G.FP8_DEFAULT, block_m=bm, block_n=bn, k_step=ks, n_split=ns)
            nm = f"m{bm}_n{bn}_k{ks}_s{ns}"
            try:
                pts[nm] = G.gemm_kernel("dgsw_" + nm, op_dtype=hk.fp8e5m2, b_dtype=hk.fp8e4m3, **p)
            except Exception as e:  # noqa: BLE001
                print(nm, type(e).__name__, str(e).splitlines()[0][:120] if str(e) else "")
    tun = torch.cuda.tunable
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
    for m, n, k in SHAPES:
        dy = (torch.randn(m, k, device="cuda") * 0.5).to(torch.float8_e5m2)
        wt = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
        sa = torch.tensor(0.75, device="cuda")
        sb = torch.tensor(1.25, device="cuda")
        ref = (dy.float() @ wt.float().T) * (0.75 * 1.25)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        def blas():
            return torch._scaled_mm(dy, wt.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

        fns = {"hk": lambda: G.scaled_mm(dy, wt, sa, sb, out=out)}
        for nm, kern in pts.items():
            fns[nm] = lambda kern=kern, c=torch.empty_like(out): (kern(dy, wt, c), c)[1]
        for nm, f in list(fns.items()):
            got = f().float() * (1.0 if nm == "hk" else 0.75 * 1.25)  # a raw kernel is unscaled
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            assert err < 1e-2, f"{m}x{n}x{k} {nm}: max error {err:.2e}"
        tun.enable(False)
        err = ((blas().float() - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, f"hipBLASLt {err}"
        # TunableOp: tune this shape online, then time the tuned arm with the lookup on
        tun.enable(True)
        tun.tuning_enable(True)
        blas()
        torch.cuda.synchronize()
        tun.tuning_enable(False)
        fns["hipblaslt"] = lambda: (tun.enable(False), blas())
        fns["hipblaslt_tuned"] = lambda: (tun.enable(True), blas())
        times = {x: [] for x in fns}
        names = list(fns)
        for r in range(args.rounds):
            for x in names[r % len(names):] + names[:r % len(names)]:
                f = fns[x]
                f()
                f()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(20):
                    f()
                e1.record()
                torch.cuda.synchronize()
                times[x].append(e0.elapsed_time(e1) / 20)
        tun.enable(False)
        fl = 2 * m * n * k
        med = {x: statistics.median(t) for x, t in times.items()}
        best = min(med["hipblaslt"], med["hipblaslt_tuned"])
        print(f"== M{m} N{n} K{k}: hipBLASLt {med['hipblaslt']:.4f} ms ({fl / med['hipblaslt'] / 1e9:.0f} TF), "
              f"tuned {med['hipblaslt_tuned']:.4f} ({fl / med['hipblaslt_tuned'] / 1e9:.0f}), "
              f"hk {med['hk']:.4f} ({fl / med['hk'] / 1e9:.0f}) -> {best / med['hk']:.2f}x of the better", flush=True)
        for x, ms in sorted(med.items(), key=lambda t: t[1])[:6]:
            if x not in ("hk", "hipblaslt", "hipblaslt_tuned"):
                print(f"   {ms:.4f} ms {fl / ms / 1e9:5.0f} TF  {x}")


if __name__ == "__main__":
    main()
