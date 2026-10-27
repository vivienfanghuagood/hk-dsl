"""fp8 decode GEMM on gfx1201: hk.ops.decode_gemm's kernel with fp8 operands (no epilogue: raw x @ w.T),
schedules swept, against scaled_mm's 128-row tile and FlyDSL's rdna_fp8_preshuffle_gemm (M=32 only), one process."""
import sys, os, statistics, itertools
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))
import hk
from hk.ops import decode_gemm as D, gemm as G
from hk.types import fp8e4m3

def bench_all(fns, rounds=5, it=50):
    t = {a: [] for a in fns}; names = list(fns)
    for r in range(rounds):
        for a in names[r % len(names):] + names[:r % len(names)]:
            f = fns[a]; f(); f(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(it): f()
            e1.record(); torch.cuda.synchronize(); t[a].append(e0.elapsed_time(e1) / it * 1e3)
    return {a: statistics.median(v) for a, v in t.items()}

KS = {}
def kern(tokens, slices, warps, resident, splits=1):
    key = (tokens, slices, warps, resident, splits)
    if key not in KS:
        KS[key] = D.decode_gemm_kernel(f"dec8_t{tokens}_s{slices}_w{warps}_r{resident}_k{splits}", tokens=tokens,
                                       slices=slices, warps=warps, resident=resident, splits=splits, op_dtype=fp8e4m3)
    return KS[key]

for n, k in ((4096, 4096), (14336, 4096), (4096, 14336)):
    w = (torch.randn(n, k, device="cuda") * .5).to(torch.float8_e4m3fn)
    for m in (1, 16, 32, 64):
        x = (torch.randn(m, k, device="cuda") * .5).to(torch.float8_e4m3fn)
        ref = x.float() @ w.float().T
        fns = {}
        tokens = D.tokens_for(m)
        for slices, warps, resident in itertools.product((2, 4, 8), (8, 16), (0, 2)):
            if k % (16 * slices * warps):
                continue
            name = f"s{slices}w{warps}r{resident}"
            try:
                kk = kern(tokens, slices, warps, resident)
                o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                kk(x, w, o); torch.cuda.synchronize()
                err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
                if err > 1e-2:
                    print(m, n, k, name, "WRONG", err); continue
                fns[name] = (lambda kk=kk, o=o: kk(x, w, o))
            except Exception as e:
                print(m, n, k, name, type(e).__name__, str(e).splitlines()[0][:140] if str(e) else "")
        one = torch.ones(m, 1, device="cuda"); oneb = torch.ones(n, device="cuda")
        fns["scaled_mm"] = lambda: G.scaled_mm(x, w, one, oneb)
        if m % 32 == 0:
            from kernels.gemm.rdna_fp8_preshuffle_gemm import compile_fp8_gemm, preshuffle_b_fp8
            launch = compile_fp8_gemm(M=m, N=n, K=k)
            bs = preshuffle_b_fp8(w.T.contiguous()).view(torch.float32).contiguous()
            of = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            fns["flydsl"] = lambda: launch(of, x.view(torch.float32), bs, one.view(m), oneb, torch.cuda.current_stream())
        t = bench_all(fns)
        best = sorted(t.items(), key=lambda kv: kv[1])
        line = "  ".join(f"{a}={v:.1f}" for a, v in best[:5])
        print(f"{m}x{n}x{k}: {line}  | scaled_mm={t['scaled_mm']:.1f}" + (f" flydsl={t['flydsl']:.1f}" if 'flydsl' in t else "")
              + f"  (W read {n*k/1e6:.0f} MB -> {n*k/best[0][1]/1e3:.0f} GB/s)", flush=True)
