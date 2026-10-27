"""fp8 decode on gfx1201 with W streamed from GDDR6, as a model's decode step reads it: 6 copies of W rotated
(past the 64 MB Infinity Cache), profiler kernel time. hk scaled_mm decode schedules vs FlyDSL's fp8 GEMM."""
import sys, os, itertools
import torch
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python")); sys.path.insert(0, str(_ROOT / "tools/hk-bench"))
sys.path.append(os.environ.get("FLYDSL_ROOT", "/workspace/FlyDSL"))
import scoreboard as sb, _rounds
from hk.ops import gemm as G, decode_gemm as D
from hk.templates.gemm import _preparer
_rounds.use("gfx1201")
COPIES = 6
for m, n, k in ((32, 14336, 4096), (32, 4096, 4096), (16, 14336, 4096), (32, 4096, 14336), (1, 14336, 4096)):
    a = (torch.randn(m, k, device="cuda") * .5).to(torch.float8_e4m3fn)
    ws = [(torch.randn(n, k, device="cuda") * .5).to(torch.float8_e4m3fn) for _ in range(COPIES)]
    sa = torch.rand(m, 1, device="cuda") + .5; sbv = torch.rand(n, device="cuda") + .5
    fns = {}
    tokens = D.tokens_for(m)
    for slices, warps in ((8, 8), (4, 8), (4, 4), (8, 4), (4, 16)):
        if k % (16 * slices * warps):
            continue
        try:
            ep = G._scale_epilogue(False)
            kern = D.decode_gemm_kernel(f"dec8sw_t{tokens}_s{slices}w{warps}k1", tokens=tokens, op_dtype=G.DTYPES["fp8e4m3"],
                                        epilogue=ep, slices=slices, warps=warps, combine_depth=0, resident=0, splits=1)
            prep = dict(_preparer(ep, m)); args = [prep["scale_a"](sa), prep["scale_b"](sbv)]
            o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            st = {"i": 0}
            def f(kern=kern, o=o, args=args, st=st):
                st["i"] = (st["i"] + 1) % COPIES
                kern(a, ws[st["i"]], o, *args)
            f(); torch.cuda.synchronize()
            fns[f"hk s{slices}w{warps}"] = f
        except Exception as e:
            print(m, n, k, slices, warps, type(e).__name__)
    if m % 32 == 0:
        from kernels.gemm.rdna_fp8_preshuffle_gemm import compile_fp8_gemm, preshuffle_b_fp8
        launch = compile_fp8_gemm(M=m, N=n, K=k)
        bss = [preshuffle_b_fp8(w.T.contiguous()).view(torch.float32).contiguous() for w in ws]
        of = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        sa1 = sa.view(m).contiguous(); s = torch.cuda.current_stream(); st2 = {"i": 0}
        def ff():
            st2["i"] = (st2["i"] + 1) % COPIES
            launch(of, a.view(torch.float32), bss[st2["i"]], sa1, sbv, s)
        fns["flydsl"] = ff
    one = torch.ones(1, n, device="cuda"); st3 = {"i": 0}
    def ft():
        st3["i"] = (st3["i"] + 1) % COPIES
        torch._scaled_mm(a, ws[st3["i"]].T, scale_a=sa, scale_b=sbv.view(1, n), out_dtype=torch.bfloat16)
    if m > 16 or True:
        fns["hipblaslt"] = ft
    raw = sb.gpu_rounds(fns, 5, 12)
    t = {}
    for x, (ts, recs) in raw.items():
        kept, _ = _rounds.screen(ts, recs)
        ms, _, _ = _rounds.summarize(ts, kept, "median")
        if ms: t[x] = ms * 1e3
    best = sorted(t.items(), key=lambda kv: kv[1])
    print(f"{m}x{n}x{k} (W {n*k/2**20:.0f} MiB x{COPIES}): " + "  ".join(f"{x}={v:.1f}" for x, v in best)
          + f"   | best {n*k/best[0][1]/1e3:.0f} GB/s", flush=True)
