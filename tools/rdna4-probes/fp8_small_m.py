"""scaled_mm's kernel at block_m 256 and 128 vs torch._scaled_mm across M (per-token x per-channel scales)."""
import sys, statistics
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
from hk.ops import gemm as G
def bench_all(fns, rounds=5, it=20):
    t = {a: [] for a in fns}; names = list(fns)
    for r in range(rounds):
        for a in names[r % len(names):] + names[:r % len(names)]:
            f = fns[a]; f(); f(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(it): f()
            e1.record(); torch.cuda.synchronize(); t[a].append(e0.elapsed_time(e1) / it * 1e3)
    return {a: statistics.median(v) for a, v in t.items()}
for n, k in ((4096, 4096), (14336, 4096), (4096, 14336)):
    b = (torch.randn(n, k, device="cuda") * .5).to(torch.float8_e4m3fn)
    sb = torch.rand(n, device="cuda") + .5
    for m in (1, 16, 64, 128, 256, 1024, 4096):
        a = (torch.randn(m, k, device="cuda") * .5).to(torch.float8_e4m3fn)
        sa = torch.rand(m, 1, device="cuda") + .5
        ref = (a.float() @ b.float().T) * sa * sb
        fns, outs = {}, {}
        for bm in (256, 128):
            kern, ep = G._scaled_kernel("fp8e4m3", False, bm)
            o = outs[bm] = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            from hk.templates.gemm import _preparer
            prep = dict(_preparer(ep, m))
            args = [prep["scale_a"](sa), prep["scale_b"](sb)]
            fns[f"hk{bm}"] = (lambda kern=kern, o=o, args=args: kern(a, b, o, *args))
            fns[f"hk{bm}"]()
        fns["torch"] = lambda: torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb.view(1, n), out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        errs = {bm: ((outs[bm].float() - ref).abs().max() / ref.abs().max()).item() for bm in outs}
        t = bench_all(fns)
        print(f"{m}x{n}x{k}: err {errs[256]:.1e}/{errs[128]:.1e}  hk256 {t['hk256']:.1f}  hk128 {t['hk128']:.1f}  torch {t['torch']:.1f} us  best-hk/torch {t['torch']/min(t['hk256'], t['hk128']):.2f}x", flush=True)
