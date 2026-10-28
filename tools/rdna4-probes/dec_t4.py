"""decode_gemm at 33-64 rows (tokens=4) on gfx1201: which schedules build without spilling under this image's
compiler, and their time (bf16, W rotated past the cache), one process."""
import sys, itertools
import torch
_ROOT = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT + "/python"); sys.path.insert(0, _ROOT + "/tools/hk-bench")
import scoreboard as sb, _rounds
from hk.ops import decode_gemm as D
_rounds.use("gfx1201")
for n, k in ((6144, 4096), (4096, 12288)):
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** .5 for _ in range(3)]
    for m in (33, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().T
        fns = {}
        for slices, warps, cd in itertools.product((1, 2), (4, 8), (0, 2, 4)):
            s = dict(tokens=4, slices=slices, warps=warps, combine_depth=cd, resident=0, splits=1)
            if k % (16 * slices * warps):
                continue
            try:
                kern = D.kernel_for(**s)
                o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                kern(x, ws[0], o); torch.cuda.synchronize()
                if ((o.float() - ref).abs().max() / ref.abs().max()).item() > 2e-2:
                    print("WRONG", s); continue
                i = [0]
                def f(kern=kern, o=o, i=i):
                    i[0] = (i[0] + 1) % 3; kern(x, ws[i[0]], o)
                fns[f"s{slices}w{warps}c{cd}"] = f
            except Exception as e:
                print(m, n, k, s, type(e).__name__, (str(e).splitlines() or [""])[1][:120] if len(str(e).splitlines()) > 1 else "")
        raw = sb.gpu_rounds(fns, 5, 10)
        t = {}
        for a, (ts, recs) in raw.items():
            kept, _ = _rounds.screen(ts, recs); ms, _, _ = _rounds.summarize(ts, kept, "median")
            if ms: t[a] = ms * 1e3
        print(f"{m}x{n}x{k}: " + "  ".join(f"{a}={v:.1f}" for a, v in sorted(t.items(), key=lambda kv: kv[1])), flush=True)
