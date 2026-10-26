"""int8 GEMM schedules on gfx1201 (the fp8 sweep with iu8 operands), interleaved with torch._int_mm, one process."""
import sys, statistics, itertools, traceback
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python"))
import hk
from hk.ops import gemm as G

pts = []
for bm, bn, ks, ds, buf, ns in itertools.product((128, 256), (128, 256), (64, 128), (16, 32), (1, 2), (2, 4)):
    if bm == 256 and bn == 256:
        continue
    pts.append(dict(G.DEFAULT, block_m=bm, block_n=bn, k_step=ks, dot_slice=ds, lds_buffers=buf, n_split=ns))
shapes = [(4096, 4096, 4096), (2048, 6144, 4096)]
data = {}
for m, n, k in shapes:
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    data[(m, n, k)] = (a, b, a.float() @ b.float().T)
kerns = {}
for p in pts:
    nm = "i8sw_" + "_".join(f"{x}{p[x]}" for x in ("block_m", "block_n", "k_step", "dot_slice", "lds_buffers", "n_split"))
    try:
        kern = G.gemm_kernel(nm, op_dtype=hk.i8, **p)
        a, b, ref = data[shapes[0]]
        c = torch.empty(a.shape[0], b.shape[0], device="cuda", dtype=torch.int32)
        kern(a, b, c); torch.cuda.synchronize()
        err = ((c.float() - ref).abs().max() / ref.abs().max()).item()
        if err > 1e-2:
            print(nm, "WRONG", err); continue
        kerns[nm] = kern
    except Exception as e:
        print(nm, type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
print(len(kerns), "built", flush=True)
one = torch.ones((), device="cuda")
for shp in shapes:
    a, b, ref = data[shp]
    m, n, k = shp
    outs = {}
    fns = {nm: (lambda kern=kern, c=torch.empty(m, n, device="cuda", dtype=torch.int32): kern(a, b, c)) for nm, kern in kerns.items()}
    fns["scaled_mm"] = lambda: torch._int_mm(a, b.T)
    times = {x: [] for x in fns}
    names = list(fns)
    for r in range(5):
        for x in names[r % len(names):] + names[:r % len(names)]:
            f = fns[x]
            f(); f(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20): f()
            e1.record(); torch.cuda.synchronize()
            times[x].append(e0.elapsed_time(e1) / 20)
    fl = 2 * m * n * k
    print(f"== {m}x{n}x{k}")
    for x, ts in sorted(times.items(), key=lambda t: statistics.median(t[1]))[:12]:
        ms = statistics.median(ts)
        print(f"  {ms:.4f} ms {fl / ms / 1e9:5.0f} TF  {x}")
    print(f"  scaled_mm: {statistics.median(times['scaled_mm']):.4f}", flush=True)
