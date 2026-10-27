"""Chapter 48's small-M loss with W read as a model reads it: hk.ops.gemm.matmul (NT, a linear layer's
layout) against hipBLASLt in NT, NN (W stored K x N) and TT (C^T = W A^T), each with one W read back to back
("hot", resident in the 64 MB Infinity Cache once it fits) and with copies of W rotated past twice the
cache ("cold", from GDDR6). Profiler kernel time, median of queued rounds, TFLOP/s."""
import sys
import torch
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "python")); sys.path.insert(0, str(_ROOT / "tools/hk-bench"))
import scoreboard as sb, _rounds
from hk.ops import gemm as G
_rounds.use("gfx1201")
BF16 = torch.bfloat16
SHAPES = [(m, n, k) for n, k in ((6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288)) for m in (32, 64, 128, 256, 512)]
for m, n, k in SHAPES if len(sys.argv) < 2 else [tuple(map(int, s.split("x"))) for s in sys.argv[1:]]:
    a = torch.randn(m, k, device="cuda", dtype=BF16)
    w = torch.randn(n, k, device="cuda", dtype=BF16)
    ref = (a.float() @ w.float().T)
    copies = max(2, -(-2 * _rounds.IC_BYTES // (n * k * 2)))
    fns = {}
    for temp, nw in (("hot", 1), ("cold", copies)):
        ws = [w] + [w.clone() for _ in range(nw - 1)]
        wkn = [x.T.contiguous() for x in ws]
        o = torch.empty(m, n, device="cuda", dtype=BF16)
        nxt, nxt_kn = sb._rotating(ws), sb._rotating(wkn)
        fns[f"hk NT {temp}"] = lambda nxt=nxt, o=o: G.matmul(a, nxt(), out=o)
        fns[f"blas NT {temp}"] = lambda nxt=nxt: torch.nn.functional.linear(a, nxt())
        fns[f"blas NN {temp}"] = lambda nxt=nxt_kn: a @ nxt()
        fns[f"blas TT {temp}"] = lambda nxt=nxt: (nxt() @ a.T).T
        for x in (f"hk NT {temp}", f"blas NT {temp}"):
            got = fns[x]().float()
            err = (got - ref).abs().max().item() / ref.abs().max().item()
            assert err < 2e-2, (x, m, n, k, err)
    torch.cuda.synchronize()
    raw = sb.gpu_rounds(fns, 5, 12)
    t = {}
    for x, (ts, recs) in raw.items():
        kept, _ = _rounds.screen(ts, recs)
        ms, _, _ = _rounds.summarize(ts, kept, "median")
        if ms:
            t[x] = 2 * m * n * k / (ms * 1e-3) / 1e12
    for temp in ("hot", "cold"):
        row = {x.rsplit(" ", 1)[0]: v for x, v in t.items() if x.endswith(temp)}
        best = max((v, x) for x, v in row.items() if x != "hk NT")
        print(f"{m}x{n}x{k} {temp:<4} (W {n * k * 2 / 2**20:.0f} MiB x{1 if temp == 'hot' else copies}): "
              + "  ".join(f"{x}={v:.1f}" for x, v in row.items())
              + f"   | hk / best blas {row.get('hk NT', 0) / best[0]:.2f}x ({best[1]})", flush=True)
