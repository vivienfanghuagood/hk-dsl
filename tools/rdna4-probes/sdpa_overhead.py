"""Where hk.ops.sdpa's drop-in loses at ViT shapes on gfx1201: GPU kernel time (profiler) vs wall time per call."""
import sys, time, statistics
import torch, torch.nn.functional as F
_ROOT = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT + "/python"); sys.path.insert(0, _ROOT + "/tools/hk-bench")
import scoreboard as sb, _rounds
from hk.ops import sdpa as S, attn as A
_rounds.use("gfx1201")
for n, h, d in ((577, 16, 64), (1025, 16, 64)):
    qb, kb, vb = (torch.randn(1, n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    q, k, v = (x.permute(0, 2, 1, 3) for x in (qb, kb, vb))
    qc, kc, vc = (x.contiguous() for x in (q, k, v))
    fns = {"sdpa_dropin": lambda: S.scaled_dot_product_attention(q, k, v, scale=d ** -0.5),
           "attention_contig": lambda: A.attention(qc, kc, vc),
           "torch": lambda: F.scaled_dot_product_attention(q, k, v, scale=d ** -0.5)}
    raw = sb.gpu_rounds(fns, 5, 20)
    gpu = {}
    for a, (ts, recs) in raw.items():
        kept, _ = _rounds.screen(ts, recs); ms, _, _ = _rounds.summarize(ts, kept, "median")
        gpu[a] = (ms * 1e3 if ms else None, recs[0] if recs else None)
    wall = {}
    for a, f in fns.items():
        for _ in range(5): f()
        torch.cuda.synchronize(); ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(50): f()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 50 * 1e6)
        wall[a] = statistics.median(ts)
    S.reset_stats(); S.scaled_dot_product_attention(q, k, v, scale=d ** -0.5)
    print(f"n{n} h{h} d{d}: stats {dict(S.STATS)}")
    for a in fns:
        print(f"   {a:<18} GPU {gpu[a][0]:7.1f} us ({gpu[a][1]} kernels/round)  wall {wall[a]:7.1f} us/call", flush=True)
