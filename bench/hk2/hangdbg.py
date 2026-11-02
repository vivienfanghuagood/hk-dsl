"""Run one GEMM config; dump all Python stacks after a timeout (for hangs in compile or launch)."""
import faulthandler
import os
import sys
import time

faulthandler.dump_traceback_later(int(os.environ.get("HK2_DBG_TIMEOUT", "90")), exit=True)
sys.path.insert(0, os.path.dirname(__file__))
import torch  # noqa: E402

import hk2 as hk  # noqa: E402
from gemm import matmul  # noqa: E402

cfg = sys.argv[1] if len(sys.argv) > 1 else "256x256x64x8"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
bm, bn, bk, nw = map(int, cfg.split("x"))
a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
w = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
t0 = time.time()
k = matmul.warmup(a, w, c, BM=bm, BN=bn, BK=bk, TRANS_B=True, num_warps=nw)
print(f"compiled in {time.time() - t0:.1f}s", k.stats, flush=True)
matmul[(hk.cdiv(n, bm), hk.cdiv(n, bn))](a, w, c, BM=bm, BN=bn, BK=bk, TRANS_B=True, num_warps=nw)
torch.cuda.synchronize()
ref = a.float() @ w.float().T
print("ran; rel err", ((c.float() - ref).abs().max() / ref.abs().max()).item(), flush=True)
