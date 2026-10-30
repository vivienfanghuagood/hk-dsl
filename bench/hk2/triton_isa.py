"""Instruction mix of the autotuned Triton attention kernel's loops, next to hk2's, for one shape."""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import triton_ref  # noqa: E402

from hk2 import isa  # noqa: E402

B, H, S, D = 1, 32, 4096, int(sys.argv[1]) if len(sys.argv) > 1 else 128
q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
triton_ref.attention(q, k, v)
best = triton_ref._fa.best_config
print("triton best config:", best)
cache = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
cands = sorted(glob.glob(os.path.join(cache, "*", "_fa.amdgcn")), key=os.path.getmtime)
want = f"BM': {best.kwargs['BM']}"
for path in reversed(cands):
    meta = os.path.join(os.path.dirname(path), "_fa.json")
    j = json.load(open(meta)) if os.path.exists(meta) else {}
    if j.get("num_warps") == best.num_warps and j.get("num_stages") == best.num_stages and \
            str(best.kwargs["BM"]) in json.dumps(j.get("constants", j)):
        asm = open(path).read()
        break
else:
    asm = open(cands[-1]).read()
for key in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy", "LDSByteSize"):
    for line in asm.splitlines():
        if f"; {key}:" in line:
            print(line.strip())
            break
print(isa.summary(asm))
