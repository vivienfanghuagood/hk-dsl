#!/usr/bin/env python3
"""End to end against the unmodified engine. Baseline 1.0x.

This is the number the IR exists to move: take a serving engine as it ships,
replace the kernels it has no good one for, and see whether the *request* got
faster. Not the kernel, not the decode step -- the request.

`vllm_sweep.py` deliberately cancels prefill to isolate a decode step, which
was the right tool for asking "is the paged kernel any good" and the wrong one
for asking "is the patch worth installing". A profile of this exact workload
(tools/hk-bench/vllm_profile.py, Qwen3-8B, batch 16, ctx 4096) says where the
time is:

    57.6%  hipBLASLt GEMM
    39.1%  Triton prefill attention
     2.3%  silu-and-mul, RMSNorm, RoPE, cache writes, everything else

-- so a patch that only touches decode attention is arguing over the last
percent, and the dense prefill kernel is sitting in front of 39%.

The harness itself ships in the package, `python -m hk.integration ab`
(hk/integration/ab.py); this is the same command from a checkout:

    python3 tools/hk-bench/vllm_e2e.py --context 4096 --batch 16 --tokens 72 \\
        --arms hk,gemm,hk+gemm
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from hk.integration.ab import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
