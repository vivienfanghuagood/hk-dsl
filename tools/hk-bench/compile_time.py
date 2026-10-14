#!/usr/bin/env python3
"""Cold compile time of one kernel each way hk can build it: the pybind
module `Kernel.__call__` loads, the libtorch registration
`hk.torch_op` loads by default, and the device-only code object
`hk.runtime.codeobj` launches. Each build goes to an empty cache.

    python3 tools/hk-bench/compile_time.py           # torch arm needs torch
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from hk.ops import attn, gemm, norm, paged  # noqa: E402

KERNELS = {
    "rms_norm (weighted)": lambda: norm.KERNELS["rmsw_bf16_w4"].specialize(
        TPW=1, FOLD=16, EXACT=1),
    "gemm 128x128": lambda: gemm.KERNELS["gemm_bf16_128x128"],
    "attention d128": lambda: attn.attn_kernel("ct_attn", head_dim=128),
    "paged decode walk": lambda: paged.fused_kernel(128, 32),
}


def _cold(fn):
    with tempfile.TemporaryDirectory() as d:
        os.environ["HK_CACHE_DIR"] = d
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t


def main():
    try:
        import torch  # noqa: F401, PLC0415
        have_torch = True
    except ImportError:
        have_torch = False
    from hk.runtime import codeobj, torch_ext  # noqa: PLC0415

    print(f"{'kernel':<22}{'pybind s':>10}{'torch op s':>12}{'code object s':>15}")
    for name, make in KERNELS.items():
        k = make()
        k.trace()
        py = _cold(lambda: k.build())
        to = (_cold(lambda: torch_ext.build(k, name=f"ct_{k.name}"))
              if have_torch else float("nan"))
        co = _cold(lambda: codeobj.load(k))
        print(f"{name:<22}{py:>10.2f}{to:>12.2f}{co:>15.2f}", flush=True)


if __name__ == "__main__":
    main()
