"""Time the register -> occupancy model on the GPU, not just read the compiler's.

granule_sweep.py reads LLVM's Occupancy remark; this checks it against the
part. Each kernel pins exactly N VGPRs (and M AGPRs) with an asm clobber, then
runs a latency-bound loop: each wave chases a chain of dependent loads through
an L2-resident table, 64 consecutive dwords a step (one coalesced access; a
gather of 64 random entries is bound by the cache's address rate instead, and
times flat whatever the occupancy). One wave cannot hide that latency, so the
time to push a
fixed number of waves through the GPU is inversely proportional to how many are
resident per SIMD. The measured occupancy at N is occ(32) * t(32) / t(N), with
occ(32) taken as the hardware maximum; the model's is 512 // round8(align4(V) +
A), capped at 8.

    python3 tools/cdna-probes/occupancy_timed.py        # in rocm/pytorch, one GPU

Prints per register count the model, the measured estimate, and the ratio.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

ARCH = os.environ.get("HK_ARCH", "gfx950")
#: 64-dword blocks in the table: 1 MiB, past L1, inside one XCD's 4 MB L2.
BLOCKS = 4096
POINTS = ([("v", n, 0) for n in (32, 56, 64, 65, 72, 73, 80, 81, 96, 97, 112, 128, 129,
                                 168, 169, 256)] +
          [("va", 64, a) for a in (1, 8, 64)] + [("va", 128, a) for a in (1, 64)])


def source() -> str:
    out = ["#include <hip/hip_runtime.h>", '#include <cstdint>', f"#define BLOCKS {BLOCKS}"]
    for kind, n, a in POINTS:
        clob = [f'"v{n - 1}"'] + ([f'"a{a - 1}"'] if a else [])
        out.append(f"""
extern "C" __global__ __launch_bounds__(64) void k_{kind}_{n}_{a}(const uint32_t *t, uint32_t *o, int iters) {{
    asm volatile("" ::: {", ".join(clob)});
    uint32_t i = ((blockIdx.x * 64) & (BLOCKS * 64 - 1)) + threadIdx.x;
    for (int s = 0; s < iters; ++s) i = t[i];
    o[blockIdx.x * 64 + threadIdx.x] = i;
}}""")
    return "\n".join(out) + "\n"


def main():
    import torch  # noqa: PLC0415

    from hk.runtime import codeobj, compile as hc  # noqa: PLC0415
    from hk.runtime.cocheck import check  # noqa: PLC0415
    from hk.target import get_target  # noqa: PLC0415

    tgt = get_target(ARCH)
    d = Path(tempfile.mkdtemp(prefix="hk-occ-"))
    (d / "o.cpp").write_text(source())
    co = d / "o.co"
    subprocess.run([hc.hipcc(), f"--offload-arch={ARCH}", "-O3", "--offload-device-only",
                    "--genco", "--no-gpu-bundle-output", str(d / "o.cpp"), "-o", str(co)],
                   check=True, capture_output=True)
    res = {k.name: k for k in check(co)}
    hip = codeobj.hip()
    mod = ctypes.c_void_p()
    codeobj._check(hip.hipModuleLoadData(ctypes.byref(mod), co.read_bytes()), "load")
    # A random cycle through the blocks; lane l of block b points at lane l
    # of the next block, so a wave's step is one contiguous 256-byte read.
    perm = torch.randperm(BLOCKS, dtype=torch.int64)
    nxt = torch.empty(BLOCKS, dtype=torch.int64)
    nxt[perm] = torch.roll(perm, 1)
    table = (nxt[:, None] * 64 + torch.arange(64)[None, :]).reshape(-1).to(torch.int32)
    t = table.cuda()
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = cus * 4 * 8 * 8          # 8 rounds of a full GPU at 8 waves/SIMD
    o = torch.empty(blocks * 64, dtype=torch.int32, device="cuda")
    iters = 400
    rows = []
    for kind, n, a in POINTS:
        name = f"k_{kind}_{n}_{a}"
        fn = ctypes.c_void_p()
        codeobj._check(hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()), name)
        args = (ctypes.c_void_p * 3)()
        tp, op, it = ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(o.data_ptr()), ctypes.c_int(iters)
        args[0], args[1], args[2] = (ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in (tp, op, it))
        best = float("inf")
        for _ in range(7):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            codeobj._check(hip.hipModuleLaunchKernel(fn, blocks, 1, 1, 64, 1, 1, 0,
                                                     torch.cuda.current_stream().cuda_stream,
                                                     args, None), name)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
        k = next(v for key, v in res.items() if name in key)
        rows.append((kind, n, a, k.vgpr, best))
    t32 = rows[0][4]
    occ_max = tgt.max_waves_per_simd
    print(f"{ARCH}: {cus} CUs, {blocks} waves, {iters} dependent loads each; "
          f"model = {tgt.vgprs_per_simd} // round{tgt.vgpr_granule}(charged), cap {occ_max}")
    print(f"{'V':>4} {'A':>4} {'charged':>7} {'model':>5} {'measured':>8} {'ms':>7}")
    for kind, n, a, charged, ms in rows:
        model = tgt.occupancy(charged)
        meas = occ_max * t32 / ms
        print(f"{n:4d} {a:4d} {charged:7d} {model:5d} {meas:8.2f} {ms:7.3f}")


if __name__ == "__main__":
    main()
