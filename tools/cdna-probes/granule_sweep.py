"""Measure the VGPR allocation granule and the register->occupancy curve the
compiler applies on a CDNA target, from LLVM's own Occupancy remark.

Each kernel clobbers exactly one register (v<N-1> or a<N-1>) so the allocator
must reserve N of that kind. Compile-only.

    python3 tools/cdna-probes/granule_sweep.py gfx950

Measured with ROCm 7.1 hipcc: gfx950 and gfx942 agree -- 8 waves/SIMD up
to 64 registers, then 512 // round8(align4(VGPRs) + AGPRs).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

arch = sys.argv[1]
HIPCC = os.environ.get("HK_HIPCC") or shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
cases = []
for n in list(range(1, 129, 1)) + list(range(129, 257, 1)):
    cases.append(("v", n, 0))
for a in (0, 1, 8, 9, 16, 64, 128, 129, 200, 256):
    for v in (32, 64, 128, 129, 256):
        cases.append(("va", v, a))
src = ["#include <hip/hip_runtime.h>"]
for kind, n, a in cases:
    name = f"k_{kind}_{n}_{a}"
    clob = [f'"v{n-1}"']
    if a:
        clob.append(f'"a{a-1}"')
    src.append(f'__global__ __launch_bounds__(256, 1) void {name}(int *p) {{ asm volatile("" ::: {", ".join(clob)}); p[threadIdx.x] = 0; }}')
d = Path(tempfile.mkdtemp(prefix="hk-granule-sweep-"))
(d / "s.cpp").write_text("\n".join(src) + "\n")
r = subprocess.run([HIPCC, f"--offload-arch={arch}", "-O3", "--offload-device-only", "--genco",
                    "--no-gpu-bundle-output", "-Rpass-analysis=kernel-resource-usage",
                    str(d / "s.cpp"), "-o", str(d / "s.co")], capture_output=True, text=True, timeout=900)
log = r.stdout + r.stderr
if r.returncode:
    print(log[-3000:]); sys.exit(1)
recs, cur = {}, None
for line in log.splitlines():
    m = re.search(r"Function Name: (\S+)", line)
    if m:
        cur = m.group(1); recs[cur] = {}; continue
    m = re.search(r"remark:\s+(VGPRs|AGPRs|Occupancy \[waves/SIMD\]): (\d+)", line)
    if m and cur:
        recs[cur][m.group(1)] = int(m.group(2))
rows = []
for kind, n, a in cases:
    key = next(k for k in recs if f"k_{kind}_{n}_{a}" in k)
    f = recs[key]
    rows.append((kind, n, a, f.get("VGPRs"), f.get("AGPRs"), f.get("Occupancy [waves/SIMD]")))
# occupancy steps for v-only
prev = None
print(f"{arch}: arch-VGPR-only sweep, occupancy steps (VGPRs -> waves/SIMD):")
for kind, n, a, vg, ag, occ in rows:
    if kind == "v" and occ != prev:
        print(f"  vgpr={vg:4d} agpr={ag} -> occ {occ}")
        prev = occ
print(f"  last: vgpr={rows[255][3]} occ={rows[255][5]}")
print(f"{arch}: combined VGPR+AGPR:")
for kind, n, a, vg, ag, occ in rows:
    if kind == "va":
        print(f"  vgpr={vg:4d} agpr={ag:4d} -> occ {occ}")
