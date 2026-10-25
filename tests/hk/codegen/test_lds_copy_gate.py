"""The wait gate: hk refuses a kernel whose code reads registers an LDS load is
still filling, before the wait that retires the load.

hk's LDS reads are inline asm, and LLVM takes their outputs as ready when the
asm ends. Under register pressure, the compiler can copy a destination (a
v_mov to assemble an operand tuple, a v_accvgpr_write to park it) ahead of
hk's lgkmcnt wait. That kernel is silently wrong with ScratchSize 0, so it
gets the spill gate's treatment.

- The synthetic cases pin the rule on every compiler: one inline-asm copy
  before the wait is refused, the same copy after it builds, and
  HK_WAIT_GATE=0 lets it through.
- The ROCm 10 cases (hipcc clang >= 23, the vLLM image's) are the kernels
  where it was found:
  - experiment 2's MXFP8 GEMM at 128x256 with byte A scales and K=1152,
    0.83 off when it ran;
  - hk.ops.gemm_cdna_dp's 128x256 setprio split-K configurations.
  rocm/pytorch 7.2.3's hipcc, CI's, does not make these copies.
Compile-only: hipcc and the ROCm LLVM tools.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("hk._mlir")

from hk.runtime import compile as hc  # noqa: E402
from hk.runtime.resources import ResourceError  # noqa: E402

PROBE = Path(__file__).resolve().parents[3] / "tools" / "cdna-probes" / "lds_copy_wq.py"

# One LDS read of four dwords, and a copy of its first register either side of the wait.
SYNTH = r"""
#include <hip/hip_runtime.h>
extern "C" __global__ void lds_copy(float *out) {
    __shared__ float4 s[64];
    s[threadIdx.x] = make_float4(float(threadIdx.x), 1.f, 2.f, 3.f);
    __syncthreads();
    const unsigned addr = threadIdx.x * 16u;
    float4 v;
    float c;
    asm volatile(READ " %0, %1" : "=v"(v) : "v"(addr) : "memory");
#ifdef EARLY_COPY
    asm volatile("v_mov_b32 %0, %1" : "=v"(c) : "v"(v.x));
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#else
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    asm volatile("v_mov_b32 %0, %1" : "=v"(c) : "v"(v.x));
#endif
    out[threadIdx.x] = c + v.y;
}
"""
READ = {"gfx950": '"ds_read_b128"', "gfx1100": '"ds_load_b128"'}


def _synth(arch: str, early: bool) -> str:
    return ("#define EARLY_COPY 1\n" if early else "") + f"#define READ {READ[arch]}\n" + SYNTH


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100"])
def test_a_copy_of_an_lds_reads_registers_before_the_wait_is_refused(arch):
    with pytest.raises(ResourceError, match="LDS load is\\s+still filling"):
        hc.build(_synth(arch, True), arch, name=f"lds_copy_early_{arch}", kind="co")


@pytest.mark.parametrize("arch", ["gfx950", "gfx1100"])
def test_the_same_copy_after_the_wait_builds(arch):
    hc.build(_synth(arch, False), arch, name=f"lds_copy_late_{arch}", kind="co")


def test_the_gate_can_be_turned_off_to_look_at_such_a_kernel(monkeypatch):
    monkeypatch.setenv("HK_WAIT_GATE", "0")
    hc.build(_synth("gfx950", True), "gfx950", name="lds_copy_early_off", kind="co")
    monkeypatch.delenv("HK_WAIT_GATE")
    # the entry built with the gate off is not marked checked: its next use is gated
    with pytest.raises(ResourceError, match="LDS load is\\s+still filling"):
        hc.build(_synth("gfx950", True), "gfx950", name="lds_copy_early_off", kind="co")


def _rocm10():
    m = re.search(r"clang version (\d+)", hc._hipcc_version())
    if not m or int(m.group(1)) < 23:
        pytest.skip("the copies are ROCm 10's (hipcc clang >= 23, the vLLM image's); "
                    "rocm/pytorch 7.2.3's hipcc does not make them")


def _gfx950(code: str) -> str:
    """Run `code` in a process whose arch is gfx950 (this one is pinned to gfx1100, and the
    kernels below trace arch-dependent code), returning its stdout."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, HK_ARCH="gfx950")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_rocm10_refuses_the_mxfp8_gemm_that_was_wrong_at_k1152():
    _rocm10()
    out = _gfx950(f"import runpy; runpy.run_path({str(PROBE)!r}, run_name='__main__')")
    assert "K=1152: refused -- generated kernel rejected" in out, out
    assert "K=1024: built" in out and "K=1280: built" in out, out      # exact when run


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_rocm10_refuses_the_setprio_split_k_dp_gemms(splits):
    _rocm10()
    out = _gfx950(f"""
from hk.codegen import scaffold
from hk.ops import gemm_cdna_dp as D
from hk.runtime import compile as hc
from hk.runtime.resources import ResourceError
s = dict(D.DEFAULT, block_m=128, block_n=256, setprio=True, splits={splits}, xcd_chunk=0)
k = D.kernel_for(s)
try:
    hc.build(k.with_defines(scaffold.device(k.trace())), "gfx950", name=k.name, kind="co")
    print("built")
except ResourceError as e:
    print("refused:", str(e).splitlines()[0])
""")
    assert "refused: generated kernel rejected" in out and "LDS load" in out, out
