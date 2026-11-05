# Runtime records (to 2026-11)

> **A dated record.** Measurements and history that used to sit in
> [the runtime](../runtime.md) page, with the hardware and versions of their
> time. The page itself says how the runtime behaves today.

## The cache key's cost

Measured on the warm path (20 cached gfx950 kernels, one process, interleaved
with the previous key, two runs): the key costs 0.27 ms a kernel against 0.13
(+0.14-0.18 ms; ~70 headers to stat for a CDNA kernel), and +0.7-0.8 ms over
all 20 on a process's first pass, against a whole cache hit of 1.1-2.6 ms.

Through 0.8.0 the key hashed a fixed set of files (kittens.cuh, pyutils/, the
arch tree, include/hk/cdna.cuh), and an edit to `include/hk/cdna_agpr.cuh` or
`include/pyutils/hk_bind.cuh` served the old kernel.

## The VOPD interlock hazard, measured

In `tools/rdna3-probes/vopd_interlock_parity.hip`, the same window gives a wrong
value in 88-93 of 100 launches under a partial EXEC when the VSRC1s share a
parity, whether or not Y reads X's result, and in none when they differ or
EXEC is full. A bounded strip RMSNorm built by AMD clang 23 (TheRock 7.14)
had such a VOPD (`v_dual_add_f32 v0, v14, v22 :: v_dual_lshlrev_b32 v53,
16, v0`) and got element 1 of every lane's run wrong in 25-65% of launches.

## Warming the cache, measured

24 norm kernels into an empty `HK_CACHE_DIR` on a 256-core box: 102.9 s at
`-j 1` and 5.4 s at `-j 24`.

## Cold compile times by artifact

Cold compile time, one kernel each
(`tools/hk-bench/compile_time.py`; the pod has hipcc 7.2 and torch, the
workstation hipcc 7.1 and no torch):

| kernel | pybind, workstation | code object, workstation | pybind, pod | torch op (libtorch), pod | code object, pod |
|---|---|---|---|---|---|
| weighted RMSNorm | 4.15 s | 0.62 s | 6.80 s | 14.69 s | 1.09 s |
| GEMM 128x128 | 4.24 s | 0.75 s | 7.01 s | 15.05 s | 1.35 s |
| attention d128 | 4.39 s | 0.83 s | 7.14 s | 15.14 s | 1.46 s |
| paged decode walk | 4.48 s | 0.88 s | 7.27 s | 15.20 s | 1.52 s |

## The torch op's host time

The torch op used to cost 27.3 µs of host time per call: it packed the
argument field by field, built a ctypes buffer per launch and entered a device
context. With one precompiled `struct`, a reused argument buffer and no device
switch it cost 17.8 µs, before the host fast paths.
