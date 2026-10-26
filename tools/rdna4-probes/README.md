# gfx1201 probes

Standalone programs behind the numbers in `python/hk/target/gfx1201.py` and the
gfx1201 ceilings (`tools/hk-bench/_rounds.py`, `hk.check.CARDS`,
`hk.profile.analyze.CEILINGS`), and what FlyDSL runs there. None uses hk or HipKittens. The layout probes
are HipKittens' (`third_party/HipKittens/tools/rdna-probes`: `wmma_layout.hip`
with `-DPROBE_GFX12_W32` measured the gfx12 fragments, `srsrc3.hip` the buffer
descriptor, `mma_gate.hip` and friends the library on top).

| | |
|---|---|
| `wmma_peaks.hip` | Back-to-back WMMAs, no memory, every CU, for bf16, f16, fp8 x fp8, iu8 and the 2:4 sparse bf16 SWMMAC. On a Radeon AI PRO R9700: 197.1, 198.5, 371.8 TFLOP/s, 373.8 TOP/s, 391.5 (dense-equivalent). |
| `lds_pool.hip` | One-wave workgroups at 8-64 KB of LDS each, the peak resident at once: 4 MB over the R9700's 32 WGPs at every size, so 128 KB a WGP. |
| `fly_probe.py` | FlyDSL outside hk (`FLYDSL_ROOT`): its gfx12 WMMA GEMM against fp32, then its dense attention, which aborts the process on gfx12 (LLVM cannot lower `raw_ptr_buffer_load_lds`). The abort leaves an apport crash report on a host whose `core_pattern` pipes to apport, core limit or not. |
| `fp8_sweep.py` | hk's GEMM with fp8 operands over 52 schedules against `torch._scaled_mm`, one process: the source of `hk.ops.gemm.FP8_DEFAULT` (256x128x64, 240 TFLOP/s at 4096³ against hipBLASLt's 236). |
| `fp8_small_m.py` | `scaled_mm`'s kernel at 256 and 128 rows a block against `torch._scaled_mm` with per-token x per-channel scales, M 1-4096: where `FP8_SMALL_M` comes from. |
| `fp8_decode.py` | hk.ops.decode_gemm's kernel with fp8 operands over its schedules, against `scaled_mm`'s tiled kernel and FlyDSL's RDNA4 fp8 GEMM, M 1-64: the source of `FP8_DECODE` / `FP8_DECODE_M`. |
| `i8_sweep.py` | The fp8 sweep with iu8 operands against `torch._int_mm`: the same point (256x128x64) wins, 217 TOP/s at 4096³ against 215. |
| `fp8_decode3.py` | The fp8 decode kernels with W read from GDDR6 (six copies rotated past the Infinity Cache), profiler time: hk 585-617 GB/s, ahead of FlyDSL's 32-row kernel, which is ahead only when W stays in the cache. |
| `vgpr_sweep.sh ARCH LO HI` | A kernel clobbering v0..v(N-1) for each N, and LLVM's VGPRs / Occupancy remark: the allocation granule (24 on gfx1201: 216 -> 7 waves, 217 -> 6, 241 -> 5). Compile only. |

```bash
hipcc -O3 --offload-arch=gfx1201 wmma_peaks.hip -o wmma_peaks && ./wmma_peaks
hipcc -O3 --offload-arch=gfx1201 lds_pool.hip -o lds_pool && ./lds_pool
./vgpr_sweep.sh gfx1201 200 256
```

The cache and DRAM ceilings are `tools/hk-bench/scoreboard.py --probe-ceilings`:
on the R9700 an add peaks at 1.81 TB/s at a 48 MB footprint, inside the 64 MB
Infinity Cache, and every size past it runs at 0.62 TB/s.
