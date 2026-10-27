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
| `vgpr_sweep.sh ARCH LO HI` | A kernel clobbering v0..v(N-1) for each N, and LLVM's VGPRs / Occupancy remark: the allocation granule (24 on gfx1201: 216 -> 7 waves, 217 -> 6, 241 -> 5). Compile only. |

```bash
hipcc -O3 --offload-arch=gfx1201 wmma_peaks.hip -o wmma_peaks && ./wmma_peaks
hipcc -O3 --offload-arch=gfx1201 lds_pool.hip -o lds_pool && ./lds_pool
./vgpr_sweep.sh gfx1201 200 256
```

The cache and DRAM ceilings are `tools/hk-bench/scoreboard.py --probe-ceilings`:
on the R9700 an add peaks at 1.81 TB/s at a 48 MB footprint, inside the 64 MB
Infinity Cache, and every size past it runs at 0.62 TB/s.
