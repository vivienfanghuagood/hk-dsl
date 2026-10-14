# Development record

The sections below are a record, in the order the work happened, written while
hk lived inside the HipKittens repository: paths such as `kernels/rdna3`,
`include/rdna3` and `tools/rdna-probes` are HipKittens', now under
`third_party/HipKittens`. They also name the API of their time: before [the last section](15-mlir-dialect.md) the IR was Python and kernels
were written as flat calls (`hk.load(x, idx, ty, bounded=True)` is now
`x[idx].load(ty, bounded=True)`). The kernels themselves have all been ported;
the numbers are as measured then, and [the MLIR chapter](15-mlir-dialect.md) re-measures the ones
that matter against the port.

For the results in one place, with the hardware and baseline of each, see [performance](../performance.md).

| | chapter | in one line |
|---|---|---|
| 01 | [Phases 1 and 2: Python to torch, end to end](01-phases-1-2.md) | 291 shipped kernels, every one spill-free; 13 of 15 cases faster than `torch.compile` and two ties; the worst Python wrapper cut to 9.9 µs. |
| 02 | [Phase 3: the generated GEMM matches the handwritten one](02-phase-3-gemm.md) | `hk.ops.matmul` compiles to the same 209 VGPRs as the handwritten GEMM and runs within noise of it (75.0 and 77.9 TFLOPs); the first 2.4-3.0% gap was one instruction. |
| 03 | [Phase 4: the generated attention kernel matches the handwritten one](03-phase-4-attention.md) | Attention at parity with the handwritten kernel, ahead under causal, at occupancy 6 against 5; three findings moved into the library or the IR. |
| 04 | [Phase 5: `hk.autotune`](04-phase-5-autotune.md) | A two-stage search with the resource gate in front of the GPU; 4 of 16 keys clear the 1% bar, and the rest stay on the default. |
| 05 | [Phase 6: torch ops, the SDPA drop-in, and vLLM's vision tower](05-phase-6-torch-ops.md) | `hk.torch_op` and the SDPA drop-in; what the patch can reach in vLLM; a biased bf16 conversion found and fixed; what is left is a tie. |
| 06 | [Phase 7: the decoder, as a vLLM attention backend](06-phase-7-paged-decode.md) | A paged decode backend that loses to Triton (up to 1.92x slower); splitting the KV axis brings it to a flat 1.14-1.19x, which is the wrapper, not the kernel. |
| 07 | [Against the engine, which is the number that matters](07-against-the-engine.md) | A profile of Qwen3-8B puts prefill attention at 39%; 1.29x end to end on the prefill-heavy shape and 0.87x on the decode-heavy one. |
| 08 | [The other 57.6%: the linear layers](08-linear-layers.md) | The DSL GEMM beats hipBLASLt on Qwen3-8B's prefill shapes; the GEMM patch; 1.37x with both patches; decode turns out to be GEMV. |
| 09 | [The decode kernel](09-decode-kernel.md) | A decode GEMV at about 2x hipBLASLt per step up to 16 tokens; end to end up to 1.72x; residency and a 128 KB LDS pool, found the hard way. |
| 10 | [Row-predicated loads and stores](10-row-predicated.md) | Buffer-range predication: the GEMM takes any M and the decode GEMM stops padding (1.08-1.10x); five wrong versions, each measured. |
| 11 | [A reduction across workgroups, and the decode GEMV that did not need it](11-cross-workgroup-reduction.md) | `hk.wg_arrive`; split-K for the decode GEMV is slower everywhere (a negative result); end to end 1.10-1.84x with nothing losing. |
| 12 | [Strided globals, and the two patches they were for that lost](12-strided-globals.md) | One runtime row stride per global; the SiLU-mul and weighted RMSNorm patches were dropped because the kernels lose to Inductor. |
| 13 | [gfx1201: no arch baked into `hk.ops`, and fp8 WMMA tiles](13-gfx1201.md) | Kernels resolve their arch on first use; fp8 dtypes and WMMA for gfx1201, checked by type rules, build gate and ISA, not on hardware. |
| 14 | [One patch layer, and a routing table that is measured rather than written](14-routing-table.md) | `Op`, `Patch` and `RouteTable`; the measured table beats the hand rule by 7-34% where the rule fell back; the CUDA-graph check. |
| 15 | [The IR as an MLIR dialect, and kernels that read like the math](15-mlir-dialect.md) | The Python IR replaced by an MLIR dialect, gated three times (byte-identical C++, resources and A/B, vLLM end to end); tests, wheels, limits, what is next. |
| 16 | [Typical kernels checked at full size, and the race they found](16-validation-and-the-cross-warp-race.md) | Examples and `validate.py`; the cross-warp combine raced at full size since there were norms -- a plain barrier after ordinary LDS stores; drained, and now refused by the pass, at no measurable cost. |
| 17 | [The decode step captured whole, and where hk's decode still loses](17-the-decode-step-captured-whole.md) | The backend under vLLM's defaults: decode captured whole, paged prefill 3.6-7.9x Triton's; a one-layer A/B hid a 10 us a layer loss below 16 requests, which now runs Triton's kernel on hk's cache. |
| 18 | [Against Triton and FlyDSL: the scoreboard, and what it moved](18-the-scoreboard.md) | One process, every way to compute each op; the GEMM was one LDS buffer away from 1.03-1.08x Triton and FlyDSL; weighted norms and online softmax; small-M norms still lose, to fixed latency. |
