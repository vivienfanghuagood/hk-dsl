# Phase 3: the generated GEMM matches the handwritten one

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Phase 3 is done: the generated GEMM matches the handwritten one.** The hard
gate was three numbers at once -- same `ScratchSize`, same occupancy, and
77.2 TFLOPs. On the pod's ROCm 7.2.4, `hk.ops.matmul` and
`kernels/rdna3/gemm/bf16fp32/gemm.cpp`'s `config<128,128,64,16,8,4,8,4>` compile
to *the same* 209 VGPRs, 0 scratch, 0 spill, 7 waves/SIMD, 64 KB LDS, and run
within noise of each other:

```
shape                 variant         ms    TFLOPs   vs C++
4096x4096x4096        hk (DSL)     1.831      75.0    1.00x
4096x4096x4096        C++          1.833      75.0    1.00x
4096x4096x4096        rocBLAS      1.975      69.6    0.93x

8192x8192x4096        hk (DSL)     7.055      77.9    1.00x
8192x8192x4096        C++          7.047      78.0    1.00x
8192x8192x4096        rocBLAS      7.661      71.8    0.92x
```

Same-process interleaved, bit-identity checked before timing
(`tools/hk-bench/gemm_ab.py`). Disassembled, the two kernels issue the identical
64 WMMAs, 96 `ds_read`s, 16 `ds_write`s and 24 global loads; the generated one
has 52 *fewer* scalar and vector ALU instructions.

It did not start there -- the first measurement was 2.4% and 3.0% behind, and
the whole of it was one instruction. `hk.ops.barrier` is documented as ordering
execution and not memory, but the emitter was lowering it to `__syncthreads()`,
which promises to order global memory too and on gfx11 pays for that with a
`buffer_gl0_inv`: a per-WGP vector L0 flush, once per K-tile, throwing away the
B-tile lines the other workgroup on the WGP was about to read. Emitting
`__builtin_amdgcn_s_barrier()` instead -- which is what the handwritten kernel
calls, and what the op always meant -- closed the entire gap. The LDS side is
not weakened by this: `lds_pipeline` proves it separately, either by finding a
wait that already retired the queue or by making the author write
`barrier(drain=True)`.

---

← [Phases 1 and 2](01-phases-1-2.md) · [Development record](README.md) · [Phase 4](03-phase-4-attention.md) →
