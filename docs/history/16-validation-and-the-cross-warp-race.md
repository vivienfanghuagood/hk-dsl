# Typical kernels checked at full size, and the race they found

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

After the split into its own repository, hk had a test suite but no
examples, and nothing that ran the shipped ops at the sizes a model
produces. Both arrived together, and the second found a bug on its first run.

**The examples.** Six kernels written from scratch, in the order they
introduce the language: vector add, RMSNorm, softmax, a GEMM through LDS,
flash attention, and `for` / `if` / `while` on a count read from memory.
Each one checks itself against an fp32 torch reference before it times
anything. It then times itself in one process, interleaved, against the
torch call it would replace and the tuned kernel in `hk.ops`. Without a GPU
it runs as `--dump ir|cpp` or `--build`. The test suite holds every one of
them at three tiers: it traces, it compiles without spilling, and its script
passes on the GPU.

**validate.py, and what it found.** `examples/validate.py` runs every
shipped op at 7-8B decoder shapes -- a 4096-wide hidden state, 12288-wide MLP
projections, 32 heads of 128, a 4096-token context, single-token decode rows
-- check first. On its first run rmsnorm, layernorm and softmax at 8192x4096
came back with NaN in the output, and quantize was off by 81% of the
reference's magnitude. Everything else passed.

None of it was new. A probe filled the output with NaN, ran rmsnorm and
counted the rows still NaN:
- From 1024 rows up, a different few hundred rows came out NaN or scaled
  wrong on each run.
- The bad rows always fell in a band, rows 770-3070, whatever the tensor's
  size.
- The tree from before the MLIR move did the same.
- A `torch.cuda.synchronize()` before the call changed nothing: torch and hk
  launch on the same null stream.

So it was in the kernel, and had been since there were norms. The test
suite's shapes -- 64x256, 100x200, 17x65 -- fit in a handful of workgroups,
which is why it never showed.

**The cause.** Every one of the failing kernels combines a row across warps
through `hk.collective.cross_warp` or `fold_rows`. Each warp stores its
partial into LDS, passes `hk.arch.barrier()`, and reads the other warps'
partials. The barrier was the plain one, `__builtin_amdgcn_s_barrier()`, and
on gfx11 that orders execution, not memory:
- The vector store is ordinary ds_writes, which the compiler retires before
  this warp's own reads and not before a barrier. `__syncthreads()` would
  have carried a fence; the raw builtin does not.
- A 16-warp workgroup spans both CUs of a WGP. A warp on the other CU
  therefore read, past the barrier, what an earlier workgroup had left in
  that LDS.

That is why it took a full-size run: enough workgroups have to cycle through
each WGP's LDS for the stale values to be some other row's.

The library already knew this in two places. The vectorised register-to-LDS
tile store ends in `s_waitcnt lgkmcnt(0)` ("retire the writes before the
caller's barrier"), and the group load waits for its own writes. The decode
GEMV, which stores an fp32 tile, already used `barrier(drain=True)`. The
combine was the one place nobody had followed it through.

**The fix, in two parts:**
- **The combine drains.** Both barriers in `cross_warp`, and the one in
  `fold_rows`, are `barrier(drain=True)`: one `s_waitcnt lgkmcnt(0)` before
  the `s_barrier`.
- **hk-lds-pipeline refuses the pattern** instead of leaving it to luck. An
  ordinary LDS store is a `store_shared_vec`, or a `store_shared` that cannot
  take the vectorised path, such as an fp32 tile. When one is followed by a
  plain barrier, with nothing in between that ends in lgkmcnt(0), that is now
  an error that says to drain. Those writes are not counted in any derived
  wait, because the compiler owns them and an estimate could only under-wait.
  What the pass tracks is whether one is still unretired at a barrier.

None of the 422 shipped instantiations trips the check now. The 283 norm and
quantize ones changed by the drains, and every other one is byte-identical.

**What it costs.** The before build is wrong at this size, but it can still
be timed. Before and after ran in one process, interleaved for 7 rounds, on
the W7900D:

| kernel | shape | after / before |
|---|---|---|
| rmsnorm (16 warps, TPW 4) | 8192x4096 | 1.002x |
| softmax (4 warps, FOLD 16) | 8192x4096 | 1.010x |

Nothing measurable. The first version of that comparison showed the
softmax's before build at 0.044 ms, which is 3 TB/s on an 864 GB/s part. The
harness had passed the tensor unfolded to a FOLD=16 kernel, which then ran a
sixteenth of it. A time faster than the memory allows is a wrong harness,
every time.

**Tests.**
- New regression tests run rmsnorm, layernorm and softmax at 8192x4096 and
  16384x1024, and quantize at 4096x4096, each twice; they pass on every row.
- tests/hk/gpu: 401 passed, which includes the seven new tests and eight
  that run the examples.
- tests/hk/ir and tests/hk/codegen: 1010 passed.
- lit covers the new refusal: a vector store and an fp32 tile store followed
  by a plain barrier are errors; a vectorised store or a drain is not.

**The shipped ops against torch, after the fix.** W7900D, PyTorch 2.10
(ROCm 7.2), every check passing:

| case | shape | hk vs torch |
|---|---|---|
| add / exp | 8192x4096 | 0.98x / 0.92x |
| rmsnorm / layernorm / softmax | 8192x4096 | 0.96x / 0.96x / 0.88x |
| silu_mul | 8192x4096 | 1.57x |
| rope (NeoX) | 32 heads x 2048 x 128 | 9.19x |
| quantize int8 | 4096x4096 | 2.78x |
| gemm | 4096³ / M=2048 N=12288 K=4096 | 1.12x / 1.12x |
| decode GEMV | M=1, N=4096 / N=12288, K=4096 | 4.38x / 3.90x |
| attention | 1x32x4096x128, non-causal / causal | 2.76x / 3.94x |

The torch side is eager: `F.rms_norm`, `F.layer_norm`, `torch.softmax`,
`torch.matmul`, `F.linear`, SDPA. Against those the norms trail by 4-12%.
That does not contradict Phase 2's "13 of 15 faster", which was measured
against `torch.compile`; torch's own fused norm kernels are a different
baseline. The silu_mul, rope and quantize columns compare a single hk kernel
with torch's unfused eager ops.

The examples on the same card, as ratios to torch:

| example | this kernel | hk.ops | torch |
|---|---|---|---|
| vector add, 8192² | 0.98x | -- | 1.00x |
| RMSNorm, 8192x4096 | 0.52x | 0.97x | 1.00x |
| softmax, 8192x4096 | 0.48x | 0.85x | 1.00x |
| GEMM, 4096³ | 0.88x (60.4 TF) | 1.07x (73.6 TF) | 1.00x (68.8 TF) |
| attention, N=4096 d=128 | 0.96x | 2.78x | 1.00x |
| attention, causal | 1.42x | 3.99x | 1.00x |

Single-buffered, the tutorial GEMM already reaches 88% of hipBLASLt. Its
attention reads K and V from global memory in every warp, and trails the
shipped kernel by 2.9x non-causal and 2.8x causal, which is what sharing them
through LDS buys.
