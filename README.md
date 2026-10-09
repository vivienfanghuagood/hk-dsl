# hk — HipKittens as a Python kernel IR

Write Radeon kernels in Python, get HipKittens C++, get a `.so`, call it from
torch. The point is not convenience: it is that three classes of *silently
wrong* kernel on RDNA3 stop being expressible.

```python
import hk
from hk import bf16

@hk.kernel(arch="gfx1100",
           grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def add(a: hk.GL[bf16], b: hk.GL[bf16], o: hk.GL[bf16], *, ROWS=16, COLS=64):
    t   = hk.rt(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.block_idx.y, hk.block_idx.x)
    hk.store(o, hk.load(a, idx, t) + hk.load(b, idx, t), idx)

add(a, b, out)          # traces, compiles (once), launches
```

Three entry points, in increasing order of what they need:

| call | needs | does |
|---|---|---|
| `k.trace()` | nothing | runs the body once, returns the IR |
| `k.source()` | nothing | the generated HipKittens C++ |
| `k.build()` | hipcc | compiles and **gates on the resource remarks** |
| `k(*tensors)` | a Radeon + torch | launches |

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop.

## Why this exists

`include/rdna3` is 9494 lines of verified C++ hitting 77.2 TFLOPs on GEMM and
64–65 on attention. Using it meant writing C++: `kernels/rdna3/attn/fwd/attn.cpp`
is 1237 lines plus 20 `#define` knobs, and retuning meant editing macros,
`make clean`, rebuild, remeasure. vLLM and SGLang could not touch it.

But the real argument is the three traps, each of which cost days:

1. **`s_waitcnt` carries no register dependence.** The compiler will hoist a
   WMMA above the wait that was supposed to guard it. Results are randomly
   wrong, `ScratchSize` is 0, and nothing is reported. The emitter binds the
   fragment with a volatile asm after every wait — you cannot forget to.
2. **A VGPR spill is silent corruption, not slowness**, because this library
   hand-places `s_waitcnt` and scratch traffic reorders against it.
   `hk.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
   return* a spilling kernel.
3. **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
   degrades to 16 scalar `ds_read_u16`, eight times the instructions. The
   verifier rejects it while tracing, not at benchmark time.

Plus the arch facts that used to live in `RDNA.md`, kernel comments and
notebooks, now in `target/gfx1100.py` as data shared by the verifier, the LDS
allocator, the register model and the emitter — the VGPR granule is 24, so
occupancy 6 needs ≤240 registers, not ≤256.

An op that the library does not implement for the requested dtype is the same
kind of late failure, so it is caught the same way: `hk.gelu` on a bf16 tile
raises while tracing rather than producing `undefined hidden symbol` at the end
of a 30-second compile.

## Layout

```
hk/lang/      the DSL surface -- @kernel, tile types, ops
hk/ir/        Value/Op/KernelIR, the tracing builder, passes
hk/target/    gfx1100 / gfx1201 facts, as data
hk/codegen/   IR -> HipKittens C++ (cpp.py) and the module boundary (scaffold.py)
hk/runtime/   arch detection, hipcc + content-hash cache, the resource gate
hk/ops/       kernels written in the DSL and shipped with the package
```

The generated C++ is kept readable — named tile aliases, value names shared
with the IR dump, a provenance header — because it is the artifact you take to
the disassembler when something spills.

`codegen/scaffold.py` is separate from `codegen/cpp.py` so that the torch
registration path can reuse the identical kernel body instead of becoming a
second emitter that drifts.

## Caching

Key = hipcc version + a content hash of `include/rdna3` + the full flag list +
the generated source. A build lands in `~/.cache/hk/<key>/` via a rename, so a
killed compile never leaves an importable half-written `.so`. Failed builds are
kept as `failed-<key>/` with their source and log.

The thresholds (`max_vgprs`, `min_occupancy`) are *not* in the key — they are
re-applied on every cache hit, so a kernel with a strict occupancy requirement
is not let through by whoever happened to compile the same bytes first.

First compile of a small kernel is ~5 s; an attention kernel is 20–40 s.

## Tests

Three tiers, by what they need. The first two run anywhere:

```bash
python3 -m pytest tests/hk/ir tests/hk/codegen -q   # no GPU, no torch
```

| tier | needs | what it catches |
|---|---|---|
| `tests/hk/ir/` | pure Python | tracing and emitted C++ |
| `tests/hk/codegen/` | hipcc | compiles; spill/occupancy gates fire |
| `tests/hk/gpu/` | Radeon + torch | numerics vs torch |

A tier that cannot run is skipped with a reason, never silently passed.

## Status

**Phases 1 and 2 are done.** End to end: Python → IR → C++ → hipcc → `.so` →
torch. 291 shipped kernels — elementwise (5 binary × 3 dtypes plus the unary ops
the library specialises), RMSNorm / LayerNorm / softmax, RoPE, SiLU-mul, and
per-row int8 quantize/dequantize. Every one builds with `scratch=0 spill=0`
(a hard gate, not a check) and matches torch elementwise on a W7900D, including
shapes that do not divide the tile.

Phase 2's gate was "do not require a win". It won anyway: against
`torch.compile` on a W7900D, 13 of 15 cases are faster and the other two lose by
under 1%.

```
case                        hk      compile    eager
rmsnorm   4096x4096      0.060 ms   0.093     0.418
rmsnorm  16384x5120      0.486      0.775     2.823
layernorm 4096x4096      0.061      0.081     0.063
softmax   8192x16384     0.707      0.874     0.738
silu_mul  8192x11008     0.761      0.763     1.218
rope     128x1024x128    0.076      0.107     0.583
quantize 16384x5120      0.367      0.369     6.118
```

Getting there was mostly *not* kernel work. A 4096x4096 quantize is 60 µs of
GPU; the Python wrapper around it was 13.9 µs. Three rounds — memoising the
launcher on the kwargs as passed, caching the plan on the `torch.dtype` object
rather than `str(dtype)`, and a pybind converter that interns the attribute
names and reads a tensor in five Python calls (`pyutils/hk_bind.cuh`) — took the
worst wrapper to 9.9 µs and `rmsnorm` from 11.3 to 6.6. See
`tools/hk-bench/host_overhead.py`, which measures host time, queue depth and
device time separately so the three cannot be confused for each other.

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

Next: attention parity (Phase 4). See `.claude/plans/` for the full plan and its
gates.
