# hk2: a tile language for AMD GPUs, CDNA and RDNA from one source

hk2 is the rewrite of hk-dsl's language and compiler (branch `hk2/main`). A kernel is a Python function over
**tiles**: blocks of a tensor that the compiler spreads over the workgroup. The kernel never names a
wavefront size, a matrix instruction, a register layout, an LDS buffer, a wait or a barrier. The compiler
picks all of them for the GPU it runs on, so the same source runs on MI355X (gfx950, wave64, MFMA) and on
Radeon W7900 (gfx1100, wave32, WMMA).

```python
import hk2 as hk

@hk.jit
def matmul(a, b, c, BM: hk.constexpr = 128, BN: hk.constexpr = 128, BK: hk.constexpr = 64):
    m0, n0 = hk.program_id(0) * BM, hk.program_id(1) * BN
    acc = hk.zeros((BM, BN), hk.f32)
    for k in range(0, a.shape[1], BK):
        x = hk.load(a, (m0, k), (BM, BK))
        y = hk.load(b, (n0, k), (BN, BK)).T          # b is (N, K), as nn.Linear stores it
        acc = hk.dot(x, y, acc)
    hk.store(c, (m0, n0), acc.to(c.dtype))

matmul[(hk.cdiv(M, 128), hk.cdiv(N, 128))](a, b, c)
```

That kernel is 1.2x hipBLASLt on a W7900D and level with autotuned Triton on an MI355X
([performance](#performance)). Flash attention is [45 lines](../../python/hk2/kernels/attention.py) and runs
at Triton's speed on MI355X.

**What is different from hk-dsl 0.x**:
- **Pure Python.** Kernels go to LLVM IR text and are compiled by the ROCm installation's own LLVM (`clang`,
  `ld.lld`): no MLIR build, no hipcc, no per-Python-version wheel. A kernel compiles in about 0.15 s.
- **Compiler-chosen layouts** instead of HipKittens tile types. Data movement, LDS staging and barriers are
  derived, not written.
- **Errors at your source line**, and a CPU [interpreter](#debugging).

## Requirements

- ROCm 7.x (its LLVM 21 or newer) and PyTorch for ROCm.
- Tested on gfx950 (MI355X) and gfx1100 (W7900D).
- Tables exist for gfx942 and gfx1101/1102/1151/1200/1201; these compile but are not yet checked on
  hardware.

Put `python/` on `PYTHONPATH` and `import hk2`.

## The language

**Kernels and launches.** `@hk.jit` makes a kernel; `kernel[grid](args...)` launches it on torch's current
stream.
- Positional tensor arguments are torch tensors of 1 to 4 dimensions.
- `int` and `float` arguments are runtime scalars.
- Parameters annotated `hk.constexpr`, or keyword-only, are compile-time values. Each distinct value gets
  its own compiled kernel.
- `num_warps=` is a launch option (default 4).
- Inside, `t.shape[d]` and `t.stride(d)` are runtime scalars.

**Tiles.** Tile shapes are powers of two, 1-D or 2-D.
- `hk.load(t, offsets, shape, other=0)` reads a tile.
  - Give one offset per dimension of `t`. When `t` has more dimensions than the tile, the leading offsets
    index and the tile covers the trailing ones: `hk.load(q, (b, h, m0, 0), (BM, D))`.
  - Elements outside `t` read `other`.
- `hk.store(t, offsets, tile)` writes a tile, dropping the parts outside `t`.
- Tiles combine with `+ - * / // % & | ^ << >>`, comparisons, unary `-`, numpy-style broadcasting,
  `t[:, None]` / `t[None, :]`, `t.T` and `t.to(dtype)`.
- Functions: `hk.exp exp2 log log2 sqrt rsqrt abs maximum minimum where`.
- Constructors: `hk.zeros / full(shape, v, dtype)` and `hk.arange(n)`.
- Reductions: `t.sum(axis) / max / min`. Sums of 16-bit floats accumulate in f32.
- `hk.dot(a, b, acc=None)` runs on the matrix cores, with bf16/f16 operands and an f32 result.

**Scalars.** `hk.program_id(axis)`, arguments, loop indices and arithmetic on them are uniform across the
workgroup.

**Control flow.** These are plain Python, rewritten to run on the GPU:
- `for i in range(lo, hi, step)` is a device loop. Variables assigned in the body are carried.
- `if cond:` on a runtime scalar is a device branch.
- On Python values (constexprs), `if` and `for` run at trace time as usual.
- `hk.static_range` unrolls a loop at trace time.
- `@hk.func` marks a helper with the same rules. [Attention](../../python/hk2/kernels/attention.py) uses one
  for its masked and unmasked key blocks.

## How it compiles

1. **Trace.** The function runs once per specialization and records tile IR (`hk2/ir.py`). An AST rewrite
   turns `for`/`if` on runtime values into structured ops with carried values.
2. **Pipeline** (`hk2/passes.py`). In loops that feed the matrix cores, the next iteration's loads are
   issued right after this iteration's tiles reach LDS.
3. **Layouts** (`hk2/layout.py`, `hk2/compiler.py`). A layout is a linear map over GF(2) from (register,
   lane, warp) bits to tile-coordinate bits, the representation Triton's linear layouts use.
   - CDNA vs RDNA differ only in the basis tables of `hk2/arch.py`.
   - Loads are coalesced; dot results are matrix-core accumulators; reductions keep their input's layout.
   - Constants, `arange` and broadcasts are rebuilt in whatever layout each use needs.
   - Layout changes that only renumber registers move no data. Changes that stay within each warp use LDS
     without `s_barrier`. Other changes go through LDS.
4. **LDS and barriers.**
   - The emitter tracks which LDS bytes were written or read since the last barrier and puts barriers only
     where a hazard exists.
   - Inside loops, conversions are double-buffered: two slots that swap each iteration. The loop-head state
     is computed as a fixpoint, so a GEMM gets one barrier per K step.
5. **Dots.**
   - The atom and warp grid are chosen per GPU; MI355X uses 32x32 MFMAs when every warp owns at least
     64x64 outputs.
   - In row-reduction kernels (softmax, attention), dots are issued transposed. An operand already in an
     operand layout, up to a permutation of K, is used in place, with the other operand permuted to match.
     So attention's P never leaves registers.
   - On gfx950, operands that need a transpose (V in P@V) are written to LDS as 128-bit rows and read with
     `ds_read_b64_tr_b16`. That instruction's lane mapping was probed: `bench/hk2/probe_tr.py`.
6. **LLVM IR → code object** with the installed ROCm LLVM; launched through the HIP runtime torch has loaded.
   - Spills are reported as a warning, with the register count.
   - `kernel.cache[...].stats` holds VGPRs, LDS, occupancy and the barrier count.
   - `hk2.isa` gives the instruction mix of each loop.

## Debugging

- **Interpreter.** `HK2_INTERPRET=1`, or `kernel[grid](..., interpret=True)`, runs the traced IR on the
  CPU with torch, one program at a time. `hk2.print("x =", tile)` shows values there; compiled kernels
  ignore it. Tests use the interpreter as an oracle for the compiler.
- **Errors** name the kernel source line.
- **Dumps:** `HK2_DUMP=1` prints the tile IR and the LLVM IR.
- **Escape hatches for A/B checks:**

  | variable | disables |
  |---|---|
  | `HK2_PIPELINE=0` | load pipelining |
  | `HK2_NO_DOUBLE` | LDS double buffering |
  | `HK2_NO_TR` | transposed LDS reads |
  | `HK2_NO_TRANS_DOT` | transposed dots |
  | `HK2_NO_LAZY` | lazy operand reads |
  | `HK2_NO_DPP` | DPP shuffles |
  | `HK2_PRECISE_EXP` | fast `exp2` / `log2` |

## Performance

Medians of interleaved rounds in one process, bf16, against torch and autotuned Triton 3.6 (the configs of
Triton's own AMD tutorials). Scripts: `bench/hk2/compare.py`, `bench/hk2/rowwise.py`. Numbers from
2026-10-30.

| TF/s | MI355X torch | MI355X Triton | MI355X hk2 | W7900D torch | W7900D Triton | W7900D hk2 |
|---|---|---|---|---|---|---|
| GEMM 8192³ (A·Bᵀ) | 1610 | 1076–1102 | 1115–1121 | 67 | 81 | 83 |
| GEMM 4096³ | 1393 | 951–978 | 942–970 | 69 | 78 | 81 |
| attention 1x32x4096x128 | 622 (SDPA) | 808 | 821 | 21 | 60 | 39 |
| attention 1x32x8192x64 | 721 | 755 | 764 | 21 | 62 | 46 |
| attention 4096x128, causal | 324 | 439 | 430 | 12 | 53 | 32 |

Row-wise kernels (GB/s):
- **MI355X:** hk2 is level with Triton or ahead, e.g. softmax at 65536x1024: 6237 vs 4372.
- **W7900D:** all three reach the memory limit, ~730 GB/s.

**Open gaps:**
- **MI355X GEMM** is ~0.7x hipBLASLt, the same as Triton. Closing it needs direct-to-LDS loads and the 8-wave
  ping-pong schedule.
- **W7900D attention** is ~0.65x Triton: register pressure and global-load waits; see the thread traces
  from `bench/hk2/att.py`.
- **Causal D=64 on MI355X** is 0.84x Triton.
- **Small row-wise shapes on MI355X** (4096x4096 RMSNorm) are 0.5x Triton.

## Not there yet

- fp8, int8 and MX dots.
- Atomics, scans, gathers, sort.
- Tensors over 2 GB (buffer offsets are 32-bit).
- An autotuner.
- The schedules-as-data and correctness gates of hk-dsl 0.x. The compiler-derived waits and barriers here
  replace the hand-written ones those gates checked; an ISA-level gate pass is still to be added.
- Backward kernels.
- Packaging as its own pure-Python wheel.

## Layout of the code

| path | what |
|---|---|
| `python/hk2/language.py` | everything a kernel calls; records IR |
| `python/hk2/jit.py` | `@jit`, AST rewrite, specialization, cache, launch |
| `python/hk2/layout.py`, `arch.py` | linear layouts; per-GPU atoms and constants |
| `python/hk2/compiler.py` | layouts, LDS/barriers, emission to LLVM IR |
| `python/hk2/passes.py` | load pipelining |
| `python/hk2/runtime.py` | ROCm LLVM invocation, code-object cache, HIP launch |
| `python/hk2/interpret.py` | CPU interpreter |
| `python/hk2/kernels/` | attention, RMSNorm/LayerNorm/softmax |
| `tests/hk2/` | 93 tests; all pass on gfx950 and gfx1100 |
| `bench/hk2/` | GEMM, attention, row-wise, Triton baselines, thread traces, ISA mixes |
