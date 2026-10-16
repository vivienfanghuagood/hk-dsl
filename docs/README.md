# hk-dsl documentation

| page | what it covers |
|---|---|
| [Getting started](getting-started.md) | install from a wheel or from source, a first kernel, the four entry points, running the tests and the examples |
| [The language](language.md) | kernels and parameters, types, operators, memory, control flow, namespaces, `@hk.func`, errors, limits |
| [The IR](ir.md) | the hk dialect's types and ops, the kernel and grid functions, the four passes, the C++ translation, `hk-opt` / `hk-translate` |
| [The runtime](runtime.md) | arch detection, hipcc and the content-addressed cache, the resource gates, warming the cache, autotune, `torch.ops` |
| [Integration](integration.md) | the vLLM and SGLang patches, the environment variables that turn them on, `Op` / `Patch` / `RouteTable`, the CUDA-graph contract |
| [Performance](performance.md) | the measured results in one place, each with its hardware, baseline and source chapter |
| [Building](building.md) | MLIR and the dialect, the wheels and what they carry, the repository layout |
| [Testing](testing.md) | the four test tiers, how to run each, and the three gates the MLIR move passed |
| [Development record](history/README.md) | the chronological record, Phases 1-7 and the chapters after them, verbatim |
| [CDNA3/4 (design)](design/cdna.md) | what is RDNA3-only, what HipKittens offers on gfx942/gfx950, the op-family estimates, and what runs on an MI355X |
| [HipKittens on CDNA4 (design)](design/cdna-hipkittens-ops.md) | HipKittens' gfx950 GEMM and GQA forward as hk ops: constraints, the toolchain matrix, the on-device check, measurements against hipBLASLt / SDPA / FlyDSL, the route table |

New to hk: read [getting started](getting-started.md), then
[the language](language.md). Deciding whether to use it:
[performance](performance.md), then the [development record](history/README.md)
for how each number was taken.

## Why this exists

`include/rdna3` is 9494 lines of verified C++ hitting 77.2 TFLOPs on GEMM and
64–65 on attention. Using it meant writing C++: `kernels/rdna3/attn/fwd/attn.cpp`
is 1237 lines plus 20 `#define` knobs, and retuning meant editing macros,
`make clean`, rebuild, remeasure. vLLM and SGLang could not touch it.

But the real argument is the three traps, each of which cost days:

1. **`s_waitcnt` carries no register dependence.** The compiler will hoist a
   WMMA above the wait that was supposed to guard it. Results are randomly
   wrong, `ScratchSize` is 0, and nothing is reported. The translation binds
   the fragment with a volatile asm after every wait — you cannot forget to.
2. **A VGPR spill is silent corruption, not slowness**, because this library
   hand-places `s_waitcnt` and scratch traffic reorders against it.
   `k.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
   return* a spilling kernel.
3. **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
   degrades to 16 scalar `ds_read_u16`, eight times the instructions. The
   passes flag it while tracing, not at benchmark time.

Plus the arch facts that used to live in `RDNA.md`, kernel comments and
notebooks, now data: `target/gfx1100.py` for Python and
`mlir/lib/Target/Target.cpp` for the passes, with a test that the two tables
agree field by field — the VGPR granule is 24, so occupancy 6 needs ≤240
registers, not ≤256.

An op that the library does not implement for the requested dtype is the same
kind of late failure, so it is caught the same way: `gelu` on a bf16 tile
raises while tracing rather than producing `undefined hidden symbol` at the end
of a 30-second compile.

Where each of these is enforced: the translation ([IR](ir.md#translation)),
the resource gate ([runtime](runtime.md#the-resource-gates)), the passes
([IR](ir.md#passes)) and the target tables ([IR](ir.md#target-facts)).
