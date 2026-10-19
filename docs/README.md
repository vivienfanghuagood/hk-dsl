# hk-dsl documentation

| page | what it covers |
|---|---|
| [Getting started](getting-started.md) | install from a wheel or from source, a first kernel, the four entry points, running the tests and the examples |
| [The language](language.md) | kernels and parameters, types, operators, memory, control flow, namespaces, `@hk.func`, errors, limits |
| [The IR](ir.md) | the hk dialect's types and ops, the kernel and grid functions, the passes, the C++ translation, `hk-opt` / `hk-translate` |
| [The runtime](runtime.md) | arch detection, hipcc and the content-addressed cache, the resource gates, the code-object checker, warming the cache, autotune, `torch.ops` |
| [Integration](integration.md) | the vLLM and SGLang patches, the environment variables that turn them on, `Op` / `Patch` / `RouteTable`, FlyDSL on the routes, the CUDA-graph contract |
| [Performance](performance.md) | the measured results in one place, each with its hardware, baseline and source chapter |
| [Building](building.md) | MLIR and the dialect, the wheels and what they carry, the repository layout |
| [Testing](testing.md) | the test tiers and how to run each, CI, the engine contract tests, the performance gate, and the gates for a change to the generated code |
| [For agents](agents.md) | the loop without a harness of your own: `headroom` (where a step's time goes, against what hk has), `new-op` (a kernel on the rails), `ab --json` (win, loss, tie or noise), and the numbers not to trust |
| [Development record](history/README.md) | the chronological record, Phases 1-7 and the chapters after them, verbatim |
| [CDNA3/4 (design)](design/cdna.md) | what is RDNA3-only, what HipKittens offers on gfx942/gfx950, the op-family estimates, and what runs on an MI355X |
| [HipKittens on CDNA4 (design)](design/cdna-hipkittens-ops.md) | HipKittens' gfx950 GEMM and GQA forward as hk ops: constraints, the toolchain matrix, the on-device check, measurements against hipBLASLt / SDPA / FlyDSL, the route table |

New to hk: read [getting started](getting-started.md), then
[the language](language.md). Deciding whether to use it:
[performance](performance.md), then the [development record](history/README.md)
for how each number was taken.

## Why this exists

The [README](../README.md#why-it-exists) gives the argument in three lines:
the serving engines do not use AMD's fast kernels, those kernels fail
silently, and writing them by hand is slow. This is the longer form, with what
hk-dsl does about each, and where it stops.

### The engines leave Radeon on slow paths

vLLM and SGLang are tuned for Instinct. On a W7900D (gfx1100), each kernel
below was measured against what the engine runs:

- **SGLang's linear layers** are `F.linear` on ROCm, which is hipBLASLt. At
  decode it uses 38-44% of the memory bandwidth; hk's GEMV uses 87% on qkv.
  Per layer that is 1.18-2.30x, and end to end 1.49-1.81x the output
  throughput ([SGLang](history/24-sglang.md)).
- **SGLang's MoE** does not run on gfx1100: sgl_kernel's top-k faults. hk's
  patch replaces the top-k, and its expert GEMMs give OLMoE 1.06-1.32x the
  output throughput of SGLang's Triton GEMMs
  ([SGLang MoE](history/25-sglang-moe.md)).
- **vLLM's 4-bit linear layers** run a Triton AWQ kernel on ROCm. hk's W4A16
  GEMM is 1.40-2.61x faster, and Qwen3-14B-AWQ up to 4.43x per request
  ([4-bit weights](history/22-int4.md)).
- **vLLM's MoE** is Triton's `fused_moe`. hk's MoE GEMM is 1.00-2.77x up to
  256 tokens ([MoE](history/21-moe.md)).
- **vLLM's prefill attention** is Triton's `unified_attention`. hk's paged
  prefill is 3.6-7.9x ([captured whole](history/17-the-decode-step-captured-whole.md)).

Each patch is turned on by an environment variable, sends each shape to the
kernel that measured fastest for it, and keeps to the CUDA-graph contract
([integration](integration.md)).

### Why not HipKittens' C++ directly

HipKittens' `include/rdna3` is about 12,000 lines of tested C++, reaching
77 TFLOP/s on GEMM and 64-65 on attention. Using it meant writing C++:
`kernels/rdna3/attn/fwd/attn.cpp` is 1108 lines with 18 `#define` knobs, and
retuning meant editing macros, `make clean`, rebuild, remeasure. vLLM and
SGLang could not touch it.

hk-dsl keeps HipKittens as the backend: its tiles, MMAs and waits, in the C++
it generates. In front of it are a typed Python layer, an MLIR dialect whose
passes check and derive what the C++ left to the author, a tuner, and the
engine layer. A kernel written in hk-dsl runs at the C++'s speed: attention
0.999-1.002x ([Phase 4](history/03-phase-4-attention.md)), GEMM 1.00x
([Phase 3](history/02-phase-3-gemm.md)).

### Why not only Triton or FlyDSL

On gfx1100, with every contender tuned and measured in one process
([the scoreboard](history/26-the-scoreboard-retuned.md)), each covers part of
the space:

- **FlyDSL**'s GEMM dispatcher beat hk's GEMM by 2-5% until hk's schedule
  changed; now they tie, 0.99-1.01x
  ([the GEMM schedule](history/28-gemm-schedule.md)). FlyDSL has no attention
  on gfx11, because its flash attention stages K/V through a global-to-LDS
  load gfx11 does not have.
- **Triton** schedules the tile itself. Against Triton tuned over gfx1100's
  space, hk's GEMM is 1.09-1.11x faster, and hk's register strips tie or beat
  its row kernels ([strips](history/27-strips.md)). The attention torch runs
  on ROCm is aotriton.
- **hk-dsl** has the attention: 2.7-3.8x torch 2.10's SDPA, and 1.16x causal
  and 0.99x non-causal against torch 2.13's, which is 3.3x faster
  ([chapter 30](history/30-flydsl-routes-gate-contracts.md)).

So hk-dsl is built to sit beside them. A route table may name FlyDSL's
kernel, HipKittens' own or the engine's for the shapes where they measured
fastest, and the scoreboard keeps measuring all of them.

### The traps

Each of the first three cost days to find, and none produces a diagnostic:

1. **`s_waitcnt` carries no register dependence.** The compiler will hoist a
   WMMA above the wait that was supposed to guard it. Results are randomly
   wrong, `ScratchSize` is 0, and nothing is reported. The translation binds
   the fragment with a volatile asm after every wait — you cannot forget to.
   CDNA4 is no different: on an MI355X, every LDS GEMM built without the bind
   returned NaN ([chapter 29](history/29-cdna.md)).
2. **A VGPR spill is silent corruption, not slowness**, because this library
   hand-places `s_waitcnt` and scratch traffic reorders against it.
   `k.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
   return* a spilling kernel.
3. **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
   degrades to 16 scalar `ds_read_u16`, eight times the instructions. The
   passes flag it while tracing, not at benchmark time.
4. **On CDNA4, a copy from global memory into LDS is still in flight at the
   barrier after it.** It counts on `vmcnt`, gfx950's back-off barrier makes
   LLVM put no wait before `s_barrier`, and the reads after it are inline asm
   that LLVM's wait insertion cannot see. The first LDS GEMM on an MI355X gave
   a different wrong answer on each run. The translation now follows every
   such copy with its wait ([the race](history/29-cdna.md#the-race)).

`python -m hk.runtime.cocheck` runs the resource and wait gates on any AMDGPU
code object, hk's or not ([runtime](runtime.md#any-code-object-cocheck)).

### Arch facts as data

The arch facts that used to live in `RDNA.md`, kernel comments and notebooks
are now data: `target/gfx1100.py`, `gfx1201.py`, `gfx942.py` and `gfx950.py`
for Python, and `mlir/lib/Target/Target.cpp` for the passes, with a test that
the two tables agree field by field. Two examples: on gfx1100 the VGPR granule
is 24, so occupancy 6 needs ≤240 registers, not ≤256; on gfx9 lgkmcnt has
4 bits, so a derived wait above 15 is emitted as the cap.

An op that the library does not implement for the requested dtype is the same
kind of late failure, so it is caught the same way: `gelu` on a bf16 tile
raises while tracing rather than producing `undefined hidden symbol` at the end
of a 30-second compile.

Where each of these is enforced: the translation ([IR](ir.md#translation)),
the resource gate ([runtime](runtime.md#the-resource-gates)), the passes
([IR](ir.md#passes)) and the target tables ([IR](ir.md#target-facts)).

### Where it stops

- **One GPU measured end to end.** Every engine result is from one W7900D.
  On an MI355X the elementwise and row kernels run and pass, GEMM and
  attention are HipKittens' own CDNA4 kernels behind hk's engine layer, and
  hk's own CDNA GEMM reaches 0.41x hipBLASLt ([CDNA3/4](design/cdna.md)).
- **Some kernels lose.** The MoE GEMM loses above 256 tokens, so the route
  leaves those steps on Triton. Paged decode attention is 0.83-0.95x Triton's
  at 16-64 requests on vLLM 0.31, yet stays on hk's, because routing those
  steps to Triton slowed the captured step. LayerNorm backward is 0.66-0.92x
  Triton's at 2048-8192 wide, and the W4A16 GEMM 0.69-0.90x vLLM's
  `gptq_gemm_rdna3` at 1-4 tokens. [Performance](performance.md) lists every
  loss.
