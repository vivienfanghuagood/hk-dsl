# What hk-dsl is, and what it is not

**hk-dsl is a general-purpose Python DSL for AMD GPU kernels on RDNA and CDNA,
with build-time gates against silent tile-level errors (spills, missing waits,
MFMA hazards).** Today its largest end-to-end gains are on Radeon (RDNA3 and
RDNA4), where its kernels plug into vLLM and SGLang.

The gates are its one unique feature. This page says what hk-dsl is, how it
compares with the alternatives, and where its numbers come from.

## How it works

You write a kernel in Python at the level of tiles in registers and LDS,
matrix-core instructions (WMMA on RDNA, MFMA on CDNA), asynchronous copies and
the waits for them. hk-dsl traces the kernel into its own MLIR dialect and runs
its passes there. It then translates the result to
[HipKittens](https://github.com/HazyResearch/HipKittens) C++, compiles that
with ROCm's hipcc, and calls it from torch.

```
Python kernel -> hk MLIR dialect (passes) -> HipKittens C++ -> hipcc -> code object -> torch
```

## The one unique feature: build-time gates

A kernel written at this level can build, run and return wrong numbers, with
nothing to say so. hk-dsl catches these errors at build time. Several checks
run on the final ISA, because that is where the compiler can still break a
kernel:

- **Register spills.** A spilling build is an error (`hk.ResourceError`), not a
  warning, for every instantiation of the kernel
  ([the resource gates](runtime.md#the-resource-gates)).
- **LDS reads before their wait.** The passes derive every wait and prove that
  each asynchronously loaded tile is waited for before anything reads it. The
  code object is checked again after the compiler has scheduled it
  ([passes](ir.md#passes), [`cocheck`](runtime.md#any-code-object-cocheck)).
- **MFMA hazards (CDNA).** A read of an MFMA result too few wait states after
  the MFMA, on any path through the control flow. The gate exists because
  hipcc's LLVM missed such a hazard across a branch and computed wrong values
  ([ROCm/llvm-project#5153](https://github.com/ROCm/llvm-project/issues/5153)).
  By default the gate adds the missing padding and checks again;
  `HK_MFMA_GATE=refuse` refuses the kernel instead.
- **EXEC (CDNA).** A warning for an MFMA that the compiler moved into a
  lane-divergent branch.
- **Schedules are data.** A GEMM or attention schedule is checked for legality
  before anything compiles, and the check gives its reasons
  ([schedules](schedules.md)).

`python3 -m hk.runtime.cocheck` runs the code-object checks on any AMDGPU code
object: hk's, HipKittens', Triton's or FlyDSL's.

Writing at this level is also what creates most of these bugs. Most of the
silent errors the gates have caught were in hk's own kernels or in its
HipKittens fork. The gates are the price of tile-level control.

## Three layers

1. **The language** (`hk`, `hk.templates`, `hk.check`, `hk.profile`,
   `hk.schedules`) is general: tile types, MMA, data movement, schedules as
   data, and the gates. New kernels are written with its general primitives;
   when a kernel needs something the language lacks, the language is extended,
   not given a one-off helper ([CONTRIBUTING](https://github.com/vivienfanghuagood/hk-dsl/blob/main/CONTRIBUTING.md)).
2. **Kernels** (`hk.ops`) are written in the language. They show that the
   language works, and they are how its gains reach a model. They are not the
   product's identity.
3. **Engine routes** (`hk.integration`) are thin adapters that send some of
   vLLM's or SGLang's calls to `hk.ops`. A route is on by default only on a GPU
   where it was checked against the engine's own output with real weights.

**The routes are on by default.** When hk-dsl is installed next to vLLM or
SGLang, their plugin loaders apply hk's routes for your GPU, so the next
`vllm serve` runs some layers on hk's kernels. SGLang-Diffusion needs one
`python3 -m hk.integration sglang-diffusion install` first. On an MI355X
(gfx950) the vLLM routes on by default are DeepSeek-V4.1's MXFP8 linear
layers, its a8w4 MoE, its sparse MLA prefill (`HK_VLLM_SPARSE_MLA=0` turns it
off; inactive while `VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, where vLLM runs
AITER's sparse MLA instead), its index layers under `--monokernel`, and AWQ
linear layers. Every GPU's list is in
[integration](integration.md#on-by-default-and-turning-it-off).

```bash
python3 -m hk.integration --status   # every route: on or off, and why
HK_DISABLE=1 vllm serve ...          # turns every route off
```

If the installed engine is not the build the routes were checked against, hk
patches nothing and prints one warning line. The details are in
[integration](integration.md#on-by-default-and-turning-it-off).

## What it is not

- **Not an inference engine,** and not an engine-routing framework. The routes
  are thin adapters over vLLM and SGLang.
- **Not an agentic or self-optimizing system.** It is a language and its
  tools.
- **Not a faster code generator than the C++ it emits.** Its ceiling is
  HipKittens' schedules and the LLVM that hipcc ships. It cannot choose another
  LLVM. Expect the speed of well-written HipKittens C++, not more.
- **Not cheaper to write than Triton.** In our own tests the same kernel took
  several times the code and the effort in hk-dsl.

## How it compares

| | what it is | where it fits |
|---|---|---|
| [Triton](https://github.com/triton-lang/triton) | Block-level Python. The compiler manages layouts, LDS and waits (Gluon exposes layouts). Targets CDNA and RDNA3/RDNA4. | Much cheaper to write. Start here unless you need control of tiles, LDS and waits. |
| [FlyDSL](https://github.com/ROCm/FlyDSL) | AMD's MLIR-based Python DSL, with layout algebra and instruction-level control, built with its own LLVM. CDNA first; its official targets are MI300X/MI308X, MI350/MI355X and the Radeon AI PRO R9700. | The level of control closest to hk's. AITER and vLLM use its kernels. |
| [AITER](https://github.com/ROCm/aiter) | AMD's kernel library: asm, CK, Triton and FlyDSL kernels behind one API, turned on in vLLM with `VLLM_ROCM_USE_AITER=1`. | The tuned reference on Instinct GPUs. Its README marks RDNA support as experimental. |
| **hk-dsl** | Tile- and instruction-level Python, emitted as readable HipKittens C++ for hipcc. | Kernels that need tile-level control, checked by the build-time gates. RDNA3 and RDNA4 are first-class targets. |

## Performance

We quote only end-to-end serving numbers from the
[scorecard](performance/scorecard.md):

- **Instinct MI355X:** DeepSeek-V4.1-Flash on vLLM (TP2, real weights). Stock
  vLLM already uses AITER's tuned paths. hk's default routes on top of it give
  **about +5% on 8k-token prefill**. Decode is unchanged.
- **Radeon (RDNA3/RDNA4):** end-to-end results will come with the RDNA
  scorecard, which first sets fair baselines.

Three admissions explain why we quote nothing else:

1. **Older kernel-level comparisons used some weak baselines.** One opponent
   ran with a tuning table made for a different routing distribution. One
   comparison bypassed the opponent's tuned entry point, one left out work that
   hk itself needed, and one used an older library version. They are in the [development record](history/README.md) and the
   older [performance page](performance.md). Against fixed baselines, most of
   hk's kernel-level leads on gfx950 are within noise.
2. **On Radeon, much of what hk's routes gained came from the engines' weak
   paths there,** not from better code generation. A competent kernel in any
   language would have taken much of it, and the engines are closing these
   gaps.
3. **The language grew kernel-specific C++ helpers for its hardest kernels.**
   The DSL-surface work folds them into general primitives
   ([migrating](migrating.md)).

## Supported hardware and toolchains

<!-- hw-table:begin -- generated by scripts/gen_hardware_docs.py from python/hk/target/*.py and switches.py; edit those and rerun it -->
| GPU | arch | level | engine routes on by default |
|---|---|---|---|
| Radeon PRO W7900, Radeon PRO W7900D, Radeon RX 7900 | gfx1100 (RDNA3) | tested | [11](supported-hardware.md#engine-routes-on-each) |
| Radeon AI PRO R9700, Radeon RX 9070 | gfx1201 (RDNA4) | tested | [3](supported-hardware.md#engine-routes-on-each) |
| Instinct MI300X, Instinct MI325X | gfx942 (CDNA3) | compile-only | [none](supported-hardware.md#engine-routes-on-each) |
| Instinct MI355X, Instinct MI350X | gfx950 (CDNA4) | tested | [5](supported-hardware.md#engine-routes-on-each) |

Tested: measured on the part and run by CI's GPU tier. What the language and each engine route do on each GPU: [supported hardware](supported-hardware.md).
<!-- hw-table:end -->

- **Compilers (hipcc's clang):**
  - clang 22 (ROCm 7.2.x) is what CI's GPU tier runs.
  - clang 23 (ROCm 10.0, vLLM's ROCm image) is tested on gfx950.
  - clang 24 (ROCm 10.1) needs main after 0.8.0; 0.8.0 fails to compile
    there. **Known issue:** on gfx1100, clang 24 allocates more registers, so
    some kernels spill and the gate refuses them. These are the 16-bit norms,
    rope, and examples 02 and 03
    ([#140](https://github.com/vivienfanghuagood/hk-dsl/pull/140)).
- **Python:** wheels for CPython 3.10 and 3.12 (Linux x86_64, glibc 2.35 or
  newer). Other versions build from source.

Details: [getting started](getting-started.md#install),
[integration](integration.md#on-by-default-and-turning-it-off).

## Who it is for

- **AMD GPU kernel engineers** who would otherwise write HipKittens or CK C++
  by hand, and who want the gates, derived waits and checked schedules that C++
  and Triton do not give.
- **Radeon deployments of vLLM and SGLang,** where the engines' own RDNA paths
  are thin and hk's kernels can be plugged in.
