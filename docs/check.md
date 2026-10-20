# Checking a kernel: `hk.check`

One call answers the three questions a kernel's author -- a person or a coding
agent -- asks after every edit: what does it cost (registers, occupancy, LDS,
spills), is it right (against a torch reference), and is it fast (against
torch eager and `torch.compile`, interleaved in one process).

```bash
python -m hk.check my_kernel.py:gelu_mul --ref "F.gelu(a, approximate='tanh') * b" \
    --inputs "bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]" --json out.json
```

```python
import hk
report = hk.check(gelu_mul, reference, inputs)   # the same, from Python
print(report)                                    # the summary below
report.json()                                    # the JSON below, as a dict
```

- [The loop](#the-loop)
- [What it checks](#what-it-checks)
- [Telling it what to check](#telling-it-what-to-check)
- [Reading the report](#reading-the-report)
- [Without a GPU, and the other GPU](#without-a-gpu-and-the-other-gpu)
- [Rules it keeps, and its limits](#rules-it-keeps-and-its-limits)

## The loop

Write the kernel in a file with its reference and its inputs:

```python
# gelu_mul.py
import torch
import torch.nn.functional as F
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def gelu_mul(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    x = a[idx].load(hk.RT(hk.fp32, 16, 64))
    x.gelu(into=x)                       # hk.math.gelu is the tanh form
    x *= b[idx].load(hk.RT(hk.fp32, 16, 64))
    o[idx] = x.to(hk.bf16)

def reference(a, b):                     # takes what the kernel reads, returns what it writes
    return F.gelu(a, approximate="tanh") * b

SHAPES = [(16, 14336), (4096, 14336)]    # decode- and prefill-sized

def inputs(device, shape=(4096, 14336)):
    a = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    return a, torch.randn_like(a), torch.empty_like(a)
```

then run it after every change:

```bash
python -m hk.check gelu_mul.py:gelu_mul            # reference, inputs and SHAPES from the file
```

and act on what it says: a `FAIL` names the first wrong element; an
occupancy below the card's maximum names what binds it; a `loss` against
`torch.compile` at a small fraction of the DRAM peak says the kernel is not
moving bytes fast enough (a streaming kernel wants register strips or a
template, not 16x64 tiles); the `hint:` lines say which page covers it.
Exit status 0 is pass, 1 a wrong answer, 2 an error, so a script or an agent
can branch on it; `--json -` puts the report on stdout.

The same kernel, measured on an MI355X (gfx950, ROCm 7.2.3, torch 2.10) at
4096 x 14336:

```text
hk.check gelu_mul  gfx950 (AMD Radeon Graphics)  [gpu]  hk 0.4.0  torch 2.10.0+rocm7.2.3.git1a270074
  build  gelu_mul: warps=1 lds=0 B  0.0 s (from the disk cache)
         vgpr=35 (charged 40, headroom 29)  occupancy=8/8 (regs 8, lds 8)  scratch=0 spill=0
  case 0: bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]
    correct ok   out0 bfloat16: max |err| 0.0313 (0.002 of max|ref|), mean 0.000447; allowed 0.0155 + 0.016*|ref|
            eager's own max |err| 0.0564
            compile's own max |err| 0.0312
    time    hk 114.4 us ±0.9%   eager 102.2 us ±0.6%   compile 62.0 us ±1.2%
    speed   hk is 0.90x eager (loss, band 2%)   0.54x compile (loss, band 2%)
    rate    hk 3081 GB/s = 39% of DRAM peak (inputs + outputs once)
    first   hk 0.05 s   eager 0.00 s   compile 1.14 s  (first call: trace, build or cache load, launch)
  hint: case 0: 1.84x slower than torch.compile (...); at 39% of the DRAM peak a streaming
        kernel is not moving bytes fast enough: start from hk.templates (docs/templates/), or
        put rows on register strips (hk.RS, docs/language.md#register-strips) with 16-byte runs
PASS
```

Correct, fully occupied and half the speed of `torch.compile`: the tile
kernel the first page of the docs teaches is the wrong shape for a streaming
op on this card ([the goal](design/dx-goal.md) measured the same, 0.61x).
`hk.ops.silu_mul`, the same pass on strips, measured 56.8 us and a tie with
`torch.compile` in the same harness.

## What it checks

1. **Build** (no GPU). Every kernel instantiation the call launches is
   traced and compiled by hipcc, and its resources reported: VGPRs (and
   AGPRs on CDNA), the count occupancy is charged on and the headroom to the
   next step down, occupancy by registers and by LDS and the smaller of the
   two, LDS bytes, scratch and spills, and the passes' warnings. A spilling
   build is an error, as it is everywhere in hk ([the resource
   gates](runtime.md#the-resource-gates)). A trace error (`hk.VerifyError`,
   a `TypeError` from the language) and a hipcc error are reported with
   their message and the line they point at, and stop there.
2. **Correct** (a GPU). The outputs against the reference at a tolerance
   for the output's dtype ([below](#rules-it-keeps-and-its-limits)): max
   and mean absolute error, the error relative to the reference's largest
   magnitude, the count of mismatches and the first one's index, got and
   want. torch eager's and `torch.compile`'s own error against the same
   reference is printed beside it, so a tolerance can be judged.
3. **Time** (a GPU). The kernel, torch eager (the reference itself) and
   `torch.compile(reference)`, plus any `--baseline`, in one process:
   warmup, then rounds that rotate the contenders, each round a burst of
   back-to-back calls filling about a millisecond. Each contender's median
   over the rounds and its noise (median absolute deviation over the
   median), the ratio to each baseline (the median of the per-round
   ratios; above 1 hk is faster) with a verdict inside a noise band, the
   bandwidth (inputs and outputs counted once), and the first call of each
   timed on its own (trace and hipcc, or `torch.compile`'s compile).

## Telling it what to check

**What is checked** (`TARGET`) is one of:

| form | example |
|---|---|
| a raw `@hk.kernel` | `my.py:gelu_mul` -- the outputs are the tensors it stores into |
| a function that calls hk kernels | `my.py:gelu_mul_fn` -- returns the output(s) |
| a shipped op or a template's product | `hk.ops.silu_mul`, `hk.ops:rmsnorm` |
| a file | `my.py` -- its `CHECK` dict, else its `kernel` or `fn` |

For a raw kernel the reference takes the tensors the kernel only reads and
returns the ones it writes, in order (`TracedKernel.written_tensors`).
Constexprs are `--const NAME=VALUE`. A function's reference takes the same
arguments the function does. A function that writes into an argument
instead of returning names it: `--outputs 2` (outputs are NaN-filled before
the call and not given to the reference). An argument updated in place --
read, then written, as `fused_add_rms_norm`'s `x` and `residual` -- is
`--inplace NAME` (or an index): the reference gets it as it was and returns
its new value, in argument order with the outputs.

**The reference** (`--ref`): `FILE.py:name`, `module:name`, a `lambda`, or an
expression over the reference's argument names -- the target's own parameter
names -- with `torch`, `F` (`torch.nn.functional`), `math` and `hk` in scope:
`--ref "F.silu(a) * b"`. Default: the file's `reference`.

**The inputs** (`--inputs`, one per case, repeat it for several shapes): a
generator `FILE.py:name` called as `gen(device)` or `gen(device=, shape=)`,
or a spec:

```text
bf16[4096,14336]                 torch.randn of that shape, as bf16
fp32[64]@rand                    uniform [0, 1)
i32[64]@randint(0,128)           integers in [0, 128); ints default to randint(0,100)
bf16[16,64]@zeros  @ones  @full(2.5)  @arange  @empty
bf16[4096,4096]@randn*0.1        scaled
1e-6                             a scalar argument
```

Default: the file's `inputs`, once per entry of its `SHAPES` if it has them.

**More contenders**: `--baseline NAME=REF` (`--baseline ops=hk.ops:silu_mul`);
in Python, `extra={"name": callable}`. They take the reference's arguments,
and are checked against the reference too, so a baseline that computes
something else shows it. `--no-eager`, `--no-compile`, `--compile-mode
max-autotune-no-cudagraphs`.

In Python every option is a keyword of `hk.check(fn, reference, inputs, ...)`
([the API](api/hk.check.md)); `cases=[...]` takes several input generators,
and `report.raise_for_status()` makes it a test.

**Tuning a constexpr**: `--sweep CW=512,1024,2048` (repeatable; every
combination) builds, checks and times a raw kernel at each value beside the
default, as `hk[CW=1024]`, and names the fastest (`best`). A value that does
not build -- a strip width that is not whole chunks on this arch, a tile
that spills -- is a note, not an error; a wrong one is reported and not
timed. On an MI355X, llms.txt's strip kernel at 4096x14336: `CW=512` 61.3
us, 1024 64.6, 2048 72.0 (`CW=256` is not a whole 64-lane chunk there).

## Reading the report

`--json` writes (abridged; this is the run above):

```json
{
 "schema": 1, "name": "gelu_mul", "status": "pass", "arch": "gfx950", "mode": "gpu",
 "device": "AMD Radeon Graphics",
 "versions": {"python": "3.12.3", "hk": "0.4.0", "torch": "2.10.0+rocm7.2.3.git1a270074",
              "hip": "7.2.53211", "triton": "3.6.0+rocm7.2.3.git4ed88892"},
 "builds": [{"kernel": "gelu_mul", "consts": {}, "arch": "gfx950", "warps": 1, "threads": 64,
             "lds_bytes": 0, "warnings": [], "build_s": 0.045, "cached": true,
             "source": ".../hk-cache/f43225c4.../kernel.cpp",
             "instantiations": [{"name": "gelu_mul_kernel(globals)", "vgpr": 35, "agpr": 0,
                                 "sgpr": 27, "vgpr_charged": 40, "vgpr_headroom": 29, "lds": 0,
                                 "scratch": 0, "spill": 0, "occupancy_by_regs": 8,
                                 "occupancy_by_lds": 8, "occupancy": 8,
                                 "max_waves_per_simd": 8}]}],
 "cases": [{"case": 0, "inputs": "bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]",
            "first_call_s": {"hk": 0.0515, "eager": 0.0002, "compile": 1.1398},
            "correct": {"ok": true,
                        "outputs": [{"dtype": "bfloat16", "shape": [4096, 14336],
                                     "numel": 58720256, "ok": true, "rtol": 0.016,
                                     "atol": 0.01547, "scale": 15.47, "max_abs": 0.03125,
                                     "mean_abs": 0.000447, "max_rel": 0.00202,
                                     "mismatches": 0, "nan": 0}],
                        "baselines": {"eager": {"max_abs": 0.0564, "ok": true},
                                      "compile": {"max_abs": 0.0312, "ok": true}}},
            "time": {"rounds": 15, "calls_per_round": {"hk": 9, "eager": 10, "compile": 13},
                     "bytes": 352321536, "flops": null,
                     "ms": {"hk": {"median": 0.11436, "min": 0.11286, "noise": 0.0089,
                                   "gb_s": 3080.7},
                            "eager": {"median": 0.10215, "min": 0.10135, "noise": 0.0055,
                                      "gb_s": 3449.0},
                            "compile": {"median": 0.06196, "min": 0.06111, "noise": 0.012,
                                        "gb_s": 5686.0}},
                     "vs": {"eager": {"x": 0.8954, "band": 0.02, "verdict": "loss"},
                            "compile": {"x": 0.544, "band": 0.0209, "verdict": "loss"}},
                     "of_dram_peak": 0.385}}],
 "errors": [], "notes": [], "hints": ["case 0: 1.84x slower than torch.compile ..."]
}
```

| key | meaning |
|---|---|
| `status` | `pass` (correct at every case), `fail` (a wrong answer), `error` (did not trace, build or run: see `errors`), `built` (no GPU: the build is all there is) |
| `mode` | `gpu`, `build-only` (no GPU, or a GPU of another arch) or `trace-only` (no hipcc) |
| `builds[]` | one per kernel and constexpr set the call launched; `cached` is true when hipcc did not run (the disk cache had it); `build_s` is null when the first call built it (then it is in `first_call_s`); `source` is the generated C++ |
| `instantiations[]` | the compiler's numbers: `occupancy` is waves per SIMD, the smaller of `occupancy_by_regs` and `occupancy_by_lds`, out of `max_waves_per_simd`; `vgpr_charged` is what occupancy is charged on (rounded to the granule, AGPRs included on CDNA); `vgpr_headroom` is how many more VGPRs fit before it drops a step |
| `correct.outputs[]` | per output: `ok` when every element has `abs(got - want) <= atol + rtol * abs(want)`; `scale` is max abs(want), `max_rel` is max_abs / scale; `first_mismatch` is `{index, got, want}`; `nan` counts NaNs the reference does not have |
| `correct.baselines` | torch eager's and torch.compile's own error against the same reference |
| `time.ms` | per contender: `median` and `min` ms a call over the rounds, `noise` the rounds' spread, `gb_s` bytes / median |
| `time.vs` | per baseline: `x` = baseline time / hk time (above 1 hk is faster), `band`, `verdict` |
| `errors[]` | `{stage, type, message, where}`; stage is `trace`, `build`, `inputs`, `reference`, `run` or `time` |
| `hints[]` | what to try next, read off the numbers (heuristics; the timing is the verdict) |

## Without a GPU, and the other GPU

With no GPU, `hk.check` traces and builds and stops: the resources need
hipcc, not a device (`mode: build-only`, `status: built`). A raw kernel needs
only its constexprs; a function is run once on CPU tensors with launching
switched off, to see which kernels and constexprs it would launch. Without
hipcc either, it traces (`mode: trace-only`): the warps, the LDS bytes and
the passes' warnings.

`--arch gfx1100,gfx950` reports each arch, one process each with `HK_ARCH`
set: the arch of the GPU in the machine is checked and timed, the other is
built. `--arch` naming only another arch builds for it. So on a gfx950 box:

```bash
python -m hk.check gelu_mul.py:gelu_mul --arch gfx1100,gfx950 --json both.json
```

gives the gfx1100 build's register count and occupancy (92 VGPRs,
occupancy 16/16 on the W7900D's wave32 SIMDs) beside the gfx950 run. Per-arch
facts that change a kernel's plan are in [hk.target](api/hk.target.md).

## Rules it keeps, and its limits

- **The reference runs in fp32** on fp32 copies of the 16-bit inputs
  (`--ref-precision native` to run it as given), and the tolerance is the
  output dtype's, about two units in the last place: bf16 `rtol 1.6e-2`,
  `atol 1e-3 x max|ref|`; fp16 `2e-3`, `1e-4 x`; fp32 `1e-4`, `1e-5 x`;
  integers exact (`hk.check.TOLERANCES`). `--rtol`/`--atol` override
  (`--atol` is then absolute), `--exact` asks for bit equality. A tolerance
  is where a wrong kernel hides: when the kernel's error is larger than
  torch eager's own, find out why before loosening it, and keep one case with
  an exact answer in the kernel's tests ([for agents](agents.md#numbers-not-to-trust)).
- **A raw kernel's outputs are filled with NaN** before it runs, so an
  element it never writes is a mismatch, not a stale value that happens to
  match.
- **Timing is in one process, interleaved**, the only comparison this
  project trusts: across processes a kernel A/B manufactured 13-16% out of
  nothing ([phase 5](history/04-phase-5-autotune.md)). A verdict is `tie`
  inside a band of max(2%, the noise), and `noise` when the band passes 15%.
  Back-to-back calls find a footprint smaller than the cache (96 MB on a
  W7900D, 256 MB on an MI355X) resident, for every contender alike; the
  `rate` line flags a rate past the card's ceiling.
- **The bandwidth counts every input and output once.** It is what a
  streaming kernel's roofline is drawn from, not what a GEMM or attention
  moves; pass `--bytes`/`--flops` for those.
- **First calls** include the build: hipcc (seconds) or the disk cache's
  load (milliseconds), and `torch.compile`'s compile. A kernel built in this
  process before `hk.check` ran reports its load, not its build.
- **What it can see**: kernels launched through `Kernel.__call__`,
  `Kernel._launcher` or built through `Kernel.build` during the call --
  every `@hk.kernel`, every `hk.ops` op written in the DSL, and the
  templates (checked against `hk.templates.rowwise` and
  `hk.templates.gemm(epilogue=...)` on their branches: a rowwise RMSNorm's
  two plans, a GEMM's split-K kernel, its reduction carrying the epilogue and
  its pingpong). A launcher a template cached before `hk.check` ran is not
  seen again: check in a fresh process. The HipKittens C++ kernels wrapped by
  `hk.ops.hipkittens` are launched otherwise and have no build row. A
  function's no-GPU discovery runs it on CPU tensors, so a function that
  needs a GPU tensor to decide what to launch is reported as not
  discoverable; check its kernel directly.
- `hk.__version__` of a checkout is its last release's; the report's
  `versions` carries `hk_commit` where git can see the checkout, and
  `hk_src` (a hash of the package's Python sources) everywhere, so two
  reports from the same code say so.
