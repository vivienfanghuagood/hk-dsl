# The runtime: compile, cache, gate, tune, launch

Everything after tracing: finding the GPU and the compiler, the
content-addressed cache, the resource gate that refuses a spilling kernel and
the same gates for any code object, filling the cache ahead of time, schedule
search, and the `torch.ops` boundary a framework calls. The code is `python/hk/runtime/` and
`python/hk/autotune/`; their module docstrings are the source for this page.

- [What a launch does](#what-a-launch-does)
- [Which GPU](#which-gpu)
- [Compiling](#compiling)
- [The cache](#the-cache)
- [The resource gates](#the-resource-gates)
- [Warming the cache](#warming-the-cache)
- [Autotune](#autotune)
- [Torch ops and the SDPA drop-in](#torch-ops-and-the-sdpa-drop-in)
- [Environment variables](#environment-variables)

## What a launch does

`k(*tensors)` traces, verifies, translates, compiles (once) and launches. Each
step is cached: the trace per set of constexprs, the build by its content
hash, the loaded module by path, and the launcher on the constexpr keyword
arguments as passed — so after the first call everything before the launch
is a dict lookup (`python/hk/dsl/kernel.py`, `python/hk/runtime/module.py`).
That matters at these sizes: a 4096x4096 quantize is 60 µs of GPU, so ten
microseconds of Python in front of it is sixteen percent. The host-overhead
work is in [Phases 1 and 2](history/01-phases-1-2.md).

**The stream.** On CDNA (gfx942, gfx950) the pybind entry point takes the
stream to launch on, and `Kernel.__call__` passes torch's current stream for
the first tensor's device (`scaffold.takes_stream`). So every hk op there --
`hk.ops.gemm`'s gfx950 routes, the norms, the decode GEMV -- is ordered
against torch's work and is captured by a CUDA graph like any torch op;
`tests/hk/gpu/test_stream_cdna.py` captures and replays them. It costs
0.2 µs a call on the host (2.97 → 3.18 µs for a tiny kernel, eager). The
RDNA targets' pybind entry point still launches on the null stream, as their
generated source always has; their serving paths go through
[`torch_op`](#torch-ops-and-the-sdpa-drop-in), which takes torch's stream.

What that means on gfx1100 when you call `k(...)` yourself: eager calls on
torch's default stream run and time correctly, but a CUDA graph does not
capture the kernel (a replay runs nothing, and a timing of it measures
nothing), and work on another stream is not ordered against it. Nothing
reports either. To capture or time an hk kernel in a graph on gfx1100, or to
run it on a stream of your own, set `HK_LAUNCH=codeobj` before its first
call -- every `@hk.kernel`, and so every template, then launches its
[device-only code object](#over-a-device-only-code-object) on torch's current
stream -- or call it through `hk.torch_op(k)`. The DX suite's runner does the
former on RDNA (`bench/dx/run.py`).

A compiled `.so` is never overwritten in place: every build writes a new
content-hashed path, because rebuilding a `.so` while a process has it mapped
corrupts that process's view of the code (`python/hk/runtime/module.py`).

## Which GPU

A kernel that names no `arch` is built for `hk.target.current_arch()`:
`$HK_ARCH`, else the GPU in the machine (`hk.runtime.arch.detect`), else
gfx1100. It is resolved on first use, so importing `hk.ops` still builds no
IR and probes no GPU ([gfx1201](history/13-gfx1201.md)).

`detect()` tries, in order:
1. `HK_ARCH`;
2. `HCC_AMDGPU_TARGET` or `PYTORCH_ROCM_ARCH`, only when it names exactly
   one arch;
3. the first GPU node of the KFD topology in sysfs
   (`/sys/class/kfd/kfd/topology/nodes/*/properties`), which creates no HIP
   context, so an engine plugin can decide in every process;
4. the device torch sees, then `rocminfo`;
5. the first entry of a `HCC_AMDGPU_TARGET` / `PYTORCH_ROCM_ARCH` list, as a
   last resort.

A list says what a build targets, not which GPU is here: vLLM's ROCm image
sets `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`, and until 0.7.0, which
read its first entry second, an MI355X was taken for gfx90a and hk built for
gfx1100 (#54). `HK_ARCH` is the only way to build for a part that is not in
the machine, which is how the gfx1201 and gfx942 kernels get their compile
gate on a gfx1100 box. The gfx1100 fallback exists so that a box with no GPU
gets the source a W7900 would; a detected GPU that is not one of the targets
(`hk.target.TARGETS`: gfx1100, gfx1201, gfx942, gfx950) falls back the same
way (`python/hk/target/__init__.py`).

Only launch on the arch you built for. A code object for another arch does
not fail cleanly: a gfx1100 kernel launched on a gfx950 crashed the process.
So before a kernel is loaded, hk compares its arch with the device's and
raises `hk.runtime.arch.ArchMismatchError` if they differ; the check costs
nothing after the first call per device (`python/hk/runtime/arch.py`).

## Compiling

The backend is the stock hipcc compiling the translated C++ against
HipKittens' `kittens.cuh`, so nothing at run time needs LLVM. `kittens.cuh`
picks `include/rdna3`, `cdna3` or `cdna4` by the arch it is compiled for; on
CDNA, hk's `include/hk/cdna.cuh` adds what that tree lacks (the target's
`extra_includes`).

hipcc is `$HK_HIPCC`, else the one on `PATH`, else `/opt/rocm/bin/hipcc`. The
headers come from the wheel's `hk/include`, or in a checkout from `include/`
and the `third_party/HipKittens` submodule; `HK_INCLUDE_DIR` overrides them
with a path list. `ROCM_PATH` is used for `include/hip` when set, else it is
found by walking up from hipcc, then from the clang hipcc runs (the
`InstalledDir` of `hipcc --version`: TheRock's pip layout, as in
`rocm/pytorch` 10.x, has a hipcc launcher in `/usr/local/bin` and the toolchain
under `site-packages/_rocm_sdk_devel`). The LLVM tools hk reads code objects
with (`llvm-objdump`, `llvm-readelf`, `clang-offload-bundler`, ...) are taken
from beside that clang (`HIP_CLANG_PATH` when set), then `ROCM_PATH`'s
`llvm/bin`, then `PATH`; a missing one is an error that says where it looked.
The flags include `--offload-arch=<arch>`,
`-std=c++20 -O3`, `-shared -fPIC`, and
`-Rpass-analysis=kernel-resource-usage`, which the gate depends on; the full
list is `flags()` in `python/hk/runtime/compile.py`. The gate reads the
remarks in both shapes compilers print them: clang's diagnostics
(`k.cpp:3:1: remark: VGPRs: 2 [-Rpass-analysis=kernel-resource-usage]`, clang 22
and 23, and clang 24's `--genco`), and clang 24's bare lines for a `-shared`
module, whose device code its new offload driver generates at link time
(`k.cpp:3:0:     VGPRs: 2`).

A cold compile takes about 4-7 s as a pybind module and about 1 s as a code
object, attention included; the libtorch op takes about 15 s
([the table](#over-a-device-only-code-object)).

## The cache

Key = hipcc version + the full flag list (and what hipcc appends from
HIPCC_COMPILE_FLAGS_APPEND and the like) + the dialect version + the generated
source (torch's and pybind11's versions too, when their headers are on the
include path) + a content hash of **every header that kernel's compile reads**.
The header list is the compiler's own: on a miss, a preprocessor pass per side
(device, and host for a `.so`) writes clang's dependency file, and the list is
kept as `deps/<part of the key without headers>` beside the entries. A lookup
reads that list and hashes the files again (once a process; a file re-read only
if its stat changes), so an edit to any header the kernel includes, however
deep, is a miss, and an edit to one it does not include is not. ROCm's, Python's,
torch's and pybind11's own headers are keyed by those versions rather than read.
A build lands in `~/.cache/hk/<key>/` via a rename, so a killed compile never
leaves an importable half-written `.so`; a header that changes while it compiles
is noticed and the build redone. Failed builds are kept as `failed-<key>/` with
their source and log.

Measured on the warm path (20 cached gfx950 kernels, one process, interleaved
with the previous key, two runs): the key costs 0.27 ms a kernel against 0.13
(+0.14-0.18 ms; ~70 headers to stat for a CDNA kernel), and +0.7-0.8 ms over
all 20 on a process's first pass, against a whole cache hit of 1.1-2.6 ms.

The thresholds (`max_vgprs`, `min_occupancy`) are *not* in the key — they are
re-applied on every cache hit, so a kernel with a strict occupancy requirement
is not let through by whoever happened to compile the same bytes first.

The directory is `$HK_CACHE_DIR`, else `$XDG_CACHE_HOME/hk`, else
`~/.cache/hk`. Entries are never invalidated, only added: a key that hashes
everything does not need eviction for correctness, and the directory is one
you can delete. The header hash is the part that matters most —
editing a header changes the kernel while leaving the generated source
byte-identical; through 0.8.0 the key hashed a fixed set of files
(kittens.cuh, pyutils/, the arch tree, include/hk/cdna.cuh) and an edit to
`include/hk/cdna_agpr.cuh` or `include/pyutils/hk_bind.cuh` served the old
kernel. The dialect version is the dialect's semantic version, not a
hash of the library, so a rebuild that changes nothing it means does not throw
the cache away (`python/hk/runtime/compile.py`).

## The resource gates

**A VGPR spill is silent corruption, not slowness**, because this library
hand-places `s_waitcnt` and scratch traffic reorders against it.
`k.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
return* a spilling kernel.

The gate is `python/hk/runtime/resources.py`. What it enforces:

* **No spill, ever.** A memory spill -- any `ScratchSize`, or a VGPR spill --
  raises `hk.ResourceError`, and nothing importable is left behind. The
  generated source is kept beside the failure for inspection. SGPR spills to
  VGPR lanes (ScratchSize 0) are no memory traffic and pass. One exception
  (#68): a kernel with ScratchSize > 0 but no VGPR spill passes when its code
  object never touches the frame -- no `scratch_*` instruction, no `flat_*`
  memory instruction, no flat scratch, no private segment buffer
  (`hk.runtime.cocheck.unused_scratch_frame`). The compiler can reserve such a
  frame and never use it (K1's indexer build under ROCm 10: 36 bytes). The
  build then records `scratch_note = "reserved-unused scratch frame"`, in its
  report and on stderr. A code object that cannot be read gets no verdict and
  is refused.
* **Every instantiation**, not just the one being benchmarked — reading only
  the shipped shape's line is how a regression hides in a build nobody times.
* **`max_vgprs` and `min_occupancy`**, when the kernel asks for them
  (`@hk.kernel(max_vgprs=..., min_occupancy=...)`), so a change that quietly
  costs a wave fails the build instead of the benchmark. Occupancy is charged
  on the register count rounded up to the target's granule — 24 on gfx1100 —
  not on the raw count. On CDNA (gfx90a and later) the AGPRs share the
  VGPRs' file: the charge is the VGPRs rounded up to a multiple of four plus
  the AGPRs, then rounded to the granule, 8 there
  (`Target.charged_vgprs`). The report prints both (`vgpr=20+16a`).
* **No remarks is an error.** A build that emits no resource remarks cannot
  be certified, so it is fatal rather than skipped.
* **MFMA hazards (CDNA).** The hardware does not interlock on an MFMA's
  result: whatever reads or overwrites it must come enough wait states after
  it, and LLVM pads with `s_nop`. Before llvm/llvm-project#225989 (ROCm 7.2.3,
  10.0 and 10.1 included) it measured the wrong path into a block with two
  predecessors and padded too little; gfx950's paged decode read its scores 2
  wait states after the QK MFMA where 8 are needed and computed some wrong
  (ROCm/llvm-project#5153). The gate (`hk.runtime.mfmahaz`) walks the code
  object's control flow backwards from every instruction that touches an
  MFMA's registers, shortest path first, against LLVM's windows. A kernel
  short anywhere is rebuilt through hipcc's own pipeline with the padding the
  fixed compiler would emit added to its assembly, and checked again (the
  default, `HK_MFMA_GATE=fix`). The build log, the entry's `MFMAHAZ` file and
  `python3 -m hk.integration --status` (on a CDNA GPU) say which kernels were
  padded and where. `HK_MFMA_GATE=refuse` refuses such a kernel instead, naming the
  instruction and the wait states missing; `report` builds it and says so,
  for a sweep; `0` turns the gate off. A compiler that pads the gate's probe
  (a two-path case, through its own `llc`) is not checked. hk's inline-asm
  LDS reads are the one case it leaves out: a load writes its destination
  when its data returns, long after any window.

The gate runs at `k.build()`, which needs hipcc but no GPU: the failure most
likely to bite — generated code that compiles but spills — is caught on a
laptop. `Build.report()` prints one line per instantiation:

```
add_kernel(globals): vgpr=71->72 occ=16 lds=0 scratch=0 spill=0 hdrm=25
```

(`71->72` is the raw and the rounded register count; `hdrm` is the headroom
before the next occupancy step.)

### Any code object: `cocheck`

Neither the resource rule nor the waits need hk's source. Both can be read
off a compiled code object, so `hk.runtime.cocheck` runs them on any AMDGPU
kernel — hk's, HipKittens', Triton's, FlyDSL's:

```bash
python -m hk.runtime.cocheck kernel.co [more.co ...]
python -m hk.runtime.cocheck --bundle lib.so     # a host binary with an embedded HIP fat binary
```

- **Resources**, from the metadata note: scratch or a spill fails; the VGPR,
  AGPR and SGPR counts and the occupancy they allow are reported.
- **MFMA hazards**, from the disassembly over its control flow: a read or an
  overwrite of an MFMA result too few wait states after the MFMA on some path
  (the gate above).
- **Waits**, from the disassembly: an instruction that reads a VGPR that an
  LDS or vector-memory load is still writing fails. The compiler inserts the
  waits for loads it emitted, and nothing does for loads written as inline
  asm. On gfx950, with hk's bind removed, LLVM hoisted the MFMAs above the
  waits and every configuration returned NaN
  ([chapter 29](history/29-cdna.md)).

It exits 1 if any kernel fails a gate. The wait scan is a linear walk,
done twice for loops, so a report can be a false alarm, and a clean pass is
only as good as that walk is for the kernel's control flow. gfx950's copies
into LDS (`buffer_load … lds`) write no register but count on vmcnt, so they
take their place in its queue ([chapter 34](history/34-cdna-attention.md)).

## Warming the cache

`python3 -m hk.runtime.warm -j 32` fills the cache in parallel (19x on 24
kernels), and `python3 -m hk.autotune --aot` does the same for every schedule a
tuning record can reach. Those two are how the serving path ends up with no
hipcc in it.

Measured, 24 norm kernels into an empty `HK_CACHE_DIR` on a 256-core box:
102.9 s at `-j 1` and 5.4 s at `-j 24`. Tracing stays serial — the builder
keeps its state in a module-global stack — and only the hipcc invocations run
in parallel; the default is one per core, capped at 32
(`python/hk/runtime/warm.py`).

A third pass covers the torch ops: `python3 -m hk.integration --warm -j 32`.
A torch op is a device-only code object ([below](#over-a-device-only-code-object)),
a different artifact from the pybind module, so a `--aot` pass built for
pybind does not spare a server its first attention compile. Run both
([integration](integration.md#ahead-of-time-builds)); with
`HK_TORCH_OP=library` the pass builds the libtorch registrations instead.

## Autotune

`hk.autotune` replaces HipKittens' `sweep.sh`. A `Tuner` is a kernel factory,
a schedule space and a record; the tuners live next to the kernels they tune
(`hk.ops.gemm`, `hk.ops.attn`).

```
python3 -m hk.autotune --list
python3 -m hk.autotune --tune gemm_bf16 -j 32      # search
python3 -m hk.autotune --ship                      # promote what is real
python3 -m hk.autotune --aot  -j 32                # prebuild it all
```

`--tune NAME` also takes `--key` (default: the tuner's own keys) and
`--rounds` (default 3); `--ship` takes `--min-gain` and `--dry-run`.

**Stage 1 compiles the whole space and never touches the GPU.** The factory's
own arithmetic refuses a tiling that does not divide (`invalid`), and the
verifier or the resource gate refuses an illegal layout or a spill
(`rejected`); of attention's 32 schedules, 26 never reach the GPU. **Stage 2
times what is left in one process**, interleaved, reversing the order on odd
rounds and scoring each candidate by its minimum, because cross-process A/B on
this chip manufactures 13-16% differences out of nothing.

**The production path never searches.** `Tuner.kernel(key)` is a dict probe
and a memoised factory call. A missing, stale or foreign record costs
performance and never correctness.

**Records.** `--tune` writes to `$HK_TUNE_DIR` (default: `autotune/` inside
the compile cache), with the schedule, its time, the default's time, the arch
and the toolchain fingerprint. Records shipped in the package live in
`python/hk/tuned/`; a record measured on this machine wins over a shipped
one, and a record from another arch is ignored. A record changes nothing
unless it beat the default by `MIN_GAIN` (1%); the bar is applied at `--ship`
and again on lookup, `HK_AUTOTUNE_MIN_GAIN` moves it, and `HK_AUTOTUNE=0`
ignores records entirely. `python/hk/tuned/README.md` says why there are so
few: in [Phase 5](history/04-phase-5-autotune.md) 4 of 16 keys cleared the
bar, and three attention records ship today (`gemm_bf16.json` is empty).

`--aot` prebuilds every schedule production can reach — the recorded ones and
the default, since the default is what an unrecorded key gets.

## Torch ops and the SDPA drop-in

`hk.torch_op(kernel)` builds a kernel as `torch.ops.hk.<name>`, which is the
boundary vLLM and SGLang call:

```python
import hk
op = hk.torch_op(hk.ops.elementwise.KERNELS["add_bf16"])
op(a, b, out)                 # == torch.ops.hk.add_bf16(a, b, out)
```

By default the op runs over a device-only code object
([below](#over-a-device-only-code-object)). Either way:
* The op takes its outputs as arguments. The schema is derived from the IR:
  the tensors the kernel stores into become `Tensor(a!)` out-parameters.
* It launches on torch's current stream and has a `Meta` implementation so
  `torch.compile` can trace it.
* Building it needs no GPU, and it still gates on spills. It is memoised on
  (namespace, name), because torch refuses to register the same op twice.

With `HK_TORCH_OP=library`, the op is a libtorch registration instead:
* It is built with plain `hipcc -shared -fPIC` and loaded with
  `torch.ops.load_library` — not `torch.utils.cpp_extension.load`, which
  would hipify source that is already HIP.
* It registers with `TORCH_LIBRARY_FRAGMENT` (every kernel is its own `.so`
  in one namespace), and building it needs torch's headers.

The reasons for each of these are in [Phase 6](history/05-phase-6-torch-ops.md),
with the measurement that the registration costs nothing.

### Over a device-only code object

Both the pybind module and the libtorch registration compile a host half, and
that is most of a build. `hk.runtime.codeobj` compiles the kernel alone
(`hipcc --offload-device-only --genco`) and does the host half in Python:

* The code object carries a table of the `globals` struct's layout -- each
  tensor's offset, and each of its gl's fields -- emitted by
  `codegen.scaffold.device` and asked of the compiler rather than
  reconstructed. It is read out of the ELF, so no GPU is needed to read it.
* A launch packs the tensors' pointers, extents and row strides at those
  offsets, evaluates the traced grid over the extents, and calls
  `hipModuleLaunchKernel` through the HIP runtime torch already loaded, on
  torch's current stream. So it is captured into a CUDA graph like any other
  launch.
* The torch op over it (`codeobj.torch_op`) has the same schema, the same
  out-parameters and the same tensor checks as the libtorch one.

Same cache, same gate. Cold compile time, one kernel each
(`tools/hk-bench/compile_time.py`; the pod has hipcc 7.2 and torch, the
workstation hipcc 7.1 and no torch):

| kernel | pybind, workstation | code object, workstation | pybind, pod | torch op (libtorch), pod | code object, pod |
|---|---|---|---|---|---|
| weighted RMSNorm | 4.15 s | 0.62 s | 6.80 s | 14.69 s | 1.09 s |
| GEMM 128x128 | 4.24 s | 0.75 s | 7.01 s | 15.05 s | 1.35 s |
| attention d128 | 4.39 s | 0.83 s | 7.14 s | 15.14 s | 1.46 s |
| paged decode walk | 4.48 s | 0.88 s | 7.27 s | 15.20 s | 1.52 s |

`hk.torch_op` registers over the code object by default; inside vLLM's
decode graphs the backend's layer takes the same time either way
(`paged_ab.py --layers 36`, 0.0215 against 0.0214 ms at one request), and
`HK_TORCH_OP=library` builds the libtorch registration instead.
`HK_LAUNCH=codeobj` makes `Kernel.__call__` launch the code object too,
instead of loading the pybind module. That is opt-in because its launch is
Python where the pybind entry point is C++: nothing inside a captured graph,
measurable in front of a microsecond kernel run eagerly.

What an eager launch costs on the host is what an eager server pays per hk
call whenever the CPU, not the GPU, sets the pace. SGLang on RDNA3 decodes
eagerly. The torch op used to cost 27.3 µs of host time per call on the pod:
it packed the argument field by field, built a ctypes buffer per launch and
entered a device context. It now costs 17.8 µs ([chapter
25](history/25-sglang-moe.md)):
- one precompiled `struct` packs the whole argument in one call;
- each thread reuses one argument buffer, since the launch copies the
  argument out before it returns;
- there is no device switch when the device is already current.

**The host fast paths** take it further (`hk.runtime.hostpath`). A call's
checks, grid and argument depend on its tensors' dtypes, device, shapes and
strides; only the pointers change from one call of a shape to the next. So
the op's implementation is one function written out per kernel
(`CodeObjectKernel.launcher`): it checks the dtypes and the device as before,
reads the grid and the argument from a memo of the shapes it has seen, and
writes the pointers into it. A new shape, a wrong dtype or device, or a view
it has not seen takes `tensor_args`, which checks everything and raises the
same errors as before; a shape it accepts is then memoized (up to 4096 a
kernel). Code that is itself eager -- an engine's opaque op -- calls that
implementation without the dispatcher (`codeobj.direct`); hk.ops calls it
through `torch_ext.launcher`, which goes through the op whenever
torch.compile is tracing. The engine routes keep their per-shape decisions
the same way (see [integration](integration.md#host-time-per-call)).

Where hk._mlir has it, that function is C (`hk._mlir._mlir_libs._hkLaunch`,
built with the bindings, plain CPython API, no HIP: it calls
`hipModuleLaunchKernel` through the address `codeobj.hip()` read). It takes
the common call -- every tensor of its dtype, contiguous, on the current GPU
-- reads the tensors as the pybind entry point does and computes the grid
from the traced expressions; anything else it hands to the Python function.

The kernels and their arguments are the same, so outputs are bitwise the
same. `HK_HOST_FASTPATH=0` turns all of it off: every call then takes the
path it took before, line for line, which is also what a same-process A/B
flips (`hostpath.set_enabled`). Host time of one launch of a code-object
kernel (w4a16, one token), µs, each column one process, pointers changing
every call:

| | W7900D | MI355X |
|---|---|---|
| `tensor_args` and `launch`: the op's implementation before | 9.4 | 9.7 |
| the launcher, in Python | 5.4 | 6.6 |
| the launcher, in C (`_hkLaunch`) | 2.9 | 4.6 |
| the pybind entry point (`Kernel.__call__`'s) | 3.4 | 5.3 |
| `hipModuleLaunchKernel` through ctypes, argument pre-packed | 2.5 | 4.2 |

Through `torch.ops.hk.<op>` the dispatcher adds 1-3 µs to any of these.
| the pybind entry point (`Kernel.__call__`'s) | 2.7-4.4 | 4.7 |
| `hipModuleLaunchKernel` through ctypes, argument pre-packed | 2.3-3.0 | 3.9 |

`hk.ops.sdpa` is `F.scaled_dot_product_attention` on the generated kernel
where it fits, and torch where it does not:

```python
import torch.nn.functional as F
from hk.ops import sdpa

sdpa.patch()                 # F.scaled_dot_product_attention -> hk, where it fits
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

`sdpa.unpatch()` reverses it. `hk.attention` raises when a call does not fit;
the drop-in falls back to torch silently, and `HK_SDPA_VERBOSE=1` prints each
distinct reason once. `hk.ops.sdpa.STATS` counts which branch each call took,
because a patch that silently falls back produces identical answers and no
speedup. `sdpa.warm()` and `python3 -m hk.autotune --aot` move the first
call's compile to startup (`python/hk/ops/sdpa.py`). Patching a running
framework with it is [integration](integration.md).

## Environment variables

Read by `python/hk/runtime`, `python/hk/autotune` and `python/hk/ops`. The
ones that turn on engine patches, and the ops' own plan switches
(`HK_SPLITS`, `HK_NORM_LAYOUT`, `HK_NORM_FOLD`, `HK_HIPKITTENS_KERNELS`), are
listed in [integration](integration.md#environment-variables).

| variable | read by | effect |
|---|---|---|
| `HK_ARCH` | `runtime/arch.py`, `target/__init__.py` | the arch to build for; first in the detection order |
| `HCC_AMDGPU_TARGET`, `PYTORCH_ROCM_ARCH` | `runtime/arch.py` | next in the detection order when it names one arch; a list's first entry only after the KFD topology, torch and rocminfo |
| `HK_XCDS` | `dsl/placement.py` | the XCD count workgroup placement deals over (`hk.arch.xcd_count()`), instead of KFD's `num_xcc` |
| `HK_HIPCC` | `runtime/compile.py` | the hipcc to use |
| `ROCM_PATH` | `runtime/compile.py` | where `include/hip` is, and the second place LLVM tools are looked for (`llvm/bin`) |
| `HIP_CLANG_PATH` | `runtime/compile.py` | hipcc's own: the directory of the clang it runs. hk takes `llvm-objdump` and the other LLVM tools from it first, and keys the compile cache on it |
| `HK_INCLUDE_DIR` | `runtime/compile.py` | header directories, as a path list, instead of the packaged or checkout ones |
| `HK_CACHE_DIR` | `runtime/compile.py` | the compile cache (else `$XDG_CACHE_HOME/hk`, else `~/.cache/hk`) |
| `HK_TORCH_OP` | `runtime/torch_ext.py` | `library`: `hk.torch_op` builds the libtorch registration (default: over a device-only code object) |
| `HK_LAUNCH` | `dsl/kernel.py` | `codeobj`: `Kernel.__call__` launches the device-only code object (else the pybind module) |
| `HK_HOST_FASTPATH` | `runtime/hostpath.py` | `0`/`off`/`no`/`false`: no host fast paths (per-shape memos, launches without the dispatcher); every call takes the path it took before them. Default on |
| `HK_TUNE_DIR` | `autotune/__init__.py` | where tuning records are written (default: `autotune/` in the cache) |
| `HK_AUTOTUNE` | `autotune/__init__.py` | `0`, `off` or `no` pins every tuner to its default schedule |
| `HK_AUTOTUNE_MIN_GAIN` | `autotune/__init__.py` | the smallest recorded win that changes what launches (default 0.01) |
| `HK_SDPA_VERBOSE` | `ops/sdpa.py` | non-empty and not `0`: print each distinct fallback reason once |
