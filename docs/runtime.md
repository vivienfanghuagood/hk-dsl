# The runtime: compile, cache, gate, tune, launch

Everything after tracing: finding the GPU and the compiler, the
content-addressed cache, the resource gate that refuses a spilling kernel,
filling the cache ahead of time, schedule search, and the `torch.ops`
boundary a framework calls. The code is `python/hk/runtime/` and
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

A compiled `.so` is never overwritten in place: every build writes a new
content-hashed path, because rebuilding a `.so` while a process has it mapped
corrupts that process's view of the code (`python/hk/runtime/module.py`).

## Which GPU

A kernel that names no `arch` is built for `hk.target.current_arch()`:
`$HK_ARCH`, else the GPU in the machine (`hk.runtime.arch.detect`), else
gfx1100. It is resolved on first use, so importing `hk.ops` still builds no
IR and probes no GPU ([gfx1201](history/13-gfx1201.md)).

`detect()` tries, in order, `HK_ARCH`, `HCC_AMDGPU_TARGET` and
`PYTORCH_ROCM_ARCH` (the first entry of a list), then the device torch sees,
then `rocminfo`. `HK_ARCH` is the only way to build for a part that is not in
the machine, which is how the gfx1201 kernels get their compile gate on a
gfx1100 box. The gfx1100 fallback exists so that a box with no GPU gets the
source a W7900 would; a detected GPU that is not one of the targets
(`hk.TARGETS`: gfx1100 and gfx1201) falls back the same way
(`python/hk/target/__init__.py`). Only launch on the arch you built for: a
wrong `--offload-arch` fails at load time or, on a near-enough arch, computes
the wrong answer (`python/hk/runtime/arch.py`).

## Compiling

The backend is the stock hipcc compiling the translated C++ against
HipKittens' `include/rdna3`, so nothing at run time needs LLVM.

hipcc is `$HK_HIPCC`, else the one on `PATH`, else `/opt/rocm/bin/hipcc`. The
headers come from the wheel's `hk/include`, or in a checkout from `include/`
and the `third_party/HipKittens` submodule; `HK_INCLUDE_DIR` overrides them
with a path list. `ROCM_PATH` is used for `include/hip` when set, else it is
found by walking up from hipcc. The flags include `--offload-arch=<arch>`,
`-std=c++20 -O3`, `-shared -fPIC`, and
`-Rpass-analysis=kernel-resource-usage`, which the gate depends on; the full
list is `flags()` in `python/hk/runtime/compile.py`.

First compile of a small kernel is ~5 s; an attention kernel is 20–40 s.

## The cache

Key = hipcc version + a content hash of the include directories + the full flag list +
the dialect version + the generated source. A build lands in
`~/.cache/hk/<key>/` via a rename, so a killed compile never leaves an
importable half-written `.so`. Failed builds are kept as `failed-<key>/` with
their source and log.

The thresholds (`max_vgprs`, `min_occupancy`) are *not* in the key — they are
re-applied on every cache hit, so a kernel with a strict occupancy requirement
is not let through by whoever happened to compile the same bytes first.

The directory is `$HK_CACHE_DIR`, else `$XDG_CACHE_HOME/hk`, else
`~/.cache/hk`. Entries are never invalidated, only added: a key that hashes
everything does not need eviction for correctness, and the directory is one
you can delete. The include-directory hash is the part that matters most —
editing a header changes the kernel while leaving the generated source
byte-identical. The dialect version is the dialect's semantic version, not a
hash of the library, so a rebuild that changes nothing it means does not throw
the cache away (`python/hk/runtime/compile.py`).

## The resource gates

**A VGPR spill is silent corruption, not slowness**, because this library
hand-places `s_waitcnt` and scratch traffic reorders against it.
`k.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
return* a spilling kernel.

The gate is `python/hk/runtime/resources.py`. What it enforces:

* **No spill, ever.** Any `ScratchSize`, VGPR spill or SGPR spill raises
  `hk.ResourceError`, and nothing importable is left behind. The generated
  source is kept beside the failure for inspection.
* **Every instantiation**, not just the one being benchmarked — reading only
  the shipped shape's line is how a regression hides in a build nobody times.
* **`max_vgprs` and `min_occupancy`**, when the kernel asks for them
  (`@hk.kernel(max_vgprs=..., min_occupancy=...)`), so a change that quietly
  costs a wave fails the build instead of the benchmark. Occupancy is charged
  on the register count rounded up to the target's granule — 24 on gfx1100 —
  not on the raw count.
* **No remarks is an error.** A build that emits no resource remarks cannot
  be certified, so it is fatal rather than skipped.

The gate runs at `k.build()`, which needs hipcc but no GPU: the failure most
likely to bite — generated code that compiles but spills — is caught on a
laptop. `Build.report()` prints one line per instantiation:

```
add_kernel(globals): vgpr=71->72 occ=16 lds=0 scratch=0 spill=0 hdrm=25
```

(`71->72` is the raw and the rounded register count; `hdrm` is the headroom
before the next occupancy step.)

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
few: 4 of 16 keys clear the bar, and the measurements behind that are in
[Phase 5](history/04-phase-5-autotune.md).

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

* The op takes its outputs as arguments. The schema is derived from the IR:
  the tensors the kernel stores into become `Tensor(a!)` out-parameters.
* It is built with plain `hipcc -shared -fPIC` and loaded with
  `torch.ops.load_library` — not `torch.utils.cpp_extension.load`, which
  would hipify source that is already HIP.
* It launches on torch's current stream, registers with
  `TORCH_LIBRARY_FRAGMENT` (every kernel is its own `.so` in one namespace),
  and has a `Meta` implementation so `torch.compile` can trace it.
* Building it needs hipcc and torch but no GPU, and it still gates on spills.
  It is memoised on (namespace, name), because torch refuses to register the
  same op twice.

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
ones that turn on engine patches are listed in
[integration](integration.md#environment-variables).

| variable | read by | effect |
|---|---|---|
| `HK_ARCH` | `runtime/arch.py`, `target/__init__.py` | the arch to build for; first in the detection order |
| `HCC_AMDGPU_TARGET`, `PYTORCH_ROCM_ARCH` | `runtime/arch.py` | next in the detection order (first entry of a list) |
| `HK_HIPCC` | `runtime/compile.py` | the hipcc to use |
| `ROCM_PATH` | `runtime/compile.py` | where `include/hip` is |
| `HK_INCLUDE_DIR` | `runtime/compile.py` | header directories, as a path list, instead of the packaged or checkout ones |
| `HK_CACHE_DIR` | `runtime/compile.py` | the compile cache (else `$XDG_CACHE_HOME/hk`, else `~/.cache/hk`) |
| `HK_TORCH_OP` | `runtime/torch_ext.py` | `library`: `hk.torch_op` builds the libtorch registration (default: over a device-only code object) |
| `HK_LAUNCH` | `dsl/kernel.py` | `codeobj`: `Kernel.__call__` launches the device-only code object (else the pybind module) |
| `HK_TUNE_DIR` | `autotune/__init__.py` | where tuning records are written (default: `autotune/` in the cache) |
| `HK_AUTOTUNE` | `autotune/__init__.py` | `0`, `off` or `no` pins every tuner to its default schedule |
| `HK_AUTOTUNE_MIN_GAIN` | `autotune/__init__.py` | the smallest recorded win that changes what launches (default 0.01) |
| `HK_SDPA_VERBOSE` | `ops/sdpa.py` | non-empty and not `0`: print each distinct fallback reason once |
