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
hk/lang/        the DSL surface -- @kernel, tile types, ops
hk/ir/          Value/Op/KernelIR, the tracing builder, passes
hk/target/      gfx1100 / gfx1201 facts, as data
hk/codegen/     IR -> HipKittens C++ (cpp.py) and the module boundary (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                and the torch.ops registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time
hk/tuned/       the schedules that search picked, as shipped data
hk/ops/         kernels written in the DSL, plus the SDPA drop-in (sdpa.py)
hk/integration/ patching a running vLLM or SGLang
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

`python3 -m hk.runtime.warm -j 32` fills the cache in parallel (19x on 24
kernels), and `python3 -m hk.autotune --aot` does the same for every schedule a
tuning record can reach. Those two are how the serving path ends up with no
hipcc in it.

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
`torch.compile` on a W7900D, 13 of 15 cases are faster and the other two
(`quantize 16384x4096` and `layernorm 16384x5120`) land within 1% behind --
which is inside the same noise band Phase 5 later had to make explicit, and so
is reported as a tie rather than as a win for either side.

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

**Phase 4 is done: the generated attention kernel matches the handwritten one,
and beats it under causal.** The gate was 64-65 TFLOPs plus elementwise
agreement with `F.scaled_dot_product_attention` across every H3 shape, a
non-dividing N, B=2, GQA and causal. Same process, interleaved, both kernels
loaded at once (`hk/ops/_attn_ab.py`):

```
shape           dsl ms   dsl TF   c++ ms   c++ TF  dsl/c++
N=4096           7.880     61.0    7.895     60.9   1.002x
N=8192          30.528     63.0   30.527     63.0   1.000x
N=16384        120.334     64.0  120.300     64.0   1.000x
N=32768        475.309     64.8  475.132     64.8   1.000x
480p/5s       1100.228     64.9 1099.149     65.0   0.999x
causal 8192     16.628     57.9   16.754     57.4   1.008x
```

Against torch SDPA on the same chip that is 2.9-3.2x, and 4.0x causal. 31
numerical cases pass (`tests/hk/gpu/test_attn.py`). The generated kernel also
runs at **occupancy 6 against the handwritten kernel's 5** -- 224 VGPRs versus
247 -- which is the one place the IR is structurally ahead rather than level.

Three findings came out of getting there, and all three now live in the library
or the IR rather than in a kernel:

* **`warpid()` costs a divergent branch.** It is `threadIdx.x >> 5`, which the
  compiler keeps in a VGPR because it cannot see that a wave is 32 consecutive
  threads. Anything derived from it inherits the VGPR, so a comparison on it
  becomes `v_cmpx` with exec save and restore rather than one `s_cbranch`, and
  every value the branch needs is pinned in a vector register across it. The
  causal kernel spilled 90 VGPRs that way and none with
  `kittens::warpid_uniform()`.

* **An index inside a tile mask costs one register per element.** Written the
  obvious way -- ask `rt_base_coord` for (row, col) and evaluate the predicate
  -- a mask holds 8 loop-invariant registers, which LLVM hoists into the
  preheader and spills. Writing it as `dbase >= thr` with a scalar threshold is
  *worse*: LLVM reassociates straight back. The fix is to fold the index into
  the lane term so the per-element part is an integer literal after unrolling,
  which is one register for the whole tile. `detail::diag_fill` and
  `detail::axis_fill` in `conversions.cuh` now do this for all eight fills.

* **Branches in the hot region are a register-allocation lever, and the trade
  is asymmetric.** A mask that fires on one block in the whole loop (the
  backed-up last KV block) is worth a uniform branch: 224 VGPRs and no spill
  with it, 116 B/lane without. A mask that fires on nearly every block (causal)
  is worth no branch at all: 226 and none without it, 156 B/lane with it.

The last 2% was one thing the IR could not say. The handwritten kernel skips
the online-softmax rescale when the running max did not grow -- 64 `v_mul_f32`
per KV block at head_dim 128, against that block's 32 WMMAs -- and that needs a
predicate that reduces *out of* the register file and into control flow, which
no elementwise op can produce. `hk.s_any_ne` is that op, and it closed the gap
exactly. Causal then went ahead on longest-processing-time-first dispatch: the
work per workgroup is a ramp, so reversing the q index starts the long ones
while there is still short work to fill the slots they free.

**Phase 5 is done: `hk.autotune` replaces `sweep.sh`, and most of what it
measured turned out not to be worth acting on.** A `Tuner` is a kernel factory, a schedule
space and a record. Searching it is two stages, and the first one never touches
the GPU:

```
python3 -m hk.autotune --list
python3 -m hk.autotune --tune gemm_bf16 -j 32      # search
python3 -m hk.autotune --ship                      # promote what is real
python3 -m hk.autotune --aot  -j 32                # prebuild it all
```

Stage 1 builds every candidate through `runtime.warm` -- 32 hipcc invocations
at a time -- and three kinds of candidate die there for free: the factory's own
arithmetic refuses a tiling that does not divide, the verifier refuses an
illegal layout, and the resource gate refuses anything that spills. Of
attention's 32 schedules, 26 never reach the GPU. Stage 2 times what is left in
*one process*, interleaved, reversing the order on odd rounds and scoring each
candidate by its minimum, because cross-process A/B on this chip manufactures
13-16% differences out of nothing.

The production path never searches. `Tuner.kernel(key)` is a dict probe and a
memoised factory call, so a record costs nothing per launch; a missing, stale
or foreign record costs performance and never correctness.

`min` over a column of noisy numbers always names a winner, so a tuner that
ships its winner unconditionally ships noise as a finding. `tune()` therefore
also records `default_ms`, and a record changes nothing unless it beat the
default by `MIN_GAIN` (1%). The bar is applied at `--ship` *and* on the lookup
path, so a record measured here is held to the same standard as one that
arrived in a wheel; `HK_AUTOTUNE_MIN_GAIN` moves it and `HK_AUTOTUNE=0` ignores
records entirely.

Out of 16 keys across five tuners, four clear it:

```
tuner                  key                 winner                  gain
gemm_bf16              4096x4096x4096      k_step32 wgm16 pos1    +1.2%
attn_fwd_d128          n16384              qk1 pv2                +1.1%
attn_fwd_d128          n65536              qk1 pv2                +1.1%
attn_fwd_d128_causal   n4096               qk1 pv1                +2.8%
```

The other twelve stay on the default, including every `gemm_bf16` shape but
one and all six `attn_fwd_d64` keys. That is the result and not a gap: a space
whose winners are inside the noise is a space whose default was already right.

The bar had to be checked against itself before any of this was believable.
The first attention sweep reported gains of 0.10-0.21% with four *different*
winners across three lengths -- a coin. A second sweep on the same machine
reported 0.26% to 2.8%. Two runs of one tuner disagreeing by 10x is not a
result, so the three keys above were re-measured three times each at
`--rounds 5`:

```
key                        repeat 1        repeat 2        repeat 3
d128 n16384          qk1 pv2 +1.04%  qk1 pv2 +0.99%  qk1 pv2 +1.00%
d128 n65536          qk1 pv2 +1.12%  qk1 pv2 +1.12%  qk1 pv2 +1.09%
d128_causal n4096    qk2 pv1 +1.81%  qk1 pv4 +2.13%  qk1 pv2 +1.29%
```

The first two reproduce exactly -- same schedule, same gain to within 0.05 --
so `qk1 pv2` is a real if small win at head_dim 128, and the original sweep's
0.1% was the measurement that was wrong. The third says something different
and more useful: at `n4096` causal the gain is reproducible but the *winner* is
not, because the default (`qk2 pv2`) is specifically the slow one there and all
three alternatives beat it by about the same amount. The shipped record names
one of them, and this paragraph is what stops the next reader from concluding
that `qk1 pv1` is special.

`--aot` prebuilds every schedule production can reach -- the recorded ones *and*
the default, since the default is what an unrecorded key gets -- into the
content-addressed cache, which is what takes hipcc's 20-40 s off the first
request. `pip install -e .` works from the repo root; `hk/tuned/*.json` and
`hk/include/**` are package data, because the generated C++ `#include`s the
headers at JIT time.

**Phase 6 is done: a DSL kernel becomes a `torch.ops.hk.*` op, and the SDPA
entry point is a drop-in.**

```python
import hk
from hk.ops import sdpa

sdpa.patch()                 # F.scaled_dot_product_attention -> hk, where it fits
torch.ops.hk.attn_fwd_d128(q, k, v, o)
```

`hk.torch_op(kernel)` emits a third scaffold beside `pybind` and `bare`, builds
it with plain `hipcc -shared -fPIC` and loads it with `torch.ops.load_library`
-- not `cpp_extension.load`, which would hipify source that is already HIP. The
schema is derived from the IR rather than from the author: `written_tensors`
walks the ops that write memory the caller can see (`store_global`,
`store_scalar`, `group_store`, and deliberately not `store_frag`, which writes
a shared tile), and those tensors become `Tensor(a!)` out-parameters. A schema
that forgets to mark an output mutable compiles, loads, runs, and is wrong only
under `torch.compile`'s functionalization -- which is exactly the kind of bug a
derivation removes and a convention does not.

It costs nothing. Same process, interleaved, bit-identity checked before timing
(`tools/hk-bench/sdpa_paths.py`); `aotriton` is the backend torch's own SDPA
picks on this chip, and is the thing being replaced:

```
shape             path             ms      TF   host us   vs pybind
1x56x4096x128     pybind        7.875    61.1       3.7       1.00x
                  torch.ops     7.857    61.2       3.5       1.00x
                  sdpa drop-in  7.872    61.1       9.4       1.00x
                  aotriton     22.612    21.3      11.5       0.35x

1x56x16384x128    pybind      120.204    64.0       3.8       1.00x
                  torch.ops   120.449    63.9       3.1       1.00x
                  sdpa drop-in 120.466   63.9       9.5       1.00x
                  aotriton    368.805    20.9      11.7       0.33x

1x32x8192x128 c   pybind        9.769    56.3       3.3       1.00x
                  torch.ops     9.525    57.7       2.9       1.03x
                  sdpa drop-in  9.599    57.3       9.6       1.02x
                  aotriton     40.467    13.6      15.0       0.24x
```

Device time is identical across the three hk paths, which is what the shared
kernel body predicts. The surprise is the host column: torch's dispatcher is
*cheaper* than the pybind converter it replaces (3.1 us against 3.8), so the
registration is not a tax even on a 60 us elementwise op. The drop-in's 9.4 us
is the entry point's own work -- the support check and the output allocation --
and is the price of being callable as `F.scaled_dot_product_attention`.

Three details are load-bearing:

* **The kernel launches on torch's stream.** `cpp.py` now emits
  `launch_on(const globals&, hipStream_t)` with `launch(g)` as the default-stream
  wrapper, and the torch impl passes `getCurrentHIPStream()`. A default-stream
  launch breaks CUDA-graph capture and overlapped copies, silently.
* **`TORCH_LIBRARY_FRAGMENT`, not `TORCH_LIBRARY`.** Every kernel is its own
  `.so` and they share one namespace; the non-fragment form claims the namespace
  exclusively, and the second `.so` to load throws.
* **A `Meta` implementation** makes the op traceable by `torch.compile`. It is a
  no-op, because the op allocates nothing -- which is also what a serving
  framework wants.

`hk.integration` patches a running vLLM or SGLang. It rebinds **by identity**,
not by name: `_common.rebind` walks `sys.modules` under a prefix and replaces
every attribute that *is* `torch.nn.functional.scaled_dot_product_attention`,
which catches the modules that did `from torch.nn.functional import ...` at
import time and cannot be fooled by a same-named impostor. `python3 -m
hk.integration --warm -j 32` fills the cache for the torch half specifically:
libtorch's include and ABI flags are part of the cache key, so a `--aot` pass
built for pybind does not spare a server its first attention compile.

Two things are deliberately *not* patched, and the reason is the same both
times -- the patch would be slower than what it replaces:

* **RMSNorm.** Every framework's RMSNorm applies a learned weight;
  `hk.ops.rmsnorm` is unweighted, so the patch would add an elementwise pass to
  save one.
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves and `hk.ops.silu_mul` has no stride support, so the
  patch would add two copies to save one pass.

Both are written down as missing *kernel* work rather than attempted as
patches.

**What the patch can and cannot reach in vLLM**, read against vLLM at 1dfe9fb
and checked at runtime by `hk.integration.vllm.probe()`:

* **The decoder's attention is out of reach, structurally.** vLLM V1 keeps KV
  in a paged cache and dispatches through `AttentionImpl.forward(..., kv_cache,
  block_table, ...)` -- `RocmAttentionImpl` or `TritonAttentionImpl` on this
  chip. There is no dense `(B, H, N, D)` tensor on that path for an SDPA
  drop-in to intercept. Reaching it needs a *paged* kernel registered through
  `vllm.v1.attention.backends.registry.register_backend`, which is kernel work
  `hk.ops.attn` has not done.
* **The vision tower's attention is in reach, exactly.**
  `mm_encoder_attention._forward_sdpa` -> `torch.ops.vllm.torch_sdpa_wrapper`
  -> `apply_sdpa` is a literal `F.scaled_dot_product_attention` on a dense
  `(B, H, N, D)` tensor, and patching `F` reaches it because the attribute is
  resolved on the module at call time.
* **It is not the default on RDNA3.** `RocmPlatform.get_vit_attn_backend`
  prefers the Triton flash-attention whenever `flash_attn_triton_available()`;
  `--mm-encoder-attn-backend TORCH_SDPA` selects it deliberately, and that is a
  supported configuration rather than a trick.

An earlier version of this file said `VLLM_ATTENTION_BACKEND=TORCH_SDPA` was
the lever for the decode path. That was wrong twice over: the variable no
longer exists in current vLLM, and `AttentionBackendEnum.TORCH_SDPA` carries
the comment "this tag is only used for ViT".

Measured on the exact call the ViT path makes -- `(b, s, h, d)` permuted to
`(b, h, s, d)`, so never contiguous -- against aotriton on a W7900D:

```
tower                      B,H,N,D        path           ms   vs fp32   speedup
CLIP ViT-L/14-336 (LLaVA)  1,16,577,64    hk (kernel) 0.0978   0.00369     1.69x
                                          torch       0.1657   0.00185     1.00x
InternViT-300M             1,16,1025,64   hk (kernel) 0.1508   0.00243     2.49x
                                          torch       0.3759   0.00105     1.00x
SigLIP so400m              1,16,729,72    hk (back)   0.1945   0.00128     0.95x
                                          torch       0.1850   0.00128     1.00x
Qwen2.5-VL ViT             1,16,1024,80   hk (back)   0.3406   0.00113     0.97x
                                          torch       0.3295   0.00113     1.00x
```

`hk.ops.attn` has head_dim 64 and 128, so CLIP-L/14 (LLaVA, Pixtral,
InternViT) fits and SigLIP's 72 and Qwen2-VL's 80 fall back. The fallback rows
are the reason this table exists: the first measurement of them was 0.84x and
0.90x, because `scaled_dot_product_attention` copied `q`, `k` and `v`
contiguous *before* asking whether the kernel could take them -- three copies
thrown away, and torch handed the original strided tensors anyway. Deciding
first and memoising the verdict took it to 0.95x and 0.97x, the remainder
being the Python predicate itself on a 0.19 ms call.

The drop-in now counts which branch it took (`hk.ops.sdpa.STATS`), because a
patch that silently falls back produces identical answers and no speedup,
which is indistinguishable from one that works unless something counts.

`tools/hk-bench/vllm_verify.py` is the verification: `--probe` reports what the
installed vLLM actually does, `--unit` is the table above, and `--e2e MODEL`
runs a real multimodal request twice and reads the counters. The first two
need no vLLM and are green; `--e2e` has not been run, because no vLLM is
installed on the bench pod.

The layer underneath is covered by the GPU tier -- 225 tests including
`torch.compile(fullgraph=True)`, CUDA-graph capture and replay, and the
strided-ViT-input contract in both directions.

`python3 -m hk.integration --warm -j 32` builds the six distinct attention
registrations in 17.8 s, which is the number that matters for a server's first
request.

See `.claude/plans/` for the full plan and its gates.
