# The same call on gfx1100 and gfx950: an audit of `hk.ops`

Criterion G4 of [the goal](../design/dx-goal.md) is "the same op takes the
same arguments on every GPU". This page lists, for every public `hk.ops`
function, where its arguments, dtypes, layouts or behaviour differ between a
Radeon W7900D (gfx1100, RDNA3, wave32) and an Instinct MI355X (gfx950, CDNA4,
wave64), what was fixed, and what is left and for whom.

- [How it was checked](#how-it-was-checked)
- [The table](#the-table)
- [Fixed here](#fixed-here)
- [For the row-wise and elementwise work (W2)](#for-the-row-wise-and-elementwise-work-w2)
- [Real gaps, documented rather than unified](#real-gaps-documented-rather-than-unified)
- [The language itself](#the-language-itself)

## How it was checked

1. **Reading the code**: each op's wrapper to the kernel each arch reaches.
   The routing is `current_arch()` (`$HK_ARCH`, else the GPU): the row and
   elementwise ops send gfx950 to register-strip kernels through each
   module's `STRIP_ARCHS = ("gfx950",)`; `matmul`, `attention`, the paged
   decode and the paged prefill switch to separate gfx950 modules
   (`gemm_cdna`, `attn_cdna`, `paged_cdna`).
2. **Running it on both GPUs**: [`tools/api_consistency.py`](../../tools/api_consistency.py)
   asks each question below on the GPU it runs on and prints one JSON
   record per probe. Measured on an MI355X (rocm/pytorch 7.2.3, torch 2.10)
   and a W7900D (ROCm 7.2.4, torch 2.10), at this branch.
3. **Building every `hk.math` function** on fp32 and bf16 tiles and strips
   for both archs (the language, [below](#the-language-itself)).

## The table

"Class" says who owns a difference: **W2** for the row-wise and elementwise
ops (`hk.templates.elementwise`/`rowwise` are replacing their insides; listed
here, not changed), **fixed** where this branch made the two agree or the
message say which arch does what, **gap** where a kernel does not exist on
one arch.

| op | gfx1100 | gfx950 | kind | class |
|---|---|---|---|---|
| `add` `sub` `mul` `maximum` `minimum` `exp` `relu` `gelu` `neg` `abs` | 16x64 tiles: the shape must tile (last dim a multiple of 64 with rows a multiple of 16, or numel a multiple of 1024) -- `100x1000` is refused | strips below 2 GiB: any size; tiles above | shape | W2 |
| `gelu` | fp32 only | fp32 only; the error says "include/rdna3 implements it" on gfx950 too | message | W2 |
| `silu_mul` | 16x64 tiles: `a`, `b`, `out` contiguous (`a must be contiguous; call .contiguous()`), at least 16x64 | strips: rows of unit column stride -- the two halves of a `(M, 2I)` gate/up output as views -- any shape | layout, shape | W2 |
| `rmsnorm` `rms_norm` `fused_add_rms_norm` `layernorm` `softmax` | strips while a row fits 64 values a lane, then tiles (at least 16x64) | strips up to 128 values a lane, then tiles | where the fallback starts | W2 |
| `rope` | the same tile kernel | the same | none | -- |
| `rotary` | `NotImplementedError` ("use hk.ops.rope"), whose arguments differ (separate cos/sin, (seq, D/2)) | runs | availability | W2 |
| `quantize` | at least 16x64 | any shape (strips) | shape | W2 |
| `dequantize` | at least 16x64 | the same | none | -- |
| `matmul` | `hk.ops.gemm`: K a multiple of the tuned k_step (64), N of 128; K = 0 passed validation | `hk.ops.gemm_cdna`: K a multiple of 64 and > 0, N of 128 | shape | **fixed**: K = 0 refused on both |
| `matmul` | bf16 store rounds ties away from zero (`HK_BF16_STORE_ROUND`) | rounds to nearest even; split-K routes sum fp32 shares then round | numerics | gap, documented in the docstring |
| `matmul` | a non-contiguous `b` (`w.t()`) is refused | refused (a route to `hk.ops.decode_gemm` would copy it) | layout | same |
| `attention`, `sdpa.attention` | n_q >= 192 (one Q tile), n_kv >= 32 | any non-empty length | shape | gap; **fixed**: the message says gfx950 takes any length |
| `attention` | dense online softmax | lazily rescaled softmax (threshold 8) | numerics: same math, different bits | gap |
| `sdpa.scaled_dot_product_attention` | falls back to torch below one Q tile | runs the kernel | silent fallback | gap (the fallback is the point of this entry) |
| `sdpa.op_for(n=)` | `n` picks the tuned schedule | ignored (one schedule) | argument | **fixed** in the docstring |
| `paged.KERNELS[hd]` | RDNA decode; q and o padded to 16 rows (unbounded loads) | `paged_cdna`'s, bounded | layout | gap |
| `paged.KERNELS.get(hd)` | the RDNA kernel | **the RDNA kernel too** (dict.get skipped the routing `[]` does) | behaviour | **fixed** |
| `paged.split_kernel` `fused_kernel` | head_dim a multiple of 16 | 64 or 128 (`head_dim=96: have (64, 128)`) | shape | gap; **fixed**: the message names the arch |
| `paged.merge_kernel(hd, splits)` | any split count | a multiple of 8 above 8 (`splits=12: a whole number of 8-split rounds`) | shape | gap; **fixed**: the message says why and what to do |
| `paged.plan_splits` | any 1..32 | the same, which the gfx950 merge then refuses unless rounded | shape | **fixed** in the docstring |
| `paged_prefill.kernel_for(hd, group)` | `group` compiled in; head_dim a multiple of 32 up to 192 | `group` ignored (read off the shapes); 64 or 128 | argument, shape | gap; **fixed**: docstring |
| `paged_prefill.Q_TILE` | 192 | the kernel's tile is 256 (`attn_cdna.Q_TILE`); `work_list` knows | constant | **fixed**: comment |
| `moe.moe_gemm`, `w4a16.matmul`, `decode_gemm.matmul` | one code path | the same | none | -- |
| `hipkittens.gemm_bf16`, `gqa_fwd` | `WrongArch` ("these kernels run on gfx950 only") | runs | availability, by design | -- |

Measured (`tools/api_consistency.py`, max abs error against an fp32 torch
reference, or the error raised):

| probe | gfx1100 (W7900D) | gfx950 (MI355X) |
|---|---|---|
| `matmul`, bf16 tie (1 + 2^-8) | pending* | 1.0 (to even) |
| `matmul`, K = 0 | ValueError (after the fix)* | ValueError: K=0 must be a multiple of 64 |
| `matmul`, `b = w.t()` (M 16 and 1024) | RuntimeError (the same message)* | RuntimeError: a non-contiguous tensor is taken only as a slice of its last axis |
| `matmul`, K = 96 | ValueError: K=96 must be a multiple of 64* | ValueError: K=96 must be a multiple of 64 |
| `attention`, n = 64, D = 128 | ValueError: n_q=64 is below one Q tile (192) of gfx1100's kernel* | 0.0079 |
| `attention`, n = 256, D = 64 | pending* | 0.0140 |
| `silu_mul`, strided halves of a (64, 8192) gate/up | ValueError: a must be contiguous* | 0.031 |
| `silu_mul`, 8 x 100 | ValueError: a is 8x100 and silu_mul needs at least 16x64* | 0.0074 |
| `add`, 100 x 1000 | ValueError: shape (100, 1000) does not tile* | 0.0156 |
| `gelu` on bf16 | TypeError (the same message)* | TypeError: hk has no bf16 gelu kernel -- include/rdna3 implements it for fp32 only |
| `gelu` on fp32, against the tanh / erf forms | pending* | 4.8e-7 / 4.7e-4 |
| `quantize`, 8 x 100 | ValueError: hk quantize works on at least 16x64* | 0.0156 |
| `rotary`, 16 tokens x 8 heads x 128 | NotImplementedError* | runs |
| `rms_norm` 3 x 4096 / `rmsnorm` / `softmax` 3 x 100 / `layernorm` 16 x 65536 | pending* | 0.023 / 0.0078 / 0.00022 / 0.0155 |
| `paged.KERNELS.get(128)` vs `[128]` (before the fix) | the same kernel | `paged_decode_d128` vs `paged_decode_d128_cdna` |
| `paged.merge_kernel(128, 12)` | runs* | ValueError (the message above) |

\* From the code; the gfx1100 run is queued behind the GPU tier on the
shared W7900D and replaces these when it lands.

## Fixed here

Each fix leaves the arch where the call already worked as it was.

- `hk.ops.matmul` refuses K = 0 on gfx1100 with the message gfx950 gives
  (an empty reduction is not a launch).
- `paged.KERNELS.get(hd)` routes as `paged.KERNELS[hd]` does, so on gfx950 it
  is `paged_cdna`'s kernel, not the RDNA one.
- Messages that now say which arch has the limit and what the other does:
  `hk.attention`'s and `sdpa.attention`'s length limits (gfx1100's),
  `paged_cdna`'s and `attn_cdna`'s head dims (gfx950's), the gfx950 merge's
  split count.
- Docstrings that described one arch, or another time: `matmul` (the gfx950
  kernel, its K rule and rounding), `attention` (the length rule per arch),
  `paged.plan_splits` (round for the gfx950 merge), `paged_prefill.kernel_for`
  (new: what `group` and head_dim do on each), `paged_prefill.Q_TILE`,
  `sdpa.op_for` (`n` is gfx1100's), and the stale module docstrings of
  `hk.ops.hipkittens` ("hk's native GEMM and attention do not build for
  gfx950 yet"), `hk.ops.paged` ("there is still no paged prefill") and
  `hk.ops.paged_cdna` (where its splits merge).
- `hk.ops.__all__` lists `paged`, which it imported but did not export.

`tests/hk/ir/test_api_consistency.py` holds them, without a GPU.

## For the row-wise and elementwise work (W2)

Not changed here; `hk.templates.elementwise` / `rowwise` are where these ops'
insides are going, and the fix belongs there:

1. **`silu_mul`'s layout rule differs**: gfx950 takes the two halves of a
   gate/up output as strided views (`_rows_view`: unit last-axis stride, rows
   one stride apart); gfx1100 demands contiguous operands (`_as_2d`) and at
   least 16x64. The goal names this one. On gfx1100 a strided view could go
   through the same `_rows_view` and the tile kernel's global (a global
   carries one row stride, [language](../language.md#memory)), or onto strips
   there too.
2. **The elementwise ops' shape rule differs**: any size on gfx950 (bounded
   strips), "the shape must tile" on gfx1100 (`100x1000` refused). The
   module docstring ("a partial tile ... is refused") is false on gfx950.
3. **`rotary` is gfx950-only**, and `rope`, gfx1100's, takes different
   arguments (cos and sin as two (seq, D/2) tables, (batch*heads, seq, D)
   input) from `rotary` (vLLM's `cos_sin_cache`, token-major, positions).
4. **`quantize` needs 16x64 on gfx1100**, any shape on gfx950.
5. **The norms' fallback to tiles starts at different widths** (64 values a
   lane on gfx1100, 128 on gfx950), and below 16x64 the tile fallback refuses.
6. **Messages**: `gelu` on bf16 says "include/rdna3 implements it for fp32
   only" on gfx950 too (`elementwise._kernel_for`); the two contiguity
   messages differ in wording (`elementwise._as_2d` vs `elementwise`'s strip
   path).
7. **Neither arch checks `silu_mul`'s `out=`** shape or dtype in Python.

## Real gaps, documented rather than unified

These are kernels that do not exist on one arch, or numerics that follow
from a different kernel; the docstrings and [docs/api/hk.ops.md](../api/hk.ops.md)
say so per arch:

- gfx1100's attention needs a whole Q tile (192 rows) and KV block (32); a
  decode-length query has `sdpa`'s torch fallback or the paged decode.
- gfx950's paged decode, paged prefill and attention have schedules for
  head_dim 64 and 128 only; gfx1100's take multiples of 16 (decode) or 32
  (prefill).
- gfx1100's plain paged decode needs q and o padded to 16 rows.
- gfx950's merge reads splits in rounds of 8.
- The GEMMs round bf16 stores differently (ties away from zero on gfx1100,
  to even on gfx950) and gfx950's split-K accumulates in another order.
- `hk.ops.hipkittens` is gfx950-only by design.
- The RDNA kernel builders (`attn.attn_kernel`, `gemm.gemm_kernel`,
  `paged.paged_*_kernel`, `paged_prefill.paged_prefill_kernel`) are not what
  gfx950 calls; the wrappers route around them.

## The language itself

Not `hk.ops`, but the same question one level down:

- **`hk.math.sqrt` did not build on gfx950** -- neither CDNA tree has a
  `base_ops::sqrt`, and hk's `include/hk/cdna.cuh` added `rsqrt`, `silu` and
  `step` but not `sqrt`, so a kernel calling it traced and then failed in
  hipcc ("no member named 'sqrt' in namespace 'kittens'"). Fixed: every
  `hk.math` function now builds on the same dtypes on both archs (fp32 for
  gelu, dgelu, sqrt, rsqrt and silu; bf16, fp16 and fp32 for the rest), on
  tiles and on strips.
- The fp32-only error named include/rdna3 on every arch; it now says the
  headers specialise the function for fp32 on every arch.
- **`import hk` settles the arch**: `hk.ops.attn` builds its default kernels'
  plan at import, which calls `current_arch()`, so `HK_ARCH` set after
  `import hk` has no effect (the package docstring says importing traces
  nothing; it traces nothing, but it does probe the GPU). `python -m hk.check
  --arch` runs another arch in a child process for this reason; the
  pitfall is in [llms.txt](../../llms.txt). Making that plan lazy is a small
  change in `hk.ops.attn`, left for whoever next touches it.
- Strips' chunk is a wave of lanes, 32 on gfx1100 and 64 on gfx950: a strip
  of `CW` columns at `run` 8 needs `CW` a multiple of 256 on gfx1100 and of
  512 on gfx950. `CW = 512` fits both.
