# Phase 6: torch ops, the SDPA drop-in, and vLLM's vision tower

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

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

An earlier version of this record said `VLLM_ATTENTION_BACKEND=TORCH_SDPA` was
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

`tools/hk-bench/vllm_verify.py` is the verification, and it has now been run
inside a real vLLM (0.16.1.dev0, ROCm 7.2.1, W7900D) serving
`OpenGVLab/InternVL2-2B` with `--mm-encoder-attn-backend TORCH_SDPA`. The
engine reaches the kernel: **120 kernel calls, 0 fallbacks** over five
requests, which is 24 InternViT layers each, and the server answers normally.

Getting there found two failures that nothing short of a live engine would
have shown, both of which looked like success:

* **vLLM V1 runs the model in a spawned process.** `apply()` in the process
  that built `LLM(...)` patches a process the model never runs in. The first
  end-to-end run reported `{'kernel': 0, 'fallback': 0}` with byte-identical
  answers and a plausible-looking 1.001x. Production therefore needs the
  `vllm.general_plugins` entry point (`hk/integration/_vllm_plugin.py`), which
  vLLM loads in the driver *and* every worker; it is opt-in on
  `HK_PATCH_VLLM=1`, because installing a kernel library must not silently
  change the numerics of every vLLM on the machine.
* **The vision tower runs once per distinct image.** With
  `mm_processor_cache` and prefix caching on, a warm-up request consumed the
  only ViT pass and every later request was served from cache -- counters at
  zero again, for a different reason.

And one measurement was wrong until a control caught it. Running the arms in
blocks gave "1.18x end to end"; running the *same* arm twice under a different
ViT backend gave 1.16x with the kernel never called at all. The order effect
was the whole number. Interleaved, the end-to-end spread is +-25% on a ~250 ms
request of which the vision tower is ~4 ms, so an end-to-end request simply
cannot resolve this -- the tower's own number is the `--unit` table above, and
`--e2e` is a correctness and reachability check that now says so.

**And it found a real bug, which is now fixed.** Patched, InternVL2-2B
described the test image differently from unpatched -- while the unpatched run
repeated itself exactly, and vLLM's Triton flash-attention ViT backend agreed
with unpatched word for word. Two independent implementations agreeing and
ours disagreeing is the shape of a bug, so `tools/hk-bench/attn_precision.py`
went after it. Three hypotheses died first:

* **not** the non-dividing KV tail: `hk/torch` was the same 1.5-2.4x at
  N = 512, 544, 576, 608, 1024, 1056 as at 577, 578, 1000, 1025, 1026;
* **not** an artifact of the reference: torch's flash, mem-efficient and math
  backends all land within 3e-6 of each other here;
* **not** concentrated: p50, p90 and p99 were each ~2.5x torch's with the
  worst rows scattered, which is rounding and not a mishandled block.

Emulating the kernel's own algorithm in torch then ruled out the algorithm:
with the probability tile rounded to bf16, a 32-wide KV block and a 1-ulp
exp2, the emulation landed on *torch's* number exactly, not the kernel's. So
the kernel was not doing what its algorithm said.

The probe that found it has an exact answer. Set `V` to all ones: the softmax
weights sum to one, so attention must return exactly 1.0 for every element,
whatever Q and K were. torch returned exactly 1.0. **This kernel returned 1.0
off by 2^-8 -- one whole bf16 ulp -- on every element.**

`convertor<bf16, float>` in `include/rdna3/common/base_types.cuh` was
`bit_cast<uint32_t>(u) >> 16`, labelled "fast unsafe conversion (truncation
only)". Truncation toward zero is *biased*, half an ulp, always the same
direction -- and attention rounds a tile of positive probabilities through it
on every KV block while summing the fp32 ones for the denominator, so the bias
does not cancel. It accumulates into the ratio.

Rounding instead of truncating closes it completely. Measured, four ways in
one process (`tools/hk-bench/bf16_round_cost.py`), cost on attention at
head_dim 128 / 64:

```
0  truncate             1 instr      --          biased; not a supported mode
1  nearest, ties away   2 instr   +2.1% / +3.4%  unbiased          <- default
2  nearest even         3 instr   +3.5% / +6.4%  unbiased
3  nearest even + NaN   5 instr   +6.0% / +12.5% unbiased and total
```

GEMM pays 0.2% in every mode, because its conversion is once per output
element rather than once per probability. No mode changes the register count
(215 / 217 / 147 VGPRs, zero scratch, all four).

The default is ties-away: it is the cheapest unbiased rounding there is, and
it is **numerically indistinguishable from the expensive one** here -- `V = 1`
returns exactly 1.0, and hk's error against fp32 is now `1.00x` torch's at
every N in the sweep, not 1.5-2.4x. The NaN guard doubles the cost to protect
against a value attention cannot produce, so it is opt-in
(`-DHK_BF16_ROUND=3`); without it a NaN whose low mantissa bits are near
all-ones can come back as an infinity.

Two tests now hold the line, both with exact answers rather than a tolerance
(`tests/hk/gpu/test_attn.py`): `V = 1` must come back bit-exact, and `Q = 0`
must give the mean of V. The file's existing `RTOL = 5e-2` could never have
seen this -- it is sized to catch a shifted mask, and a half-ulp bias passed
all 31 of its cases while changing what a model said.

**What is left is a tie, and it was measured rather than assumed.** hk and
aotriton now differ on 0.02-0.04% of the vision tower's output elements, by
exactly one bf16 ulp -- two correctly-rounded kernels breaking ties from
different intermediates, which is as close as independent implementations get.
Whether that changes a caption is a property of how close the model's top two
tokens were, so the e2e runs six images under all three implementations:

```
image   aotriton (TORCH_SDPA)      Triton FA        hk
  1     "mosaic of shapes"         same             "crowd of people"
  2     "mosaic of shapes"         same             same
  3     "crowd of people"          same             same
  4     "crowd of people"          "digital noise"  "digital noise"
  5     "blurred and distorted"    same             same
  6     "shades of green, red"     same             same
```

vLLM's own two backends already disagree with each other on 1 of 6; hk
disagrees with aotriton on 2 of 6, and on one of those (image 4) it agrees
with Triton FA *against* aotriton. Each arm repeated itself 6/6, so the engine
is deterministic and these are real disagreements -- between three
implementations, on inputs chosen to be maximally ambiguous (synthetic noise).
hk is inside the spread that already exists between shipped backends, not
outside it. A real-image benchmark would say more; this says enough to stop
treating the remaining difference as a defect.

The layer underneath is covered by the GPU tier -- 225 tests including
`torch.compile(fullgraph=True)`, CUDA-graph capture and replay, and the
strided-ViT-input contract in both directions.

`python3 -m hk.integration --warm -j 32` builds the six distinct attention
registrations in 17.8 s, which is the number that matters for a server's first
request.

---

← [Phase 5](04-phase-5-autotune.md) · [Development record](README.md) · [Phase 7](06-phase-7-paged-decode.md) →
