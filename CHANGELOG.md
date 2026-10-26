# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## 0.7.0

hk-dsl 0.7.0: hk is on by default in vLLM and SGLang, for the routes validated
with real weights on your GPU; `HK_DISABLE=1` turns it off. Also `hk.comm` for
hand-offs between gfx950's XCDs and between GPUs, and two out-of-bounds fixes
on gfx950, one of them a silent write past the tensor in 0.6.0's DSL.

On PyPI: `pip install hk-dsl` (CPython 3.10 on glibc 2.35+, 3.12 on glibc
2.38+, Linux x86_64). The wheels on the GitHub release are the same files; the
manylinux copies are the ones on PyPI. Each carries the MLIR bindings for its
interpreter and the HipKittens headers, so at run time only hipcc is needed.

    pip install hk_dsl-0.7.0-cp312-cp312-manylinux_2_38_x86_64.whl

**hk is on by default** (#54). With hk-dsl installed, its vLLM and SGLang
plugins (and the SGLang-Diffusion boot) apply every route that was validated
with real weights on your GPU: outputs bit-identical to the engine's, or error
within the engine's own noise. Before, every route needed its own `HK_*=1`.
Turning hk off is now for comparison experiments:

```bash
HK_DISABLE=1 vllm serve <model>        # stock: nothing patched, in every engine
HK_VLLM_MXFP8=0 vllm serve <model>     # one route off (HK_<ROUTE>=0)
python3 -m hk.integration --status     # every route on or off, and which rule decided
```

- **On by default** (the evidence for each route is in
  [integration](docs/integration.md#on-by-default-and-turning-it-off)):

  | GPU | engine | routes (switch) |
  |---|---|---|
  | gfx950 (MI355X) | vLLM | K2 MXFP8 linear for DeepSeek-V4.1 at M >= 2048 (`HK_VLLM_MXFP8`); K1 MoE a8w4 for DeepSeek-V4.1 at M >= 8192 (`HK_VLLM_MOE_A8W4`); AWQ 4-bit (`HK_VLLM_INT4`) |
  | gfx1100 (W7900D) | vLLM | unquantized linear (`HK_VLLM_GEMM`); bf16 MoE (`HK_VLLM_MOE`); AWQ 4-bit (`HK_VLLM_INT4`); the encoder SDPA (`HK_VLLM_SDPA`) |
  | gfx1100 | SGLang | linear (`HK_SGLANG_LINEAR`); MoE and the gfx1100 top-k fix (`HK_SGLANG_MOE`) |
  | gfx1100 | SGLang-Diffusion | DiT attention (`HK_SGLD_ATTN`); DiT linear (`HK_SGLD_LINEAR`) |
  | gfx1100 | `hk.ops.sdpa` | K3 DiT attention (`HK_SDPA_K3`) |

  On a GPU a route was not validated on, it stays off: the gfx1100 routes on
  gfx950 (vLLM's linear, MoE and encoder SDPA among them), and every route on
  gfx942 and gfx1201.
- **Stays opt-in:**
  - **ConvRot INT8** (`HK_SGLD_CONVROT=1`, or `HK_SGLD=attn,linear,convrot`):
    it is lossy by design (PSNR 30.4 dB from bf16), a model-level choice and
    not a toolchain default.
  - **The vLLM decoder attention backend** (`HK_VLLM_BACKEND=1`): see the
    known issues.
  - **`HK_MXFP8_ANY`**, K2 at shapes it did not measure (no real-weight run).
- **Turning routes off.** The first rule that applies decides:
  1. `HK_DISABLE=1` turns every route off, in every engine. Nothing beats it.
  2. An engine's switch at `0` (`HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0`,
     `HK_SGLD=0`) turns off that engine's routes.
  3. A route's own switch decides that route: `HK_<ROUTE>=0` off, `=1` on (also
     a default-off route, or a GPU it was not validated on).
  4. The legacy switches still mean what they meant: `HK_SGLD=attn,linear,convrot`
     turns on exactly those parts, `HK_SGLD=1` attention and linear,
     `HK_PATCH_VLLM=1` also the encoder SDPA; `HK_SGLANG_MOE=topk` and
     `HK_VLLM_BACKEND=<slot>` work as before.
  5. Otherwise the default applies.

  `0`, `off`, `no` and `false` are off; an empty value counts as unset.
  `python3 -m hk.integration --status` prints every route, whether it is on,
  and which rule decided.
- **The engine contract is checked at startup** (`hk.integration.contract`).
  Before it patches anything, a plugin checks every route it is about to apply
  against the installed vLLM / AITER / SGLang: each patched or fallback entry
  point exists, with the parameters the route uses. If any check fails, for
  example after a version change, the plugin patches **nothing** in that
  engine, prints one warning line, and the engine runs fully stock:

  ```
  [hk] WARNING: vllm: engine contract FAILED -- hk patched NOTHING in this engine, it runs stock (...)
  ```

  It is all or nothing per engine, never half-patched: a route that raises
  while it is applied undoes the ones applied before it, with the same line.
- **The first JIT compile says so.** A route's kernels compile on first use,
  seconds each, in every worker. If they were not pre-built, the first
  compile in each process prints one line that suggests
  `python3 -m hk.integration warm-routes` (and `--warm` for attention), run
  with the server's `HK_CACHE_DIR` before the server starts.
- **GPU detection reads the KFD topology.** The vLLM ROCm image sets
  `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`. Without `HK_ARCH`, hk took the
  list's first entry, so it read an MI355X as gfx90a, built for gfx1100, and
  with default-on it would have left every gfx950 route off.
  `hk.runtime.arch.detect()` now goes `HK_ARCH`, then a variable naming exactly
  one arch, then the KFD topology in sysfs (no HIP context), then torch and
  rocminfo, and uses a list's first entry only as the last resort.
- **Comparison scripts must say stock.** Unset no longer means stock, so an
  A/B's stock arm sets `HK_DISABLE=1`. `python3 -m hk.integration ab` /
  `headroom` and `tools/hk-bench/` do this themselves.
- **SGLang-Diffusion:** an install made before 0.7.0 keeps its old boot, which
  needs `HK_SGLD=1`. Run `python3 -m hk.integration sglang-diffusion install`
  again for the defaults; `... sglang-diffusion status` says when the boot is
  the old one.
- **Fixed:** with several routes on in one vLLM (int4 and MXFP8 on gfx950), its
  compile cache root was nested again in every worker
  (`hk-int4/hk-mxfp8/hk-int4/...`). It is now one directory named for all the
  routes (`hk-int4+mxfp8`), the same in every process.
- `hk.ops.sdpa.reset_stats()` also resets K3's counters.

**Fixes**
- **gfx950: synchronous `g.load` / `g.store` went out of bounds at a partial
  last tile** (#49). This affects 0.6.0's DSL. In 0.6.0 on gfx950, in a
  user-written kernel whose synchronous `g.load` covers a partial last tile
  (the tile's rows run past the tensor's last row):
  - the load reads up to a tile's rows past the tensor: an intermittent memory
    fault (10 of 10 fenced cases faulted, at M = 1, 63, 65, 130 and 1000);
  - its `g.store` **silently overwrites the memory after the tensor**: with
    slack after the output, a probe found 63 of 63 rows past it overwritten,
    with no error.

  No shipped `hk.ops` op and no vLLM / SGLang route was affected, in 0.6.0 or
  in this release: all run clean under an end-fenced allocator on gfx950 and
  gfx1100, and no shipped kernel issues a `g.store`. The fix: on CDNA, the
  synchronous `g.load` stops at the tensor's last row (rows past it read zero),
  and `g.store` drops the rows past it (`group_load_sync`, `group_store_bounded`
  in `include/hk/cdna.cuh`). `hk.ops.mxfp8.gemm`, the one shipped CDNA kernel
  with a synchronous group copy, is bit-identical and within 0.6% of its speed.
  gfx942 still uses the unbounded copies (no hardware to test). RDNA3 already
  bounded them. `tests/hk/gpu/test_fenced.py` now runs every `hk.ops` family at
  ragged shapes under the end-fenced allocator, in CI on gfx950 and gfx1100.
- **`mxfp8.quant` read up to 63 rows past the activation** (#47). At
  M % 64 != 0, K2's fused quant (#41, after 0.6.0) copied whole 64-row tiles of
  x. When x ended an allocator segment, that read took a GPU memory fault in
  vLLM with `HK_VLLM_MXFP8` on: the first three real-weight DeepSeek-V4.1 runs
  with K1 and K2 on faulted, and since it depends on the allocator's layout,
  later runs of the same arm completed. It only ever affected main after
  0.6.0; the released 0.6.0 does
  not contain `mxfp8.quant`. The last block's copy now stops at row M, with the
  output unchanged: 48 of 48 ragged-M cases are bitwise equal to vLLM's quant +
  GEMM. The end-fenced allocator test (`test_fenced.py`, which #49 extended)
  came with it.

- **A stated strip-reduction order was not the documented one for products**
  (#57). With #52's `lanes=` / `xor=`, `(x * x).sum(axis=1, lanes="seq",
  xor=...)` compiled to fused multiply-adds: HIP's FMA contraction fused each
  product into the running sum across hk's inlined ops, so the sum was not the
  documented one of rounded products (4288 of 32768 lanes differed from it on
  MI355X). Only an ordered reduction of products was affected, and no `hk.ops`
  kernel uses one. #52 came after 0.6.0, so 0.6.0 is not affected. The ordered
  reduction now keeps each summand's rounding (no fma reaches in), and
  `sum(..., times=y)` is the fused form, stated.

**New in the language: hk.comm (gfx950)** -- hand-offs between the XCDs of
one GPU and between GPUs, each lowered to what a memory-model litmus suite
(20000 rounds a combination, across XCDs and across two MI355X) found correct.
The memory model is in [docs/design/comm.md](docs/design/comm.md).
- **Scoped tensors, signal / wait, symmetric memory** (#38):
  - `hk.Tensor[dtype, "device" | "system"]`: every load, store and scalar
    atomic through it carries the scope's cache-policy bits (on gfx950 both
    sides of a hand-off need them, so the scope belongs to the tensor);
  - `hk.comm.signal` (drain, barrier, one scoped flag store) and
    `hk.comm.wait` (a scoped poll with a timeout that sets an error element,
    or traps, instead of hanging);
  - `hk.comm.arrive` / `hk.comm.barrier`: an epoch'd grid barrier;
  - `hk.kernel(..., resident=True)` and `Kernel.resident_capacity()`: a grid
    that cannot all be resident raises `ResourceError` instead of deadlocking;
  - `hk.runtime.SymmetricBuffer`: one buffer per rank with every peer's mapped;
    while one is open, a launch refuses it bound to an unscoped parameter.
- **Every scalar fetch-op, and scoped col_vecs** (#45): `View.atomic_min_scalar`,
  `atomic_or_scalar`, `atomic_and_scalar`, `atomic_xor_scalar` and
  `atomic_exch_scalar` beside `atomic_add_scalar` / `atomic_max_scalar`, each
  returning the old value, plain or at a scoped tensor's scope; a row tile's 32-bit `col_vec` loaded from or
  stored to a scoped tensor (a per-row scale next to its tile).
- **Work queues and per-XCD aggregation** (#48): `hk.comm.claim` (a work
  queue's take: one scoped atomic add, no drain) for persistent kernels and
  task graphs; `hk.arch.xcd_id()`, read from the hardware; and
  `hk.comm.xcd_barrier`, a grid barrier with one global atomic per XCD a round
  (2.53 us against 3.71 us flat at 256 workgroups). The tile-signal pattern
  through a double-buffered symmetric buffer is tested across two GPUs.
- **Tagged mailboxes** (#50): `hk.comm.post` / `hk.comm.take` send a one-int32
  message and its tag as one 8-byte scoped store and one poll, no drain: a
  1.51 us round trip between two MI355X, against 3.40 us for a tile with
  signal / wait. A mailbox orders nothing else.

**New in the language: strips**
- **A stated row-reduction order** (#52, #57, gfx950): `t.sum(axis=1,
  lanes="seq" | "tree", xor=(32, 16, 8, 4, 2, 1))`, and the same for `max`,
  `min` and `prod`, for a result that must match another kernel's bit for bit.
  The guaranteed order:
  - a lane's own values are combined left to right (`"seq"`) or as a balanced
    tree (`"tree"`), then the lanes over the given xor offsets in that order,
    each of 1, 2, 4, 8, 16 and 32 once;
  - the strip's values are summed as they are: `(x * y).sum(...)` adds the
    rounded products, and the compiler does not fuse them into the sum;
  - `x.sum(axis=1, lanes="seq", xor=..., times=y)` is the fused form, spelled
    with fma: `s = fma(x0, y0, r(x1 * y1))`, then `s = fma(xi, yi, s)`, then the
    xor sums. That is what a compiled `s += a * b` loop computes, and it is
    dsv41_mono's and aiter's sum of squares in their norms (`times=x`).

  Without the arguments nothing changes (a tree, then 1 .. 32). Refused on
  wave32 targets, on columns and on WMMA tiles.

**Kernels and serving**
- **gfx950 attention: a spatial workgroup numbering, routed by measurement**
  (#55). `attn_cdna` (dense flash attention and paged prefill) gains
  `mapping="spatial"`: each XCD takes a contiguous range of (batch, KV head)
  clusters, so the workgroups resident on an XCD read one K/V stream. Outputs
  are bitwise the default numbering's. `hk.ops.attn` and the SDPA ops pick it
  when an XCD would otherwise hold five or more KV heads' streams, except for
  causal grids with fewer than twelve rounds of workgroups an XCD. Against the
  default, MHA at D 128 and 8K-128K tokens: 1.05-1.18x at 128 KV heads, up to
  1.08x at 64 and 1.05x at 40 (at 128 heads, 32K, the L2 hit rate goes from
  0.47 to 0.96). The routed choice was never below the default at the 74 timed
  points. The paged prefill accepts the option but is not
  routed (unmeasured).
- **K2's activation quant and its scale layout in one launch** (#41):
  `hk.ops.mxfp8.quant(x, packed=True)`, and `hk.mx_quant_store` writes the
  packed scale layout. The vLLM route makes one opaque call
  (`hk_vllm::mxfp8_linear`) instead of vLLM's quant, hk's scale packing and the
  GEMM: 1.01-1.58x per call at the 12 DeepSeek-V4.1 TP2 shapes, bit-identical.
- **Serving readiness** (#43, #44): the engine-contract test, a gfx950 GPU tier
  in the vLLM deployment image (ROCm 10), and `python3 -m hk.integration
  warm-routes --tp 2`, which builds the DeepSeek-V4.1 routes' kernels before
  the engine starts (31.2 s on an empty cache, 0.9 s warm).
  [docs/design/serving-readiness.md](docs/design/serving-readiness.md) lists what
  0.6 lacked for a deployment; [docs/design/comm.md](docs/design/comm.md) has
  hk.comm's roadmap (#51); [docs/design/placement-pass.md](docs/design/placement-pass.md)
  designs workgroup placement as a compiler pass (#62, design only).
- CI: pushes run only on main, and a PR's superseded runs are cancelled (#42);
  GPU-tier runs write no core dumps (#46).

**Compatibility**
- **Unset now means on.** With hk-dsl installed, vLLM and SGLang (and
  SGLang-Diffusion, after `install`) run with hk's validated routes unless
  `HK_DISABLE=1` or a route's switch is `0`. Every existing `HK_*=1` still
  works.
- No public DSL name or parameter was removed or renamed since 0.6.0 (checked
  name by name against the 0.6.0 wheel). Changed signatures only gained
  parameters: the strip reductions' `lanes=` and `xor=` (and `sum`'s
  `times=`), and `hk.kernel`'s `resident=`. One integration op was
  replaced: vLLM's MXFP8 route calls `hk_vllm::mxfp8_linear`, which quantizes
  inside, instead of `hk_vllm::mxfp8_block32` (#41).
- On gfx950, a synchronous `g.load` now reads zero for a tile's rows past the
  tensor, and `g.store` no longer writes them. In-bounds rows are unchanged.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-5`. 0.6.0's cached builds are rebuilt once.

**Known issues**
- **gfx942: the synchronous group copies are still unbounded** (#49). There
  is no gfx942 here to test the bounded form on, so on gfx942 a `g.load` /
  `g.store` at a partial last tile still reads and writes past the tensor.
- **gfx1100's defaults were not run end to end with the defaults.** Each
  gfx1100 route's evidence is its earlier opt-in run with real models. The
  default-on configuration itself was checked by the unit tests, the gfx1100
  GPU tier and the engine contract in each engine's environment, not by a
  model run on a W7900D with nothing set. The real-weight default-on smoke ran
  on MI355X (DeepSeek-V4.1-Flash, TP2).
- **The vLLM decoder attention backend stays off by default.** It replaces
  ROCm's default decoder attention for every model, and it raises rather than
  falls back on sliding windows, sinks, softcap, an fp8 KV cache, fp16 and
  other head sizes. It was validated on Qwen3-8B only, and on gfx950 it is
  slower at one request on long contexts (0.71-0.88x).
- **SGLang-Diffusion installs made before 0.7.0 stay opt-in** until
  `python3 -m hk.integration sglang-diffusion install` is run again.
- gfx1100's paged decode at `splits=1` leaves a zero-length (padded)
  request's output rows unspecified; through the merge (`splits` > 1), and on
  gfx950, they are zero. `hk.ops.convrot.linear` checks neither N % 128 nor
  K % 64 itself; the SGLang-Diffusion backend checks before calling it.
- gfx1100's default launch goes to the null stream, so a CUDA graph does not
  capture it. `HK_LAUNCH=codeobj` launches on torch's current stream.
- Under the vLLM image's hipcc (ROCm 10), `gemm_cdna`'s `dp128x224` candidate
  at 130x1792x768 spills and is refused. rocm/pytorch 7.2.3 builds it.
- gfx1201 and gfx942 are compile-only, and no route is on by default there.
- `.github/workflows/pypi.yml` cannot upload yet: PyPI's trusted publisher is
  not configured ([building](docs/building.md#wheels)). Releases go up with
  twine from the GitHub release's manylinux wheels.

See docs/api (marks "New after 0.6.0"), [integration](docs/integration.md),
[docs/design/comm.md](docs/design/comm.md) and
[docs/testing.md](docs/testing.md).
