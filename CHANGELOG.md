# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## Unreleased

What is on main after 0.7.0. `pip install hk-dsl` (0.7.0) does not have it;
the API reference marks these names *New after 0.7.0*.

**Workgroup placement as a pass** (#63, gfx950; design in
[docs/design/placement-pass.md](docs/design/placement-pass.md), #62).
Which XCD runs which workgroup, as one mechanism instead of hand-written
remaps:
- `hk.kernel(placement=None | "auto" | hk.Placement)`. `hk.Placement.cluster()`
  runs one sharing cluster an XCD at a time, its key found by the analysis
  (`hk.placement.analyze`: each global read's sharing key over the grid axes)
  or named with `hk.Digit`s; `hk.Placement.chunked(chunk, tail)` renumbers
  `block_idx.x` alone. `"auto"` launches the cluster placement where
  `hk.placement.choose` takes it: not for order-dependent or resident
  kernels, from five KV streams an XCD, and not for non-uniform work under
  twelve rounds an XCD (thresholds measured in #55). `Kernel.placed()` and
  `Kernel.placement_choice()` give the placed kernel and the decision.
- Outputs are bitwise the unplaced kernel's. `attn_cdna`'s `mapping="spatial"`
  (#55) is now `hk.Placement.cluster()`, and `gemm_cdna` / `gemm_cdna_dp`
  take `Placement.chunked` instead of their inline remaps: 11 GEMM shapes
  bitwise, new/old 0.997-1.007; 15 attention points bitwise, 0.993-1.003 of
  #55's hand-written numbering.
- `hk.arch.xcd_count()`: KFD's `num_xcc` (MI350P / DPX: 4, MI300A: 6), the
  arch's default otherwise, `HK_XCDS` to override. With 1 (gfx1100) every
  placement is the identity.
- `python -m hk.check FILE.py:NAME --placement --inputs ...` checks that a
  launch's placement covers its grid exactly once, evaluating the decode the
  kernel's own emitted ops compute ([check](docs/check.md#which-xcd-runs-which-workgroup---placement)).
- **M2** (#65): K2 (`mxfp8`), K1 (`moe_a8w4`) and the stream-K plan take it too, so hk.ops has no
  hand-written XCD remap left (and no `NUM_XCDS`); persistent kernels use
  `hk.placement.renumber` / `hk.placement.xcd_range`. K2 at the 18 DSV4.1 points bitwise,
  0.992-1.011; K1 bitwise, and **1.19-1.24x faster** at M where its old remap fell back to the
  hardware's order (stage-1 grid not a multiple of 8: M 9500, 10000, 14000, 15001), within 0.1%
  where it applied. `HK_XCDS=1` turns every placement off.
- A user guide, [docs/placement.md](docs/placement.md) (when to use it, examples, the measured
  tables, limits, turning it off); the design document rewritten as built; an
  [hk.placement](docs/api/hk.placement.md) API page. `hk.Digit.of`'s radix may be spelled
  `"tensor.dim"` (`"k.depth"`).

**`hk.map_cpp` and i32 strips** (#53, gfx950;
[the language](docs/language.md#elementwise-c-over-strips-and-i32-strips-gfx950)).
- `hk.map_cpp(expr, *args, dtype=None)` evaluates one C++ expression for each
  element of its strip arguments (`x0`, `x1`, ... bound to the elements, or
  to an `hk.Int` / Python int), into an fp32 or i32 strip. It is for what a
  kernel must reproduce bit for bit and the operators cannot say: bit
  manipulation, a fused multiply-add, one particular instruction. The
  expression compiles with floating-point contraction off (`a * b + c` is two
  roundings; an fma is `__builtin_fmaf`). Strips of fp32 and i32 together
  must say `dtype=`; wave32 targets refuse it.
- Strips may hold i32: they load, store, convert with `.to` and are
  `hk.map_cpp`'s arguments and results. Operators and reductions on them are
  refused, so integer arithmetic is written in `hk.map_cpp`.

**Fixes**
- **K1 MoE a8w4 at decode M ran 11% slower after #65** (M 64: 0.89x of before, found by the MI355X
  scoreboard). The sorted kernels' grids are sized for the most sorted blocks a launch can have,
  and #65's `Placement.chunked(0, "spread")` dealt contiguous ranges of the whole grid, so the
  live blocks (235 of 390 at M 64) landed on the first five XCDs. They now renumber in the body
  over the live blocks only (`hk.placement.renumber(..., live=)`, new). Against main before
  #65, K1 is 1.04-1.33x at every M from 64 to 16384, and bitwise the same.
- **gfx950: fp32 register tiles through LDS were FLAT loads and stores**
  (#59). `store(st<float>, rt<float>)` / `load(rt<float>, st<float>)` addressed
  the tile through a generic pointer, so every access was FLAT: slower to
  issue than `ds_read` / `ds_write`, and counted in vmcnt as well as lgkmcnt,
  so each wait for an LDS value also waited for every global load in flight.
  They are `ds_read` / `ds_write` now. Results are unchanged; affected are the
  cross-warp combines of fp32 partials (`hk.ops.decode_gemm`, `w4a16`, `moe`)
  and user kernels that move fp32 tiles through LDS on gfx950.

**Wheels** (#66)
- From the next release, cp312 is one `manylinux_2_35` wheel, built on glibc
  2.35 in vLLM's ROCm image, like cp310: through 0.7.0 cp312 was built on
  Ubuntu 24.04 and needed glibc 2.38, so `pip install hk-dsl` found no cp312
  wheel in vLLM's ROCm images until 0.7.0's extra `manylinux_2_35` wheel.
- `scripts/check_wheel_glibc.py` fails a wheel that needs a newer glibc (or
  GLIBCXX / CXXABI, with auditwheel) than its manylinux tag; CI's IR leg
  builds the cp310 wheel and runs it.
- Reproducible wheels: no build path in the binaries (`-ffile-prefix-map`,
  RUNPATH `$ORIGIN` at link time), entry times from `SOURCE_DATE_EPOCH`
  (the last commit's), fixed modes; the same commit builds byte-identical
  wheels ([building](docs/building.md#reproducible-wheels)).

**Docs**
- The API reference marks names against the 0.7.0 wheel (it marked them
  against 0.6.0, so `placement=` and `hk.arch.xcd_count` read as if 0.7.0 had
  them), and documents `hk.Placement`, `hk.Digit` and
  `hk.runtime.SymmetricBuffer`.
- [The language](docs/language.md) gains `hk.comm`, workgroup placement, the
  stated strip-reduction order (#52, #57), the bounded group copies on gfx950
  (#49) and LDS allocated in a runtime branch (#56).
- The docs site builds with `mkdocs build --strict` in CI: every page is in
  the nav (hk.profile, schedules, hk.tune and the end-to-end loop were not),
  heading ids follow GitHub's (so `#which-xcd-runs-which-workgroup---placement`
  resolves on both), and the links #54 broke (`#turning-them-on`) point at
  `#on-by-default-and-turning-it-off`.
- The install pages state 0.7.0's three wheels and that CPython 3.12 needs
  glibc 2.35, not 2.38; [the runtime](docs/runtime.md#which-gpu) has the arch
  detection order 0.7.0 introduced.
- [Releasing](docs/releasing.md): the release steps, and the documentation
  sync that must pass before a tag.
- Examples [14](examples/README.md) (an hk.comm task graph, gfx950) and
  [15](examples/README.md) (workgroup placement).

**CI and tests**
- Several gfx950 runners in parallel, each with its own GPUs and state, and
  GPU locks shared with manual measurements (#60,
  [testing](docs/testing.md#several-gfx950-runners)).
- A cancelled GPU job no longer leaves its container on the GPU (#69,
  [testing](docs/testing.md#the-gpu-tier)).
- Tests of gfx1100's measured data are pinned to gfx1100
  (`@pytest.mark.arch`, #64).

## 0.7.0

hk-dsl 0.7.0: hk is on by default in vLLM and SGLang, for the routes validated
with real weights on your GPU; `HK_DISABLE=1` turns it off. Also `hk.comm` for
hand-offs between gfx950's XCDs and between GPUs, and fixes for two silent
errors that 0.6.0's DSL has in user kernels: on gfx950 a synchronous `g.store`
at a partial last tile overwrites the memory after the tensor, and LDS
allocated inside a runtime branch reads back as 0. No shipped op or route was
affected by either.

On PyPI: `pip install hk-dsl` (CPython 3.10 or 3.12 on glibc 2.35+, Linux
x86_64). Three wheels: cp310 `manylinux_2_35`, and cp312 both
`manylinux_2_35` and `manylinux_2_38`; pip picks the newest tag the system's
glibc takes. The cp312 `manylinux_2_35` wheel was added on release day, for
glibc 2.35 systems such as vLLM's ROCm images (Ubuntu 22.04), where the
`manylinux_2_38` one does not install. The wheels on the GitHub release are
the same files; the manylinux copies are the ones on PyPI. Each carries the
MLIR bindings for its interpreter and the HipKittens headers, so at run time
only hipcc is needed (and a ROCm build of torch to launch).

    pip install hk_dsl-0.7.0-cp312-cp312-manylinux_2_35_x86_64.whl

(This paragraph first said "3.12 on glibc 2.38+", as do 0.7.0's PyPI page and
the GitHub release notes; the `manylinux_2_35` cp312 wheel makes 2.35 enough.)

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
- **LDS allocated inside a runtime branch was not part of the launch's LDS**
  (#56). The `hk-lds-alloc` pass sized a kernel's launch from the
  allocations in the kernel's own block only, while the generated C++ bumps
  its allocator along the path the kernel takes. So a tile that a kernel
  allocated with `hk.lds.alloc` inside a runtime `if` (a branch on a device
  value) lay past the end of the LDS the launch requested: its reads returned
  0 and its writes were dropped, with no error. An allocation inside a runtime
  loop was past the end the same way. Found porting DeepSeek-V4.1's K1 to hk,
  where a reduction tile allocated in a role branch read back as zeros (87.5%
  of q wrong). Every released version through 0.6.0 has the pass in that form,
  so a user kernel that allocates LDS in a runtime branch or loop is affected
  there. No shipped kernel is: a scan of every kernel the IR and codegen tiers
  translate (both arches) and the gfx950 GPU suite builds found no LDS
  allocation outside a kernel's own block, in 0.6.0 or in this release. The
  pass now walks every region: the two arms of an `if` start at the same
  offset and the larger one continues, and an allocation inside a runtime
  `for` / `while` is refused, with a message to allocate it before the loop.
  `tests/hk/gpu/test_lds_branch.py` reads such a tile back across lanes (all
  zeros before the fix, exact after).
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

- **Tests: `test_sdpa` compares against fp32 attention on torch's math
  backend** (#61), test-only: no kernel or library change. Its intermittent
  head_dim-64 failure on a shared MI355X was torch's reference, not hk: in the
  vLLM ROCm 10 image, torch's fused bf16 SDPA (AOTriton 0.13.50) gave wrong,
  nondeterministic tiles under same-GPU load, while hk's kernel gave identical
  bits on every call.

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
- An LDS allocation inside a runtime loop is now refused at build time (it
  was silently past the launch's LDS); allocate it before the loop. One inside
  a runtime branch now counts toward the launch's LDS, so such a kernel asks
  for more LDS than 0.6.0's did.
- The dialect's semantic version, part of the compile-cache key, is now
  `hk-dialect-5`. 0.6.0's cached builds are rebuilt once.

**Known issues**
- **gfx942: the synchronous group copies are still unbounded** (#49). There
  is no gfx942 here to test the bounded form on, so on gfx942 a `g.load` /
  `g.store` at a partial last tile still reads and writes past the tensor.
- **The gfx1100 defaults were not run end to end.** Each
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
