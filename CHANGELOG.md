# Changelog

Release notes for each version are also on its
[GitHub release](https://github.com/vivienfanghuagood/hk-dsl/releases).
The next version's notes collect here until it is released.

## Unreleased (draft notes for 0.7.0)

**hk is on by default.** With hk-dsl installed, its vLLM and SGLang plugins
apply every route that was validated with real weights on your GPU: outputs
bit-identical to the engine's, or error within the engine's own noise.
Before, every route needed its own `HK_*=1`. Turning hk off is now for
comparison experiments:

```bash
HK_DISABLE=1 vllm serve <model>        # stock: nothing patched, in every engine
HK_VLLM_MXFP8=0 vllm serve <model>     # one route off (HK_<ROUTE>=0)
python3 -m hk.integration --status     # every route on or off, and which rule decided
```

- **On by default** (the evidence for each is in
  [integration](docs/integration.md#on-by-default-and-turning-it-off)):
  - **gfx950 (MI355X):** K2 MXFP8 linear (`HK_VLLM_MXFP8`) and K1 MoE a8w4
    (`HK_VLLM_MOE_A8W4`, still at M >= 8192 only) for DeepSeek-V4.1, and AWQ
    4-bit (`HK_VLLM_INT4`).
  - **gfx1100 (W7900D):**
    - vLLM: linear (`HK_VLLM_GEMM`), bf16 MoE (`HK_VLLM_MOE`), AWQ 4-bit,
      and the encoder SDPA (`HK_VLLM_SDPA`);
    - SGLang: linear and MoE (`HK_SGLANG_LINEAR`, `HK_SGLANG_MOE`);
    - SGLang-Diffusion: attention and linear (`HK_SGLD_ATTN`, `HK_SGLD_LINEAR`);
    - K3, DiT attention in `hk.ops.sdpa` (`HK_SDPA_K3`).
  - On a GPU a route was not validated on, it stays off.
- **Stays opt-in:**
  - **ConvRot INT8** (`HK_SGLD=attn,linear,convrot` or `HK_SGLD_CONVROT=1`): it is
    lossy by design, a model-level choice and not a toolchain default.
  - **The vLLM decoder attention backend** (`HK_VLLM_BACKEND=1`): it replaces
    ROCm's default decoder attention for every model, and it raises rather
    than falls back on sliding windows, sinks, softcap, fp8 KV caches, fp16
    and other head sizes.
  - **`HK_MXFP8_ANY`.**
- **Precedence:**
  1. `HK_DISABLE=1` beats everything.
  2. An engine's switch at `0` (`HK_PATCH_VLLM=0`, `HK_PATCH_SGLANG=0`,
     `HK_SGLD=0`) turns off that engine's routes.
  3. A route's own switch, `0` or `1`, decides that route.
  4. Otherwise the default applies.

  The existing `HK_*=1` switches keep working. An empty value counts as unset.
- **The engine contract is checked at startup.** Before patching, a plugin checks
  every route it is about to apply against the installed vLLM / AITER / SGLang
  (`hk.integration.contract`). If one is broken, for example by a version
  mismatch, it patches **nothing** in that engine, prints one warning line, and
  the engine runs stock. It never runs half-patched: a route that raises while
  it is applied undoes the ones applied before it.
- **The first JIT compile says so.** If the routes' kernels were not pre-built,
  the first compile in each process prints one line, which suggests
  `python3 -m hk.integration warm-routes`.
- **Comparison scripts must say stock.** An A/B's stock arm sets `HK_DISABLE=1`:
  unset no longer means stock. `python -m hk.integration ab` / `headroom` and
  `tools/hk-bench/` do this themselves.
- **SGLang-Diffusion:** an existing `sglang-diffusion install` keeps its old
  boot, which needs `HK_SGLD=1`. Run `install` again for the default.
- **Fixed:** several routes in one vLLM no longer nest its compile cache root
  (`.../vllm/hk-int4+mxfp8`, the same in every process).
- **Fixed: GPU detection in images that list several build archs.** The vLLM
  ROCm image sets `PYTORCH_ROCM_ARCH=gfx90a;gfx942;gfx950;...`. Without
  `HK_ARCH`, hk took the list's first entry, so an MI355X was gfx90a, and hk
  then built for gfx1100 and left the gfx950 routes unpatched.
  `hk.runtime.arch.detect()` now reads the GPU from the KFD topology in sysfs,
  which creates no HIP context. It uses a list variable only as the last
  resort; `HK_ARCH`, or a variable naming exactly one arch, still wins.

**Fixes since 0.6.0**
- **`mxfp8.quant` read past the activation's last row** (#47). At M % 64 != 0
  it copied whole 64-row tiles, and it memory-faulted in vLLM when the
  activation ended an allocator segment. It hit 3 of 3 real-weight runs with
  K1 and K2 on. The last block's copy now stops at row M, with the output
  unchanged (48/48 ragged-M cases bitwise equal to vLLM's). `mxfp8.quant`
  came after 0.6.0, so 0.6.0 is not affected.
- **gfx950: synchronous `g.load` / `g.store` went out of bounds on a partial
  last tile** (#49). On CDNA, a DSL kernel's group copy read up to a tile's
  rows past the tensor. Worse, its `g.store` silently wrote them: 63 of 63
  rows past `y` in the probe. Both are now bounded at the tensor's last row
  (`group_load_sync`, `group_store_bounded`), and `mxfp8.gemm` stays
  bit-identical, within 0.6%. This affects 0.6.0's DSL on gfx950. No shipped
  op or route was affected. gfx942's copies are still unbounded (no hardware
  to test). `tests/hk/gpu/test_fenced.py` runs every `hk.ops` family at ragged
  shapes under an end-fenced allocator, on gfx950 and gfx1100.

**Also since 0.6.0**
- **K2's activation quant and scale layout in one launch** (#41): the route is
  1.01-1.58x faster, bit-identical.
- **Serving readiness** (#44):
  - the engine-contract test;
  - a gfx950 GPU tier in the vLLM deployment image;
  - `python3 -m hk.integration warm-routes`.
- **`hk.comm` on gfx950** (#38, #45, #48, #50):
  - scoped tensors, signal / wait, and symmetric memory;
  - every scalar fetch-op;
  - `claim`, `xcd_id` and `xcd_barrier`;
  - tagged mailboxes.
- **No core dumps from GPU-tier runs** (#46).
