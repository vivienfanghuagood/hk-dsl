---
name: hk-kernel
description: Write, check and tune an AMD GPU kernel with hk-dsl (Python, `import hk`) for gfx1100 (Radeon W7900D) and gfx950 (MI355X) -- elementwise, row-wise (norms, softmax), GEMM epilogues, attention. Use when asked to write or speed up an hk kernel, port a torch/Triton op to hk, or beat torch.compile on an AMD GPU.
---

# Writing an hk kernel

Use it in Claude Code from a checkout: copy or symlink this directory to
`.claude/skills/hk-kernel/` (project) or `~/.claude/skills/hk-kernel/` (user).
In Cursor: copy this file to `.cursor/rules/hk-kernel.mdc` and replace the
frontmatter with `description:` (the line above), `globs: **/*.py`,
`alwaysApply: false`. Other agents: read [llms.txt](../../llms.txt) first.

Paths below are relative to the repository root.

## The loop

1. **Start from a template if one fits.** `hk.templates.elementwise` /
   `rowwise` for pointwise and per-row work, `hk.templates.gemm(epilogue=...)`
   for a GEMM with a fused epilogue ([docs/templates/](../../docs/templates/),
   [docs/api/hk.templates.md](../../docs/api/hk.templates.md)). Use an
   existing `hk.ops` op when it already does the job
   ([docs/api/hk.ops.md](../../docs/api/hk.ops.md)). Write a kernel only when
   neither fits.
2. **Write the kernel in a file**, with `reference(...)` (torch, taking what
   the kernel reads, returning what it writes) and `inputs(device, shape)` and
   `SHAPES` (a decode-sized and a prefill-sized shape at least). Never in
   `python -c` or `exec`: the body must be readable by `inspect`.
3. **Check it**: `python -m hk.check FILE.py:NAME --json out.json`
   ([docs/check.md](../../docs/check.md)). Exit 0 pass, 1 wrong answer, 2 did
   not trace/build/run. Without a GPU it builds and reports resources only.
4. **Read the report**, in this order: `errors` (fix the line it names; the
   messages say how); `correct` (a FAIL names the first wrong index -- NaN
   there means the kernel never wrote it); `instantiations` (spill must be 0;
   occupancy against `max_waves_per_simd`); `time.vs.compile` (the bar is
   `win` or `tie` at every shape); `of_dram_peak` and `hints`.
5. **Iterate** on one thing at a time, re-running step 3. Make a plan
   choice a constexpr and let `--sweep NAME=v1,v2,...` time every value
   (each checked first) instead of editing and re-running. Check both
   architectures: `--arch gfx1100,gfx950` builds the one that is not in the
   machine; time on each GPU before calling it done.
6. Keep the reference exact enough: never loosen `--rtol` to make a FAIL pass
   without knowing why the error is larger than torch eager's own (printed
   beside it).
7. **When it is slower than it should be, profile it**: check -> profile ->
   change -> check. `python -m hk.profile FILE.py:NAME --json prof.json`
   (or `hk.check ... --profile`; [docs/profile.md](../../docs/profile.md))
   names the bound (`verdict.bound`: memory, compute, latency) and the
   losses with what each costs (`findings[].kind`/`cost`: `tail`,
   `epilogue`, `issue_bound`, `stall` with its `reason`, `narrow_access`,
   ...). Change the largest loss's cause, then re-check: the change is
   accepted on hk.check's interleaved timing, not the profile's.

## Strips or tiles

- **Streaming** (elementwise, norms, softmax, rotary, quantize; anything that
  reads each byte once): register strips, `hk.RS(hk.fp32, 1, CW, run)` with
  `run=8` for 16-bit data (16 bytes a lane), loads and stores by element
  (`x.at[row, c0]`), `bounded=True` for ragged edges. One workgroup per
  `CW`-column block of a row, consecutive workgroups along the row. The
  10-line example in [llms.txt](../../llms.txt) is this shape: it ties or
  beats torch.compile on both GPUs at 4096x14336 and is 3.7-4.7x it at
  16x14336; the same op on 16x64 `hk.RT` tiles is 0.54x on an MI355X.
- **Matrix core** (GEMM, attention): register tiles `hk.RT`, LDS tiles
  `hk.ST` staged by `hk.group(n)`, `acc += a @ b.T`. Prefer the GEMM template
  or `hk.ops.matmul` / `hk.attention` to writing one.
- Row reductions on strips: `(v * v).sum(axis=1)` per warp, then
  `hk.collective.cross_warp(vec, "add", WARPS)` across warps.

## Per-arch facts that change a kernel

| | gfx1100 (W7900D) | gfx950 (MI355X) |
|---|---|---|
| wave | 32 | 64 |
| strip chunk (`run=8`) | 256 columns | 512 columns |
| max waves / SIMD | 16 | 8 |
| LDS a workgroup may take | 64 KB | 160 KB |
| DRAM | 864 GB/s | 8 TB/s |
| matrix tile fragment | 16x16 | 16x32 bf16 operand (set by the passes) |
| global -> LDS DMA (`wait=False`) | no | yes |

`CW` a multiple of 512 fits both waves. Branch on `hk.arch.target().wave`
(or `.arch`) inside the body when a plan differs. All facts:
[docs/api/hk.target.md](../../docs/api/hk.target.md).

## Pitfalls

- Transcendentals (gelu, dgelu, sqrt, rsqrt, silu) are fp32-only: load into
  an fp32 tile/strip (the load converts) and store (the store converts).
  `gelu` is the tanh form. There is no tanh/sigmoid: compose from
  [hk.math](../../docs/api/hk.math.md).
- Positional parameters are tensors; constexprs are keyword-only with
  defaults, and each new value compiles anew -- do not pass a shape as one.
- A tile rebound inside a runtime loop is refused: `+=`, `into=`.
- A spill is a build error, not a warning: fewer live tiles, smaller tiles.
- `import hk` fixes the arch: set `HK_ARCH` before importing, or use
  `hk.check --arch`.
- `hk.ops` argument rules differ by arch in a few places (attention's
  lengths, the paged kernels;
  [docs/dx/api-consistency.md](../../docs/dx/api-consistency.md)); the row
  and elementwise ops are alike on both.
- On gfx1100 `k(...)` launches on the null stream: a CUDA graph does not
  capture it. Time or capture it with `HK_LAUNCH=codeobj` set (or through
  `hk.torch_op(k)`); `hk.check`'s eager timing is unaffected.
- Names marked *New after 0.4.0* in
  [docs/api](../../docs/api/README.md) need hk-dsl 0.5.0 or a checkout
  (`hk.__version__` says which you have).
- Only same-process, interleaved timing counts; under ~2% is a tie.
