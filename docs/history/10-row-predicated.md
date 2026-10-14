# Row-predicated loads and stores

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Until now a tile could not hang off the end of a tensor. The decode GEMM padded
every batch to 16 rows with an extra launch, and the GEMM backed its last M
block up over its neighbour, so M had to be at least one block. Both were
there because the staging path had no bounds check. Predicating the issue
would make the right `vmcnt` a runtime value, and no instruction accepts that.

The way out is the buffer instruction's own range check. A buffer resource
whose range ends at the tensor's last row makes the hardware return zero for
a load past it and drop a store past it, with no branch, no masked lane, and
exactly the instructions a whole tile issues. So every hand-placed wait count
stays an immediate:

* **Staging** (global -> registers -> LDS) is always bounded this way, and the
  synchronous global -> LDS load and LDS -> global store skip the same rows.
  Free: the staging descriptor now starts at the tile's row and carries the
  K offset in `soffset`, which makes it loop-invariant. The default GEMM came
  out 15 VGPRs *lighter* (226 -> 211, occupancy 6 -> 7).
* **Register loads and stores** are opt-in, `hk.load(..., bounded=True)` and
  `hk.store(..., bounded=True)`, and compile to separate buffer-instruction
  functions (`kittens::load_bounded` / `store_bounded`). Plain `load` and
  `store` compile to exactly what they did before.

What it bought:

* **The GEMM takes any M.** The last block runs past M -- its A rows past M
  stage as zeros and its bounded store writes only rows below M -- and
  `gemm.matmul` no longer refuses M < 128; at M <= 16 it picks the 16-row
  tiling.
* **The decode GEMM stops padding.** Same kernel, one process, 36 layers back
  to back, eager: 19.85 -> 18.13 ms at batch 1, 1.6-1.7 ms a step at batch
  1-8 (1.08-1.10x) -- a pad launch and a sliced 16-row output per linear call,
  144 of each per step.

Gates. All 303 shipped kernel instantiations rebuilt against HEAD:
no occupancy dropped, no scratch, no build error. Performance against the
shipped GEMM, both built and timed in one process, interleaved, at 4096^3:
the default schedule is 1.0% faster in every run. The tuned schedule `matmul`
uses there ranged from 1.7% slower to 0.9% faster across five runs, always
inside the 3-7% round-to-round spread, and its main loop is 121 instructions
against 136 shipped, wait for wait otherwise the same. 343 GPU tests pass,
including a tensor that starts 2 bytes past a 16-byte boundary.

It took five wrong versions, each measured, and the reasons are what the next
person needs:

* **Always on in the pointer path:** paged decode spilled 66 VGPRs, for tiles
  that are never ragged. Hence opt-in.
* **A predicate that is a compile-time constant** on the unbounded side still
  changed how LLVM addressed every store: 32 base+offset stores became 32
  separate address pairs, an elementwise op went from 36 to 91 VGPRs, and the
  GEMM spilled. Hence separate functions, not a flag.
* **A descriptor built from a warp-derived coordinate** looked per-lane to
  LLVM, which wrapped each of the GEMM epilogue's 64 buffer stores in a
  waterfall loop -- 8% of the kernel. `make_srsrc` now `readfirstlane`s it;
  it is uniform by construction.
* **An unaligned fallback** in the bounded load cost the decode GEMM 10
  VGPRs and an occupancy step: LLVM merges a pointer loop into wide loads but
  cannot merge buffer intrinsics, and reserves the fallback's registers
  whether it runs or not. Dropped -- gfx11 runs with unaligned access on, and
  the 2-byte-misaligned test covers it.
* **A whole-tile fast path** in the bounded store, falling back to the old
  pointer store, made the default GEMM spill 76 VGPRs. Reverted.

Scope: rows only. N and K still have to tile exactly. Also seen in passing
and not touched: both the shipped and the new GEMM wait `vmcnt(0)` right
before every iteration's barrier, so under `write_pos=1` the prefetch for
tile t+2 never overlaps tile t's math -- a lever for the GEMM itself.

---

← [The decode kernel](09-decode-kernel.md) · [Development record](README.md) · [A reduction across workgroups](11-cross-workgroup-reduction.md) →
