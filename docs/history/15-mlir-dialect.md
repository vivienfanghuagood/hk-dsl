# The IR as an MLIR dialect, and kernels that read like the math

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

Until here a kernel was a list of flat calls — `_ops.store(o, _ops.mul_row(
_ops.load(x, _ops.elem_coord(...), ty), r), idx)` — recorded into a Python
IR. That is a builder API, not an IR anyone would recognise, and nothing
else could read it. Now the IR is MLIR: an `hk` dialect in C++ (`mlir/`)
with its verifiers, the four passes and the translation to HipKittens C++,
and kernels are written in a Python that reads like the operations it
performs (`acc += a @ b.T`, `o[idx] = (v * r).to(hk.bf16)`, a plain `for`
over a device bound). The Python IR, its builder, verifier and passes, the
Python emitter and the flat API are deleted. The backend did not change:
hipcc and include/rdna3, so nothing at run time needs LLVM.

The move was gated three times, each against the code it replaced.

**Gate A — the dialect against the Python IR.** A temporary converter
turned every kernel the old front end traced into the dialect, and ran it
through the C++ passes and the C++ translation. For all 354 shipped
instantiations, and the constexpr points the tests use (708 translations),
the C++ was byte-identical to the Python emitter's. The whole no-GPU suite
ran under the gate as its corpus — 394 tests, the error paths included,
where the C++ passes had to refuse what the Python ones refused, in the same
words. Then the converter was deleted with the rest.

**Gate B — each kernel family against its old self** (elementwise, norm,
quant, fused, decode GEMV, GEMM, paged, attention):

* All 422 instantiations rendered both ways. 49 are byte-identical, 281
  differ only in a loop variable's name (`i` became the `kb` the source
  wrote), and 92 differ in constant folding, hoist order or names. Those 92
  were compiled: VGPRs, SGPRs, scratch, LDS and occupancy are identical in
  every one.
* tests/hk/gpu: 379 passed on gfx1100.
* Old and new in one process, interleaved for 7 rounds, the old as control:

  | kernel | shape | new / old |
  |---|---|---|
  | GEMM | 4096³ | 1.002x (71.6 TF) |
  | attention d128 | N=4096, non-causal / causal | 1.001x / 0.993x |
  | decode GEMV | M=1 N=4096, M=1 N=12288, M=4 N=4096 | 1.010x / 0.999x / 1.000x |
  | rmsnorm | 8192x4096, 8192x8192, 32768x1024 | 0.998x / 1.002x / 1.002x |

  Every gap is smaller than that row's own round-to-round spread, which is
  the expected answer for kernels whose C++ differs in names.

After the old front end was deleted, the 422 sources were rendered again:
byte-identical to the migration commit's, so the numbers above are the
shipped code's.

**Gate C — the engine.** vLLM 0.16.1 serving Qwen3-8B on the W7900D with the
GEMM patch (`vllm_e2e.py`, eager, context 1024, batch 1 and 16, 72 output
tokens, three rounds of each arm alternating), once from the tree before the
move and once after, each against stock vLLM in its own run:

| | before | after |
|---|---|---|
| total, batch 1 / 16 | 1.820x / 1.338x | 1.824x / 1.340x |
| prefill | 1.125x / 1.160x | 1.125x / 1.160x |
| decode per step | 1.928x / 1.735x | 1.933x / 1.746x |

The routing counters match call for call (5,472 prefill GEMMs, 102,553
decode GEMVs, no fallback). At both shapes the first request's greedy output
is identical to stock, and so to the other run's (72/72).

The first attempt at this gate measured nothing. With vLLM's engine in its
own process, the patch installed in the driver never reached the model, and
both arms timed as stock (0.995–1.000x, `hk 0, hk_decode 0`). The tool prints
the routing counters for exactly this reason. The gate is run with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, as every earlier measurement here was.

**Tests, now.** `mlir/test` (lit): 8, for the op verifiers, a round trip
through text, each of the four passes on hand-written IR (including the
workgroup-operation checks and the budget warning), and the translation of
a kernel and of a `while`. tests/hk/ir and
tests/hk/codegen: 990, no xfails. tests/hk/gpu: 386 passed on gfx1100: the
379 above, four for `while`, and three for the RMSNorm the plan for this move
set as the language's target. That one is written as the plan wrote it,
except for two spellings, both kept on purpose:
- a vector type is named by the tile it reduces, `hk.col_vec(T)`, not
  `hk.ColVec[fp32, 16]`, because the dialect ties a vector's layout to its
  tile;
- a load names the register tile it makes, `x.at[r, c].load(T)`, not
  `x.tile(16, 64)[r, c].load(fp32)`, because `x.tile[r, c]` already means a
  tile-unit coordinate.

**Wheels.** `scripts/build_wheel.sh` builds one per interpreter, because
`hk._mlir` is nanobind extension modules for one CPython:
`hk_dsl-0.2.0-cp310-cp310-linux_x86_64.whl` (11.2 MB) and
`hk_dsl-0.2.0-cp312-cp312-linux_x86_64.whl` (11.3 MB) -- named `hipkittens` when
they were measured, before the split. Each
carries the package, include/ (the generated kernels include it at JIT
time) and the bindings. A wheel built for another interpreter installs and
then fails at import, so the script refuses to build one. The link-time
`libHKPythonCAPI.so` symlink is dropped, because a wheel would carry it as a
second 34 MB copy. Installed outside the checkout, each wheel
traces, translates, compiles against its own headers and launches on the
W7900D, with a result equal to torch's bit for bit.

**What porting the tests found.** About fifteen test files used the flat
API, and porting them found real problems in the new front end, not just
spelling:

* *Error locations were wrong two ways.* The frame filter skipped every
  file whose path contained `/hk/`, which includes `hk/ops` and `tests/hk`,
  so an error in a shipped kernel pointed at its caller. That bug predates
  the DSL. And an op that recorded its location directly (`barrier`,
  `lds.alloc`, `lds.wait_for`, a store) skipped one frame too many, landing
  on the line that called `k.trace()`. The filter is now the front end's own
  directories, with no fixed frame count. `hk.scope()` no longer reports
  contextlib, and a body lifted into a loop function reports the kernel
  function's name.
* *The dump and the C++ named values differently* (`%ld_0` vs `ld1`).
  `dump()` now prints a copy renamed by the translation's own naming, so the
  two cannot drift.
* *Seven of the Python-side messages had lost information* — which mma
  variant the layouts select, that an alternative is `x[idx].store(v)`, why a
  subtile has to divide its parent, that `//` on a tile is not a rounding
  division. The old wording is back. The C++ passes never lost theirs.

**What the language does not do.** Each of these raises, with the
spelling that works:

* A loop or branch body that `return`s, `break`s or `continue`s is left as
  Python, so its condition has to be known at trace time. `while ... else`
  is not rewritten.
* A workgroup operation (`barrier`, group loads and stores, staging) cannot
  sit inside a runtime branch or a runtime `while`: a warp that leaves early
  leaves the rest waiting. A `for` over a bound every warp agrees on is fine.
* A tile rebound inside a runtime loop is refused; it is updated in place
  (`+=`, `into=`), so that every iteration writes the same registers.
* A kernel body must be source `inspect` can read. Code built with `exec`
  or a lambda traces, but its loops cannot be runtime loops.
* Pass errors name values by their base name (`lds`), not the C++'s
  deduplicated one (`lds1`), so "add lds to that lds_wait_for" can be
  ambiguous with several LDS tiles in flight. The old front end had the same
  limit.
* Nothing is eliminated before the passes, so an unused allocation still
  counts in `lds_bytes`. The old front end behaved the same way.
* MLIR is built from source (23.1.3). Wheels are Linux x86_64 only, one
  CPython each; gfx1201 is still compile-only.

**Reproducing.**

```bash
scripts/build_mlir.sh && scripts/build_hk_mlir.sh
ninja -C $HK_MLIR_BUILD check-hk
python3 -m pytest tests/hk/ir tests/hk/codegen -q
python3 -m pytest tests/hk/gpu -q                       # gfx1100
scripts/build_wheel.sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 HK_GPU_UTIL=0.85 python3 tools/hk-bench/vllm_e2e.py \
    --context 1024 --batch 1,16 --tokens 72 --short 24 --reps 3 --arms stock,gemm
```

**For whoever continues**, in order of value:

1. Name values in pass errors the way `dump()` does, by the translation's
   naming, so a message names one tile.
2. `break` and `continue` in runtime loops: a flag carried through an
   `scf.while`, or the `cf` dialect.
3. A canonicalize/DCE pass in front of the hk passes. It will change the
   C++, so it needs Gate B again, resources first.
4. Build MLIR in CI and publish the wheels; today they are built by hand on
   the machines that have the toolchain.

---

← [One patch layer and a routing table](14-routing-table.md) · [Development record](README.md)
