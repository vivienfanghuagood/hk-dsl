# The IR: the hk dialect, its passes and its tools

`k.trace()` is an MLIR module in the `hk` dialect (`mlir/`), and
`k.trace().dump()` prints it with the names the C++ will use:

```mlir
func.func @row_sumsq(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<f32> {hk.name = "o"})
    attributes {hk.arch = "gfx1100", hk.kernel, hk.warps = 4 : i64, hk.lds_bytes = 0 : i64, ...} {
  %warp_id = hk.warp_id
  %bid_y = hk.block_idx y
  %mul = arith.muli %bid_y, %c_1 : i32
  %zv = hk.zero -> !hk.rv<f32, 16x64, row, col>
  %cols = hk.extent %x cols : !hk.gl<bf16>
  %cdiv = arith.ceildivsi %cols, %c_0 : i32
  scf.for %cb = %warp_id to %cdiv step %c4_i32 : i32 {
    %idx = hk.coord %c, %c, %mul, %mul_2 : !hk.coord<element>
    %ld = hk.load %x[%idx] : !hk.gl<bf16>, !hk.coord<element> -> !hk.rt<f32, 16x64, row>
    %mul_4 = hk.mul %ld, %ld : !hk.rt<f32, 16x64, row>, !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
    hk.row_sum %mul_4 into %zv {accumulate} : !hk.rt<f32, 16x64, row> into !hk.rv<f32, 16x64, row, col>
  } {hk.iv_name = "cb"}
  ...
}
func.func @row_sumsq__grid(...) -> (i32, i32, i32) attributes {hk.grid_of = @row_sumsq}
```

That is the `row_sumsq` kernel from [the language guide](language.md#kernels-and-parameters).

- [Types](#types)
- [Ops, the kernel and its grid](#ops-the-kernel-and-its-grid)
- [Passes](#passes)
- [Target facts](#target-facts)
- [Translation](#translation)
- [Value names in the dump](#value-names-in-the-dump)
- [hk-opt and hk-translate](#hk-opt-and-hk-translate)

## Types

* **Types**: `!hk.rt<f32, 16x64, row>` register tile (on gfx950,
  `!hk.rt<bf16, 16x64, row, frag 16x32>` names the matrix-core fragment it is
  laid out in), `!hk.rt<f32, 1x1024, strip<8>>` register strip (each lane owning 8-element runs of a row;
  `kittens::rs`), `!hk.st<bf16, 128x64, 2>` LDS tile (a double buffer),
  `!hk.rv<..., col>` register vector (a strip's carries the strip),
  `!hk.sv`, `!hk.gl<bf16>` global, `!hk.coord<tile|element>`,
  `!hk.granules`, `!hk.stage`. All print in full, so a dump reads on its own.

A strip is a third layout of `!hk.rt`, not a type of its own, so the maps,
loads, stores, reductions, broadcasts and column fills take it unchanged;
the ops whose C++ exists only for WMMA layouts (the mma variants, LDS tile
loads and stores, transposes, gathers, the row-limited loads, the
upper/lower fills, `philox_uniform`, `load_w4`, `atomic_add`) refuse it in
their verifiers, and a strip load or store takes an element coordinate.

Every RDNA kernel's C++ begins with `kittens::touch_kernargs<sizeof(globals)>()`,
which reads each 64-byte line of the kernel's arguments at entry: LLVM sinks
an argument's load to its first use, and a late one is a cold miss nothing
overlaps ([chapter 27](history/27-strips.md)).

The ODS definitions are in `mlir/include/hk/Dialect/HK/IR` (`HKTypes.td`,
`HKOps.td`).

## Ops, the kernel and its grid

* **Ops**, about a hundred, each with an optional destination: `hk.add %a, %b`
  makes a tile, `hk.add %a, %b into %d` writes one. Scalars are `arith`,
  control flow is `scf`, a kernel is a `func.func` with `hk.*` attributes, and
  the launch grid is a second, host-side function of the tensors.

In the dump above the kernel carries `hk.kernel`, `hk.arch`, `hk.warps` and
`hk.lds_bytes`, and the grid function `@row_sumsq__grid` points back at it
with `hk.grid_of`. The destination form is the IR's spelling of
[tiles are buffers](language.md#values-and-operators).

## Passes

* **Passes**, in C++ (`mlir/lib/Dialect/HK/Transforms`), run by `k.trace()` as
  `hk-kernel-pipeline`, in this order:
  - `hk-verify`: op legality, dtype support, no workgroup operation inside a
    branch or a runtime while, `hk.pingpong` in the kernel's own body on a
    condition from `warp_id`;
  - `hk-fragments`: on a target whose bf16 matrix-core operand is not 16x16
    (gfx950's is 16x32), names the fragment in the type of every tile an MMA
    reads, and of every tile tied to one by a cast, transpose or broadcast;
    a no-op on RDNA and gfx942;
  - `hk-lds-alloc`: LDS offsets, against the target's limit;
  - `hk-lds-pipeline`: every `lgkmcnt` derived, every asynchronously loaded
    tile proven waited for before it is read, loops walked until they reach
    a steady state; on gfx950, every `vmcnt` of an asynchronous group load
    derived too, and each copy checked across warps by counting barriers:
    no warp reads a buffer before every warp's share of the copy into it has
    landed, and no copy goes into a buffer before every warp's reads of it
    have (two barriers inside `hk.pingpong`, one outside);
  - `hk-check-budgets`: the register estimate against the occupancy the
    kernel asked for, and the slow-layout warnings.

  Their errors come out of `k.trace()` as `hk.VerifyError`, ending
  `at file:line in function` — the kernel author's line.

The passes are also where the third of
[the traps](README.md#the-traps) is caught: on RDNA3 only row-layout 16-bit
operands reach `ds_read_b128`, column layout degrades to 16 scalar
`ds_read_u16`, and the passes flag it while tracing, not at benchmark time.
The target tables say which layouts are fast; gfx950 reads both.

## Target facts

The arch facts that used to live in `RDNA.md`, kernel comments and notebooks
are now data: `python/hk/target/` for Python and `mlir/lib/Target/Target.cpp`
for the passes, with a test that the two tables agree field by field
(`tests/hk/ir/test_mlir_dialect.py`). On gfx1100 the VGPR granule is 24, so
occupancy 6 needs ≤240 registers, not ≤256.

| target | file | status |
|---|---|---|
| gfx1100, RDNA3 | `gfx1100.py` | probed and measured on a W7900D |
| gfx1201, RDNA4 | `gfx1201.py` | a 64 KB LDS pool; the VGPR granule carried over from gfx1100, not measured, so the target is `verified=False` and budget violations are warnings ([gfx1201](history/13-gfx1201.md)) |
| gfx942, CDNA3 | `gfx942.py` | modelled from the compiler, not run |
| gfx950, CDNA4 | `gfx950.py` | modelled from the compiler, timed on an MI355X ([chapter 29](history/29-cdna.md)) |

The CDNA tables differ in more than numbers: wave64, a 16x32 bf16 operand on
gfx950 (hence `hk-fragments`), a 4-bit lgkmcnt (a derived wait above 15 is
emitted as the cap), and LDS pools of 64 KB (gfx942) and 160 KB (gfx950).

## Translation

* **Translation** (`mlir/lib/Target/Cpp/HKToCpp.cpp`): hk + scf + arith to
  HipKittens C++ — named tile aliases, the value names of the dump, a
  provenance header — because it is the artifact you take to the disassembler
  when something spills. `codegen/scaffold.py` adds the module boundary
  (pybind, a `TORCH_LIBRARY` registration, or none) around the identical body.

The C++ for the kernel above:

```cpp
    const int cdiv = (cols + c1 - 1) / c1;
    for (int cb = warp_id; cb < cdiv; cb += 4) {
        const int mul1 = cb * c1;
        const kittens::coord<> idx(c, c, mul, mul1);
        rt_fp32_16x64_row ld;
        kittens::load(ld, g.x, idx);
        rt_fp32_16x64_row mul2;
        kittens::mul(mul2, ld, ld);
        kittens::row_sum(zv, mul2, zv);
    }
```

The translation is where the first of
[the traps](README.md#the-traps) is closed: `s_waitcnt` carries no
register dependence, so it binds the fragment with a volatile asm after every
wait — you cannot forget to. `k.source()` returns this C++ with the pybind
scaffold; `k.source("torch")`, `k.source("bare")` and `k.source("device")`
(the device-only code object's, with its argument-layout table) are the
other three (`python/hk/codegen/scaffold.py`).

## Value names in the dump

`dump()` prints the module with values named as the C++ names them: `%ld1` in
the dump is `ld1` in the C++. It prints a copy renamed by the translation's
own naming, so the two cannot drift (`python/hk/dsl/kernel.py`). The pass
errors do not yet do this — they name values by their base name (`lds`), not
the deduplicated one (`lds1`); see [limits](language.md#limits).

## hk-opt and hk-translate

`hk-opt` and `hk-translate` run the same passes and translation on a `.mlir`
file, which is how the lit tests in `mlir/test` are written. Their `RUN:` lines
are the reference for how to call them, for example:

```
hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s
hk-opt %s -split-input-file --hk-verify -verify-diagnostics
hk-opt %s --pass-pipeline='builtin.module(func.func(hk-lds-alloc))' | FileCheck %s
```

The tests are in `mlir/test/Dialect` (round trip, invalid IR),
`mlir/test/Transforms` (a file each for four of the five passes;
`hk-fragments` is tested from Python, in `tests/hk/ir/test_cdna.py`) and
`mlir/test/Target` (the C++ for a kernel, for indexed and row accesses, and
for a `while`). Building the tools and running them is in
[building](building.md) and [testing](testing.md).
