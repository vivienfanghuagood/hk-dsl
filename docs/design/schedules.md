# Design: schedules as data (`hk.schedules`)

Status: K1 of [the tuning goal](tuning-goal.md), criterion T1. GEMM on both
GPUs; attention later under the same protocol.

## The problem

hk's GEMMs were parameterized but not as data. gfx1100's WMMA GEMM
(`hk.ops.gemm.gemm_kernel`, 16 keyword knobs), the decode GEMV
(`hk.ops.decode_gemm`), and gfx950's two (`hk.ops.gemm_cdna`'s pingpong,
`hk.ops.gemm_cdna_dp`'s four-warp ring) each took their own keyword set;
legality was the factories' scattered `ValueError`s plus whatever the passes
and the spill gate refused after a compile; the shape-to-kernel rules lived
in `schedule_for`, `route`, `tail_for`, `wave_cut` and `run`; and the tools
that write route tables re-encoded all of it. A searcher could not ask
"which schedules exist for this shape, and which of them will build?".

## The type

One frozen dataclass per kernel family, all following one protocol
(`hk.schedules.Schedule`, and `GemmSchedule` for GEMMs):

| family | class | kernel factory | arch |
|---|---|---|---|
| `rdna` | `Rdna` | `hk.ops.gemm.gemm_kernel` | gfx1100 |
| `decode` | `Decode` | `hk.ops.decode_gemm.decode_gemm_kernel` | gfx1100, gfx950 |
| `pp` | `PingPong` | `hk.ops.gemm_cdna.gemm_kernel` | gfx950 |
| `dp` | `Dp` | `hk.ops.gemm_cdna_dp.gemm_kernel` | gfx950 |
| `cols` | `Columns` | two schedules on column ranges of C | both |

**Why one class per family and not one `GemmSchedule` with every knob.**
The families share almost no knobs that mean the same thing: `warps` is a
wave count in `rdna` and `decode`, a fixed 8 in `pp` and a 2-D grid in `dp`;
`stages` exists only in `dp`, `write_pos` only in `rdna`. One flat type would
be a union in which most fields are meaningless for most values of a `kind`
field, and every rule would start with `if kind == ...`. A class per family
keeps each knob's meaning, default and legal values next to the factory that
reads it, while the protocol below is what a searcher programs against; it
never needs to know which family it holds.

A field is a knob, declared once:

```python
block_m: int = knob(128, values=(16, 32, 64, 128, 256), seed=(64, 128),
                    doc="rows of C a workgroup computes: ...")
```

`values` is the axis a neighbourhood steps along, `seed` the values the
default space takes, `doc` the one line `docs/schedules.md` shows; `gate=True`
marks `max_vgprs` / `min_occupancy`, which change what a build refuses and
never the C++. **Adding a knob** is one field and the factory parameter it is
passed to (`kernel()` passes every field by name); add it to the legality
check if it has a constraint, and to `_regs` if it changes the live registers.

`Columns(head, tail, cut)` is the composite: `head` on C's first `cut`
columns, `tail` on the rest. `cut` is a count, `"block"` (N less its
remainder by the head's `block_n`: the pingpong's 128-column tail of a
151936-wide lm_head) or `"wave"` (the columns whose 256x256 blocks fill whole
waves of the MI355X's 256 CUs: route `ppw`). Split-K is a knob (`splits`) of
`pp`, `dp` and `decode`; for `pp`/`dp` a split schedule's `steps()` are the
GEMM into an fp32 scratch and the reduction, for `decode` one launch (the
last workgroup to arrive sums the shares).

## The protocol (the API K3 calls)

```python
import hk.schedules as S

s = S.lookup(m, n, k, arch)              # what hk runs today (op="matmul" | "template")
s.check(m, n, k, arch, dtype="bf16")     # [] or reasons, before anything compiles
s.check_kernel(arch)                     # the shape-free half: will k.build() pass?
s.predict(arch)                          # {"vgpr", "lds", "occupancy", "tiles"}
for t in S.space(m, n, k, arch): ...     # every family's legal seed space
for t in S.space(m, n, k, arch, around=s, radius=1): ...   # s's neighbourhood
S.Dp.space(m, n, k, arch, axes={"stages": [2, 3, 4, 5], "wgm": "all"})
t.steps(m, n, k)                         # the launches: [Step(kernel, sched, cols, epilogue, into)]
t.kernel(epilogue=None, arch=None)       # the hk.Kernel; .source() / .build() need no GPU
t.kernels(m, n, k, arch=...)             # [(step, kernel, constexprs)] incl. the reduction
t.launch(a, b, out=None)                 # run the whole call (GPU)
hk.templates.gemm(a, b, epilogue=..., schedule=t)
rec = t.to_record(); S.from_record(rec) == t
t.replace(k_step=128); t.knobs(); t.changed(); type(t).axes()
```

`S.Illegal` (a `ValueError`) carries `reasons`; `t.require(m, n, k, arch)`
raises it, as `launch()` and `templates.gemm(schedule=)` do.

## Legality

`check()` refuses before compiling, with a sentence per reason:

1. **Arch and dtype** -- the family's `arches`; bf16 only for now.
2. **Knob arithmetic** -- the factories' own `ValueError`s, moved here and
   completed: warp grids, whole 16-row accumulators, whole 16-byte copies per
   thread, k_step against the MFMA/WMMA depth and the read slice, `stages`,
   `write_pos`, `stream_a`'s two fragments, `resident`'s LDS, the pingpong's
   eight warps, HipKittens' waits that hk-lds-pipeline refuses.
3. **Resources** -- LDS is computed exactly from the allocations; VGPRs are
   predicted (`hk/schedules/_regs.py`): the live tiles counted as the target
   lays them out, plus a per-family overhead fitted to sampled builds and
   held above every one of them. A prediction over 256 (or `max_vgprs`) is
   refused as a spill, which the build would refuse anyway; under
   `min_occupancy` likewise.
4. **Shape** -- K and N against `k_step`/`block_n`/`splits`, M against the
   decode GEMV's token tiles, split-K's scratch limit; only M may be ragged.

The kernel is generic in M, N and K, so "will it build" is `check_kernel`.
The acceptance test (`tests/hk/codegen/test_schedules_legality.py`) samples
each family's space on its arches, builds every point `check_kernel`
accepts, and reports the false-accept rate (accepted but refused by the
passes or the spill gate) and that every refused point has a reason.

`canonical()` folds knobs that do nothing at their value back to the
default (one LDS buffer's `write_pos`, `late_b`'s HipKittens waits, a
`combine_depth` at least `warps`), so a space holds each kernel once.

## Records

- `to_record()` is `{"family": f, knob: value, ...}` for the knobs off the
  family's defaults (`full=True`: every knob); `Columns` nests its head and
  tail. `from_record()` also reads a route table's candidate form
  `[family, {knobs}]` and a bare knob dict given to a family's class, so
  `python/hk/tuned/gemm_bf16.json`'s `schedule` fields (`Rdna.from_record`)
  and `routes_gemm_gfx950.json`'s `candidates` load unchanged.
- gfx1100: `hk.ops.gemm.DEFAULT` gained `stagger`, `stream_a`, `setprio`,
  `sched` at the factory's defaults, so a tuned record can carry them (the
  autotune lookup drops a record's keys the default lacks). Neither the names
  nor the C++ change.
- gfx950: a route table's `candidates` may now define names the code does
  not have; those are schedule records, taken exactly (`gemm_cdna`'s
  `extra_candidates()`), and `route()` / `run()` / `fits()` and
  `hk.templates.gemm`'s plan use them. The shipped names keep their
  shape-dependent meaning, which `S.from_candidate(name, m, n, k)` spells out
  as data (the pingpong's block from `schedule_for`, its N tail and `ppw` as
  `Columns`, `decode` from `decode_gemm.schedule_for`).

## Round trip

`tests/hk/codegen/test_schedules_roundtrip.py`, no GPU: every
`routes_gemm_gfx950.json` point, every candidate, every `gemm_bf16.json`
record, the decode GEMV's schedules, the RDNA defaults, and
`hk.templates.gemm`'s plans over a grid of shapes go through the new type
(`lookup` / `from_candidate` / `from_record`, then `to_record` and back) and
produce the same kernel names and byte-identical `k.source()` as the hk.ops
code paths, with and without an epilogue, on both arches.

hk.ops.matmul's launch path is unchanged. `hk.templates.gemm` builds its
kernels through the schedule types (the C++ is identical, checked above);
its plan without `schedule=` is the same function as before.

## Not yet

- Attention (`hk.ops.attn`, `attn_cdna`) under the same protocol.
- fp8 / grouped GEMMs: new families, when their kernels exist.
