# Tuning a GEMM: `hk.tune`

`hk.tune` searches hk's own GEMM kernels for the schedule that runs a shape
fastest on the GPU in front of it, within a GPU-time budget, and keeps what
it found only if the measurement says it is a win. The result is a record on
this machine that `hk.ops.matmul` and `hk.templates.gemm` read for that shape;
nothing shipped changes.

It is the baseline tuner of the [tuning goal](archive/design/tuning-goal.md) (T3):
an agent with `hk.schedules`, `hk.check` and `hk.profile` should do better;
`hk.tune` is what it has to beat, and [bench/tune](../bench/tune/README.md) is
the benchmark that judges both.

## Tune a shape

```
python -m hk.tune 2048x16384x4096                 # C = A @ B.T, A (M, K), B (N, K)
python -m hk.tune 1x4608x9216 512x7168x8192 --budget 90
python -m hk.tune --dry-run 2048x16384x4096       # the space and the pool, no GPU
python -m hk.tune --list                          # what this machine has tuned
```

```python
import hk.tune

res = hk.tune.tune(2048, 16384, 4096, budget=60)    # GPU-seconds
print(res.summary())
# 2048x16384x4096 on gfx950: route dp128 ...
#   champion pp:g4_c16 ...: win, +NN% vs the route (bar ..., band ...)
#   vs torch ...: ... at 1.0x (band ...); route 0.76x
res.accepted          # the schedule the shape now runs (the route on a tie)
res.as_dict()         # everything, JSON-able
```

`baselines={name: fn(a, b, out)}` adds arms the decision is timed beside
(default: `F.linear`, hipBLASLt); the result then carries hk's ratio to each,
with its band. `save=False` measures without writing a record.
`--profile` (`profile=True`) then runs [hk.profile](profile.md) on what
`hk.ops.matmul` now launches for the shape against hipBLASLt, 30-70 s, and
prints its verdict: why the result is as fast as it is (an underfilled grid,
an unhidden prologue, an issue-bound loop), which is where a kernel change
would have to start once the search has nothing left (`hk.tune.explain(m, n,
k)` alone does the same).

## Tune a model

```
python -m hk.tune --model Qwen/Qwen2.5-32B --tp 2 --m 1,16,128,512,2048,8192
python -m hk.tune --model ./config.json --layers gate_up,down --m 16,2048
```

The shapes are the model's linear layers (qkv, o, gate_up, down; the LM head
with `--layers lm_head`) sharded for `--tp`, as `python -m hk.integration
tune` derives them (`hk.autotune.routes.linear_shapes`), at each M given. Each
(shape, M) is one search; the compiles are shared, so the second M of a
layer is cheaper than the first.

## What it searches

The space is `hk.schedules` (docs/design/schedules.md): every knob of hk's
GEMM kernels, with legality checked before anything compiles.

| GPU | families |
|---|---|
| gfx950 (MI355X) | `pp` the 8-wave pingpong, `dp` the 4-warp ring of stages, `decode` the GEMV, `cols` two of them on column ranges of C |
| gfx1100 (W7900D) | `rdna` the WMMA GEMM, `decode` the GEMV |

## How it searches

1. **Pool.** The current route, every shipped route candidate that takes the
   shape (the seeds), and the generated space ordered by an analytic model
   (`hk.tune.cost`: wave quantization, padding, tile efficiency, split-K's
   round trip, the weight-stream floor). One point per structure (family,
   block, K split) -- the model cannot tell warp grids or swizzles apart --
   a quota per family, plus a few structures past the model's cut drawn at
   random: the model is rough, and what it ranks wrong is only found by
   measuring something it did not pick.
2. **Compile** the pool on the CPU, hipcc in parallel (`-j`); a schedule the
   passes or the spill gate refuse never reaches the GPU.
3. **Check** each built candidate against fp32 (relative max error < 2e-2).
4. **Successive halving**: every survivor timed for 3 rounds, the fastest
   third kept (and anything within its noise of the cut, and the route),
   timed again for 6 rounds on fresh activations, then 10.
5. **Local search**: every one-knob move of the best two
   (`hk.schedules.space(around=s)`), compiled and timed against the
   incumbents, while it improves (up to six times) and the budget lasts.
6. **Decide**: the champion against the route (and the baselines), in 3 sets
   of 7 rounds on fresh activations each.

How a round is timed (`hk.tune.measure.GemmBench`): one process,
interleaved, the arms' order rotated every round; a round of an arm is a pass
over enough weight copies to stream three times the Infinity Cache (a model's
weights are not resident when its next layer runs), queued behind a GPU spin
longer than the host takes to launch the pass, so what is timed is the
kernels back to back, as a captured graph runs them, and not the launcher.

## When a winner is a winner

`hk.tune.stats.decide`, on the sets of the decision:

- the champion's gain is the median over sets of route time / champion time,
  each set's time the median of its rounds;
- the noise sigma is the larger of the rounds' spread inside a set (median
  absolute deviation, both arms added) and the sets' spread about the median
  ratio;
- it is a **win** only past `max(1%, 2 sigma)`; inside that it is a **tie**,
  and a tie keeps the route -- the route is what every other shape runs, and
  a record that names a schedule by a coin flip pins it for good;
- a win is measured again on fresh sets after the decision
  (`stats.confirm`); both must say "win".

Rounds faster than the card can go (the arch's bf16 peak with 15% headroom)
are dropped first. `tests/hk/ir/test_tune.py` holds the rule on synthetic
timings: a space whose every point is the route's speed, timed with
realistic noise and set-to-set drift, comes back a tie for every one of 20
seeds; one with a real 7% winner finds it.

## The budget

`--budget` is GPU-seconds a shape (default 60): wall time of the GPU work --
checks and timed rounds. A quarter is kept for the decision, which always
runs; rungs and local search stop when the next would pass the rest. The
compile is CPU time beside it, reported as wall seconds; so is the first call
of each candidate (loading its module), which is kept out of the GPU budget.
On the benchmark's shapes a search takes 5-25 GPU-seconds and 15-90 s of wall
clock, most of it hipcc.

## Records

A win is written to `$HK_TUNE_DIR/routes_gemm_<arch>.json` (by default
beside the compile cache), in the format of the shipped gfx950 route table:

```json
{"arch": "gfx950", "kind": "gemm",
 "candidates": {"u:pp:g4_c16": {"family": "pp", "wgm": 4, "xcd_chunk": 16}},
 "routes": {"16384x4096": {"2048": "u:pp:g4_c16"}},
 "points": [{"m": 2048, "n": 16384, "k": 4096, "verdict": "win", "decision": {...},
             "route": "dp128", "route_ms": ..., "ms": ..., "baselines": {...},
             "gpu_seconds": ..., "toolchain": ..., "device": ..., "date": ...}]}
```

- A record answers its (N, K) at the Ms of its M's power-of-two bucket
  (`hk.tune.records.bucket`) and nowhere else; elsewhere the shipped route (or
  its rule) decides, as before.
- `hk.ops.matmul` and `hk.templates.gemm` read it in front of the shipped
  routes (`hk.ops.gemm_cdna.route`, `hk.ops.gemm.schedule_for`); a record
  whose schedule does not take the shape (`Schedule.check`) is ignored.
- Records are read once a process (a tuner in the same process makes the
  next call see its record). `HK_AUTOTUNE=0` ignores them, as it ignores
  every tuning record; `HK_TUNE_RECORDS=0` ignores only these.
- A record is measured on this machine's GPU, toolchain and hk; it says so
  (`points`). A stale record is a correct kernel that may no longer be the
  fastest. Nothing here changes the shipped tables: that takes measured
  evidence on the benchmark's held-out split.

## How acceptance works

[bench/tune](../bench/tune/README.md) holds, per GPU, a hidden set of real
model linear shapes that no shipped table holds, split by (N, K) into a
"tuning" half (where `hk.tune` was developed) and an "acceptance" half
(where it is judged, never tuned on during development). Each acceptance
shape is tuned with the default budget, then scored in one process: hk before
(the current route), hk after (`hk.ops.matmul` with the record), and every
baseline -- hipBLASLt (`F.linear`), hipBLASLt under TunableOp, aiter's tuned
GEMM and its Triton GEMM, FlyDSL, HipKittens, an autotuned Triton matmul --
each checked against fp32. The scoreboard is the share of shapes at >= 0.95x
the fastest correct baseline, the median and worst ratio, and the GPU-minutes
a shape, before and after.

## What it reached

The acceptance split, each shape tuned once at the default budget, then
scored against every baseline in one process. The tuner was developed on the
tuning split only and frozen before the acceptance shapes were looked at
(`3e6a971`); run 2 is the same search at `f0dff9a`, after K1's final legality
(main `783d1bd`) and the code-object launch. Full tables -- every baseline's
time, each miss's numbers -- are in [bench/tune/results](../bench/tune/results/).

| | gfx950, run 2 (`f0dff9a`) | gfx950, run 1 (`3e6a971`) | gfx950, rocm/pytorch 7.2.3 | gfx1100, run 2 | gfx1100, run 1 |
|---|---|---|---|---|---|
| software | vLLM rocm100 image: ROCm 10 (HIP 7.15), torch 2.12, aiter 0.1.25, FlyDSL 0.3.4.1, Triton 3.8 | same | ROCm 7.2.3, torch 2.10, Triton 3.6 (no aiter, FlyDSL) | ROCm 7.2.4, torch 2.10, FlyDSL 0.4.0.dev, Triton 3.6 | same |
| hk's kernels >= 0.95x the best baseline, before | 0/25 | 1/25 | 3/25 | 15/24 | 15/24 |
| **after `hk.tune`** | **13/25 (52%)** | **15/25 (60%)** | **19/25 (76%)** | **24/24 (100%)** | **24/24 (100%)** |
| hk, or a route to `F.linear` where faster | 22/25 (88%) | 22/25 (88%) | 22/25 (88%) | 24/24 | 24/24 |
| median ratio, before -> after | 0.697 -> 0.950 | 0.652 -> 0.963 | 0.709 -> 0.972 | 0.995 -> 1.036 | 1.010 -> 1.045 |
| worst ratio, before -> after | 0.244 -> 0.773 | 0.242 -> 0.775 | 0.252 -> 0.821 | 0.428 -> 0.951 | 0.425 -> 0.951 |
| GPU-minutes a shape, mean / max | 0.09 / 0.22 | 0.08 / 0.21 | 0.11 / 0.29 | 0.21 / 0.71 | 0.31 / 1.04 |
| wall-clock minutes a shape, mean | 0.22 | 0.51 | 0.23 | 0.32 | 0.91 |

**The named target, 2048x16384x4096 on gfx950**: the route rule sends it to
`dp128`, 0.61-0.62x hipBLASLt measured here (the 0.76x in tuning-goal.md is
stale; re-measured, 0.58x). `hk.tune` takes it to the 256x256
pingpong with a 4-block-row swizzle (`pp(wgm=4)`, with `xcd_chunk=16` in run
1): **0.995-1.004x hipBLASLt** in the vLLM image (0.999x the best baseline in
run 1), 0.981x hipBLASLt (0.978x TunableOp's) in rocm/pytorch 7.2.3, in 3 GPU-seconds.

**Run to run.** Two independent searches land on hk times within 2% of each
other at every gfx950 miss (most within 0.5%; the schedules sometimes differ by
a swizzle). What moved the share from 15 to 13 is the baselines: TunableOp
picks a hipBLASLt solution per session, and at 2048x4608x4096 its pick ran
60.9 us in one session and 69.3 us (worse than plain `F.linear`) in the other;
2048x4096x1024's best baseline was 5% faster in run 2, 8192x14336x8192's 2.4%.
Six shapes sit within 3% of the 0.95 line in one run or the other, so the
gfx950 share is 52-60% with this toolchain, 76% with ROCm 7.2.3's (where aiter
and FlyDSL are not available and hk's kernels compile with 7.2.3's hipcc).

**Where gfx950 misses, and why** (run 2's 12; hk.profile, K2, on run 1's 10,
against hipBLASLt; the explanations agree):

Not the search: around each miss dozens of schedules sit within 1-2% of the
winner, and the two runs agree on hk's time. The misses are what hk's GEMM
kernels can be told to do -- the *space* -- or a baseline at the roofline:

- **A grid the pingpong's blocks cannot fill** (space; 6):
  2048x4608x4096 (144 workgroups of 256x256, 44% of the CUs idle; 256x128 is
  288, 1.1 waves), 1024x5120x14336 (160, with two K shares), 128x5120x14336
  (160), 4096x3072x{4096,8192} (192), 128x4608x36864 (216, issue-bound at two
  waves a SIMD, LDS bank conflicts in 75% of its LDS cycles). The hipBLASLt
  kernels they were profiled against have macro tiles hk's GEMMs do not --
  144x256, 160x256, 160x128, 192x256, 192x128 -- and Tensile's StreamK field
  set (SK5). **Since added**:
  `PingPong` blocks of 160, 192 and 320 and a persistent stream-K form
  (`stream_k=1|2`), both in the space. They make hk 4-12% faster at
  1024x5120x14336, 4096x3072x4096, 2048x4608x4096 and 2048x4096x1024, but the
  acceptance count stays at 13/25: stream-K's fp32 shares cost about what the
  part wave did, and the gap that is left is the pingpong's main loop (1.56
  instructions a matrix instruction against hipBLASLt's 0.58).
- **One 128 KB-LDS workgroup a CU, prologue and epilogue not hidden** (space;
  2): 2048x4096x1024 and 4096x4608x4096 -- the matrix core idles 12-32% of the
  kernel before and after the main loop, with no second workgroup on the CU to
  overlap. **Missing**: a persistent pingpong (each workgroup takes several
  tiles, so the next tile's prologue runs under the last one's epilogue).
- **An issue-bound ring** (space; 1): 512x8192x3584, the `Dp` ring at one
  wave a SIMD (128 KB of LDS), 1.9 other instructions per MFMA.
- **Compute-bound, 3-6% behind hipBLASLt** (baseline at its roofline; 3):
  8192x5120x5120 (0.935), 8192x14336x8192 (0.947), 2048x55296x5120 (0.949) --
  hk's pingpong at 54-61% of the bf16 peak against hipBLASLt's 58-65%, the
  store of C being 5-8% of the pingpong. Here routing to
  hipBLASLt is the answer, which is what the "or a route" row counts: with
  `F.linear` as a route where it is faster (as hk.integration's route tables
  route to the engine), 22 of 25 shapes reach 0.95x in every run; the three
  left are where TunableOp's or aiter's per-shape pick beats plain hipBLASLt
  as well as hk.

On gfx1100 nothing misses: the decode GEMV (`hk.ops.matmul` ran the THIN WMMA
kernel at M <= 16) and smaller WMMA blocks take the decode and M = 128 shapes
from 0.43-0.88x to 0.95-1.17x TunableOp's hipBLASLt; at M >= 512 hk was
already ahead of FlyDSL, the best baseline there, and stays at 1.00-1.29x it.

**Does a small win reproduce?** In run 1, of the 72 wins across both GPUs and
both splits, the 63 above 5% all reproduced in the scoring process (each at
least half its tuning gain); of the 9 under 5%, 8 did (+1.6 to +5.2%), and
one, +1.2% at 8192x3584x5120 on gfx1100, measured -0.6% in scoring: a 1% bar
is at the edge of what two measurement contexts agree on. Such a record costs nothing measurable -- the
two schedules are a tie -- but a stricter bar for shipping a record (as
`hk.autotune --ship` has) would leave it out.

## Limits

- **The space is hk's kernels.** The tuner finds the best of what hk's
  GEMMs can be told to do; the gfx950 misses above need a kernel (stream-K,
  a 192-wide macro tile, a pingpong that hides its prologue) before any
  search can reach them.
- **Candidates launch through code objects.** Every hk kernel loaded as a
  pybind extension module takes thread-specific-storage keys, and a process
  has 1024: a process that loads a few hundred aborts ("thread_specific_storage
  constructor: could not initialize the TSS key") -- one gfx950 search in the
  rocm/pytorch 7.2.3 image did, and two gfx1100 searches in one process. So the
  search launches its candidates as device-only code objects
  (`HK_LAUNCH=codeobj`, for the search only; `bench/tune/run.py` sets it for
  its process): the same device code and device time, no module a kernel, a
  fifth of the compile (a gfx1100 search went from ~60 s to 14-19 s of wall
  clock; wall per shape fell 2-3x in run 2 above). Run 1 was measured before
  this change (pybind launches, two to three shapes a process); `python -m
  hk.tune` with several shapes also runs each in a process of its own
  (`--in-process` to override).
- **Records are this machine's**: measured with its GPU, ROCm and hk, read in
  front of the shipped routes for that shape's M bucket. The gfx950 numbers
  above are ROCm 10's hipcc; a record made there is a correct kernel under
  ROCm 7.2.3, not necessarily the fastest.
- **Correctness is checked against fp32 per candidate**, on the tuning shape;
  a record answers its M bucket, where the kernel is the same code at another
  M (ragged M is predicated).
