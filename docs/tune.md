# Tuning a GEMM: `hk.tune`

`hk.tune` searches hk's own GEMM kernels for the schedule that runs a shape
fastest on the GPU in front of it, within a GPU-time budget, and keeps what
it found only if the measurement says it is a win. The result is a record on
this machine that `hk.ops.matmul` and `hk.templates.gemm` read for that shape;
nothing shipped changes.

It is the baseline tuner of the [tuning goal](design/tuning-goal.md) (T3):
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

## What it reached (2026-10-22)

The acceptance split, tuned once at the default budget by the tuner as frozen
at `3e6a971` (it was developed on the tuning split only; the acceptance
shapes were not looked at until it was frozen), then scored against every
baseline in one process. Full tables, every baseline's time and each miss's
explanation: [bench/tune/results](../bench/tune/results/).

| | MI355X (gfx950) | W7900D (gfx1100) |
|---|---|---|
| software | vLLM rocm100 image: ROCm 10 (HIP 7.15), torch 2.12, aiter 0.1.25, FlyDSL 0.3.4.1, Triton 3.8 | ROCm 7.2.4, torch 2.10, FlyDSL 0.4.0.dev, Triton 3.6 |
| shapes >= 0.95x the best baseline, before | 1/25 (4%) | 15/24 (62%) |
| **after `hk.tune`** | **15/25 (60%)** | **24/24 (100%)** |
| median ratio, before -> after | 0.652 -> 0.963 | 1.010 -> 1.045 |
| worst ratio, before -> after | 0.242 -> 0.775 | 0.425 -> 0.951 |
| GPU-minutes a shape (mean / max) | 0.08 / 0.21 | 0.31 / 1.04 |
| wall-clock minutes a shape (mean) | 0.51 | 0.91 |
| shapes that got a record | 22/25 | 16/24 |

(The tuning split, for comparison: gfx950 2/24 -> 13/24, median 0.677 ->
0.959; gfx1100 13/24 -> 23/24, median 0.984 -> 1.035.)

**The named target, 2048x16384x4096 on gfx950**: the route rule sends it to
`dp128`, 0.624x hipBLASLt re-measured here (0.621x the best baseline,
TunableOp's hipBLASLt; the 0.76x in tuning-goal.md is stale -- the lead's
re-measurement in the rocm/pytorch image is 0.58x). `hk.tune` takes it to
the 256x256 pingpong with a 4-block-row swizzle and 16-workgroup XCD chunks
(`pp(wgm=4, xcd_chunk=16)`): **1.004x hipBLASLt** (band 0.998-1.008), 0.999x
the best baseline, in 3.0 GPU-seconds.

**Where it misses (gfx950, 10 of 25), and why.** Every miss is a kernel the
space does not have, not a point the search did not find: around each of them
dozens of schedules are within 1-2% of the winner. hk.profile (K2) on hk
after against hipBLASLt says which:

- *underfilled grids* (5): 2048x4608x4096 runs 144 workgroups of the 256x256
  pingpong (44% of the CUs idle; the 256x128 block is 288, 1.1 waves),
  1024x5120x14336 160 with two K shares, 4096x3072x{4096,8192} 192,
  128x5120x14336 160. The hipBLASLt kernels it was profiled against there
  have macro tiles hk's GEMMs do not -- 144x256 at 2048x4608x4096, 160x256 at
  1024x5120x14336, 160x128 at 128x5120x14336, 192x256 at 4096x3072x* -- and
  Tensile's StreamK field set (SK5): a cut of the work the pingpong's whole
  128/256-wide blocks and equal K shares cannot make;
- *one workgroup a CU, prologue and epilogue not hidden* (3): 2048x4096x1024,
  4096x4608x4096, 512x4096x7168 -- the matrix core idles 11-32% of the kernel
  before and after the main loop, because a 128 KB-LDS pingpong workgroup
  has no second one on its CU to overlap with (hipBLASLt there: 256x128,
  256x320, 128x256 tiles);
- *issue-bound at one or two waves a SIMD* (2): 512x8192x3584 (the dp ring at
  one wave a SIMD, 1.9 other instructions per MFMA) and 128x4608x36864 (128x128
  blocks with six K shares, 2.7 per MFMA, LDS bank conflicts in 75% of the
  LDS's active cycles; hipBLASLt: 192x128x128).

On gfx1100 nothing misses: the decode GEMV (the THIN WMMA kernel is what
`hk.ops.matmul` ran there at M <= 16) and smaller WMMA blocks with a K
stagger take the decode and M = 128 shapes from 0.43-0.88x to 0.95-1.17x
TunableOp's hipBLASLt; at prefill sizes hk was already ahead of FlyDSL, the
best baseline there for M >= 512, and stays at 1.00-1.29x it.

**Does a small win reproduce?** Of the 72 wins across both GPUs and both
splits, the 63 above 5% all reproduced in the scoring process (each at least
half its tuning gain). Of the 9 under 5%, 8 reproduced (+1.6 to +5.2%); one,
+1.2% at 8192x3584x5120 on gfx1100, measured -0.6% in scoring: a 1% bar is at
the edge of what two measurement contexts agree on (docs/history/32 saw the
same drift between sessions). Such a record costs nothing measurable -- the
two schedules are a tie -- but a stricter bar for shipping a record (as
`hk.autotune --ship` has) would leave it out.

## Limits

- **The space is hk's kernels.** The tuner finds the best of what hk's
  GEMMs can be told to do; the gfx950 misses above need a kernel (stream-K,
  a 192-wide macro tile, a pingpong that hides its prologue) before any
  search can reach them.
- **A process loads a few hundred kernels at most.** Each hk extension module
  takes thread-specific-storage keys, and a process has 1024; one gfx1100
  search loads 100-300 candidates' modules, so `python -m hk.tune` tunes each
  shape of a list in a process of its own (`--in-process` to override), and
  `bench/tune/run.py` is run a few shapes a call. A process that runs out
  aborts with "thread_specific_storage constructor: could not initialize the
  TSS key".
- **Records are this machine's**: measured with its GPU, ROCm and hk, read in
  front of the shipped routes for that shape's M bucket. The gfx950 numbers
  above are ROCm 10's hipcc; a record made there is a correct kernel under
  ROCm 7.2.3, not necessarily the fastest.
- **Correctness is checked against fp32 per candidate**, on the tuning shape;
  a record answers its M bucket, where the kernel is the same code at another
  M (ragged M is predicated).
