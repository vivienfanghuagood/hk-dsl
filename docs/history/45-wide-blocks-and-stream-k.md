# gfx950's grids: wide blocks, a persistent stream-K pingpong, and why the scoreboard did not move

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

`hk.tune` (PR #14) left
gfx950's acceptance split at 13 of 25 shapes at 0.95x the best baseline or
better, short of the goal's 80%, and named the misses as the schedule
space's, not the search's: two independent searches agreed within 2% at
every one. Six were grids the pingpong's blocks could not fill -- with only
128- and 256-wide blocks, a mid-M grid is a part wave of 256 CUs, where
hipBLASLt runs 144-192-wide stream-K tiles. This chapter adds the two things
that diagnosis asked for, puts them in `hk.schedules` and `hk.tune`, and
measures what they buy.

The answer: less than the diagnosis promised. Wide blocks make hk 4-12%
faster at the four mid-M grids where the tuner picks them, and stream-K wins
at one shape of 25. The acceptance split is again 13 of 25 (median 0.950,
geomean 0.956), because the baselines moved about as much as hk did, and at
the misses that remain the gap is the main loop, not the grid:
hipBLASLt issues 0.58 instructions a matrix instruction in its loop where
the pingpong issues 1.56.

## Wide blocks

The pingpong's copies were whole 16-byte loads *a thread*: a half block of
`rows x k_step` bf16 had to be a multiple of 512 threads x 16 bytes. A
96-row half (a 192-wide block) is 1.5 copies a thread. `group_load_async_rows`
(include/hk/cdna.cuh) now needs whole copies *a warp*: every warp issues the
full count and the first warps one more, under a wave-uniform scalar branch.
hk-lds-pipeline still counts the full number for every warp, so a wait
behind such a copy is one copy stricter than needed in the warps that issued
it, and exact in the rest. The pingpong's two halves are now warps 0-3 and
4-7 on any warp grid (a SIMD holds waves w and w + 4), which on HipKittens'
2 x 4 grid is the warp row it always tested.

`PingPong` takes `block_m` and `block_n` of 160, 192 and 320, with the warp
grids that give each warp whole 16-row fragments (160 needs 8 x 1 or 1 x 8;
192 and 320 take 4 x 2 or 2 x 4 too). On their own they lose per-CU
efficiency -- 256 x 192 on 4 x 2 warps ran 5.1 TFLOP/s a CU against
256 x 256's 7.4 on the tuning split -- so they are points to search, not a
new default.

## Stream-K

`stream_k=1` launches `sk_grid` (256) workgroups, one a CU, each running a
list of segments -- a block and a range of its K-tile pairs -- that
`gemm_cdna.sk_plan` deals: every block's K cut into equal runs.
`stream_k=2` deals the grid's whole waves of blocks whole, round robin, and
cuts only what is left. The plan is an int32 tensor the kernel reads with
scalar loads; it is made once a shape (`sk_args`) with an fp32 scratch for
the cut blocks' shares and a counter a block.

A workgroup that finishes a cut segment stores its fp32 share (transposed,
so an accumulator's four rows a lane are one 16-byte run) and arrives at
the block's counter. The last to arrive sums the shares in the plan's
order -- the same bits whichever workgroup is last, which the GPU test
checks over repeated calls -- and stores the block, epilogue included,
after its own last segment. Three choices were measured on the way:

- **Shares meet in one L2.** The plan gives each XCD a contiguous eighth
  of the cut blocks and only its own workgroups (g % 8, as the hardware deals
  them) to share them, so `hk.arch.wg_arrive(scope="xcd")` only retires the
  stores (vmcnt 0) and invalidates the reader's L1 (`buffer_inv sc0`). The
  device-wide `__threadfence` it replaces writes back and invalidates the L2
  under the XCD's other workgroups on every arrival; with it a stream-K GEMM
  whose every workgroup arrives once ran 5-50x slower.
- **Head first.** A workgroup's cut segments run with the head of its last
  block first, so an XCD's workgroups all start their stream-K work at K = 0
  and read the same K-tiles of A and B together. In run order each started
  at its own K and the L2 kept nothing for its neighbours: 1024 x 5120 x 7168
  in 128 x 256 blocks took 114 us, head first 89.5.
- **Not with `transposed_c`.** The shares are in the untransposed
  accumulators' layout; `check()` refuses the pair (the legality sample found
  three accepted ones that did not build).

`check()` also refuses `stream_k` with `splits` (it cuts K itself), K not a
multiple of 2 x `k_step`, and a grid that is not a whole number a XCD.

## The acceptance split

`python -m bench.tune.run --split acceptance` at b638a3e, in the vLLM ROCm 10
image with every baseline in the process, GPUs 1 and 2 of the MI355X
(`bench/tune/results/gfx950-k4-2026-10-23.md`, raw lines and records in
`results/raw/`):

| | #14 (f0dff9a) | wide blocks + stream-K (b638a3e) |
|---|---|---|
| shapes >= 0.95x best | 13/25 | 13/25 |
| median / geomean ratio | 0.950 / 0.955 | 0.950 / 0.956 |
| worst | 0.773 | 0.770 |
| with a route to `F.linear` where faster | 22/25 | 21/25 |

Shape by shape, hk's own time against #14's (both runs' `after_us`):

| shape | hk us, #14 -> now | best baseline us, #14 -> now | ratio | what the tuner picked |
|---|---|---|---|---|
| 1024x5120x14336 | 159.1 -> 141.4 (1.125x) | 123.2 -> 123.5 | 0.775 -> 0.873 | `pp` 192 x 256, splits 2 |
| 4096x3072x4096 | 86.1 -> 80.1 (1.076x) | 76.1 -> 73.7 | 0.884 -> 0.921 | `pp` 256 x 192, 4 x 2 warps |
| 2048x4608x4096 | 77.2 -> 72.7 (1.062x) | 69.3 -> 61.1 | 0.897 -> 0.839 | `pp` 160 x 256, 1 x 8 warps |
| 2048x4096x1024 | 20.8 -> 20.0 (1.037x) | 18.4 -> 18.4 | 0.886 -> 0.917 | `pp` 256 x 128, 4 x 2 warps |
| 512x8192x3584 | 39.3 -> 38.5 (1.021x) | 35.4 -> 35.4 | 0.906 -> 0.918 | `pp` 128 x 128, k_step 128, stream_k 2 |

Everywhere else hk's time is within 2% of #14's, except 128 x 5120 x 14336
(4% slower, a different split-K pick) and the large-M rows, where hk and the
best baseline both moved 2-3% the same way (another pair of GPUs). The baselines are not fixed points: TunableOp
searches hipBLASLt and rocBLAS at its first call in each process, and found
a 13% faster solution at 2048 x 4608 x 4096 this time (69.3 -> 61.1 us),
which turned hk's 6% gain into a lower ratio.

## Why stream-K rarely wins

8192 x 5120 x 5120 is the pingpong's clearest tail: 640 blocks of 256 x 256
are 2.5 rounds of 256 CUs, the last half empty (17% of the CUs idle,
hk.profile). `stream_k=2` deals 512 blocks whole and cuts the other 128 in
two. Interleaved against hipBLASLt (`F.linear`, 298.9 us), on GPU 1:

| schedule | us | x hipBLASLt |
|---|---|---|
| pp (the route) | 318.5 | 0.938 |
| pp, stream_k=2 | 318.7 | 0.938 |
| pp, stream_k=2, wgm 4 | 320.0 | 0.934 |
| pp, stream_k=1 | 331.1 | 0.903 |

The profile of the stream-K kernel says where the tail went: CUs busy 86%
(77% for the data-parallel grid) and the matrix core 70% busy (67%), but
15% of each wave's cycles are its epilogue -- 3,066 instructions, 417 vector
stores and 1,687 VALU -- against 2% for the data-parallel kernel's. Storing
a 256 x 256 fp32 share and summing the shares costs what the full last round
saved. Where the cut is a larger part of the work it is worse: 4096 x 3072 x
8192 (192 blocks, 0.75 of a round) takes 172.6 us in either stream-K mode
against the pingpong's 151.9, and 2048 x 4608 x 4096 99.3 us against 71.5
for the 160-row block the tuner chose. The cost model's `SK_SLOW` (1.3) is
kind to it; the search measures, so it only picked stream-K where it won.

## What is left

At the misses that remain, hk.profile against hipBLASLt (interleaved):

- 8192 x 5120 x 5120, 0.938x: compute-bound both, matrix core 67% busy
  against 75%; **0.58 loop instructions a matrix instruction for hipBLASLt,
  1.56 for the pingpong** (its 128 MFMA come with 48 LDS reads, 68 scalar
  and 16 nops a loop), and the tail above.
- 2048 x 4608 x 4096, 0.92x: both latency-bound at 44-45% of the matrix
  peak; hk 160 x 256 blocks fill 234 of 256 CUs, its loop 2.58 instructions a
  matrix one against hipBLASLt's 1.49, and 4.0x the call's bytes cross the L2
  (hipBLASLt 2.7x).

The grid was a real loss and the wide blocks recover part of it; the rest of
the gap is how many instructions the pingpong's loop spends a matrix
instruction, and stream-K's share traffic. The next steps that follow from
the numbers: a main loop with fewer LDS reads and scalar instructions a
matrix instruction (a larger MFMA shape or register-blocked fragments), and
a stream-K whose cut blocks are cheaper to share -- smaller tiles for the
cut part, or only two workgroups a block with the second's share reduced in
registers.

## How it was run

```bash
# the acceptance split, GPUs 1 and 2, the vLLM ROCm 10 image (the workstation's hkx950v wrapper)
<workspace>/ci/hk-dsl-wide/k4/loop.sh acc-a 1 acceptance 0 15
<workspace>/ci/hk-dsl-wide/k4/loop.sh acc-b 2 acceptance 15 25
python3 -m bench.tune.scoreboard RUNS/acc-a RUNS/acc-b --arch gfx950 \
    --write bench/tune/results --name gfx950-k4-2026-10-23
# a schedule against hipBLASLt, interleaved (3 sets x 9 rounds, GemmBench)
python3 bench.py 8192 5120 5120 torch '{"family":"pp"}' '{"family":"pp","stream_k":2}'
# where the time goes
python3 -m hk.profile hk.ops:matmul --inputs "bf16[8192,5120], bf16[5120,5120]" \
    --vs hipblaslt=torch.nn.functional:linear      # HK_TUNE_DIR = the run's records
```

The per-CU and head-first figures above are from the development runs on the
tuning split, in the rocm/pytorch 7.2.3 image (`hkx950`); everything in the
tables is from the vLLM image on 2026-10-23.
