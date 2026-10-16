# The GEMM schedule: sched_group_barrier, and the s_setprio that was the fence

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

[Chapter 26](26-the-scoreboard-retuned.md) left hk's GEMM at 0.95-0.98x
FlyDSL's dispatcher and traced FlyDSL's lead to two things: its
`sched_group_barrier` hints (5-6% of its own kernel) and a K rotation that
does not help hk. It ranked the work: an op for the hints, then a plan that
pays for hk's schedule; then a `dot_tile` that streams A, so FlyDSL's
four-wave 64x64 warp tile fits. This chapter does both, and finds that the
lever was a fence already in hk's schedule.

## The op

`hk.arch.sched_group_barrier(classes, size, sync_id=0)` is
`__builtin_amdgcn_sched_group_barrier(mask, size, syncid)`: a group of `size`
instructions of the named classes (`"mma"`, `"vmem_read"`, `"ds_read"`,
`"valu"`, ... or a mask), placed after the earlier groups of the same
`sync_id`. A run of them is an interleave. In the dialect it is
`hk.sched_group_barrier mask, size, syncid` with a verifier for the classes,
and `hk.sched_barrier` gained the mask it always had in LLVM (`allow=`: the
classes that may cross; none by default, so every existing kernel emits what
it did).

What it can reach in hk's GEMM is narrower than in FlyDSL's. FlyDSL's LDS
reads and writes are ordinary loads and stores, so its plan interleaves
WMMAs with them. include/rdna3's LDS fragment reads and writes are inline
asm, which belongs to no class, and stays where the program puts it; the
classes left are the WMMAs and the staging loads (buffer loads, so VMEM
reads). `gemm_kernel(sched=N)` asks for one staging load then N WMMAs,
repeated across the K-tile.

## The fence

Looking at what the scheduler did with hk's loop, before any hint: nothing.
The ISA is the DSL's program order instruction for instruction, twelve
`ds_read_b128`, a wait, `s_setprio 1`, four WMMAs, `s_setprio 0`, and so on.
`gemm_kernel` wraps each WMMA group in `s_setprio`, as the C++ did, and
`s_setprio` is a scheduling boundary in LLVM's AMDGPU backend: every group is
a region of its own, and the scheduler has nothing to move.

Without it (`setprio=False`), the same kernel, 211 VGPRs and occupancy 7 as
before, schedules differently: WMMAs in pairs between pairs of LDS reads --
the interleave FlyDSL's hints ask for -- and the K-tile's last four WMMAs
sunk past the barrier, under the next tile's LDS writes and staging loads.
`tools/hk-bench/gemm_vs_flydsl.py`, TFLOP/s, one process, each arm checked
against fp32 first:

| shape | hk with s_setprio (chapter 26) | hk without | FlyDSL dispatcher | FlyDSL, all options | hk without vs FlyDSL's best |
|---|---|---|---|---|---|
| 4096x4096x4096 | 87.57 | 90.62 | 89.61 | 89.87 | 1.008x |
| 2048x6144x4096 | 87.97 | 91.48 | 91.09 | 91.26 | 1.002x |
| 2048x4096x4096 | 84.55 | 88.43 | 89.25 | 89.53 | 0.988x |
| 2048x24576x4096 | 88.04 | 90.69 | 89.83 | 91.54 | 0.991x |
| 2048x4096x12288 | 85.07 | 88.10 | 90.51 | 90.42 | 0.973x |

That is +2.9-4.4% at the scoreboard's shapes, and +3.5-6.5% at 13 more
(M 512-16384, N and K from Qwen3-8B's layers), where it beat FlyDSL's
dispatcher at eight and lost at five, 0.95-0.99x, the largest loss at
512x12288x4096. It is now the default. The thin schedule (M up to 16) keeps
`s_setprio`: at M = 1-16 the two were within 1.2%, 0.5-1.2% its way at four
of five shapes. Attention keeps it too: without it, at n=4096, 32 heads,
d=128, it measured 0.5% (causal) and 0.8% faster, under noise.

## What did not pay

- **The hint.** On the new default, `sched=2` (one staging load, then two
  WMMAs) measured 0.988-1.004x at the five shapes: the scheduler already
  spreads the eight staging loads it can see. With the two-buffer schedule
  (`lds_buffers=2`, where the staging sits inside the K-tile's math) it is
  0.993-0.999x of the same schedule without it, and that schedule is itself
  0.80-0.84x the one-buffer default, as chapter 26 found.
- **FlyDSL's tile, by streaming A.** `gemm_kernel(stream_a=True)` holds the
  warp's A slice one 16-row fragment at a time, the next in flight behind the
  current one's WMMAs, with an accumulator per fragment row: two A fragments
  live instead of four. The four-wave 128x128x32 tile with a 64x64 warp tile,
  which needed 281 VGPRs and spilled 25 in chapter 26, then builds at exactly
  256 with no spill under ROCm 7.1's hipcc -- and spills 3 under the pod's
  ROCm 7.2, where the build gate refuses it. It was not measured. Three
  registers short is a different allocator, not a different design; it stays
  an option for the next toolchain.

## The scoreboard, tuned against tuned

`tools/hk-bench/scoreboard.py --only gemm,attn --rounds 5`, GPU ms per call,
best of 5 rounds of 10; "Triton" and "FlyDSL" are the faster of each DSL's
out-of-the-box and tuned arms; the verdict is hk's faster arm against the
fastest other, a tie within the larger of 2% and the two arms' noise.

| case | hk | hk tuned | torch | Triton | FlyDSL | tuned: hk vs best | chapter 26 |
|---|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.5229 | 1.5232 | 1.9682 | 1.6638 | 1.5423 | 1.01x tie | 0.98x loss |
| gemm 2048x6144x4096 | 1.1372 | 1.1285 | 1.5719 | 1.2523 | 1.1325 | 1.00x tie | 0.97x loss |
| gemm 2048x4096x4096 | 0.7817 | 0.7785 | 0.9988 | 0.8615 | 0.7721 | 0.99x tie | 0.95x loss |
| gemm 2048x24576x4096 | 4.6167 | 4.6041 | 5.9432 | 5.0004 | 4.6292 | 1.00x tie | 0.97x loss |
| gemm 2048x4096x12288 | 2.3428 | 2.3099 | 2.9459 | 2.5596 | 2.2950 | 0.99x tie | 0.96x loss |
| attention n4096 causal, GQA 4 | 3.0094 | -- | 10.8671 | -- | -- | 3.61x win | 3.80x |
| attention n4096, GQA 4 | 4.6924 | -- | 13.2199 | -- | -- | 2.82x win | 2.73x |

- **The GEMM ties FlyDSL at all five shapes**, from five losses: 87.9-91.3
  TFLOP/s against FlyDSL's 88.9-91.0. Against tuned Triton it is 1.09-1.11x
  (from 1.05-1.07x) and against hipBLASLt 1.28-1.39x (from 1.23-1.33x).
- **hk tuned** is hk's own tuner's records where it has them (two of the five
  keys, as in chapter 26); a record carries no `setprio` and so takes the new
  default.
- **Attention** is the same kernel as chapter 26's: its generated C++ differs
  only by `touch_kernargs` (six prologue instructions in 3,093). The causal
  row's 3.0094 ms against chapter 26's 2.847 is this run's; a direct A/B on
  the same tree an hour earlier measured 2.83.

## Tests

`tests/hk/ir/test_sched.py` (the op's lowering, the mask on
`sched_barrier`, the refusals) and `tests/hk/ir/test_gemm_schedule.py` (no
`s_setprio` in the default GEMM, the thin schedule keeping it, the names, the
interleave's groups, the streamed A's accumulators). The GPU tier's GEMM
tests run the new default.

## Reproduction

```
cd /workspace/hk-dsl
export FLYDSL_ROOT=/workspace/FlyDSL HIP_VISIBLE_DEVICES=0 PYTHONPATH=python
python3 tools/hk-bench/gemm_vs_flydsl.py --arms "hk default,hk +setprio,hk i2,hk l2,hk l2 i2,fly 128 dispatch opts,fly 128 -sched"
python3 tools/hk-bench/gemm_vs_flydsl.py --arms "hk thin,hk thin +setprio,hk default" \
    --shapes 1x4096x4096,8x4096x4096,16x4096x4096,16x12288x4096,16x4096x12288
python3 tools/hk-bench/scoreboard.py --only gemm,attn --rounds 5 --json sb.json
```
