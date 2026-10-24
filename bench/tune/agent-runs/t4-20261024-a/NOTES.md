# T4 notes

Times below from `scratch/bench.py` (my quick loop: edge + one routing checked,
then median of 5 passes of >=4 calls over rotated weight copies behind a spin)
unless marked `score.py`. "ratio" = recorded best baseline (bench/t4/results) / mine.

## 2026-10-23

- 17:55 start. Read task, reference, score.py protocol, hk language docs, hk.ops.gemm (gfx1100),
  gemm_cdna / gemm_cdna_dp (gfx950), hk.ops.moe.
- Design: one launch, no host sync. Slots s_g = offsets[g]//BM + g (prefix-free, monotonic,
  >= cdiv(size_g,BM) slots per group); workgroup finds its group with two rounds of 16 scalar
  loads (E<=256); each workgroup loops over a run of N blocks (NSPLIT runs), so empty slots
  cost ~nothing. GEMM inside = gemm_cdna_dp's pipelined ring (gfx950), gemm.py's one-buffer
  staging (gfx1100). Store bounded by group end (`store(bounded=n)`), A copy rows bounded too.
- v1 gfx950 (bm by avg group: 128/64/32, bn 128, stages 3, nsplit ~4 WG/CU): correct on all 12.
  ratios: olmoe-gu 0.785, olmoe-d 0.872, mixtral-gu 0.743, mixtral-d 0.666, qwen15-gu 0.885,
  qwen15-d 0.671, granite-gu 1.129, granite-d 0.743, ds-gu 0.643, ds-d 0.750, qwen2-gu 0.805,
  qwen2-d 0.865.
- v1 gfx1100 build: block_m=128 spills (264 vgpr); block_m=64 builds (216 vgpr, occ 7).
- gfx950 sweeps (scratch/sweep950_*.log): bm=64 tiles are slower than bm=128 even at avg group 34-64
  rows; one N block a workgroup (nsplit = NB) beats walking runs of N almost everywhere (e.g.
  qwen2-down 101 vs 118-129 us). stages 2 vs 3 matters per shape (+-10%).
- Added `pp`: hk.ops.gemm_cdna's 256x256 8-warp pingpong, grouped. hk.arch.pingpong must be at the
  kernel's top level, so the N loop runs inside it (the passes accept that). early_store=True spills
  by 2 VGPRs in the grouped body; early_store=False builds (248). pp 256x256: mixtral-gu 267 us
  (1.18), mixtral-d 240 us (1.37), qwen15-d 155 us, granite-d 168 us (nsplit 2). NB: a sweep with
  checks off scored ds-gu 1.52 with bn=256 on N=384 -- it computed 256 of 384 columns. Guard added
  (N % block_n), bench checks on by default.
- Added `smallk` (gfx950): x tile copied once (whole K in LDS), w streamed through a ring of
  whole-K N blocks. ds-down (K=192) 88.5 us (1.005) with 64x64, 2 stages, nsplit 8; no gain at K=320.
- gfx1100: the N loop costs ~50 VGPRs (LICM keeps x's addresses live across it): 128x128 spilled
  (264). nloop=False (one N block a workgroup, the K loop gated by the slot being live) builds at
  216-240 -> olmoe-gu 3675 us (1.10), mixtral 1.08-1.10; bm=64 warp_rows=2 best for small groups.
- NSPLIT/NB are runtime now (NSPLIT = cols of a cached dummy int32 tensor): no compile per shape.
- First-call tuner: shortlist by avg group size (2-4 candidates), timed on a synthetic routing
  (multinomial over a skewed popularity, made from the shape alone) and synthetic x, over weight
  copies rotated past the LLC like the scorer, 5 interleaved rounds, median; each candidate's output
  checked against the first's. Plan cached per (rows, N, K, E).
- score.py v2, gfx950: olmoe-gu 1.113, olmoe-d 1.109, mixtral-gu 1.166, mixtral-d 1.306,
  qwen15-gu 1.053, qwen15-d 0.937, granite-gu 1.119, granite-d 1.055, ds-gu 1.000, ds-d 0.997,
  qwen2-gu 1.003, qwen2-d 1.025. VERDICT failed: audit of ds-gu captured 0 kernels.
  gfx1100 v2: 1.075 1.194 1.052 1.054 1.065 1.067 | 1.226 1.119 1.118 0.933 1.099 0.978 (pass both halves).
- The audit: torch.profiler on the MI355X drops a lone kernel in ~37% of one-call profiles (44/120
  over 24 shapes, scratch/audit_probe.py); a call of two launches never lost them, nor one with host
  time after the kernel's end. Host sleep after launch: 0.3 ms fixed still lost the 350-420 us
  kernels 7/120; sleep 0.2 ms + 1.5x the tuner's measured kernel time: 0/120. Host time only (the
  scorer queues a call behind a GPU spin of 2x its host time), GPU work unchanged.
- Extra shapes (scratch/shapes_extra.json, my own: decode t1/t16, E=256, K=128, N=K=8192, K=1344,
  top-1, rows 64k with N=256, rows=1, qwen3-30b, phi3.5): all correct on gfx950. vs baselines:
  decode-t1 0.75 (aiter_ck 25.5 us), decode-t16 0.94, dsv3-t128 0.97, rows1 0.88, rest 1.0-1.67.
- score.py v3 (audit pad in): gfx950 PASS geomean 1.080 min 0.985 (olmoe-gu 1.114, olmoe-d 1.072,
  mixtral-gu 1.129, mixtral-d 1.299, qwen15-gu 1.055, qwen15-d 1.048, granite-gu 1.144, granite-d
  1.091, ds-gu 0.986, ds-d 0.985, qwen2-gu 1.063, qwen2-d 1.006), all audits clean.
  gfx1100 PASS (two halves): 1.083 1.267 1.071 1.076 1.069 1.086 | 1.228 1.126 1.113 0.932 1.097 0.981.
- Tried for gfx950 small K: a flat (N block, K tile) stream for the ring kernel (`cflat`): correct,
  no faster (ds-d 96 us). transposed_c (8-byte stores): no faster anywhere.
- hk.profile of ds-d (c128s2): prologue 49% of a wave's cycles, 820 SALU + 37 lgkm waits -- the
  two-level search's 33 scalar loads are not batched (SMEM returns out of order, every use waits
  for all) plus runtime divisions (by E, NSPLIT). Replaced by a binary search with each probe a
  scalar load behind a uniform branch (8 dependent loads at most, fewer for small E), NB/NB_PER/
  NSPLIT as the extents of a 3-D dummy tensor, a 2-D grid (split, slot): no runtime division.
  gfx950 +5-15% nearly everywhere (mixtral-d 223 us, olmoe-gu 331, qwen2-d 96).
- hk codegen bug: a Bool `a & b` is emitted as a C++ variable named `and` (a keyword); use
  int products.
- My bench (scratch/bench.py) must stub the audit pad's sleep: back-to-back calls without the
  scorer's spin showed a false 2x slowdown.
- XCD renumbering (gemm_cdna's chunked chiplet transform) on the 2-D grid's dispatch order: helps
  pp on olmoe-gu (288 vs 330 us), granite-gu (48 vs 65), granite-d (139 vs 159), qwen2-d (95 vs
  115); hurts mixtral; helps the ring kernel only on ds-gu (223 vs 262). Both kept as candidates
  (pp256 / pp256x, c128s2 / c128s2x). slot-fastest grid order (splits of a slot on one XCD): worse.
- quick loop with the tuner, gfx950: 1.36 1.16 1.28 1.48 1.13 1.11 1.46 1.37 1.01 1.10 0.99 1.15.
- score.py v4 (binary search, XCD candidates, edge routing in the tuner's cross-check), gfx950:
  PASS geomean 1.211 min 1.012: olmoe-gu 1.361, olmoe-d 1.180, mixtral-gu 1.296, mixtral-d 1.469,
  qwen15-gu 1.105, qwen15-d 1.122, granite-gu 1.570, granite-d 1.307, ds-gu 1.012, ds-d 1.063,
  qwen2-gu 1.044, qwen2-d 1.142.
- 18:47-19:20 the gfx1100 helper hung: a flat-kernel sweep printed nothing in 17 min, then even
  `t4gpu gfx1100 echo hi` timed out (100 s) repeatedly; killed my stuck ssh/kubectl processes. The
  sweep just before (old search) was already very slow (10 min, one shape). The new search is
  untested on gfx1100 until it is back; its generated C++ reads right (uniform branches, no loops).
- Extra shapes, score.py on gfx950 (v4): PASS geomean 1.295 min 0.965: decode-t1 1.32 (was 0.75 before
  the binary search), decode-t16 1.06, dsv3-t128 0.97 (weight streaming at ~6.6 TB/s), kmin 2.62,
  bigNK 1.24, oddK 1.21, top1 1.18, rows64k-N256 1.11, rows1 2.13, qwen3-30b gu/d 1.07/1.10, phi3.5 1.26.
- Hedge while gfx1100 is unreachable: `_legacy()` (default on gfx1100, T4_LEGACY=0 turns it off)
  restores exactly the v3 structure there (two-level search, 1-D grid, (1, NSPLIT) meta), the one
  scored on gfx1100. Checked correct on gfx950 with T4_LEGACY=1.
- Split-K for the ring kernel (fp32 shares + hk.ops.gemm_cdna_dp.reduce): correct, slower on ds-gu
  (254 vs 229 us), qwen2-gu, qwen15-gu -- these are traffic-bound, not tail-bound. Not a candidate.
- Weights past 2^31 elements (E=128, N=4096, K=4096: 4.3 GB) correct on gfx950 (every candidate
  agrees in the tuner's cross-check); hk's copies are based per tile with 64-bit pointers.
- 19:35 cleanup: dropped the flat ring kernel, split-K, transposed_c, slot-fastest order (none won);
  re-verified gfx950 (same numbers, all correct). Removed `r32` (gfx1100 32x64, never run there) from
  the gfx1100 shortlist: an untested kernel that faults would poison the process, not just lose the
  tuner's race. Every shortlisted candidate has now run on its GPU (public + extra shapes).
- gfx1100 still unreachable at 19:45 (5-min timeout on `t4gpu gfx1100 echo`); the gfx1100 score
  files of v3 are on the pod (t4pull when it is back).
- gfx950 weight-streaming (avg < 24 rows an expert): non-temporal, whole-line (st_16x64) weight copies
  as hk's paged decode reads its cache: dsv3-tp8-t128 252-255 us against 282 (triton 272), decode-t16
  88-92 against 94. nt alone hurts (16x32 half-line copies fetch lines twice); 16x64 without nt is
  mixed. Added c32x64l / c64s3l to the avg<24 shortlist only (they lose elsewhere: olmoe-gu, mixtral).
- score.py v7 (with the decode candidates), gfx950: PASS geomean 1.207 min 1.007 (1.341 1.156 1.279
  1.469 1.083 1.122 1.575 1.324 1.007 1.061 1.056 1.144). Audit probe on the final code: 0/120 lost.
- ds-gu (the min) leftovers tried: pp 256x128 / 128x128 with XCD chunks 8-32 (240-260 us), ring with
  XCD chunk 4/16 (222-241): within noise of c128s2x (227). It is traffic-bound (1.2 GB past L2 for
  0.9 GB unique) with a 2.06-round tail.
- Tuner routings now span router skews 0.3/0.6/1.0 (the public shapes' range); same picks on gfx950.
- Model-like extra shapes (scratch/shapes_extra2.json: mixtral-8x22b, dbrx, qwen3-235b, llama4-scout,
  dsv2-lite, jamba at tp8, 256-8192 tokens), gfx950: first pass geomean high but dbrx-gu 0.744 --
  N=2688 is not a multiple of 256, so every pingpong candidate was filtered and the large-group list
  had only the plain ring kernel left (496 us). Added (a) `part` to the ring and pingpong kernels
  (head: N's whole 256-wide blocks, tail: the last 128 columns, a second launch, as
  hk.ops.gemm_cdna.run splits a dense N) as candidates pp256s/pp256xs, (b) pp128n (128x128
  pingpong, XCD chunk 32) for the mid list (qwen3-235b-gu 195 vs 209 us). Re-score: PASS geomean
  1.210 min 0.978 (dbrx-gu 0.978, qwen3-235b-gu 0.997, jamba-gu 1.028; the rest 1.08-1.47).
- pp256x now with early_store (fits at 248 VGPRs since the binary search): equal or 1-7% faster.
- gfx1100 outage, working hypothesis: the 18:38 sweep (rflat configs) was killed locally by my
  tool's 10-min background limit, but its remote side kept running (or hung the GPU) and holds the
  pod; every later command, even `echo`, queues behind it. The helper's own timeout for that command
  was ~3.7 h (to ~22:30). Lesson: on gfx1100 run long sweeps detached (nohup) with an inner
  `timeout`, never inside a tool call that can be killed. rflat is not in any gfx1100 shortlist.
- Extra shapes with tiny K and large groups (scratch/shapes_extra3.json), gfx950: K=128/192/256 and
  N=256: 1.39-1.71 (output-write bound; the pingpong/ring candidates all store fine).

## 2026-10-24

- 00:50 12 random shapes inside the guarantees (scratch/shapes_rand.json: E 8-256, top_k 1-8, N 384-8192,
  K 192-8128 incl. odd K-tile counts, 256-8192 tokens), score.py on gfx950: PASS geomean 1.065 min
  0.969 (worst: E=256 top-1 N=4608 K=512 at 8192 tokens 0.969; E=16 N=7168 K=7232 0.998 vs aiter_gmm).
- gfx1100 still unreachable (pingloop: every 5 min since 19:47, `t4gpu gfx1100 echo` times out at 150 s).
- 02:30 final gfx950 public score (score.py, 3 sets x 7 rounds; bench/t4/out/gfx950-final.json/.md):
  PASS geomean 1.208 min 1.010 -- olmoe-gu 1.345, olmoe-d 1.200, mixtral-gu 1.274, mixtral-d 1.416,
  qwen15-gu 1.096, qwen15-d 1.108, granite-gu 1.592, granite-d 1.339, ds-gu 1.010, ds-d 1.048,
  qwen2-gu 1.044, qwen2-d 1.159; every audit clean. (Earlier score files moved to scratch/out_old/.)

## Where it ended

- **gfx950: last public score PASS, geomean 1.208, min 1.010** (above). Also on my own shapes:
  extra (decode..prefill, K=128, N=K=8192, rows=1) 1.295/0.965, model-like (mixtral-8x22b, dbrx,
  qwen3-235b, llama4-scout, dsv2-lite, jamba) 1.210/0.978, random in-range 1.065/0.969.
- **gfx1100: last public score is v3 (18:30-18:40, two halves, both PASS)**: geomean 1.107 and 1.075,
  min 0.932 (ds-down) -- olmoe-gu 1.083, olmoe-d 1.267, mixtral-gu 1.071, mixtral-d 1.076,
  qwen15-gu 1.069, qwen15-d 1.086 | granite-gu 1.228, granite-d 1.126, ds-gu 1.113, ds-d 0.932,
  qwen2-gu 1.097, qwen2-d 0.981. Its files were written on the pod (bench/t4/out/gfx1100-v3-*.json)
  and could not be pulled: the gfx1100 helper stopped answering at 18:47 and never came back
  (pingloop.log). The solution's gfx1100 path is that v3 structure (`_legacy()`); what changed for
  gfx1100 since is host-side only (the tuner's edge-routing cross-check, interleaved timing over
  rotated weight copies and three router skews, the audit pad scaled to the measured kernel time)
  and the same four candidates (r128n4, r128, r64w2, r64), all of which ran there in v3.

## With more time

- gfx1100 first: measure the binary search + 2-D grid there (T4_LEGACY=0; it builds, and r64 drops
  from 240 to 192 VGPRs), the flat (N block, K tile) stream `rflat` for short K (ds-down 0.93,
  qwen2-down 0.98 were the gfx1100 minimums), and the pingpong-style larger tiles.
- gfx950 traffic-bound mid shapes (ds-gu, qwen3-235b-gu at ~1.0): a persistent schedule over
  (slot, N block) with an L2-aware order instead of the 2.06-round grid; split-K did not help.
- Fold the x tile of a slot into LDS once across its N blocks for the pingpong (K <= 2048), and a
  grouped version of hk's decode GEMM (W straight to registers) for avg < 8 rows an expert.
- 03:25 stopped. gfx1100 never answered again after 18:47 (last ping 03:19, timeout). Final
  solution = scratch/snap/v11.py; static import audit clean; GPU time used: gfx950 ~35 min,
  gfx1100 ~14 min (ledger).
