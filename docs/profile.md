# Why a kernel is slow: `hk.profile`

[`hk.check`](check.md) says whether a kernel is right and how fast it is
against torch; `hk.profile` says **why** it runs at that speed. It runs the
call under `rocprofv3` and reports, per kernel the call launches: the time;
the achieved FLOP/s and bytes/s against the card's roofline; how busy the
matrix core was; occupancy achieved against what the resources allow; how
the grid fills the CUs; where the waves wait, and what they were doing
while the matrix core sat idle; LDS bank conflicts; the L2 hit rate; the
main loop's instruction mix from the disassembled code object; and a
**verdict** -- what bounds the kernel and what it loses, largest first -- as
one line for a person and as JSON for an agent.

```bash
python -m hk.profile my_kernel.py:gelu_mul --inputs "bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]"
python -m hk.profile hk.ops:matmul --inputs "bf16[2048,4096], bf16[16384,4096]" \
    --vs hipblaslt=torch.nn.functional:linear --json out.json
python -m hk.check my_kernel.py:gelu_mul --profile        # check, then profile
```

```python
import hk
p = hk.profile("my_kernel.py:gelu_mul", inputs="bf16[4096,14336], bf16[4096,14336], bf16[4096,14336]")
print(p)                 # the summary below
p.verdict["line"]        # one line
p.json()                 # everything, for an agent (the schema below)
```

- [The loop](#the-loop)
- [What it runs](#what-it-runs)
- [Reading the report](#reading-the-report)
- [The verdict and its findings](#the-verdict-and-its-findings)
- [The JSON](#the-json)
- [Per arch: what the counters can and cannot say](#per-arch-what-the-counters-can-and-cannot-say)
- [The thread trace](#the-thread-trace)
- [Validation: three known cases](#validation-three-known-cases)
- [Limits](#limits)

## The loop

check -> profile -> change -> check. `hk.check` first (is it right, how
far from torch and the baselines), then `hk.profile` on what lost: its
verdict names the bound (memory, compute, neither) and the losses with an
estimate of what each costs; change the one at the top (a schedule knob, a
tile shape, a route), and `hk.check` again. The timing in hk.check -- one
process, interleaved -- is what a change is accepted on; hk.profile's
explains it.

## What it runs

The target is anything `python -m hk.check` takes -- an `@hk.kernel`, an
`hk.ops` op, a template's product, any function that launches kernels
(torch's included: `torch.nn.functional:linear` profiles hipBLASLt) --
named `FILE.py:name` or `module:name`. Its inputs are a spec string
(`"bf16[1024,4096], bf16[28672,4096]"`, the syntax of `python -m hk.check
--help`), a generator `FILE.py:name` called as `gen(device)`, or in Python a
list of tensors. Constexprs of a raw kernel are `--const NAME=VALUE`.

A child process runs the call under `rocprofv3` once per **pass**: each
pass warms the call up (building or loading its kernels), launches a marker
kernel, runs the profiled calls, and launches the marker again; the
dispatches between the markers are the call's. The passes:

| pass | what | default |
|---|---|---|
| timing | `--kernel-trace` alone: each kernel's duration over 20 calls 2 ms apart (`--gap-ms`; back to back, an MI355X's GEMM slowed 40% within 4 ms under its power limit), the grid and resources of each dispatch | always |
| counters | `--pmc GROUP`, one pass per counter group (`hk.profile.counters.PASSES`: 5 on gfx950), 3 calls each, medians | on (`--no-counters`) |
| thread trace | `--att` on the slowest kernel (`--att-kernels N` for more): every instruction of every wave on one CU, with its cycle, stall and duration | on where the decoder is installed (`--no-att`) |

With `--vs NAME=TARGET` the other target is profiled the same way and the
two are also timed against each other interleaved in one process, as
hk.check times. With no GPU, `--build-only` builds and disassembles. A pass
takes 3-4 s on an MI355X (5-20 s in the W7900D pod); a GEMM's profile 25-35
s, `--vs` about twice that. A pass that fails -- a counter group the part
rejects, no decoder, a container without access, counters that read zero --
is a note, and the report keeps what the other passes measured. Without
`rocprofv3` at all the call is timed in-process (hk.check's `interleave`)
and its kernels disassembled. In an agent's loop, `--no-counters` keeps
the timing, the thread trace and the ISA -- the stall attribution and the
matrix core's idle time -- in 6.5 s for the gate_up GEMM below (25 s with
the counters).

Every hk kernel's code object is in the compile cache (`HK_CACHE_DIR`):
hk.profile takes it out of the build (`kernel.co`, or the offload bundle in
the extension's `.hip_fatbin`) and disassembles it with ROCm's
`llvm-objdump` -- no GPU needed. Kernels hk did not build (hipBLASLt,
torch's) are disassembled from the thread trace.

## Reading the report

An MI355X (gfx950, ROCm 7.2.3, torch 2.10, rocprofv3 1.1.0), Llama-3-8B's
gate_up GEMM at M = 1024 (`bench/profile/cases.py:gemm_plain`):

```text
hk.profile gemm_plain  gfx950 (AMD Radeon Graphics, 256 CUs)  hk 0.5.0  rocprofv3 1.1.0
  call      bf16[1024,4096], bf16[28672,4096], bf16[28672]  179.96 us of kernels a call, 1 kernel(s)
  sources   timing: rocprofv3 --kernel-trace; counters: rocprofv3 --pmc, 5 passes; att: rocprofv3 --att (thread trace, one CU); isa: llvm-objdump of the code object
  kernel gemm_cdna_256x256x64_w2x4_g8_x64_sp_es_lb_kernel  179.96 us ±0.3%
    grid      448 workgroups x 512 threads (8 waves), LDS 131072 B, 234 VGPR -> 1 a CU (bound by registers and lds); 1.75 rounds of 256 CUs
    roofline  240.5 GFLOP -> 1.34 PFLOP/s = 54% of the bf16 matrix peak; 1032 MB past the L2 -> 5.73 TB/s = 63% of the cache-resident rate (3.4x the call's bytes); DRAM at least 25% (the call's 302 MB once); 233 FLOP/B past the L2 (DRAM ridge 375.76)
    busy      matrix core 65% busy, 1.59 of 2 waves/SIMD achieved, CUs busy 80%, clock 1.97 GHz
    waves     waiting on a dependency 28%, waiting to issue 47%, issuing 26%, waiting to issue LDS 9% of wave time (counters)
    memory    LDS bank conflicts 0.0% of LDS cycles; L2 hit 51%; latency vmem 893, lds 47 cycles
    loop      0x2140-0x29a8, 329 instructions: 128 mfma, 8 valu, 70 salu, 16 vmem_lds, 48 ds_read, 16 barrier, 1 branch, 16 nop, 16 other, 10 waitcnt (8 lgkm, 2 vmem); 90% of a wave's cycles
    stalls    (main loop, thread trace) matrix_core 39%, barrier 26%, vmem_issue 13%, lds_issue 12%, scalar 6%, valu 2%
    matrix    (thread trace, 4 SIMDs) busy 72%; idle with the waves at barrier 12%, vmem_issue 8%, valu 2%, scalar 2%, no_instruction 1%, other 1%; idle by part main_loop 19%, prologue 5%, epilogue 2%, drain 2%; no wave 0%
    prologue  591 instructions: 309 valu, 2 valu_trans, 244 salu, 8 smem, 14 vmem_lds, 3 barrier, 4 branch, 2 nop, 5 waitcnt (3 lgkm, 2 vmem); 4% of a wave's cycles
    drain     530 instructions: 128 mfma, 197 valu, 56 salu, 1 smem, 2 vmem_lds, 64 vmem_store, 48 ds_read, 11 barrier, 3 nop, 11 other, 9 waitcnt (7 lgkm, 2 vmem); 4% of a wave's cycles
    epilogue  525 instructions: 185 valu, 14 salu, 64 vmem_store, 2 barrier, 1 branch, 257 nop, 2 other; 2% of a wave's cycles
    verdict   compute-bound: matrix core 65% busy, 1337 TFLOP/s = 54% of the bf16 peak (66% of it at the 1.97 GHz measured); tail: 1.75 rounds of 256 CUs, the last fills 192/256 (12% idle); matrix core idle at barrier 12%; matrix core idle at vmem_issue 8%; L2 re-fetch: 3.4x the call's bytes cross it
```

| line | what it says | from |
|---|---|---|
| `grid` | workgroups x threads (waves), LDS and registers; how many workgroups a *slot* holds at once (a CU on CDNA, a WGP -- the LDS pool -- on RDNA) and what binds it (`waves`, `registers`, `lds`); the grid's *rounds*: workgroups / (slots x workgroups a slot) | the dispatch (kernel trace), the compiler's resources where hk built it, hk.target |
| `roofline` | the matrix FLOPs and their rate against the matrix peak; the bytes past the L2 (read and written), their rate against the cache-resident rate, and how many times the call's own bytes that is; the call's own bytes (inputs and outputs once) against DRAM -- what DRAM moved at least -- when they do not fit the cache; the intensity | counters (FLOPs: `SQ_INSTS_VALU_MFMA_MOPS_*` x 512; bytes: `TCC_EA0_RDREQ_DRAM_32B`, `TCC_EA0_WRREQ_WRITE_DRAM_32B` x 32), else the thread trace's matrix instructions x the grid's waves, else `--flops`/`--bytes`; the ceilings below |
| `busy` | matrix core busy (gfx950: `SQ_VALU_MFMA_BUSY_CYCLES` over cycles x SIMDs; gfx1100: achieved over peak, no counter); waves a SIMD achieved (`SQ_WAVE_CYCLES` over cycles x SIMDs) of what the resources allow; CUs busy (`SQ_BUSY_CU_CYCLES`); the clock the counter pass ran at (from 50 us up) | counters |
| `waves` | the share of the waves' time waiting on a dependency (`SQ_WAIT_ANY`: s_waitcnt, a barrier, a result), waiting to issue (`SQ_WAIT_INST_ANY`: the unit is busy, often with another wave's instruction), issuing (`SQ_ACTIVE_INST_ANY`), waiting to issue to the LDS | counters |
| `memory` | LDS bank conflicts (`SQ_LDS_BANK_CONFLICT` over the LDS's active cycles), the L2 hit rate (`TCC_HIT`/(`TCC_HIT`+`TCC_MISS`)), the average latency of a vector-memory and an LDS instruction in cycles | counters |
| `loop` | the main loop -- the backward branch the traced waves spent most cycles in, with sibling loops as hot merged in (without a trace: the one with most matrix instructions) -- its instruction mix by class, s_waitcnt by what they wait on, its share of a wave's cycles | the code object; the thread trace |
| `stalls` | in the main loop, the share of the traced waves' cycles at each kind of instruction (below) | thread trace |
| `matrix` | on the traced CU's SIMDs, the matrix pipe's busy share, and its idle time split by what the waves were at in each idle cycle -- by reason and by part of the kernel | thread trace ([below](#the-thread-trace)) |
| `prologue`, `drain`, `epilogue` | before the main loop; after it up to the last matrix instruction (a pipeline's drain, and with early stores part of the epilogue's math); after the last matrix instruction (the accumulator's conversion, a template's epilogue, the stores) | the code object; the thread trace |
| `verdict` | the bound, then the losses, largest first | [below](#the-verdict-and-its-findings) |

The stall *reasons*, the same names in the JSON:

| reason | the wave was at |
|---|---|
| `matrix_core` | an MFMA/WMMA: in a loop's cycles, issuing it or waiting for the matrix pipe; in the matrix core's idle time, waiting for its operands |
| `vmem_wait` | `s_waitcnt vmcnt`: global loads (or gfx950's DMA into LDS) not back |
| `lgkm_wait` | `s_waitcnt lgkmcnt`: LDS reads/writes or scalar loads not back |
| `vmem_lgkm_wait`, `other_wait` | an s_waitcnt on both counters; on another |
| `barrier` | `s_barrier`: the workgroup's other waves not there yet |
| `vmem_issue` | a buffer/global load or store waiting to issue: the address/data path is full |
| `lds_issue` | a `ds_read`/`ds_write` waiting to issue: the LDS queue, its bandwidth, bank conflicts |
| `valu` | a vector ALU instruction (its dependency, or the VALU busy) |
| `scalar` | a scalar ALU or scalar-memory instruction |
| `nop` | `s_nop`: hazard padding |
| `other` | branches, `s_setprio`, `s_delay_alu` and other control |
| `no_instruction` | between instructions, none ready (instruction fetch, a hardware-tracked dependency) |

The ceilings (`hk.profile.analyze.CEILINGS`: the numbers
[`hk.integration.headroom`](agents.md) and `hk.check` already use, held
equal by a test):

| arch | matrix peak (bf16) | DRAM | cache-resident | cache |
|---|---:|---:|---:|---:|
| gfx950 (MI355X) | 2.48 PFLOP/s measured (fp8: 2x, the datasheet's ratio) | 6.6 TB/s measured | 9.1 TB/s | 256 MB |
| gfx1100 (W7900D) | 122.6 TFLOP/s (a measured WMMA loop: 100.5) | 864 GB/s | 2.43 TB/s | 96 MB |

Two memory readings, because no counter says whether a byte past the L2
came from the Infinity Cache (MALL) or from DRAM: the bytes past the L2
against the cache-resident rate (how hard the kernel drives the fabric
behind the L2), and the call's own bytes against DRAM when they do not fit
the cache (DRAM moved at least that much). `of_memory_peak` is the reading
nearer its bar; a kernel is memory-bound at >= 60% of DRAM or >= 70% of the
cache-resident rate. The matrix peak is at the part's top clock (2.4 GHz on
an MI355X); a kernel that ran slower (a power limit) has
`of_compute_peak_at_clock` beside it.

## The verdict and its findings

`findings` are what the numbers say, each `{kind, cost, text, evidence}`
(and `short`, the verdict's words for it); `cost` is the share of the
kernel's time the finding accounts for where it can be estimated (null for
a bound or a remark). The verdict's `bound` is the first of `memory_bound`,
`compute_bound`, `latency_bound`; its `losses` the kinds with a cost,
largest first; its `line` both in words.

| kind | when | cost |
|---|---|---|
| `memory_bound` | DRAM at >= 60% for sure, or the bytes past the L2 at >= 70% of the cache-resident rate | -- |
| `compute_bound` | matrix core busy, or matrix FLOP/s of the peak, >= 60% | -- |
| `latency_bound` | neither: the waves wait (see the stalls) | -- |
| `tail` | the grid is not a whole number of rounds of the slots, under 16 rounds | the slot-time the last round leaves idle, (ceil(r) - r) / ceil(r), equal-length workgroups assumed; `cu_busy` measured beside it |
| `underfilled` | fewer workgroups than slots | the slots with none |
| `epilogue`, `prologue` | the matrix core idles >= 3% of the kernel while the waves are after the main loop (drain + epilogue) or before it; "not hidden" when one workgroup fills a slot | that idle share (thread trace), else the parts' share of a wave's cycles |
| `stall` | the matrix core idles >= 4% of the kernel with its waves at one `reason`; without a matrix pipe, >= 10% of the main loop's (or the kernel's) cycles at it | that share |
| `issue_bound` | <= 2 waves a SIMD and the matrix core idles >= 15% *in the main loop* with its waves issuing the loop's other instructions (`vmem_issue`, `lds_issue`, `valu`, `scalar`, `nop`, `other`); `per_matrix_instruction` is the loop's other instructions per matrix one | that idle share |
| `narrow_access` | global loads/stores under 16 B a lane in a kernel that is not memory-bound and waits on vector memory >= 30% of its cycles | -- |
| `l2_refetch` | the bytes past the L2 are >= 1.8x the call's own: tiles fetched again by workgroups on other XCDs, or evicted before their reuse | -- |
| `occupancy` | achieved waves under 70% of what the resources allow, the grid not explaining it | -- |
| `lds_bank_conflict` | >= 5% of the LDS's active cycles lost to bank conflicts | -- |
| `noisy` | the timed calls spread > 5%: re-run before trusting the time | -- |
| `small` | under 8 us: launch and ramp matter | -- |
| `spill` | scratch in use | -- |

`--vs NAME=TARGET` lines the two up: `compare.interleaved` is their time
in one process (hk.check's `interleave`: rounds of ~1 ms bursts, rotated,
medians), `compare.metrics` is `{path: [this, that]}` over
`hk.profile.COMPARED` and the stall shares, `compare.line` says what
differs (the time ratio, the matrix core's idle time after the main loop,
busy, VALU instructions, waves a SIMD, the loop's instructions per matrix
one), and `compare.b_report` is the other's whole report.

## The JSON

`--json PATH` (or `p.json()`), schema 1. Keys are added, never renamed or
re-meant without bumping `schema`. A metric that was not measured is
absent or null, never 0.

```text
schema 1, tool "hk.profile", name, arch, device, seconds
agent      {name, product, cus, simds, xcc, ses, simd_per_cu, max_waves_per_simd,
            max_waves_per_cu, lds_kb, max_clock_mhz, wave}            (rocprofv3's agent info)
versions   hk.check.versions() + rocprofv3
call       {inputs, footprint_bytes, kernels, us (kernel time a call), iters}
verdict    {line, kernel, share, bound, losses[]}                      (the slowest kernel's)
sources    {timing, counters, att, isa}: what each part came from, or why it is off
cannot     [what this arch's counters cannot attribute]
ceilings   {matrix_flops{dtype}, dram_bytes_s, cache_bytes_s, cache_bytes, source}
passes     {name: {ok, seconds, why?}}
notes[], errors[{stage, type, message}]
compare    {a, b, us[2], ratio, kernels[2], metrics{path: [a, b]}, line, b_report,
            interleaved{a_us, b_us, b_over_a_time, band, verdict_for_a, rounds, how}}  (--vs)
kernels[]  in a call's launch order:
  name, hk (built by hk), share (of the call's kernel time), launches_per_call
  time       {us (median), min_us, noise (MAD/median), samples}
  build      {kernel, consts, warps, lds_bytes, source, binary, resources{vgpr, agpr, sgpr,
              vgpr_charged, lds, scratch, spill, occupancy_by_regs, occupancy_by_lds,
              occupancy, max_waves_per_simd}}                          (hk kernels)
  grid       {workgroups, threads_per_workgroup, waves_per_workgroup, lds_bytes, vgpr, agpr,
              slot ("CU"|"WGP"), slots, workgroups_per_slot, bound_by, theoretical_waves_per_simd,
              rounds, idle_share, last_round{workgroups, of}}
  roofline   {flops, flops_source, flop_s, matrix_dtype, compute_peak, of_compute_peak,
              clock_mhz, of_compute_peak_at_clock, bytes, bytes_source, bytes_s, of_dram,
              of_cache, dram_bytes_min, of_dram_min, refetch, memory_ceiling ("dram"|"cache"),
              memory_peak, of_memory_peak, memory_bound_at, intensity, ridge}
  matrix_core {busy, source}
  occupancy  {theoretical_waves_per_simd, bound_by, achieved_waves_per_simd,
              achieved_of_theoretical, cu_busy, se_busy}
  stalls     {counters{dependency_wait, issue_wait, lds_issue_wait, active, active_valu,
              active_vmem, active_lds, active_scalar},
              att{region, cycles, idle, by_reason{reason: {cycles, stall, share}}},
              matrix_idle{simds, span_cycles, busy, idle{reason: share},
                          idle_by_region{prologue|main_loop|drain|epilogue: share},
                          idle_in{region: {reason: share}}, no_wave, model},
              top[{reason, share}]}
  lds        {bank_conflict, addr_conflict, unaligned_stall, instructions}
  cache      {l2_hit, mall_hit (always null: not measurable), fetch_bytes, write_bytes}
  latency_cycles {vmem, lds}
  instructions, instructions_per_wave  {valu, mfma, salu, lds, smem, vmem_load, vmem_store,
              branch, flat}                                            (counters)
  clock_mhz
  isa        {function, instructions, source, matrix_dtype, total{mix}, loops[{start, end,
              instructions, mix, iterations, cycles}], main_loop{start, end, instructions, mix,
              cycles, cycle_share, att, which, merged_loops?}, prologue|drain|epilogue{start,
              instructions, mix, cycle_share, att}, att_matrix_insts_per_wave,
              att_matrix_flops_per_wave, matrix_idle, att{total_cycles, waves_traced}}
    mix      {instructions, classes{mfma, wmma, valu, valu_trans, salu, smem, vmem_load,
              vmem_lds, vmem_store, vmem_atomic, ds_read, ds_write, ds_other, waitcnt, barrier,
              branch, nop, other}, waits{vmem, lgkm, vmem+lgkm, exp, other}, mma_flops,
              bytes_per_lane{class: {bytes: count}}, per_matrix_instruction}
  counters   {raw counter: median value over the profiled dispatches}
  findings[] {kind, cost, text, short?, evidence{...}, reason?}
  verdict    {line, bound, losses[]}
```

For an agent the contract is `verdict.bound`, `verdict.losses`,
`findings[].kind`/`cost`/`evidence`, and the paths in `COMPARED`; the rest is
evidence to quote. `--record PATH` writes the raw measurements (without the
trace); `hk.profile.replay(PATH)` recomputes the per-kernel reports from
them with no GPU -- how the verdict logic is tested (tests/hk/ir).

## Per arch: what the counters can and cannot say

**gfx950 (MI355X, ROCm 7.2.3).** Five counter passes
(`counters.PASSES`), collected in an unprivileged container; every ratio is
taken inside one pass (each carries its own `GRBM_GUI_ACTIVE` and
`SQ_WAVE_CYCLES`). Read with care, each checked against a kernel whose
answer is known: `GRBM_GUI_ACTIVE` is summed over the 8 XCDs;
`SQ_WAVE_CYCLES`, `SQ_WAIT_*`, `SQ_ACTIVE_INST_*` are quad-cycles summed over
SIMDs; `SQ_BUSY_CU_CYCLES` reads as cycles summed over CUs although its
description says quad-cycles (x4 would put the gate_up GEMM's CUs at 317%
busy; x1 gives 80% for its 1.75-wave grid); `SQ_INSTS_MFMA` is exactly 2MNK
/ 16384 for a 16x16x32 bf16 GEMM, and `SQ_INSTS_VALU_MFMA_MOPS_BF16` x 512
exactly its FLOPs. **rocprofv3's `FETCH_SIZE` reads half the bytes on
gfx950**: a 4096x14336 bf16 elementwise kernel fetches its two inputs (235
MB) and `FETCH_SIZE` says 117 MB -- it counts the 128-byte requests as 64 --
while `TCC_EA0_RDREQ_DRAM_32B` x 32 is exactly 235 MB; hk.profile uses the
latter (and `TCC_EA0_WRREQ_WRITE_DRAM_32B` for writes: the GEMM's C to
0.03%). **Cannot:** split an L2 miss between the MALL and HBM
(`TCC_EA0_RDREQ_DRAM` equals `TCC_EA0_RDREQ`), so `mall_hit` is null and
DRAM traffic is bounded from below by the call's bytes; attribute a wait to
its instruction (the counters give dependency / issue / active shares --
the thread trace does the rest). `MeanOccupancyPerCU` and its relatives are
not used: their units did not check out against the known kernel.

**gfx1100 (W7900D, ROCm 7.2.4).** No counter counts WMMA instructions or
their busy cycles, nor per-type active cycles, CU busy or VMEM latency; the
groups that exist (`counters.PASSES["gfx1100"]`) fit one pass each. **On the
W7900D pod this was built on, every counter but `SQ_WAVES` reads 0**
(`GRBM_GUI_ACTIVE` included, for kernels that ran; the legacy `rocprof`
fails and device-wide collection segfaults; the pod is privileged, so it is
not a permission): hk.profile detects it after the first pass, says so,
skips the rest, and reports from the timing, the thread trace -- which works
there -- and the ISA. The matrix core's activity on gfx1100 is the
achieved FLOP/s over the peak, and from the thread trace (WMMA executions x
FLOPs; the pipe's busy share on the traced SIMD).

## The thread trace

`rocprofv3 --att` records every instruction of every wave on one CU (SE 0,
CU 1; all four SIMDs on gfx950, one on gfx1100) for one dispatch: its
cycle, how long it stalled, how long it took. Its decoder,
`librocprof-trace-decoder.so`, is a separate download
([ROCm/rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder/releases),
0.1.6 here: unpack the manylinux tarball and put the `.so` in
`~/.local/lib`, or point `HK_ATT_DECODER` at its directory). Without it
hk.profile notes "no thread trace" and attributes waits from the counters.

From the trace: per instruction of the code, its executions and cycles
(`isa.*.att`), the main loop by cycles, the share of each part, and the
**matrix-idle attribution** (`hk.profile.timeline`): per SIMD, one matrix
pipe serves the matrix instructions in issue order, each for FLOPs / (the
SIMD's peak FLOPs a cycle) cycles -- 16 cycles for a 16x16x32 bf16 MFMA on
gfx950, which is how far apart back-to-back MFMAs are in the trace; 32 for a
16x16x16 WMMA on gfx1100, whose trace stamps a WMMA when it is queued, often
one cycle after the last (`bench/profile/att_spacing.py` measures it) -- and
each idle cycle is attributed to what the SIMD's waves were at, split evenly
between them. It is a model on one CU of many, not a counter: the busy share
it gives is the traced CU's (72% against the counters' 65% GPU-wide for the
GEMM above, whose traced CU ran two full workgroups while a quarter of the
CUs ran one; on gfx1100's GEMM 73% against 69% from the FLOP rate).

## Validation: three known cases

`python3 bench/profile/run.py` profiles them (`bench/profile/cases.py`),
writing `bench/profile/results/<arch>/`. Each was named by the tool from its
own measurements; the reports below are abridged to their verdict and
comparison lines.

**(a) The tail and the unhidden epilogue -- gfx950, Llama-3-8B gate_up
(N=28672, K=4096) at M=1024**, `hk.ops.matmul` against `hk.templates.gemm`
with bias + GELU:

```text
VERDICT compute-bound: matrix core 65% busy, 1337 TFLOP/s = 54% of the bf16 peak (66% of it at the 1.97 GHz measured); tail: 1.75 rounds of 256 CUs, the last fills 192/256 (12% idle); matrix core idle at barrier 12%; matrix core idle at vmem_issue 8%; L2 re-fetch: 3.4x the call's bytes cross it
  vs        bias_gelu takes 1.062x the time of gemm_plain (interleaved in one process: 199.0 vs 187.1 us, band 6%); profiled calls: bias_gelu 192.0 us vs gemm_plain 180.0 us (1.067x the time); matrix core idle after the main loop (drain + epilogue) 4.2% -> 9.9% of the kernel; matrix core busy 65% -> 60%; matrix idle at VALU 2% -> 6%; VALU instructions 1.804e+07 -> 2.222e+07
  bias_gelu: compute-bound: matrix core 60% busy, 1253 TFLOP/s = 51% of the bf16 peak (62% of it at the 1.97 GHz measured); matrix core idle at barrier 13%; tail: 1.75 rounds of 256 CUs, the last fills 192/256 (12% idle); matrix core idle after the main loop 10% (one workgroup a CU: not hidden); L2 re-fetch: 3.4x the call's bytes cross it
```

The grid: 448 workgroups of 512 threads, one a CU (234 VGPRs and 128 KB of
LDS each), 1.75 rounds of the 256 CUs -- the last round fills 192, and the
CUs are measured busy 80% of the kernel. The epilogue: the template adds
4.2 M VALU instructions a call (+23%, 1166 a wave), all after the main loop;
the matrix core's idle time there goes from 4.2% to 9.9% of the kernel, and
at one workgroup a CU no other workgroup's main loop runs beside it -- the
5.7 points are the 6% the template costs. Measured here 1.06x the plain
GEMM interleaved (hk.check gave 1.049x on the same day); the docs' 1.09x
(docs/templates/gemm.md) was CUDA-graph replays on 2026-10-20.

**(b) The shape outside the route table -- gfx950, 2048x16384x4096**:
`route()` has no entry for N x K = 16384 x 4096 and its rule sends M = 2048
to `dp128`:

```text
VERDICT latency-bound (neither roof): 38% of the matrix peak, 33% of the cache-resident rate (past the L2); matrix core 43% busy; issue-bound at 1 wave(s) a SIMD (lds): the matrix core idles 34% while its waves issue the loop's 2.4 other instructions per matrix one; matrix core idle before the main loop 7% (one workgroup a CU: not hidden); matrix core idle after the main loop 4% (one workgroup a CU: not hidden); L2 re-fetch: 4.0x the call's bytes cross it
  vs        hipblaslt takes 0.576x the time of matmul (interleaved in one process: 176.7 vs 304.7 us, band 6%); ...; matrix core busy 43% -> 78%; matrix idle at VALU 8% -> 4%; VALU instructions 3.002e+07 -> 1.936e+07; workgroups 2,048 -> 256; loop instructions per matrix one 2.43 -> 0.777; CUs busy 94% -> 89%
  hipblaslt: compute-bound: matrix core 78% busy, 1511 TFLOP/s = 61% of the bf16 peak (80% of it at the 1.83 GHz measured); ...
  vs        pp takes 0.590x the time of matmul (interleaved in one process: 181.8 vs 312.5 us, band 4%); ...; matrix core busy 43% -> 75%; waves a SIMD 1 -> 2; workgroups 2,048 -> 512; loop instructions per matrix one 2.43 -> 1.57
```

The cause: `dp128` is a 128x128 block of four waves whose four LDS stages
take 128 KB, so one workgroup fits a CU -- **one wave a SIMD** -- and its
main loop issues 2.4 other instructions per MFMA (32 DMA loads, 64
`ds_read`s, 76 VALU and 116 SALU against 128 MFMAs a pass). With no second
wave on the SIMD to issue MFMAs meanwhile, that issue time is the matrix
core's idle time: it is busy 43%, and idle 34% of the kernel while its wave
waits to issue a load (20%), VALU (8%), SALU (7%), LDS (6%). hipBLASLt's
kernel is also one wave a SIMD but its loop has 0.78 other instructions per
MFMA (16 DMA loads, 32 `ds_read`s, 2 VALU, 38 SALU for 128 MFMAs): 78% busy.
The 256x256 pingpong forced (`gemm_cdna.run("pp")`) puts two waves on each
SIMD and 1.57 per MFMA: 75% busy, level with hipBLASLt (181.8 vs 176.7 us).
dp128 measured 0.58x hipBLASLt here (the 0.76x quoted in the goal was an
earlier measurement); routing the shape to `pp` is the fix (K3's tuner, or a
route-table entry).

**(c) gfx950 and gfx1100: GELU-and-mul at 4096x14336**, the 16x64-tile
kernel the first page of the docs teaches against llms.txt's strip kernel:

```text
gfx950:
VERDICT latency-bound (neither roof): 46% of HBM; waves at vmem_issue 69% of their cycles (whole kernel); waves at vmem_wait 11% of their cycles (whole kernel); waves at valu 11% of their cycles (whole kernel); narrow global accesses (loads 8 B; stores 8 B a lane); occupancy 5.1 of 8 waves/SIMD
  vs        strips takes 0.516x the time of gelu_mul_tiles (interleaved in one process: 62.1 vs 121.9 us, band 5%); ...
  strips: memory-bound: 86% of HBM at least (the call's 352 MB in 62.3 us, 5.66 TB/s); ...
```

On the MI355X the tile kernel's eight loads and four stores are 8 bytes a
lane at 32-byte strides (`buffer_load_dwordx2 ... offset:32/64/96`): its
waves spend 69% of their cycles unable to issue them (the vector memory
path full), the average load takes 2541 cycles, and DRAM runs at 46%; the
strip kernel moves the same bytes in 16-byte runs (two `dwordx4` loads, one
store a wave) and is memory-bound at 86% of HBM -- 0.52x the tile kernel's
time.

On the W7900D (gfx1100, ROCm 7.2.4; no counters there, so from the
timing, the thread trace and the ISA) the same pair is much closer: both
are memory-bound, at 77% and 83% of DRAM at least, the strips 0.94x the
tiles' time -- 864 GB/s of GDDR6 is slow enough that 8-byte accesses keep
up with it:

```text
gfx1100:
VERDICT memory-bound: 77% of DRAM at least (the call's 352 MB in 528.2 us, 0.67 TB/s); waves at vmem_wait 78% of their cycles (main loop); waves at vmem_issue 11% of their cycles (main loop)
  vs        strips takes 0.942x the time of gelu_mul_tiles (interleaved in one process: 506.4 vs 533.6 us, band 2%); ...
  strips: memory-bound: 83% of DRAM at least (the call's 352 MB in 488.4 us, 0.72 TB/s); ...
```

**gfx1100: the GEMM at M = 4096 against hipBLASLt** (4096x4096x4096,
`bench/profile/cases.py:matmul` vs `hipblaslt`):

```text
VERDICT compute-bound: matrix core 72% busy, 89 TFLOP/s = 72% of the bf16 peak; matrix core idle at vmem_wait 17%; tail: 7.11 rounds of 48 WGPs, the last fills 16/144 (11% idle); matrix core idle at no_instruction 9%
  vs        hipblaslt takes 1.288x the time of matmul (interleaved in one process: 2086.6 vs 1612.5 us, band 3%); ...; matrix core busy 72% -> 57%; waves a SIMD 6 -> 4; workgroups 1,024 -> 1,849; loop instructions per matrix one 3.062 -> 4.056
  hipblaslt: latency-bound (neither roof): 57% of the matrix peak, ...; matrix core 57% busy; matrix core idle at vmem_wait 50%; tail: 9.63 rounds of 48 WGPs, the last fills 121/192 (4% idle)
```

hk's 128x128 WMMA kernel runs at 89 TFLOP/s, 72% of the 122.6 peak and 88%
of the 100.5 a register-only WMMA loop measured on this card; the traced
SIMD's matrix pipe is busy 73% (the FLOP rate says 72%), idle mostly with
its waves at `s_waitcnt vmcnt` (17%): the K loop's global loads not yet
back. Its grid is 7.11 rounds of 48 WGPs at three workgroups a WGP
(registers: 218 VGPRs), the last round 16 of 144. hipBLASLt is 1.29x slower
here: four waves a SIMD against six, 4.06 other instructions per WMMA in its
loop against 3.06, its pipe idle half the time at vmem waits.

## Limits

- One CU is traced; a kernel whose workgroups differ (a ragged last block,
  split-K shares) is seen through the ones that landed there.
- Counter passes serialize the kernels they count and run at the clock they
  get: their cycles are the counter pass's (`clock_mhz`), the time is the
  timing pass's. Another job on the GPU during a pass shows as `noisy`
  timing or as counters that disagree with the timing.
- A thread-traced kernel runs a little slower (the trace's own traffic);
  shares within it are what the trace is for, not its time.
- The cost of a finding is an estimate from a model (equal-length
  workgroups for a tail; the matrix-idle attribution); an accepted change is
  measured with `hk.check` (interleaved, medians), not read off these.
- The child imports the target: a function must be defined at a module's
  top level (not a lambda or a closure) or named `FILE.py:name`.
