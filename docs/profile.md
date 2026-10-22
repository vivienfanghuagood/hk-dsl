# Why a kernel is slow: `hk.profile`

[`hk.check`](check.md) says whether a kernel is right and how fast it is
against torch; `hk.profile` says **why** it runs at that speed. It runs the
call under `rocprofv3` and reports, per kernel the call launches: the time;
the achieved FLOP/s and bytes/s against the card's roofline; how busy the
matrix core was; occupancy achieved against what the resources allow; how
the grid fills the CUs; where the waves wait; LDS bank conflicts; the L2 hit
rate; the main loop's instruction mix from the disassembled code object;
and a **verdict** -- what bounds the kernel and what it loses, largest
first -- as one line for a person and as JSON for an agent.

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
tile shape, a route), and `hk.check` again -- the timing in hk.check, in
one process and interleaved, is the measurement a change is accepted on;
hk.profile's timing explains it.

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
| timing | `--kernel-trace` alone: each kernel's duration over 20 calls, 2 ms apart (`--gap-ms`; back to back an MI355X's GEMM slowed 40% within 4 ms under its power limit), the grid and resources of each dispatch | always |
| counters | `--pmc GROUP`, one pass per counter group (`hk.profile.counters.PASSES`: 5 on gfx950) | on (`--no-counters`) |
| thread trace | `--att` on the slowest kernel (`--att-kernels N` for more): every instruction of every wave on one CU, with its cycle, stall and duration | on where the decoder is installed (`--no-att`) |

and, with no GPU at all, the code object alone. Each pass takes 3-4 s on
an MI355X; a GEMM's whole profile 25-35 s (`--vs` doubles it). A pass that
fails -- a counter group the part rejects, no decoder, a container without
access -- is a note, and the report keeps what the other passes measured.
Every hk kernel's code object is in the compile cache (`HK_CACHE_DIR`):
hk.profile takes it out of the build (`kernel.co`, or the offload bundle in
the extension's `.hip_fatbin`) and disassembles it with ROCm's
`llvm-objdump` -- no GPU needed (`--build-only`). Kernels hk did not build
(hipBLASLt, torch's) are disassembled from the thread trace.

## Reading the report

An MI355X (gfx950, ROCm 7.2.3, torch 2.10, rocprofv3 1.1.0), Llama-3-8B's
gate_up GEMM at M = 1024 (`bench/profile/cases.py`):

```text
hk.profile gemm_plain  gfx950 (AMD Radeon Graphics, 256 CUs)  hk 0.5.0  rocprofv3 1.1.0
  call      bf16[1024,4096], bf16[28672,4096], bf16[28672]  181.98 us of kernels a call, 1 kernel(s)
  sources   timing: rocprofv3 --kernel-trace; counters: rocprofv3 --pmc, 5 passes; att: ...
  kernel gemm_cdna_256x256x64_w2x4_g8_x64_sp_es_lb_kernel  181.98 us ±0.4%
    grid      448 workgroups x 512 threads (8 waves), LDS 131072 B, 234 VGPR -> 1 a CU (bound by registers and lds); 1.75 rounds of 256 CUs
    roofline  240.5 GFLOP -> 1.32 PFLOP/s = 53% of the bf16 matrix peak; 546 MB -> ... ; 441 FLOP/B (ridge 376)
    busy      matrix core 64% busy, 1.59 of 2 waves/SIMD achieved, CUs busy 80%, clock 1.98 GHz
    ...
```

| line | what it says | from |
|---|---|---|
| `grid` | workgroups x threads (waves), LDS and registers; how many workgroups a *slot* holds at once (a CU on CDNA, a WGP -- the LDS pool -- on RDNA) and what binds it (`waves`, `registers`, `lds`); the grid's *rounds*: workgroups / (slots x workgroups a slot) | the dispatch (kernel trace), the compiler's resources where hk built it, hk.target |
| `roofline` | the matrix FLOPs and their rate against the matrix peak; bytes past the L2 (read and written) and their rate against DRAM and against the cache-resident rate; the intensity against the ridge | counters (FLOPs: `SQ_INSTS_VALU_MFMA_MOPS_*` x 512; bytes: `FETCH_SIZE`+`WRITE_SIZE`'s expressions), else the thread trace's matrix instructions x the grid's waves, else `--flops`/`--bytes`; the ceilings below |
| `busy` | matrix core busy (gfx950: `SQ_VALU_MFMA_BUSY_CYCLES` over cycles x SIMDs; gfx1100: achieved over peak, no counter); waves a SIMD achieved (`SQ_WAVE_CYCLES` over cycles x SIMDs) of what the resources allow; CUs busy (`SQ_BUSY_CU_CYCLES`); the clock the counter pass ran at | counters |
| `waves` | the share of the waves' time waiting on a dependency (`SQ_WAIT_ANY`: s_waitcnt, a barrier, a result), waiting to issue (`SQ_WAIT_INST_ANY`: the unit is busy with another wave's instruction), issuing (`SQ_ACTIVE_INST_ANY`), waiting to issue to the LDS | counters |
| `memory` | LDS bank conflicts (`SQ_LDS_BANK_CONFLICT` over the LDS's active cycles), the L2 hit rate (`TCC_HIT`/(`TCC_HIT`+`TCC_MISS`)), the average latency of a vector-memory, LDS (and scalar) instruction in cycles | counters |
| `loop` | the main loop -- the backward branch the traced waves spent most cycles in (without a trace: the one with most matrix instructions) -- its instruction mix by class, s_waitcnt by what they wait on, and its share of a wave's cycles | the code object; the thread trace |
| `stalls` | in the main loop, the share of the traced waves' cycles at each kind of instruction (below) | thread trace |
| `matrix` | on the traced CU's SIMDs, the matrix pipe's busy share, and its idle time split by what the waves were at in each idle cycle -- by reason and by part of the kernel | thread trace ([below](#the-thread-trace)) |
| `prologue`, `drain`, `epilogue` | before the main loop; after it up to the last matrix instruction (a pipeline's drain); after the last matrix instruction (the accumulator's conversion, a template's epilogue, the stores) | the code object; the thread trace |
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

The ceilings (`hk.profile.analyze.CEILINGS`, the numbers
[`hk.integration.headroom`](agents.md) and `hk.check` already use):

| arch | matrix peak (bf16) | DRAM | cache-resident | cache |
|---|---:|---:|---:|---:|
| gfx950 (MI355X) | 2.48 PFLOP/s measured (fp8: 2x, datasheet ratio) | 6.6 TB/s measured | 9.1 TB/s | 256 MB |
| gfx1100 (W7900D) | 122.6 TFLOP/s (a measured WMMA loop: 100.5) | 864 GB/s | 2.43 TB/s | 96 MB |

A call whose inputs and outputs fit the cache is measured against the
cache-resident rate (back-to-back calls find it resident), else DRAM; both
percentages are in the JSON. The matrix peak is at the part's top clock
(2.4 GHz on an MI355X); a kernel that ran at a lower clock (a power limit)
has `of_compute_peak_at_clock` beside it.

## The verdict and its findings

`findings` are what the numbers say, each `{kind, cost, text, evidence}`;
`cost` is the share of the kernel's time the finding accounts for where it
can be estimated (null for a bound). The verdict's `bound` is the first of
`memory_bound`, `compute_bound`, `latency_bound`; its `losses` the kinds with
a cost, largest first; its `line` both in words.

| kind | when | cost |
|---|---|---|
| `memory_bound` | bytes past the L2 at >= 60% of the applicable memory ceiling | -- |
| `compute_bound` | matrix core busy, or matrix FLOP/s of the peak, >= 60% | -- |
| `latency_bound` | neither: the waves wait (see the stalls) | -- |
| `tail` | the grid is not a whole number of rounds of the slots, under 16 rounds | the slot-time the last round leaves idle, (ceil(r) - r) / ceil(r); with `cu_busy` measured beside it |
| `underfilled` | fewer workgroups than slots | 1 - rounds |
| `epilogue`, `prologue` | the matrix core idles >= 3% of the kernel while the waves are past their last matrix instruction (before the main loop); "not hidden" when one workgroup fills a slot | that idle share (thread trace), else the part's share of a wave's cycles |
| `stall` | the matrix core idles >= 4% of the kernel with its waves at one reason (`reason`); without a matrix pipe, >= 10% of the main loop's cycles at it | that share |
| `issue_bound` | <= 2 waves a SIMD and the matrix core idles >= 15% with its waves issuing the loop's other instructions (`vmem_issue`, `lds_issue`, `valu`, `scalar`, `nop`, `other`); `per_matrix_instruction` is the loop's non-matrix instructions per matrix one | that idle share |
| `occupancy` | achieved waves under 70% of what the resources allow, the grid not explaining it | -- |
| `lds_bank_conflict` | >= 5% of the LDS's active cycles lost to bank conflicts | -- |
| `small` | under 8 us: launch and ramp matter | -- |
| `spill` | scratch in use | -- |

`--vs NAME=TARGET` profiles another target on the same inputs and lines
their slowest kernels up: `compare.metrics` is `{path: [this, that]}` over
`hk.profile.COMPARED` and the stall shares, `compare.line` says what differs
(the time ratio, the matrix core's idle time after the main loop, busy,
VALU instructions, waves a SIMD, the loop's instructions per matrix one).

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
compare    {a, b, us[2], ratio, kernels[2], metrics{path: [a, b]}, line, b_report}  (--vs)
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
              of_cache, memory_ceiling ("dram"|"cache"), memory_peak, of_memory_peak,
              intensity, ridge}
  matrix_core {busy, source}
  occupancy  {theoretical_waves_per_simd, bound_by, achieved_waves_per_simd,
              achieved_of_theoretical, cu_busy, se_busy}
  stalls     {counters{dependency_wait, issue_wait, lds_issue_wait, active, active_valu,
              active_vmem, active_lds, active_scalar, active_branch_msg},
              att{region, cycles, idle, by_reason{reason: {cycles, stall, share}}},
              matrix_idle{simds, span_cycles, busy, idle{reason: share},
                          idle_by_region{prologue|main_loop|drain|epilogue: share}, no_wave},
              top[{reason, share}]}
  lds        {bank_conflict, addr_conflict, unaligned_stall, instructions}
  cache      {l2_hit, mall_hit (always null: not measurable), fetch_bytes, write_bytes}
  latency_cycles {vmem, lds, smem, ifetch}
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
              per_matrix_instruction}
  counters   {raw counter: median value over the profiled dispatches}
  findings[] {kind, cost, text, evidence{...}, reason?}
  verdict    {line, bound, losses[]}
```

For an agent the contract is `verdict.bound`, `verdict.losses`,
`findings[].kind`/`cost`/`evidence`, and the paths in `COMPARED`; the rest is
evidence to quote.

## Per arch: what the counters can and cannot say

**gfx950 (MI355X, ROCm 7.2.3).** Five counter passes (`counters.PASSES`),
all collected in an unprivileged container. Read with care, each checked
against a kernel whose answer is known (the 1.75-wave GEMM above):
`GRBM_GUI_ACTIVE` is summed over the 8 XCDs; `SQ_WAVE_CYCLES`,
`SQ_WAIT_*`, `SQ_ACTIVE_INST_*` are quad-cycles summed over SIMDs;
`SQ_BUSY_CU_CYCLES` reads as cycles summed over CUs although its description
says quad-cycles (x4 would put the GEMM's CUs at 317% busy); `SQ_INSTS_MFMA`
is exactly 2MNK / 16384 for a 16x16x32 bf16 GEMM. **Cannot:** split an L2
miss between the MALL (Infinity Cache) and HBM -- `TCC_EA0_RDREQ_DRAM`
equals `TCC_EA0_RDREQ`, so `fetch_bytes` is bytes past the L2 and
`mall_hit` is null; attribute a wait to its instruction -- the counters give
dependency vs issue vs active shares; the thread trace does the rest.
`MeanOccupancyPerCU` and friends are not used: their units did not check
out against the known kernel.

**gfx1100 (W7900D, ROCm 7.2.4).** No counter counts WMMA instructions or
their busy cycles, nor per-type active cycles, CU busy or VMEM latency; the
groups that exist (`counters.PASSES["gfx1100"]`) fit one pass each. **On the
W7900D pod this was built on, every counter but `SQ_WAVES` reads 0**
(`GRBM_GUI_ACTIVE` included, for kernels that ran; the legacy `rocprof`
fails and device-wide collection segfaults): hk.profile detects it, says so,
and reports from the timing, the thread trace -- which works there -- and the
ISA. The matrix core's activity on gfx1100 comes from the thread trace (WMMA
executions x FLOPs each) or from the achieved FLOP/s over the peak.

## The thread trace

`rocprofv3 --att` records every instruction of every wave on one CU (SE 0,
CU 1, all SIMDs) for one dispatch: its cycle, how long it stalled, how long
it took. Its decoder, `librocprof-trace-decoder.so`, is a separate download
([ROCm/rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder/releases),
0.1.6 used here: unpack the manylinux tarball and put the `.so` in
`~/.local/lib`, or point `HK_ATT_DECODER` at its directory). Without it
hk.profile notes "no thread trace" and attributes waits from the counters.

From the trace: per instruction of the code, its executions and cycles
(`isa.*.att`), the main loop by cycles, the share of each part, and the
**matrix-idle attribution** (`hk.profile.timeline`): per SIMD, the matrix
pipe is busy for FLOPs / (the SIMD's peak FLOPs a cycle) cycles from each
matrix instruction's issue -- 16 cycles for a 16x16x32 bf16 MFMA on gfx950,
which is what back-to-back MFMAs show in the trace -- and each idle cycle is
attributed to what the SIMD's waves were at, split evenly between them. It
is a model on one CU of many, not a counter: the busy share it gives is the
traced CU's (72% against the counters' 64% GPU-wide for the GEMM above, whose
traced CU ran two full workgroups while a quarter of the CUs ran one).

## Validation: three known cases

(filled in from `bench/profile/run.py`'s reports)

## Limits

- One CU is traced; a kernel whose workgroups differ (a ragged last block,
  split-K shares) is seen through the ones that landed there.
- Counter passes serialize the kernels they count and run at the clock they
  get: their cycles are the counter pass's (`clock_mhz`), the time is the
  timing pass's.
- A thread-traced kernel runs a little slower (the trace's own traffic);
  shares within it are what the trace is for, not its time.
- The cost of a finding is an estimate from a model (equal-length
  workgroups for a tail; the matrix-idle attribution); an accepted change is
  measured with `hk.check` (interleaved, medians), not read off these.
- The child imports the target: a function must be defined at a module's
  top level (not a lambda or a closure) or named `FILE.py:name`.
