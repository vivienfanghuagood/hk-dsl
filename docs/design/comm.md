# hk.comm: hand-offs between XCDs and GPUs on gfx950

> Hardware: MI355X (gfx950), two GPUs of an 8-GPU node -- one xGMI link between any pair,
> 59-61 GB/s a direction. Phase 1 of the MegaKernel / comm-compute overlap study; the
> feasibility study that motivates it (why TP2 prefill communication is the first target)
> is `~/hk-dx/mk/phase0-report.md` on the dev box.

## What is here

| piece | where |
|---|---|
| scoped tensors: `hk.Tensor[dtype, "device" \| "system"]`; every load, store and atomic through one carries the scope's cache-policy bits | `hk/types.py` (`GlobalType.scope`), `hk/dsl/values.py` (`scope_aux`), `aux` on `hk.load` / `hk.store` / `hk.atomic_scalar` |
| `hk.comm.signal(flag.at[i], v)`: this workgroup's stores acknowledged (vmcnt(0)), a barrier, one thread stores the flag with the scope | `hk.signal` op, `kittens::signal<AUX>` |
| `hk.comm.wait(flag.at[i], v, cmp=">=", timeout_s=2, error=None)`: every wave polls with the scope and `s_sleep`; after the timeout an error element is set, or the kernel traps | `hk.wait` op, `kittens::wait<AUX, CMP>` |
| scalar atomics with every fetch-op: `atomic_{add,max,min,or,and,xor,exch}_scalar` (one a warp, the value before it), plain or through a scoped tensor at its scope (relaxed atomics of that scope) | `hk.atomic_scalar` kinds, `kittens::atomic_scalar{,_scoped}` |
| scoped col_vecs: a row tile's `hk.col_vec` loaded from / stored to a 32-bit scoped tensor (a per-row scale next to its tile); each element a relaxed atomic of the scope, i.e. the SC1 / SC0\|SC1 access | `aux` on `hk.load` / `hk.store` of a vector, `kittens::{load,store}_scoped` (CDNA4 ortho layout) |
| `hk.comm.arrive(cnt.at[i], v)` (signal's drain, then one atomic add of the scope; the old value) and `hk.comm.barrier(cnt.at[i], target)` (arrive + wait >= target; with an epoch the counter never resets): a grid barrier with a "device" counter, a barrier across GPUs with a "system" one | `hk.arrive` op, `kittens::arrive<AUX>` |
| `hk.comm.claim(q.at[i], n=1)`: a work queue's take -- one scoped atomic add, the old value to the whole workgroup, no drain. With `resident=True` and a `while t < n:` loop it is the persistent + task-queue pattern (Hazy's interpreter, MPK's workers); tasks wait on per-layer counters (`wait`) and publish with `arrive` | `hk.arrive` with `nodrain`, `kittens::claim<AUX>` |
| `hk.arch.xcd_id()`: the XCD a workgroup runs on, read from the hardware (HW_REG_XCC_ID) -- for per-XCD aggregation instead of assuming block id mod 8 | `hk.xcd_id` op, `kittens::xcd_id()` |
| `hk.comm.xcd_barrier(cnt, epoch, workgroups)`: a grid barrier aggregated per XCD -- arrive at the XCD's counter, the XCD's last arrival at the top one, wait for eight: 8 global atomics a round instead of one a workgroup. 256 workgroups: **2.53 us a barrier vs 3.71 flat** | composition of `xcd_id`, `arrive`, `wait` |
| `hk.comm.post(box.at[i], value, tag)` / `hk.comm.take(box.at[i], tag)`: a one-int32 message and its tag in one 8-byte scoped store, polled by tag -- no drain, no separate flag (section below) | `hk.post` / `hk.take` ops, `kittens::post<AUX>` / `kittens::take<AUX>` |
| `hk.kernel(..., resident=True)`: a launch whose grid does not fit on the GPU at once (CUs x workgroups a CU holds by registers and LDS, from the build's resource report: `Kernel.resident_capacity()`) raises `ResourceError` instead of deadlocking in a barrier | `hk/dsl/kernel.py` |
| `hk.runtime.SymmetricBuffer(nbytes, group)`: one buffer a rank, every peer's mapped (hipIpc), as torch tensors; while one is open a launch refuses its memory bound to an *unscoped* parameter | `hk/runtime/symmetric.py`, `Kernel.__call__` |
| the litmus suite the lowering follows | `tools/cdna-probes/litmus_comm.hip` |
| tests: IR (what is emitted, what is refused); GPU: message passing between two XCDs and between two GPUs through symmetric memory, the negative control, a lost signal as an error | `tests/hk/ir/test_comm.py`, `tests/hk/gpu/test_comm.py` |

## Why the scope is the tensor's

The litmus suite (message passing, 20000 rounds a combination, the reader having read the
old data first so its caches hold it; push and pull across two GPUs, and two XCDs of one GPU;
coarse-grained and uncached memory):

| writer / reader | result |
|---|---|
| data sc0\|sc1, vmcnt(0), flag sc0\|sc1 / poll sc0\|sc1, data sc0\|sc1 | 0 failures everywhere |
| C++ `release` system store (emits `buffer_wbl2 sc0 sc1`: writes back the whole L2) / C++ `acquire` | 0 failures (costly in a busy kernel) |
| **plain** data stores, vmcnt(0), flag sc0\|sc1 | stale every round on coarse memory: the data sits in the writer's L2 |
| correct writer / poll sc0\|sc1, **plain** data reads | stale every round -- also on uncached memory: remote writes do not invalidate the reader's L1/L2 |
| plain poll | never sees the flag |
| remote store then local load, no vmcnt(0) (store buffering) | the SB outcome in 38% of rounds |

Both sides need the bits, on every access to the shared data -- not only on the flag. An API
that puts the scope on the access leaves one forgotten access to be a silent stale read; on
the tensor, no access through it can lack it, and `signal` / `wait` refuse an unscoped flag.
The remaining hole -- a symmetric buffer passed to an unscoped parameter -- is closed at
launch by `SymmetricBuffer`'s binding check.

Not seen but not relied on: data then flag both sc0|sc1 with no vmcnt(0) between (0 failures);
`signal` keeps the wait. 8 B and 16 B stores never tore over the link (~60 M loads each), so a
(value, tag) pair in one store is a safe mailbox (dsv41_mono's idiom): `post` / `take`.

## Tagged mailboxes

A message that is one int32 -- a count, a task id, a token's expert, a partial scalar -- does not
need the data + flag protocol. `hk.comm.post(box.at[i], value, tag)` stores (value, tag) into
elements i, i + 1 as ONE 8-byte store with the tensor's scope; `hk.comm.take(box.at[i], tag)`
polls that slot with one 8-byte load a poll until the tag matches and returns the value. The value
cannot arrive without its tag. So the writer skips the drain (`s_waitcnt vmcnt(0)`: a link round
trip before the flag may go), and the reader skips the dependent data load after the flag. The
slot must be 8-byte aligned (an even element of a tensor that starts aligned); `post` / `take`
trap on a misaligned slot, because the store could tear. A slot holds one message: reuse it after
the reader has taken it (its reply or an ack), or post round r into slot r % S with tag r. A
mailbox orders nothing else; a message about other data is still `signal`'s.

## Lowering

| scope | aux (CPol) | where it is enough |
|---|---|---|
| `"device"` | 16: SC1 | other XCDs of this GPU |
| `"system"` | 17: SC0\|SC1 | other GPUs over xGMI (and everything `"device"` covers) |

Register-tile loads and stores of a scoped tensor go through hk's descriptor path
(`detail::hk_load_tile` / `hk_store_tile`) with `AUX` threaded to every buffer access;
`signal` is `s_waitcnt vmcnt(0)`, `s_barrier`, one `buffer_store_dword ... sc*`; `wait` a
`buffer_load_dword ... sc*` loop with `s_sleep 1` and an empty `asm volatile` memory clobber, so
the poll is re-issued and no later load is hoisted above it (two buffer-load intrinsics at one
address with no store between are otherwise CSE'd -- it faked staleness in the litmus suite's
first run). Scoped accesses are gfx950's: on gfx1100 a peer's writes stay stale in the L2 and
no shader instruction invalidates it, so a scoped tensor there is a trace-time error.

## Measured

- A work queue: 64 resident workgroups take 20000 tasks by `claim`, each exactly once, every
  workgroup some. A task graph of 16 layers x 64 tasks from one queue, each task waiting on its
  layer's predecessor counter: every last-layer value correct. `xcd_id` over 256 workgroups:
  eight dies, 32 workgroups each.
- A grid barrier over 64 workgroups, 50 rounds, each reading its neighbour's value of the round:
  all correct; the full resident capacity meets at one barrier, one more workgroup is refused. A
  barrier across the two GPUs, 200 rounds: every round each rank read the peer's value of it.
- Message passing in hk, 2000 rounds: two XCDs, 0 stale; two GPUs through `SymmetricBuffer`,
  0 stale, no timeout; the same kernel re-reading through a plain alias (the negative control
  the test asserts): 2000 of 2000 rounds stale.
- Tagged mailboxes, one MI355X pair, the GPU tests' ping-pong kernels timed over 4000 rounds: a
  round trip between GPUs is **1.51 us** with `post` / `take` and 3.40 us with a 16 x 64 tile +
  `signal` / `wait` (2.25x); between two XCDs of one GPU 1.37 us (the two-GPU test prints its own
  2000-round figures with `pytest -s`). 2000-round
  ping-pongs between XCDs and between GPUs: every value right, no timeout.
- Links (phase 0): 59-61 GB/s a direction between a pair, from one workgroup of remote stores;
  remote loads need ~16 workgroups; a flag round trip 1.5-1.9 us. Relaying through two more
  GPUs' links with remote writes only (push-push) gives 153 GB/s a direction for a TP2
  exchange (2.7x); pulling from the relays only 120 -- remote reads lose to writes.

## The NVIDIA designs, in hk.comm

Each pattern is a test in `tests/hk/gpu/test_comm.py` that runs on two MI355X.

| design (NVIDIA) | in hk | test |
|---|---|---|
| tile-level signals: a producer flags each tile as it lands, a consumer waits per tile, in its own order (TileLink, Flux, ParallelKittens) | scoped tile stores into a peer's `SymmetricBuffer`, `signal` per tile, `wait` per tile | `test_tile_signals_through_a_double_buffered_symmetric_buffer`: 8 x 8 workgroups, 64 tiles x 40 rounds, consumed in reverse order |
| double-buffered symmetric memory with flow control | buffer parity = round % 2; the consumer `arrive`s a round on a counter in the producer's memory, and the producer `wait`s for round r - 2 before reusing the buffer | the same test |
| persistent kernel + task queue + dependencies (Hazy's interpreter, MPK) | `resident=True`, `claim` from a queue in a `while` loop, per-layer counters with `wait` / `arrive` | `test_a_work_queue_hands_every_task_out_once`, `test_a_persistent_task_graph_respects_its_dependencies` |
| grid / cross-GPU barrier | `barrier` on a "device" / "system" counter with an epoch'd target | `test_a_grid_barrier_orders_every_workgroup`, the barrier in the two-GPU test |
| tagged mailboxes: a scalar message with its epoch in one store (NVSHMEM signal ops with a value, dsv41_mono's peer mailboxes) | `post` / `take` | `test_tagged_mailboxes_ping_pong_between_xcds`, `..._between_gpus` |

## Roadmap

A plan only: nothing below is implemented. It lists what hk.comm still lacks to express the NVIDIA
megakernel and comm-compute designs, ordered by value, with each item's cost and how it will be
validated. Each item is its own PR, and each PR ships with its litmus or numeric tests and measured
numbers, as #38-#50 did.

**How the items are ranked:**

1. A design that cannot be written at all ranks above one that can be written but is slow or easy to
   get wrong.
2. A gap that has been measured ranks above one that is only suspected.
3. Work that AMD's stack already covers (RCCL, quickreduce, aiter) ranks last.

**Cost** is the work for one person, including tests and docs: S ≤ 3 days, M ~1-2 weeks, L ~3 weeks.

| # | capability | what it unlocks (NVIDIA design) | today | value | cost |
|---|---|---|---|---|---|
| R1 | scoped tile-level bulk `put` / `get` with completion (`put` + signal, `quiet`) | NVSHMEM `putmem_nbi` / `putmem_signal` / `getmem`; ParallelKittens' tile put/get; TileLink's tile push/pull | hand-written tile loops; bf16 tiles move at half link rate; no put+signal atom | high | M |
| R2 | channels: credit-based slot rings and role barriers | cross-layer pipelines (weight loaders k stages ahead of compute), Hazy's per-instruction counters, TileLink producer/consumer flow control | the double-buffer pattern exists only as a hand-written test with its own epoch arithmetic | high | S-M |
| R3 | co-residency and forward progress for persistent grids | every persistent design assumes all of its workgroups run at once (cooperative launch, one block per SM) | `resident=True` checks that the grid *fits* an idle GPU, not that it *is* resident | high (correctness) | M |
| R4 | multi-rank symmetric memory: a device-side peer table | TP4 / TP8, all-to-all (EP dispatch/combine), ring steps; NVSHMEM's PE index, the NCCL device API's peer pointers | N ranks are mapped, but each peer is a separate parameter chosen at trace time; only TP2 is tested | high | M-L |
| R5 | device-side dynamic task graph: ready queues, dependency counts, dispatch | MPK (events trigger tasks), Hazy's interpreter | `claim` + per-layer counters: a static schedule in which a taken task spins on its predecessors | high for megakernels; depends on R2 + R3 | L |
| R6 | wave-level signals for tile epilogues | TileLink / Flux per-tile flags from a GEMM epilogue | `signal` drains and barriers the whole workgroup | medium-high | S |
| R7 | wider and per-lane mailboxes | DeepEP-style dispatch metadata (counts, token -> slot), several scalars per message | `post` / `take` carry one int32 and a tag | medium | S |
| R8 | observability: in-kernel clock, which wait timed out | the stamp timelines that decomposed dsv41_mono's chain; debugging hangs without core dumps | no clock in the DSL; a timeout ORs 1 into the error element | medium | S |
| R9 | collectives on hk.comm (all-gather, reduce-scatter, all-to-all) | in-megakernel collectives (fused, not stream-level) | an exact / INT8 all-reduce on an unmerged branch | medium-low: AMD ships RCCL / quickreduce, and at TP2 they are at the link | L |

### R1 -- scoped tile-level bulk put / get with completion

**Gap.** Moving a region to a peer means writing a tile loop of loads and scoped stores. Three problems
were measured:

- bf16 register tiles store 8-byte row pieces, so a remote bf16 copy runs at 33.7 GB/s, against
  56 GB/s through fp32 views of the same bytes. The all-reduce prototype had to use the fp32 trick.
- Remote *reads* need ~16 workgroups to fill a link; remote writes fill it from one.
- Nothing ties a put to its completion signal, so every put is followed by a hand-written `signal` and
  its whole-workgroup drain.

**Plan.**

- **`hk.comm.put(dst_view, src_view, signal=flag.at[i], value=r)`:** a region copied in dtype-blind
  16-byte lane chunks with the destination's scope. With `signal`, it is the drain followed by the flag.
- **`hk.comm.get(dst_view, src_view)`:** the read side, with the workgroup count to fill a link
  documented.
- **Non-blocking form plus `hk.comm.quiet()`:** the copy issues now and completes at the next `quiet`
  (the vmcnt wait moves there), so compute can run between them.

**Cost.** M. New ops, plus a byte-copy lowering through the buffer path with `aux` that handles ragged
tails.

**Validation.**

- Bitwise copies for every dtype, with ragged tails, from 1, 4, 16 and 64 workgroups, within one GPU and
  across GPUs.
- Bandwidth: one workgroup's `put` at ≥ 55 GB/s a direction (the link is 59-61), and `get` at ≥ 55 GB/s
  with 16 workgroups.
- A `put`-with-signal message-passing run of 2000 rounds with 0 stale, alongside the negative control
  (an unscoped alias that must read stale).

### R2 -- channels: credit-based slot rings and role barriers

**Gap.** Flow-controlled buffers are written by hand today. The double-buffer test uses parity
r % 2, an ack counter, and a wait for round r - 2, and each user re-derives that arithmetic. An
off-by-one there either overwrites a slot that has not been read or deadlocks. There is also no barrier
over a *subset* of workgroups (one role, one pipeline stage), and no arrive-now / wait-later form with a
name on it.

**Plan.**

- A ring of `S` slots: `hk.comm.channel(slots, flags, credits, S)`.
- Producer: `acquire(r)` (waits for the credit of round r - S), write the slot, `commit(r)`.
- Consumer: `ready(r)`, read the slot, `release(r)`.
- `group_barrier(cnt, epoch, members)`, plus split `arrive` / `wait` halves.
- All of it intra-GPU (XCD to XCD) or cross-GPU, depending on the scopes of the tensors passed in.
- This is composition over `signal` / `wait` / `arrive` / `post`, so no new op is expected.

**Cost.** S-M.

**Validation.**

- Rings with S = 1, 2 and 4, run for 10^4 rounds, with the producer slowed by injected sleeps and then
  the consumer. Sequence numbers in the payload prove nothing was overwritten or lost.
- The same runs between XCDs and between GPUs.
- A negative test with a deliberate credit bug must end in `error`, not a hang.
- Hand-off latency compared with the hand-written double-buffer test, which the channel must not be
  slower than.

### R3 -- co-residency and forward progress for persistent grids

**Gap.** `resident=True` refuses a grid larger than the GPU holds when idle. Three cases it does not
cover:

- Another stream's kernel holding CUs. This was measured in the overlap study: a partially resident
  comm kernel stalled *both* ranks.
- CU masking.
- Two persistent kernels at once.

There is also no notion of roles: which workgroups produce and which consume, and on which XCD.

**Plan.**

- With `resident=True`, launch through HIP's cooperative launch, which refuses a grid that cannot be
  co-resident.
- An optional census at kernel entry: every workgroup arrives, and if not all arrive within a timeout
  the kernel reports "not resident" in `error` instead of deadlocking later. It costs one barrier,
  ~2.5-3.7 us per launch.
- Role helpers that map a workgroup to (role, XCD) from `xcd_id`.

**Cost.** M. Most of it is changes to the runtime launch path.

**Validation.**

- The existing test that refuses a grid one past capacity.
- A hog kernel occupying part of the GPU before the launch: the census must time out into `error` and
  never hang.
- Role placement over 256 workgroups: an even split per XCD, checked against `xcd_id`.

### R4 -- multi-rank symmetric memory: a device-side peer table

**Gap.** `SymmetricBuffer` already maps N ranks. A kernel, though, receives each peer as its own tensor
parameter, chosen when the kernel is traced, so it cannot address "the rank that owns this expert" or
"the next rank in the ring" at runtime. Every test so far uses two GPUs, and an MI355X node is a full
mesh of 8 with 7 links per GPU.

**Plan.**

- A parameter type `hk.Peers[dtype, "system"]`: a device table of N base addresses.
- `peers[rank].at[...]` with a runtime `rank`.
- `SymmetricBuffer.peers(dtype, shape, offset)` builds the table.
- Every access keeps the tensor's scope, as scoped tensors do now.

**Cost.** M-L: a new type and its lowering (a table load, then a buffer resource per access), the
runtime side, and tests. **It needs more than GPUs 6 and 7**, so the 4- and 8-rank tests have to be
scheduled with the lead session.

**Validation.**

- All-pairs message passing and a cross-GPU barrier at 4 and 8 ranks.
- A 1-to-7 put whose aggregate bandwidth is ~7x one link.
- A runtime-rank ring with a sequence check.

### R5 -- device-side dynamic task graph

**Gap.** With `claim` plus per-layer counters, the task order is fixed. A workgroup takes task t and
spins until t's predecessors are done, holding its CU the whole time. MPK and Hazy instead run a task
only once it is ready:

- A task's dependency count reaching zero makes it runnable.
- Its completion pushes its successors into a ready queue, which is a bounded ring from R2.
- Workers take only ready tasks: one queue per XCD for locality, with stealing when a queue is empty.
- Each worker dispatches on the task's type with a `switch` helper over the DSL's runtime `if`.

**Plan.**

- A task table tensor holding each task's type, arguments, successors and in-degree.
- `hk.comm.task_queue` with `push_ready` / `take_ready`.
- Completion is an `arrive` on each successor's count; whichever arrival brings a count to zero pushes
  that task.

**Cost.** L. On top of the code, it needs a forward-progress argument (bounded queues, resident
workers from R3, a graph with no cycles) written down and tested.

**Validation.**

- Random DAGs of 10^4 tasks with fan-in and fan-out up to 16. Every task runs exactly once, and each
  task records the completion order of its predecessors, which must precede its own.
- Throughput (tasks per second) and dispatch latency compared with `claim` plus spinning.
- A megakernel-shaped graph (40 layers x stage tasks) timed against the static schedule.

It ranks fifth despite its value for megakernels because it builds on R2 and R3. Also, the decode
megakernel study measured the launch boundaries it would remove at only ~4 us a layer.

### R6 -- wave-level signals for tile epilogues

**Gap.** `signal` is vmcnt(0), then a workgroup barrier, then one store. In a GEMM epilogue that flags
tile by tile, that barrier serializes the workgroup's waves. A wave that stored its own subtile only
needs its own vmcnt(0) before flagging it.

**Plan.** A wave-scope variant of `signal` and `arrive`, without the barrier, with documentation of
what it orders: only that wave's stores.

**Cost.** S.

**Validation.**

- The tile-signal two-GPU test with one flag per wave (64 tiles x 40 rounds), alongside the negative
  control.
- The latency of a flagged epilogue tile, which must be below the workgroup variant's.

### R7 -- wider and per-lane mailboxes

**Plan.**

- `post` / `take` of 16 bytes: three int32 values and a tag. The litmus suite saw 16-byte stores never
  tear either.
- A per-lane `post`, where each lane writes its own slot, so that one wave publishes 64 routing entries
  (token -> expert slot) in a single store instruction.

**Cost.** S.

**Validation.**

- A ping-pong with the wider message.
- A 64-lane batch in which every slot is checked against its tag.
- A litmus run of the 16-byte slot.

### R8 -- observability: in-kernel clock, which wait timed out

**Plan.**

- `hk.arch.realtime()`, the 100 MHz s_memrealtime clock, for in-kernel stamp timelines. The decode study
  had to instrument a copy of dsv41_mono by hand to get them.
- A timed-out `wait` / `take` writes an id identifying itself into the error element, not just 1.

**Cost.** S.

**Validation.**

- The clock is monotonic and agrees across XCDs within a stated bound.
- A deliberately lost signal reports the id of the wait that timed out.

### R9 -- collectives on hk.comm

**Plan.** All-gather, reduce-scatter and all-to-all (EP dispatch / combine) as library ops built from
R1, R2 and R4. The exact and INT8 all-reduce on `claude/mk-allreduce` is the starting point.

**Value.** Low as standalone kernels. RCCL already runs at 90% of the link, quickreduce INT8 beats hk's
INT8, and at TP2 bytes are time. The value lies only *inside* fused megakernels, so this waits for a
kernel that needs it.

**Cost.** L.

**Validation.** Bitwise results against RCCL, and bandwidth against RCCL / quickreduce at the same
numerics.

### Not planned

- **Hardware multicast and in-switch reduction (NVLS, multimem).** xGMI has no equivalent: a broadcast
  is N puts.
- **TMA-style asynchronous bulk tensor copies.** gfx950 has no such engine. R1's non-blocking `put` plus
  `quiet` is the closest form.
