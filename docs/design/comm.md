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

## Next

- Per-tile signals from a GEMM epilogue (store the tile to a peer, count it).
- Wider mailboxes (three int32 values and a tag in one 16-byte store; 16 B stores never tore either).
