# XCDs, placement and `hk.comm` (gfx950)

An MI355X is eight dies (XCDs) on one package, each with its own 4 MB L2.
Two things follow for a kernel, and hk-dsl has one tool for each:

- **Which XCD runs a workgroup** decides how often bytes that several
  workgroups share are fetched: once per XCD that reads them. That is
  **placement**.
- **A hand-off between workgroups on different XCDs** (or between GPUs) is not
  ordered by plain loads and stores: a plain store stays in the writer's L2,
  and a plain re-read on the reader's side can be stale even after the flag it
  waited for arrived. That is **`hk.comm`**.

On a part with one L2 (gfx1100, gfx1201) neither question arises: every
placement is the identity and `hk.arch.xcd_count()` is 1.

## Placement: which XCD runs which workgroup

The hardware deals workgroup `w` to XCD `w % 8`. Workgroups that read the same
bytes -- attention's K and V of one KV head, a GEMM's panels -- each pull them
into their own XCD's L2, and numbered the wrong way each L2 holds many
unrelated streams at once. `hk.kernel(placement=...)` renumbers the grid at
kernel entry and changes nothing else: each workgroup computes what it did
before, so outputs are bitwise the same, and only the time changes.

There are three ways to ask for it: a cluster found by an analysis of the
kernel's reads, fixed chunks of consecutive ids, or `"auto"`, which applies the
cluster where a measured policy says it pays
([the language](language.md#which-xcd-runs-a-workgroup-placement)).

`python3 -m hk.check FILE.py:NAME --placement --inputs ...` checks, with no
GPU, that a placed launch covers its grid exactly once. A placement is a
measured choice per kernel: [placement](placement.md) says when it pays, and
[`examples/15_placement.py`](../examples/15_placement.py) shows it on a kernel
of your own.

## `hk.comm`: hand-offs between XCDs and GPUs

Each hand-off is lowered to the instruction sequence a memory-model test suite
found correct on gfx950 ([the design](design/comm.md)):

The scope is part of the tensor's type (`hk.Tensor[dtype, "device"]` between
the XCDs of one GPU, `"system"` between GPUs), so every access through it
carries the cache-policy bits both sides need. On top of it: signals and grid
barriers that time out instead of hanging, work queues (`hk.comm.claim`),
one-int32 mailboxes, a residency check for grids that wait on themselves
(`resident=True`), and `hk.runtime.SymmetricBuffer` for buffers shared between
GPUs.

The names are in [the API reference](api/hk.comm.md), the language's view in
[the language](language.md#between-workgroups-xcds-and-gpus-hkcomm-gfx950),
and [`examples/14_task_graph.py`](../examples/14_task_graph.py) is a
persistent task graph with dependencies across the XCDs.

`hk.comm` is gfx950's: on RDNA a peer's writes are not made visible to a
reader's L2, and no shader instruction invalidates it, so the hand-offs above
have no RDNA lowering.
