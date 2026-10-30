"""A persistent task graph on hk.comm (gfx950): a work queue, per-layer
dependencies, and which XCD ran each task.

One launch runs L layers of K tasks. Task (l, k) needs two results of layer
l - 1, so a layer may start only when the one before it is done:

    val[0][k] = seed[k]
    val[l][k] = val[l-1][k] + val[l-1][(k + 1) % K]

The grid is `resident=True` -- every workgroup on the GPU at once, so one can
wait for another without deadlocking (a grid that cannot fit raises
`ResourceError` at launch instead). Each workgroup loops: take the next task id
from a queue (`hk.comm.claim`: one atomic add, a distinct id every call),
wait until the layer before it is done (`hk.comm.wait` on that layer's
counter), compute, publish (`hk.comm.arrive`: its stores acknowledged, then one
atomic add to its own layer's counter), take the next.

The workgroups run on all eight XCDs, and each XCD has its own L2. A plain
store stays in the writer's L2 and a plain re-read is served from the reader's,
so a hand-off between XCDs needs the cache-policy bits on *every* access to the
shared data, on both sides -- not only on the flag. That is why the scope
belongs to the tensor: `val`, `q` and `cnt` are `hk.Tensor[hk.i32, "device"]`,
and every access through them carries the bits; `wait` / `arrive` / `claim`
refuse an unscoped tensor. Here each value is one int32, read and written
with scalar atomics of that scope (`atomic_max_scalar(0)` changes nothing and
returns the value: a read at the tensor's scope). The memory model and the
litmus suite behind the lowering: docs/design/comm.md.

`wait(..., timeout_s=1.0, error=err.at[0])` turns a lost signal into an error
the host can check, instead of a hang. `hk.arch.xcd_id()` reads the XCD a
workgroup runs on from the hardware; the script prints how the tasks spread.

hk.comm is gfx950's (MI355X): on gfx1100 a peer's writes stay stale in the
L2 and no shader instruction invalidates it, so the kernel is pinned to
gfx950 -- `--dump` and `--build` work on any machine, and on another GPU the
script says so and exits.

    python3 examples/14_task_graph.py
    python3 examples/14_task_graph.py --dump cpp
"""

import _common

import hk
from hk import i32

#: The GPUs this example runs on, and why only those: the script says so elsewhere and returns,
#: and the GPU tier skips it there (examples/_catalog.py reads these two without importing).
ARCHES = ("gfx950",)
WHY = ("hk.comm needs gfx950 (Instinct MI355X): on RDNA a peer's writes stay stale in the L2, "
       "and no shader instruction invalidates it")

K, L = 64, 16        # tasks a layer, layers
WORKGROUPS = 64      # persistent workgroups: 8 an XCD


@hk.kernel(arch="gfx950", warps=1, grid=lambda p: (WORKGROUPS, 1, 1), resident=True)
def task_graph(seed: hk.Tensor[i32], q: hk.Tensor[i32, "device"],
               val: hk.Tensor[i32, "device"], cnt: hk.Tensor[i32, "device"],
               xcd: hk.Tensor[i32], err: hk.Tensor[i32], *, K: int = K, L: int = L):
    t = hk.comm.claim(q.at[0])                          # the first task this workgroup takes
    while t < K * L:
        layer = t // K
        k = t % K
        if layer > 0:
            # layer - 1 done: all K of its tasks arrived at its counter
            hk.comm.wait(cnt.at[16 * (layer - 1)], K, timeout_s=1.0, error=err.at[0])
            a = val.at[t - K].atomic_max_scalar(0)                         # val[l-1][k]
            b = val.at[(layer - 1) * K + (k + 1) % K].atomic_max_scalar(0)  # val[l-1][(k+1) % K]
            val.at[t].atomic_max_scalar(a + b)
        else:
            val.at[t].atomic_max_scalar(seed.at[k].load_scalar())
        xcd.at[t].atomic_max_scalar(hk.arch.xcd_id())   # which XCD ran task t
        hk.comm.arrive(cnt.at[16 * layer], 1)           # publish: one more task of this layer
        t = hk.comm.claim(q.at[0])


def reference(seed):
    rows = [list(seed)]
    for _ in range(1, L):
        p = rows[-1]
        rows.append([p[k] + p[(k + 1) % K] for k in range(K)])
    return rows


def main():
    args = _common.parse(__doc__)
    if _common.static(args, task_graph):
        return
    _common.require_gpu()
    if not _common.runs_here(ARCHES, WHY):
        return
    import torch

    seed = torch.randint(1, 10, (K,), dtype=torch.int32, device="cuda")
    q = torch.zeros(16, dtype=torch.int32, device="cuda")       # the queue's head
    val = torch.zeros(K * L, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(16 * L, dtype=torch.int32, device="cuda")  # a counter a layer, 64 B apart
    xcd = torch.full((K * L,), -1, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    print(f"task graph: {L} layers x {K} tasks on {WORKGROUPS} persistent workgroups "
          f"(resident capacity {task_graph.resident_capacity()})")
    task_graph(seed, q, val, cnt, xcd, err)
    torch.cuda.synchronize()

    assert int(err[0]) == 0, "a wait timed out: a dependency was never published"
    assert int(q[0]) == K * L + WORKGROUPS, "every task claimed once, then one empty claim each"
    assert cnt.view(L, 16)[:, 0].tolist() == [K] * L, "every task arrived once"
    want = reference(seed.tolist())
    got = val.view(L, K).tolist()
    bad = sum(g != w for gr, wr in zip(got, want) for g, w in zip(gr, wr))
    if bad:
        raise AssertionError(f"{bad} of {K * L} task results differ from the host's")
    print(f"  {'every task vs the host':<28} ok   exact, {K * L} tasks")
    per = torch.bincount(xcd.long(), minlength=8).tolist()
    print(f"  tasks run per XCD (hk.arch.xcd_id): {per}")
    if args.no_bench:
        return
    # The queue and counters start at zero each launch, so the timing resets them.
    def run():
        q.zero_()
        cnt.zero_()
        val.zero_()
        task_graph(seed, q, val, cnt, xcd, err)
    best = _common.compare({"hk task graph": run}, args.rounds)
    print(f"  one launch, {K * L} tasks with 3 zero-fills: {best['hk task graph'] * 1e3:.1f} us")


if __name__ == "__main__":
    main()
