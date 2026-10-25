"""hk.comm on gfx950: message passing between two XCDs of one GPU and between two GPUs,
written in hk -- the litmus suite's correct protocol (KernelWiki pitfalls/gfx950-cross-gpu-
memory-model.md), checked end to end, and the negative control that shows the test can fail.

Message passing, R rounds: the writer stores round r's tile (slot r % S of `src`, a value
no other round has) into `data` and signals r; the reader -- which first read `data` through a
plain alias, so its caches hold the old tile -- waits for r, reads `data`, keeps the tile in
`out[r - 1]`, and acks r. Every out[r - 1] must be src[r % S]."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import i32  # noqa: E402

pytestmark = pytest.mark.gpu

T = hk.RT(hk.bf16, 16, 64)
SLOTS = 8
ACK = 64     # the ack word, a cache line away from the flag


def _xcd_kernel(neg: bool):
    def body(src: hk.Tensor[hk.bf16], data: hk.Tensor[hk.bf16, "device"],
             alias: hk.Tensor[hk.bf16], sync: hk.Tensor[i32, "device"],
             out: hk.Tensor[hk.bf16], junk: hk.Tensor[hk.bf16], *, ROUNDS: int = 2000):
        at0 = hk.tile_coord(0, 0, 0, 0)
        if hk.arch.block_idx.x == 0:
            for r in hk.range(1, ROUNDS + 1):
                hk.comm.wait(sync.at[ACK], r - 1, timeout_s=1.0)
                data[at0] = src[hk.tile_coord(r % SLOTS, 0, 0, 0)].load(T)
                hk.comm.signal(sync.at[0], r)
        if hk.arch.block_idx.x == 1:
            for r in hk.range(1, ROUNDS + 1):
                junk[at0] = alias[at0].load(T)        # plain read: the old tile into L1 / L2
                hk.comm.wait(sync.at[0], r, timeout_s=1.0)
                if neg:
                    out[hk.tile_coord(r - 1, 0, 0, 0)] = alias[at0].load(T)   # plain re-read
                else:
                    out[hk.tile_coord(r - 1, 0, 0, 0)] = data[at0].load(T)    # scoped read
                hk.comm.signal(sync.at[ACK], r)

    body.__name__ = f"comm_xcd_mp{'_neg' if neg else ''}"
    # 16 workgroups: 0 and 1 are dealt to different XCDs (round robin over 8)
    return hk.kernel(body, warps=1, name=body.__name__, grid=lambda p: (16, 1, 1))


def _src():
    return torch.stack([torch.full((1, 16, 64), float(s + 1), dtype=torch.bfloat16, device="cuda")
                        for s in range(SLOTS)])


def _stale_rounds(out, src, rounds):
    want = torch.stack([src[r % SLOTS] for r in range(1, rounds + 1)])
    return int((out != want).flatten(1).any(1).sum())


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_message_passing_between_xcds_is_never_stale():
    rounds = 2000
    src = _src()
    data = torch.zeros(1, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    sync = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.zeros(rounds, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    junk = torch.zeros_like(data)
    _xcd_kernel(False)(src, data, data, sync, out, junk, ROUNDS=rounds)
    torch.cuda.synchronize()
    assert int(sync[0]) == rounds and int(sync[ACK]) == rounds
    assert _stale_rounds(out, src, rounds) == 0


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_the_negative_control_sees_stale_data():
    # the same protocol with the reader's re-read through an unscoped alias: the line it read
    # before the wait is still in its caches. If this ever reads 0 the test above proves nothing.
    rounds = 500
    src = _src()
    data = torch.zeros(1, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    sync = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.zeros(rounds, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
    _xcd_kernel(True)(src, data, data, sync, out, torch.zeros_like(data), ROUNDS=rounds)
    torch.cuda.synchronize()
    assert _stale_rounds(out, src, rounds) > rounds // 2


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_lost_signal_is_an_error_not_a_hang():
    def body(sync: hk.Tensor[i32, "device"], err: hk.Tensor[i32], seen: hk.Tensor[i32]):
        v = hk.comm.wait(sync.at[0], 1, timeout_s=0.01, error=err.at[0])
        seen.at[0].atomic_max_scalar(v + 100)
    k = hk.kernel(body, warps=1, name="comm_timeout", grid=lambda p: (1, 1, 1))
    sync = torch.zeros(64, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    seen = torch.zeros(1, dtype=torch.int32, device="cuda")
    k(sync, err, seen)
    torch.cuda.synchronize()
    assert int(err[0]) == 1 and int(seen[0]) == 100


def _grid_barrier_kernel():
    def body(slots: hk.Tensor[i32, "device"], cnt: hk.Tensor[i32, "device"], out: hk.Tensor[i32],
             err: hk.Tensor[i32], *, NWG: int = 64, ROUNDS: int = 50):
        w = hk.arch.block_idx.x
        for r in hk.range(1, ROUNDS + 1):
            # round r: workgroup w publishes r * 1000 + w, every workgroup meets, then reads its
            # right neighbour's -- which only the barrier makes visible
            slots.at[16 * w].atomic_max_scalar(r * 1000 + w)
            hk.comm.barrier(cnt.at[0], r * NWG, timeout_s=1.0, error=err.at[0])
            v = slots.at[16 * ((w + 1) % NWG)].atomic_max_scalar(0)
            out.at[(r - 1) * NWG + w].atomic_max_scalar(v)
            hk.comm.barrier(cnt.at[64], r * NWG, timeout_s=1.0, error=err.at[0])
    return hk.kernel(body, warps=1, name="comm_grid_barrier", grid=lambda p: (64, 1, 1), resident=True)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_grid_barrier_orders_every_workgroup():
    k = _grid_barrier_kernel()
    nwg, rounds = 64, 50
    slots = torch.zeros(16 * nwg, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.full((rounds * nwg,), -1, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    k(slots, cnt, out, err, NWG=nwg, ROUNDS=rounds)
    torch.cuda.synchronize()
    want = torch.tensor([r * 1000 + (w + 1) % nwg for r in range(1, rounds + 1) for w in range(nwg)],
                        dtype=torch.int32, device="cuda")
    assert int(err[0]) == 0 and torch.equal(out, want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_an_xcd_aggregated_barrier_orders_every_workgroup():
    def body(slots: hk.Tensor[i32, "device"], cnt: hk.Tensor[i32, "device"], out: hk.Tensor[i32],
             err: hk.Tensor[i32], *, NWG: int = 256, ROUNDS: int = 50):
        w = hk.arch.block_idx.x
        for r in hk.range(1, ROUNDS + 1):
            slots.at[16 * w].atomic_max_scalar(r * 1000 + w)
            hk.comm.xcd_barrier(cnt, 2 * r - 1, NWG, timeout_s=1.0, error=err.at[0])
            out.at[(r - 1) * NWG + w].atomic_max_scalar(slots.at[16 * ((w + 1) % NWG)].atomic_max_scalar(0))
            hk.comm.xcd_barrier(cnt, 2 * r, NWG, timeout_s=1.0, error=err.at[0])
    k = hk.kernel(body, warps=1, name="comm_xcd_barrier_gpu", grid=lambda p: (256, 1, 1), resident=True)
    nwg, rounds = 256, 50
    slots = torch.zeros(16 * nwg, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(16 * 9, dtype=torch.int32, device="cuda")
    out = torch.full((rounds * nwg,), -1, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    k(slots, cnt, out, err, NWG=nwg, ROUNDS=rounds)
    torch.cuda.synchronize()
    want = torch.tensor([r * 1000 + (w + 1) % nwg for r in range(1, rounds + 1) for w in range(nwg)],
                        dtype=torch.int32, device="cuda")
    assert int(err[0]) == 0 and torch.equal(out, want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_resident_grid_that_cannot_fit_is_refused():
    def body(cnt: hk.Tensor[i32, "device"], *, N: int = 1):
        hk.comm.barrier(cnt.at[0], N, timeout_s=0.1)
    k = hk.kernel(body, warps=4, name="comm_resident_cap", grid=lambda p: (p.cnt.cols, 1, 1),
                  resident=True)
    cap = k.resident_capacity()
    assert 256 <= cap <= 256 * 32           # an MI355X: 256 CUs
    ok = torch.zeros(cap, dtype=torch.int32, device="cuda")
    k(ok, N=cap)                             # the whole capacity meets
    torch.cuda.synchronize()
    assert int(ok[0]) == cap
    with pytest.raises(hk.ResourceError, match="deadlock"):
        k(torch.zeros(cap + 1, dtype=torch.int32, device="cuda"), N=cap + 1)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_every_fetch_op_is_atomic_across_workgroups():
    def body(cnt: hk.Tensor[i32, "device"], old: hk.Tensor[i32], bits: hk.Tensor[i32]):
        w = hk.arch.block_idx.x
        cnt.at[0].atomic_add_scalar(1)
        cnt.at[16].atomic_max_scalar(w)
        cnt.at[32].atomic_min_scalar(w)
        cnt.at[48].atomic_or_scalar(bits.at[w].load_scalar())             # 1 << (w % 31)
        cnt.at[64].atomic_and_scalar(bits.at[256 + w].load_scalar())      # ~(1 << (w % 31))
        cnt.at[80].atomic_xor_scalar(w)
        old.at[w].atomic_max_scalar(cnt.at[96].atomic_exch_scalar(w))
    k = hk.kernel(body, warps=1, name="comm_fetch_ops_gpu", grid=lambda p: (p.old.cols, 1, 1))
    n = 256
    cnt = torch.zeros(128, dtype=torch.int32, device="cuda")
    cnt[32] = 1 << 30; cnt[64] = -1; cnt[96] = -7
    old = torch.full((n,), -100, dtype=torch.int32, device="cuda")
    b = [1 << (w % 31) for w in range(n)]
    bits = torch.tensor(b + [~x for x in b], dtype=torch.int32, device="cuda")
    k(cnt, old, bits)
    torch.cuda.synchronize()
    xor = 0
    for w in range(n):
        xor ^= w
    assert int(cnt[0]) == n and int(cnt[16]) == n - 1 and int(cnt[32]) == 0
    assert int(cnt[48]) == (1 << 31) - 1 and int(cnt[64]) == ~((1 << 31) - 1) and int(cnt[80]) == xor
    # exchange: the values handed back are the initial one and every workgroup's but the last
    got = sorted(old.tolist() + [int(cnt[96])])
    assert got == sorted([-7] + list(range(n)))


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_scoped_vector_crosses_xcds():
    vt = hk.col_vec(hk.RT(hk.fp32, 16, 32))

    def body(src: hk.Tensor[hk.fp32], data: hk.Tensor[hk.fp32, "device"], alias: hk.Tensor[hk.fp32],
             sync: hk.Tensor[i32, "device"], out: hk.Tensor[hk.fp32], junk: hk.Tensor[hk.fp32],
             *, ROUNDS: int = 1000):
        if hk.arch.block_idx.x == 0:
            for r in hk.range(1, ROUNDS + 1):
                hk.comm.wait(sync.at[ACK], r - 1, timeout_s=1.0)
                data[hk.elem_coord(0, 0, 0, 0)] = src[hk.elem_coord(0, 0, 0, 16 * (r % SLOTS))].load(vt)
                hk.comm.signal(sync.at[0], r)
        if hk.arch.block_idx.x == 1:
            for r in hk.range(1, ROUNDS + 1):
                junk[hk.elem_coord(0, 0, 0, 0)] = alias[hk.elem_coord(0, 0, 0, 0)].load(vt)
                hk.comm.wait(sync.at[0], r, timeout_s=1.0)
                out[hk.elem_coord(0, 0, 0, 16 * (r - 1))] = data[hk.elem_coord(0, 0, 0, 0)].load(vt)
                hk.comm.signal(sync.at[ACK], r)
    k = hk.kernel(body, warps=1, name="comm_xcd_vec", grid=lambda p: (16, 1, 1))
    rounds = 1000
    src = torch.arange(SLOTS * 16, dtype=torch.float32, device="cuda")
    data = torch.zeros(16, dtype=torch.float32, device="cuda")
    sync = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.zeros(rounds * 16, dtype=torch.float32, device="cuda")
    k(src, data, data, sync, out, torch.zeros_like(data), ROUNDS=rounds)
    torch.cuda.synchronize()
    want = torch.cat([src[16 * (r % SLOTS):16 * (r % SLOTS) + 16] for r in range(1, rounds + 1)])
    assert torch.equal(out, want)


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_work_queue_hands_every_task_out_once():
    def body(q: hk.Tensor[i32, "device"], done: hk.Tensor[i32], who: hk.Tensor[i32], *, N: int = 1):
        t = hk.comm.claim(q.at[0])
        while t < N:
            done.at[t].atomic_add_scalar(1)
            who.at[t].atomic_max_scalar(hk.arch.block_idx.x)
            t = hk.comm.claim(q.at[0])
    k = hk.kernel(body, warps=1, name="comm_queue", grid=lambda p: (64, 1, 1), resident=True)
    n = 20000
    q = torch.zeros(16, dtype=torch.int32, device="cuda")
    done = torch.zeros(n, dtype=torch.int32, device="cuda")
    who = torch.full((n,), -1, dtype=torch.int32, device="cuda")
    k(q, done, who, N=n)
    torch.cuda.synchronize()
    assert bool((done == 1).all())
    assert len(set(who.tolist())) == 64          # every workgroup took some


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_a_persistent_task_graph_respects_its_dependencies():
    # L layers of K tasks, all taken from one queue; task (l, k) waits for layer l-1 to be done,
    # reads its neighbour's value and adds one: every value of the last layer must be L
    def body(q: hk.Tensor[i32, "device"], val: hk.Tensor[i32, "device"], cnt: hk.Tensor[i32, "device"],
             err: hk.Tensor[i32], *, K: int = 64, L: int = 16):
        t = hk.comm.claim(q.at[0])
        while t < K * L:
            l = t // K
            kk = t % K
            if l > 0:
                hk.comm.wait(cnt.at[16 * (l - 1)], K, timeout_s=1.0, error=err.at[0])
                prev = val.at[(l - 1) * K + (kk + 1) % K].atomic_max_scalar(0)
                val.at[t].atomic_max_scalar(prev + 1)
            else:
                val.at[t].atomic_max_scalar(1)
            hk.comm.arrive(cnt.at[16 * l], 1)
            t = hk.comm.claim(q.at[0])
    k = hk.kernel(body, warps=1, name="comm_task_graph", grid=lambda p: (32, 1, 1), resident=True)
    K, L = 64, 16
    q = torch.zeros(16, dtype=torch.int32, device="cuda")
    val = torch.zeros(K * L, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(16 * L, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    k(q, val, cnt, err, K=K, L=L)
    torch.cuda.synchronize()
    assert int(err[0]) == 0
    assert val.view(L, K)[-1].tolist() == [L] * K


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_xcd_id_names_eight_dies():
    def body(out: hk.Tensor[i32]):
        out.at[hk.arch.block_idx.x].atomic_max_scalar(hk.arch.xcd_id())
    k = hk.kernel(body, warps=1, name="comm_xcd_ids", grid=lambda p: (p.out.cols, 1, 1))
    out = torch.full((256,), -1, dtype=torch.int32, device="cuda")
    k(out)
    torch.cuda.synchronize()
    ids = out.tolist()
    assert set(ids) == set(range(8)) and all(ids.count(x) == 32 for x in range(8))


# ---------------------------------------------------------------- two GPUs

def _gpu_kernels():
    def writer(src: hk.Tensor[hk.bf16], data: hk.Tensor[hk.bf16, "system"],
               flag: hk.Tensor[i32, "system"], ack: hk.Tensor[i32, "system"],
               err: hk.Tensor[i32], *, ROUNDS: int = 2000):
        at0 = hk.tile_coord(0, 0, 0, 0)
        for r in hk.range(1, ROUNDS + 1):
            hk.comm.wait(ack.at[0], r - 1, timeout_s=2.0, error=err.at[0])
            data[at0] = src[hk.tile_coord(r % SLOTS, 0, 0, 0)].load(T)
            hk.comm.signal(flag.at[0], r)

    def reader(data: hk.Tensor[hk.bf16, "system"], flag: hk.Tensor[i32, "system"],
               ack: hk.Tensor[i32, "system"], out: hk.Tensor[hk.bf16], err: hk.Tensor[i32],
               *, ROUNDS: int = 2000):
        at0 = hk.tile_coord(0, 0, 0, 0)
        for r in hk.range(1, ROUNDS + 1):
            hk.comm.wait(flag.at[0], r, timeout_s=2.0, error=err.at[0])
            out[hk.tile_coord(r - 1, 0, 0, 0)] = data[at0].load(T)
            hk.comm.signal(ack.at[0], r)

    def meet(mine: hk.Tensor[i32, "system"], theirs: hk.Tensor[i32, "system"],
             cnt: hk.Tensor[i32, "system"], out: hk.Tensor[i32], err: hk.Tensor[i32],
             *, RANK: int = 0, ROUNDS: int = 200):
        # a barrier across the two GPUs: each round, publish into the peer's slot, meet on the
        # counter (on rank 0's memory), read what the peer published into mine
        for r in hk.range(1, ROUNDS + 1):
            theirs.at[0].atomic_max_scalar(r * 10 + RANK)
            hk.comm.barrier(cnt.at[0], 2 * r, timeout_s=2.0, error=err.at[0])
            out.at[r - 1].atomic_max_scalar(mine.at[0].atomic_max_scalar(0))
            hk.comm.barrier(cnt.at[64], 2 * r, timeout_s=2.0, error=err.at[0])

    w = hk.kernel(writer, warps=1, name="comm_gpu_writer", grid=lambda p: (1, 1, 1))
    r = hk.kernel(reader, warps=1, name="comm_gpu_reader", grid=lambda p: (1, 1, 1))
    m = hk.kernel(meet, warps=1, name="comm_gpu_meet", grid=lambda p: (1, 1, 1))
    return w, r, m


def _two_gpu_rank(rank, port, rounds, q):
    import torch.distributed as dist

    from hk.runtime import SymmetricBuffer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        # rank 1's buffer: [data tile | flag line]; rank 0's: [ack line]
        buf = SymmetricBuffer(64 * 1024, None)
        w, r, m = _gpu_kernels()
        err = torch.zeros(1, dtype=torch.int32, device="cuda")
        data = buf.peer(1, torch.bfloat16, (1, 1, 16, 64))
        flag = buf.peer(1, torch.int32, (64,), offset=32 * 1024)
        ack = buf.peer(0, torch.int32, (64,), offset=48 * 1024)
        dist.barrier()
        if rank == 0:
            src = _src()
            w(src, data, flag, ack, err, ROUNDS=rounds)
            torch.cuda.synchronize()
            q.put(("w", int(err[0]), 0))
        else:
            out = torch.zeros(rounds, 1, 16, 64, dtype=torch.bfloat16, device="cuda")
            r(data, flag, ack, out, err, ROUNDS=rounds)
            torch.cuda.synchronize()
            q.put(("r", int(err[0]), _stale_rounds(out, _src(), rounds)))
        dist.barrier()
        # the cross-GPU barrier: slots at 40 KiB of each rank's buffer, the counter on rank 0's
        mine = buf.local(torch.int32, (64,), offset=40 * 1024)
        theirs = buf.peer(1 - rank, torch.int32, (64,), offset=40 * 1024)
        cnt = buf.peer(0, torch.int32, (128,), offset=44 * 1024)
        out = torch.full((200,), -1, dtype=torch.int32, device="cuda")
        err.zero_(); dist.barrier()
        m(mine, theirs, cnt, out, err, RANK=rank, ROUNDS=200)
        torch.cuda.synchronize()
        want = torch.tensor([r_ * 10 + (1 - rank) for r_ in range(1, 201)], dtype=torch.int32, device="cuda")
        q.put((f"m{rank}", int(err[0]), int((out != want).sum())))
        dist.barrier()
        buf.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_message_passing_between_gpus_through_symmetric_memory():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    import torch.multiprocessing as mp

    rounds = 2000
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_two_gpu_rank, args=(29561, rounds, q), nprocs=2)
    got = {k: (e, s) for k, e, s in (q.get(timeout=60) for _ in range(4))}
    assert got["w"][0] == 0 and got["r"][0] == 0, f"a wait timed out: {got}"
    assert got["r"][1] == 0, f"{got['r'][1]} of {rounds} rounds read stale data"
    # the barrier across GPUs: every round each rank saw the peer's value of that round
    assert got["m0"] == (0, 0) and got["m1"] == (0, 0), got


# ---------------------------------------------------------------- patterns (two GPUs)
# Tile-level signals with a double-buffered symmetric buffer and flow control -- the TileLink / Flux
# shape: rank 0's workgroups produce tiles round after round into rank 1's buffer (parity r % 2),
# flag each tile; rank 1 consumes them in the reverse order, tile by tile as their flags arrive, and
# acks each round on a counter in rank 0's memory, so rank 0 reuses a buffer only two rounds later.

TT = 64          # tiles a round
TR = 40          # rounds
TF = hk.RT(hk.fp32, 16, 32)


def _pattern_kernels():
    def producer(src: hk.Tensor[hk.fp32], buf: hk.Tensor[hk.fp32, "system"], flags: hk.Tensor[i32, "system"],
                 ack: hk.Tensor[i32, "system"], err: hk.Tensor[i32]):
        w = hk.arch.block_idx.x
        for r in hk.range(1, TR + 1):
            if r > 2:      # the consumer is done with round r - 2, whose buffer this one reuses
                hk.comm.wait(ack.at[0], (r - 2) * 8, timeout_s=2.0, error=err.at[0])
            for t in hk.range(w, TT, 8):
                buf[hk.tile_coord(0, 0, (r % 2) * TT + t, 0)] = src[hk.tile_coord(0, 0, (r % 4) * TT + t, 0)].load(TF)
                hk.comm.signal(flags.at[16 * ((r % 2) * TT + t)], r)       # this tile, now

    def consumer(buf: hk.Tensor[hk.fp32, "system"], flags: hk.Tensor[i32, "system"],
                 ack: hk.Tensor[i32, "system"], out: hk.Tensor[hk.fp32], err: hk.Tensor[i32]):
        w = hk.arch.block_idx.x
        for r in hk.range(1, TR + 1):
            for t0 in hk.range(w, TT, 8):
                t = TT - 1 - t0                                   # the reverse order
                hk.comm.wait(flags.at[16 * ((r % 2) * TT + t)], r, timeout_s=2.0, error=err.at[0])
                out[hk.tile_coord(0, 0, (r - 1) * TT + t, 0)] = buf[hk.tile_coord(0, 0, (r % 2) * TT + t, 0)].load(TF)
            hk.comm.arrive(ack.at[0], 1)

    return (hk.kernel(producer, warps=1, name="comm_tile_producer", grid=lambda p: (8, 1, 1), resident=True),
            hk.kernel(consumer, warps=1, name="comm_tile_consumer", grid=lambda p: (8, 1, 1), resident=True))


def _pattern_rank(rank, port, q):
    import torch.distributed as dist

    from hk.runtime import SymmetricBuffer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        nb = 2 * TT * 16 * 32 * 4
        sym = SymmetricBuffer(nb + 2 * TT * 64 + 4096, None)
        buf = sym.peer(1, torch.float32, (2 * TT * 16, 32))
        flags = sym.peer(1, torch.int32, (2 * TT * 16,), offset=nb)
        ack = sym.peer(0, torch.int32, (16,), offset=nb + 2 * TT * 64)
        err = torch.zeros(1, dtype=torch.int32, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(5)
        src = torch.randn(4 * TT * 16, 32, device="cuda", generator=g)   # round r sends block r % 4
        prod, cons = _pattern_kernels()
        dist.barrier()
        if rank == 0:
            prod(src, buf, flags, ack, err)
            torch.cuda.synchronize()
            q.put(("p", int(err[0]), 0))
        else:
            out = torch.zeros(TR * TT * 16, 32, device="cuda")
            cons(buf, flags, ack, out, err)
            torch.cuda.synchronize()
            want = torch.cat([src[(r % 4) * TT * 16:(r % 4 + 1) * TT * 16] for r in range(1, TR + 1)])
            q.put(("c", int(err[0]), int((out != want).any(dim=1).sum())))
        dist.barrier()
        sym.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_tile_signals_through_a_double_buffered_symmetric_buffer():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    import torch.multiprocessing as mp

    q = mp.get_context("spawn").Queue()
    mp.spawn(_pattern_rank, args=(29563, q), nprocs=2)
    got = {k: (e, n) for k, e, n in (q.get(timeout=60) for _ in range(2))}
    assert got["p"][0] == 0 and got["c"][0] == 0, f"a wait timed out: {got}"
    assert got["c"][1] == 0, f"{got['c'][1]} rows of {TR} rounds read the wrong tile"


# ---------------------------------------------------------------- tagged mailboxes
# post / take: a one-int32 message as ONE 8-byte store (value, tag). A ping-pong of R rounds: one side
# posts round r's value with tag r, the other takes it and posts value + 1 back with tag r; the slot is
# reused only after its message was taken (the reply is the ack).

@hk.func
def _pingpong(a, b, out, err, side, rounds):
    # side 0 posts into `b` and takes from `a`; side 1 takes from `b` and posts into `a`
    if side == 0:
        for r in hk.range(1, rounds + 1):
            hk.comm.post(b.at[0], 3 * r, r)
            v = hk.comm.take(a.at[0], r, timeout_s=2.0, error=err.at[0])
            out.at[r - 1].atomic_max_scalar(v)
    if side == 1:
        for r in hk.range(1, rounds + 1):
            v = hk.comm.take(b.at[0], r, timeout_s=2.0, error=err.at[0])
            hk.comm.post(a.at[0], v + 1, r)
            out.at[rounds + r - 1].atomic_max_scalar(v)


def _pingpong_xcd_kernel():
    def body(a: hk.Tensor[i32, "device"], b: hk.Tensor[i32, "device"], out: hk.Tensor[i32],
             err: hk.Tensor[i32], *, ROUNDS: int = 2000):
        _pingpong(a, b, out, err, hk.arch.block_idx.x, ROUNDS)    # workgroups 0 and 1
    # 16 workgroups: 0 and 1 are dealt to different XCDs (round robin over 8)
    return hk.kernel(body, warps=1, name="comm_mailbox_xcd", grid=lambda p: (16, 1, 1))


def _pingpong_gpu_kernel():
    def body(a: hk.Tensor[i32, "system"], b: hk.Tensor[i32, "system"], out: hk.Tensor[i32],
             err: hk.Tensor[i32], *, SIDE: int = 0, ROUNDS: int = 2000):
        _pingpong(a, b, out, err, SIDE, ROUNDS)
    return hk.kernel(body, warps=1, name="comm_mailbox_gpu", grid=lambda p: (1, 1, 1))


def _flag_pingpong_kernel():
    # the same ping-pong through the data + signal / wait protocol (a 16 x 64 bf16 tile and a flag),
    # for the latency comparison: side 0 stores into `db` and flags `fb`, side 1 answers into `da`
    def body(da: hk.Tensor[hk.bf16, "system"], db: hk.Tensor[hk.bf16, "system"],
             fa: hk.Tensor[i32, "system"], fb: hk.Tensor[i32, "system"], err: hk.Tensor[i32],
             *, SIDE: int = 0, ROUNDS: int = 2000):
        at0 = hk.tile_coord(0, 0, 0, 0)
        if SIDE == 0:
            for r in hk.range(1, ROUNDS + 1):
                db[at0] = da[at0].load(T)
                hk.comm.signal(fb.at[0], r)
                hk.comm.wait(fa.at[0], r, timeout_s=2.0, error=err.at[0])
        if SIDE == 1:
            for r in hk.range(1, ROUNDS + 1):
                hk.comm.wait(fb.at[0], r, timeout_s=2.0, error=err.at[0])
                da[at0] = db[at0].load(T)
                hk.comm.signal(fa.at[0], r)
    return hk.kernel(body, warps=1, name="comm_flag_pingpong", grid=lambda p: (1, 1, 1))


def _pingpong_want(rounds):
    r = torch.arange(1, rounds + 1, dtype=torch.int32, device="cuda")
    return torch.cat([3 * r + 1, 3 * r])     # side 0 got the replies, side 1 the posts


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_tagged_mailboxes_ping_pong_between_xcds():
    rounds = 2000
    # the two boxes a cache line apart
    box = torch.zeros(128, dtype=torch.int32, device="cuda")
    out = torch.full((2 * rounds,), -1, dtype=torch.int32, device="cuda")
    err = torch.zeros(1, dtype=torch.int32, device="cuda")
    _pingpong_xcd_kernel()(box[:64], box[64:], out, err, ROUNDS=rounds)
    torch.cuda.synchronize()
    assert int(err[0]) == 0
    assert int((out != _pingpong_want(rounds)).sum()) == 0


def _mailbox_rank(rank, port, rounds, q):
    import torch.distributed as dist

    from hk.runtime import SymmetricBuffer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        buf = SymmetricBuffer(64 * 1024, None)
        # each rank's box: 64 ints at offset 0 of its own buffer
        mine = buf.local(torch.int32, (64,), offset=0)
        theirs = buf.peer(1 - rank, torch.int32, (64,), offset=0)
        out = torch.full((2 * rounds,), -1, dtype=torch.int32, device="cuda")
        err = torch.zeros(1, dtype=torch.int32, device="cuda")
        k = _pingpong_gpu_kernel()
        # rank 0 is side 0 (posts into the peer's box, takes from its own), rank 1 side 1
        a, b = (mine, theirs) if rank == 0 else (theirs, mine)
        k(a, b, out, err, SIDE=rank, ROUNDS=rounds)     # compile (ROUNDS is a constant) and warm up
        torch.cuda.synchronize(); dist.barrier()
        mine.zero_(); out.fill_(-1); torch.cuda.synchronize(); dist.barrier()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        k(a, b, out, err, SIDE=rank, ROUNDS=rounds)
        e1.record(); e1.synchronize()
        half = _pingpong_want(rounds)[:rounds] if rank == 0 else _pingpong_want(rounds)[rounds:]
        got = out[:rounds] if rank == 0 else out[rounds:]
        us_box = e0.elapsed_time(e1) * 1e3 / rounds
        # the data + flag protocol: tiles at 16 KiB, flags at 32 KiB of each rank's buffer
        d_mine = buf.local(torch.bfloat16, (1, 1, 16, 64), offset=16 * 1024)
        d_theirs = buf.peer(1 - rank, torch.bfloat16, (1, 1, 16, 64), offset=16 * 1024)
        f_mine = buf.local(torch.int32, (64,), offset=32 * 1024)
        f_theirs = buf.peer(1 - rank, torch.int32, (64,), offset=32 * 1024)
        da, db, fa, fb = (d_mine, d_theirs, f_mine, f_theirs) if rank == 0 else (d_theirs, d_mine, f_theirs, f_mine)
        kf = _flag_pingpong_kernel()
        e2 = torch.zeros(1, dtype=torch.int32, device="cuda")
        kf(da, db, fa, fb, e2, SIDE=rank, ROUNDS=rounds)
        torch.cuda.synchronize(); dist.barrier()
        f_mine.zero_(); torch.cuda.synchronize(); dist.barrier()
        e0.record()
        kf(da, db, fa, fb, e2, SIDE=rank, ROUNDS=rounds)
        e1.record(); e1.synchronize()
        us_flag = e0.elapsed_time(e1) * 1e3 / rounds
        q.put((rank, int(err[0]) | int(e2[0]), int((got != half).sum()), us_box, us_flag))
        dist.barrier()
        buf.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_tagged_mailboxes_ping_pong_between_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    import torch.multiprocessing as mp

    rounds = 2000
    q = mp.get_context("spawn").Queue()
    mp.spawn(_mailbox_rank, args=(29565, rounds, q), nprocs=2)
    got = {k: (e, bad, us, usf) for k, e, bad, us, usf in (q.get(timeout=60) for _ in range(2))}
    assert got[0][0] == 0 and got[1][0] == 0, f"a take or wait timed out: {got}"
    assert got[0][1] == 0 and got[1][1] == 0, f"messages lost or torn: {got}"
    print(f"round trip between GPUs: mailbox {got[0][2]:.2f} us, data tile + signal/wait {got[0][3]:.2f} us")
