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

    w = hk.kernel(writer, warps=1, name="comm_gpu_writer", grid=lambda p: (1, 1, 1))
    r = hk.kernel(reader, warps=1, name="comm_gpu_reader", grid=lambda p: (1, 1, 1))
    return w, r


def _two_gpu_rank(rank, port, rounds, q):
    import torch.distributed as dist

    from hk.runtime import SymmetricBuffer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        # rank 1's buffer: [data tile | flag line]; rank 0's: [ack line]
        buf = SymmetricBuffer(64 * 1024, None)
        w, r = _gpu_kernels()
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
    got = {k: (e, s) for k, e, s in (q.get(timeout=60) for _ in range(2))}
    assert got["w"][0] == 0 and got["r"][0] == 0, f"a wait timed out: {got}"
    assert got["r"][1] == 0, f"{got['r'][1]} of {rounds} rounds read stale data"
