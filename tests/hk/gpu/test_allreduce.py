"""hk.ops.allreduce.AllReduce on two gfx950 GPUs: bitwise the fp32 sum rounded once, the same on both
ranks; repeated calls of different sizes (the epoch, the y parity); a CUDA graph replay."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


def _rank(rank, port, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from hk.ops.allreduce import AllReduce

    bad = []
    try:
        ar = AllReduce(None, 4096 * 5120)
        for i, m in enumerate([4096, 128, 2048, 4096, 1024, 512, 4096]):
            g = torch.Generator(device="cuda").manual_seed(1000 * i + rank)
            x = torch.randn(m, 5120, device="cuda", generator=g).to(torch.bfloat16)
            xs = [torch.zeros(m * 5120, dtype=torch.bfloat16) for _ in range(2)]
            dist.all_gather(xs, x.flatten().cpu())
            ref = (xs[0].float() + xs[1].float()).to(torch.bfloat16).view(m, 5120)
            y = ar(x)
            torch.cuda.synchronize()
            if not torch.equal(y.cpu(), ref):
                bad.append(f"m={m} call {i}")
        # a CUDA graph of two calls, replayed
        x = torch.randn(1024, 5120, device="cuda").to(torch.bfloat16)
        ar(x); torch.cuda.synchronize(); dist.barrier()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            y1 = ar(x)
            y2 = ar(x)
        for rep in range(3):
            x.normal_(); dist.barrier()
            gph.replay(); torch.cuda.synchronize()
            xs = [torch.zeros(1024 * 5120, dtype=torch.bfloat16) for _ in range(2)]
            dist.all_gather(xs, x.flatten().cpu())
            ref = (xs[0].float() + xs[1].float()).to(torch.bfloat16).view(1024, 5120)
            if not (torch.equal(y1.cpu(), ref) and torch.equal(y2.cpu(), ref)):
                bad.append(f"graph replay {rep}")
        q.put((rank, ar.error(), bad))
        dist.barrier()
        ar.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950", reason="hk.comm is gfx950's")
def test_allreduce_is_exact_on_both_ranks():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    import torch.multiprocessing as mp

    q = mp.get_context("spawn").Queue()
    mp.spawn(_rank, args=(29571, q), nprocs=2)
    got = [q.get(timeout=60) for _ in range(2)]
    for rank, err, bad in got:
        assert err == 0, f"rank {rank}: a wait timed out"
        assert not bad, f"rank {rank}: {bad}"
