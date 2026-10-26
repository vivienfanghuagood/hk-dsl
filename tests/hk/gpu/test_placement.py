"""hk.dsl.placement on the GPU: a placed grid runs every point once, a
cluster's workgroups share an XCD, and a placed kernel computes bitwise what
the unplaced one does. The host-side checks (bijection, analysis, policy) are
tests/hk/ir/test_placement.py."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import i32  # noqa: E402

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(os.environ.get("HK_ARCH") != "gfx950",
                                 reason="XCDs: gfx950's")]


def _probe(placement):
    def body(shape: hk.Tensor[i32], out: hk.Tensor[i32]):
        x, y, z = hk.arch.block_idx.x, hk.arch.block_idx.y, hk.arch.block_idx.z
        i = (z * 6 + y) * shape.depth + x
        out.at[i, 0].atomic_max_scalar(1)                       # ran
        out.at[i, 1].atomic_max_scalar(hk.arch.xcd_id())        # where
    return hk.kernel(body, warps=1, name=f"pl_probe{placement.suffix(8)}",
                     grid=lambda p: (p.shape.depth, 6, p.shape.batch), placement=placement)


@pytest.mark.parametrize("gx,gz", [(40, 1), (37, 3), (8, 2)])
def test_a_cluster_placed_grid_runs_every_point_once_its_clusters_on_one_xcd(gx, gz):
    """Key (z, y): 6 * gz clusters of gx workgroups. Every (x, y, z) runs, and a
    cluster sits on one XCD unless a range boundary cuts it (then two)."""
    k = _probe(hk.Placement.cluster(key=(hk.Digit.of("z"), hk.Digit.of("y"))))
    n = gx * 6 * gz
    shape = torch.zeros(gz, gx, 1, 1, dtype=torch.int32, device="cuda")
    out = torch.full((n, 2), -1, dtype=torch.int32, device="cuda")
    k(shape, out)
    torch.cuda.synchronize()
    ran, xcd = out[:, 0].cpu(), out[:, 1].cpu()
    assert bool((ran == 1).all()), "a grid point did not run"
    for c in range(6 * gz):
        dies = set(xcd[c * gx:(c + 1) * gx].tolist())
        assert len(dies) <= 2, (c, dies)
    if n % 8 == 0 and (n // 8) % gx == 0:      # ranges a whole number of clusters
        assert all(len(set(xcd[c * gx:(c + 1) * gx].tolist())) == 1 for c in range(6 * gz))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,h,h_kv,n", [(1, 40, 40, 4097), (2, 48, 48, 2048), (3, 24, 3, 777)])
def test_the_placed_attention_is_bitwise_the_unplaced(causal, b, h, h_kv, n):
    from hk.ops import attn_cdna

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(b, h, n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(b, h_kv, n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(b, h_kv, n, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    base = attn_cdna.attn_kernel(f"pl_attn_gpu_{int(causal)}", causal=causal)
    outs = []
    for kern in (base, base.placed()):
        o = torch.full_like(q, 7.0)
        kern(q, k, v, o)
        outs.append(o)
    assert torch.equal(outs[0], outs[1])


def test_auto_launches_the_cluster_placement_where_the_policy_takes_it():
    from hk.ops import attn_cdna

    k = attn_cdna.attn_kernel("pl_attn_auto", causal=False)
    auto = hk.Kernel(k.fn, warps=k.warps, grid=k.grid_fn, name="pl_attn_auto",
                     placement="auto")
    q = torch.randn(1, 128, 8192, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 128, 8192, 128, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    c = auto.placement_choice(q, kv, kv, o)
    assert c.placed and c.streams == 16, c
    small = torch.randn(1, 64, 8192, 128, device="cuda", dtype=torch.bfloat16)
    kv8 = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)
    assert not auto.placement_choice(small, kv8, kv8, torch.empty_like(small)).placed
    auto(q, kv, kv, o)
    ref = torch.empty_like(q)
    k(q, kv, kv, ref)
    assert torch.equal(o, ref)
