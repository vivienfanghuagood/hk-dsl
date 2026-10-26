"""Workgroup placement: which XCD runs which workgroup (hk.dsl.placement).

An MI355X deals workgroup w to the XCD of w % 8, and each XCD has its own
4 MB L2. Bytes that several workgroups read are fetched once per XCD that
reads them, so the order a kernel numbers its work in decides how often they
cross to HBM -- with no change to what any workgroup computes.

The kernel here has attention's shape of sharing without its math: workgroup
(x, y, z) is query tile x of head y of batch z; it loads its own q tile and
then walks every tile of its head's kv, adding them up. The 16 query tiles of
a head all read that head's kv (the cluster); in the hardware's order the
heads of a round are spread over all eight XCDs, so each head's kv is fetched
by up to eight L2s.

- `hk.kernel(..., placement=hk.Placement.cluster())` renumbers the grid so
  that each XCD takes a contiguous range of clusters. With no `key=`, the
  cluster key comes from the kernel's own reads (here (z, y): batch and head;
  `hk.Digit.of("z")`, `hk.Digit.of("y")` spell it out). `Kernel.placed()` is
  the same kernel with that placement installed.
- `placement="auto"` launches the placed kernel where the policy takes it,
  from the measured thresholds (five shared streams an XCD;
  `Kernel.placement_choice(...)` says what it decided and why).
- A placement changes only the order: the outputs are bitwise the unplaced
  kernel's, which this script checks.
- On a GPU with one L2 (gfx1100), `hk.arch.xcd_count()` is 1 and every
  placement is the identity.

It is not faster here, and the script's timing shows that. On an MI355X the
two orders took the same time (1.00x at 4K, 16K and 32K kv rows a head, 64
and 128 heads): the whole grid is resident at once (0.25 rounds an XCD), and
one warp a workgroup adding tiles one after another is bound by latency, not
by what the L2s fetch. The policy still says "placed" -- its thresholds were
measured on attention, where the same renumbering gains 5-18% at 128 KV heads
(docs/design/placement-pass.md). A placement is a measured choice per kernel.

`hk.check --placement` checks, with no GPU, that the decode the kernel's own
emitted ops compute covers its grid exactly once (for "auto": the kernel and
its cluster placement):

    python3 examples/15_placement.py
    python3 examples/15_placement.py --dump cpp
    python -m hk.check examples/15_placement.py:shared_sum_auto --placement \\
        --inputs "bf16[1,128,256,64], bf16[1,128,4096,64], bf16[1,128,256,64]"
    # case 0: shared_sum_auto: the hardware's numbering (cluster key (z, y))
    # case 0: shared_sum_auto_x8cl0p: a bijection over 2048 workgroups, grid (16, 128, 1), order (z, y, x), 8 XCDs

(built for gfx950; for gfx1100 only the first line, since one L2 has nothing to
place).

The design, the policy and the measurements behind it:
docs/design/placement-pass.md.
"""

import _common

import hk

T = hk.RT(hk.fp32, 16, 64)
GRID = lambda p: (hk.cdiv(p.q.rows, 16), p.q.depth, p.q.batch)   # (query tile, head, batch)


def _shared_sum(q: hk.Tensor[hk.bf16], kv: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    x, h, b = hk.arch.block_idx.x, hk.arch.block_idx.y, hk.arch.block_idx.z
    acc = q[hk.tile_coord(b, h, x, 0)].load(T)               # this workgroup's own tile
    for j in hk.range(0, kv.rows // 16):                     # every tile of the head's kv:
        acc += kv[hk.tile_coord(b, h, j, 0)].load(T)         # read by all the head's workgroups
    o[hk.tile_coord(b, h, x, 0)] = acc


# The hardware's order, the cluster placement, and the policy's choice.
shared_sum = hk.kernel(_shared_sum, warps=1, grid=GRID, name="shared_sum")
shared_sum_placed = hk.kernel(
    _shared_sum, warps=1, grid=GRID, name="shared_sum_placed",
    placement=hk.Placement.cluster(key=(hk.Digit.of("z"), hk.Digit.of("y"))))
shared_sum_auto = hk.kernel(_shared_sum, warps=1, grid=GRID, name="shared_sum_auto",
                            placement="auto")


def main():
    args = _common.parse(__doc__)
    if _common.static(args, shared_sum_placed):
        return
    _common.require_gpu()
    import torch

    b, h, n, nk, d = 1, 128, 256, 4096, 64
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(b, h, nk, d, device="cuda", dtype=torch.bfloat16)
    print(f"shared sum: {b} x {h} heads, {n // 16} query tiles a head each reading the "
          f"head's {nk} x {d} bf16 kv")
    print(f"  XCDs: {hk.placement.xcd_count(shared_sum.arch)}"
          + ("  (one L2: every placement is the identity)"
             if hk.placement.xcd_count(shared_sum.arch) == 1 else ""))
    tr = shared_sum.placed().trace()
    print(f"  the analysis's cluster key: "
          f"{list(tr.plan.key) if tr.plan else 'none (one L2: nothing to place)'}")
    choice = shared_sum_auto.placement_choice(q, kv, torch.empty_like(q))
    print(f"  placement=\"auto\": {'placed' if choice.placed else 'not placed'} ({choice.reason})")

    outs = {}
    for name, k in (("hardware order", shared_sum), ("cluster placement", shared_sum_placed),
                    ("auto", shared_sum_auto)):
        o = torch.full_like(q, 7.0)
        k(q, kv, o)
        outs[name] = o
    ref = (q.float() + kv.float().view(b, h, nk // 16, 16, d).sum(2).repeat(1, 1, n // 16, 1))
    _common.check("hardware order vs torch", outs["hardware order"], ref, rel=2e-2)
    for name in ("cluster placement", "auto"):
        if not torch.equal(outs[name], outs["hardware order"]):
            raise AssertionError(f"{name}: not bitwise the hardware order's output")
        print(f"  {name + ' vs hardware order':<28} ok   bitwise equal")
    if args.no_bench:
        return
    o = torch.empty_like(q)
    best = _common.compare({
        "hardware order": lambda: shared_sum(q, kv, o),
        "cluster placement": lambda: shared_sum_placed(q, kv, o),
    }, args.rounds)
    _common.report("time", best, baseline="hardware order",
                   work=(q.numel() * 2 + kv.numel() * (n // 16)) * 2, unit="TB/s (as read)")


if __name__ == "__main__":
    main()
