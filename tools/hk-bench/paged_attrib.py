#!/usr/bin/env python3
"""Where the paged decode's time goes, one deleted stage at a time.

/kernel-attribution, in the DSL: `paged.fused_kernel(..., ablate=...)` drops
a stage of the walk at trace time, every variant is wrong on purpose, and all
of them are timed in one process, interleaved. Read the table as shares --
work deleted, not time recoverable -- and ask what covers a stage before
acting on it. Shares are of the walk alone; `decode` adds the merge launch.

  walk      the paged walk, splits' partials out, no merge
  -page     pages contiguous, no block_table read per page
  -kload    one K fragment loaded once and reused: the QK math stays
  -vload    the same for V and the PV math
  -loads    both: what is left is math and scalar work
  -qk, -softmax, -pv    the stage's math (and the loads only it consumes)
  merge     the merge launch alone, over the walk's partials
  decode    walk + merge, the two launches the backend makes, at every
            split count in the case

GPU time per launch: `--reps` launches back to back between two events, so a
few-microsecond launch is not charged to the kernel.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/paged_attrib.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

from hk.ops import paged                           # noqa: E402
from _timing import burn_in                        # noqa: E402

HKV, G, D = 8, 4, 128
PAGE = paged.KV_BLOCK

ABLATIONS = {
    "walk": (),
    "-page": ("page",),
    "-kload": ("kload",),
    "-vload": ("vload",),
    "-loads": ("kload", "vload"),
    "-qk": ("qk",),
    "-softmax": ("softmax",),
    "-pv": ("pv",),
}

#: (ctx, batch, split counts to time `decode` at; the rest run at the first).
CASES = (
    (1024, 1, (32, 16, 8)),
    (4096, 4, (32, 16, 8)),
    (4096, 16, (8, 16, 4)),
    (16384, 16, (8, 16, 4)),
)


def setup(n, ctx):
    g = torch.Generator(device="cuda").manual_seed(ctx + n)
    pages = ctx // PAGE
    nb = n * pages
    kc = torch.randn(nb, HKV, PAGE, D, device="cuda", dtype=torch.bfloat16,
                     generator=g)
    vc = torch.randn_like(kc)
    table = torch.randperm(nb, device="cuda", generator=g).view(n, pages)
    lens = torch.full((n,), ctx, device="cuda", dtype=torch.int32)
    q = torch.randn(n, HKV, G, D, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    return q, kc, vc, table.to(torch.int32).contiguous(), lens


def per_launch(fn, reps):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cases", default="",
                    help="ctx:batch:s1/s2/..., comma separated; the first "
                         "split count runs the ablations")
    args = ap.parse_args()
    cases = CASES
    if args.cases:
        cases = tuple((int(c), int(b), tuple(int(s) for s in ss.split("/")))
                      for c, b, ss in (x.split(":") for x in args.cases.split(",")))

    walks, merges = {}, {}
    for _, _, splits in cases:
        for s in splits:
            walks[(s, ())] = paged.fused_kernel(D, s)
            merges[s] = paged.merge_kernel(D, s)
        for v in ABLATIONS.values():
            walks[(splits[0], v)] = paged.fused_kernel(D, splits[0], v)
    with ThreadPoolExecutor(16) as ex:           # hipcc is one process each
        list(ex.map(lambda k: k.build(),
                    [*walks.values(), *merges.values()]))
    print(f"built {len(walks) + len(merges)} kernels", flush=True)
    burn_in()

    for ctx, n, splits in cases:
        q, kc, vc, table, lens = setup(n, ctx)
        out = torch.empty_like(q)
        fns = {}
        for s in splits:
            sc = paged.fused_scratch(n * s, HKV, D, q.device, group=G)
            if s == splits[0]:
                for name, v in ABLATIONS.items():
                    fns[(s, name)] = (lambda k=walks[(s, v)], sc=sc:
                                      k(q, kc, vc, table, lens, out, *sc))
                walks[(s, ())](q, kc, vc, table, lens, out, *sc)
                fns[(s, "merge")] = (lambda mk=merges[s], sc=sc:
                                     mk(lens, out, *sc))
            fns[(s, "decode")] = (
                lambda w=walks[(s, ())], mk=merges[s], sc=sc, s=s:
                paged.decode(w, mk, q, kc, vc, table, lens, out, sc, s))
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
        best = {key: float("inf") for key in fns}
        for _ in range(args.rounds):
            for key, fn in fns.items():
                best[key] = min(best[key], per_launch(fn, args.reps))
        gb = n * ctx * HKV * D * 2 * 2 / 1e9
        walk = best[(splits[0], "walk")]
        print(f"\nctx {ctx} batch {n}: {gb * 1e3:.1f} MB of K+V")
        for (s, name), t in best.items():
            share = (f"{100 * (1 - t / walk):7.1f}%"
                     if name.startswith("-") else "")
            print(f"  s{s:<3}{name:<10}{t * 1e3:9.1f} us"
                  f"{gb / (t * 1e-3):8.0f} GB/s{share}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
