#!/usr/bin/env python3
"""How many ways should the KV axis be cut, at each (batch, context)?

`plan_splits` started as a guess -- aim for ~96 workgroups -- and the vLLM
sweep showed the guess is wrong where it matters: at batch 16 it asks for one
split, and one split means 128 workgroups of a single warp each walking 128
pages serially. Counting workgroups is the wrong unit when a workgroup is one
warp.

So measure it. This times the kernel alone, including the merge, across split
counts, and prints the achieved bandwidth against the card's 864 GB/s -- which
is the ceiling that matters, because decode attention reads the KV cache once
and does almost no arithmetic with it.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/paged_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                   # noqa: E402

from hk.ops import paged                       # noqa: E402
from _timing import interleave                 # noqa: E402

HBM_GB_S = 864.0
PAGE, TILE = paged.KV_BLOCK, paged.Q_TILE
#: Qwen3-8B's attention shape.
HKV, GROUP, D = 8, 4, 128


def case(n_req, n_ctx, splits):
    pages = n_ctx // PAGE
    g = torch.Generator(device="cuda").manual_seed(0)
    n_blocks = n_req * pages
    kc = torch.randn(n_blocks, HKV, PAGE, D, device="cuda",
                     dtype=torch.bfloat16, generator=g)
    vc = torch.randn_like(kc)
    table = torch.arange(n_blocks, device="cuda",
                         dtype=torch.int32).reshape(n_req, pages)
    q = torch.zeros(n_req, HKV, TILE, D, device="cuda", dtype=torch.bfloat16)
    q[:, :, :GROUP] = torch.randn(n_req, HKV, GROUP, D, device="cuda",
                                  dtype=torch.bfloat16, generator=g)
    lens = torch.full((n_req,), n_ctx, device="cuda", dtype=torch.int32)
    out = torch.empty(n_req, HKV, TILE, D, device="cuda",
                      dtype=torch.bfloat16)

    if splits == 1:
        k = paged.KERNELS[D]
        return lambda: k(q, kc, vc, table, lens, out)

    o_part = torch.empty(n_req * splits, HKV, TILE, D, device="cuda",
                         dtype=torch.float32)
    ml = torch.empty(n_req * splits, HKV, 2, TILE, device="cuda",
                     dtype=torch.float32)
    k = paged.split_kernel(D, splits)
    o32 = torch.empty(n_req, HKV, TILE, D, device="cuda", dtype=torch.float32)

    def run():
        k(q, kc, vc, table, lens, o_part, ml)
        paged.merge_splits(o_part, ml, splits, o32)

    return run


def main() -> int:
    print(f"{'ctx':>6}{'batch':>7}  " +
          "".join(f"{('s%d' % s):>9}" for s in SPLITS) +
          f"{'best':>7}{'GB/s':>9}{'%HBM':>7}")
    for n_ctx in (1024, 4096, 16384):
        for n_req in (1, 4, 8, 16, 32):
            # K and V, read once each, is the traffic the kernel exists to move.
            gb = n_req * n_ctx * HKV * D * 2 * 2 / 1e9
            fns, ok = {}, []
            for s in SPLITS:
                if s > n_ctx // PAGE:
                    continue
                try:
                    fns[f"s{s}"] = case(n_req, n_ctx, s)
                    ok.append(s)
                except Exception:        # noqa: BLE001 -- OOM at the big end
                    break
            if not fns:
                continue
            best = interleave(fns, rounds=3)
            row = "".join(f"{best.get('s%d' % s, float('nan')):9.3f}"
                          for s in SPLITS)
            bs = min(ok, key=lambda s: best[f"s{s}"])
            ms = best[f"s{bs}"]
            print(f"{n_ctx:>6}{n_req:>7}  {row}{bs:>7}"
                  f"{gb / (ms * 1e-3):>9.1f}{100 * gb / (ms * 1e-3) / HBM_GB_S:>6.0f}%",
                  flush=True)
    return 0


SPLITS = (1, 2, 4, 8, 16, 32)

if __name__ == "__main__":
    raise SystemExit(main())
