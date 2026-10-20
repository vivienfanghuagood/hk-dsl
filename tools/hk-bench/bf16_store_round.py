#!/usr/bin/env python3
"""How RDNA's stores round fp32 to bf16: agreement with torch, and the cost.

include/rdna3 narrows a register tile, vector or strip to a bf16 global
through `store_convertor` (HK_BF16_STORE_ROUND, default 2: nearest even),
and converts in registers through `convertor` (HK_BF16_ROUND, default 1:
nearest, ties away). Before this the stores used `convertor` too. Every case
is built three times in this process -- the store at 1 (ties away, as
before), at the default, and at 3 (nearest even, NaN kept) -- and compared:

* `= torch`: the share of output elements equal to torch's bf16 answer,
  bit for bit. For an add of two bf16 rows torch's answer is the round to
  nearest even of the exact fp32 sum, so 100% is reachable; for a norm or a
  GEMM the fp32 values themselves differ in the last bits, and only the ties
  among equal ones can move.
* `ties`: elements where the old and new builds differ -- each an exact tie
  the kernel's fp32 value sat on.
* time: the three builds interleaved, rounds rotating, the median of each
  round's best. A kernel's own `defines` stay unless a mode overrides them:
  the GEMM keeps ties away, so its `new` is its `old`.

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/bf16_store_round.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

import hk                                          # noqa: E402
from _timing import burn_in                        # noqa: E402

OLD = {"HK_BF16_STORE_ROUND": 1}
NAN = {"HK_BF16_STORE_ROUND": 3}
#: (tag, defines): ties away as before, the default (nearest even), and
#: nearest even with NaN kept
MODES = (("old", OLD), ("new", {}), ("nan", NAN))
BF16 = torch.bfloat16


def variant(k, defs, tag):
    return type(k)(k.fn, arch=k._arch, warps=k.warps, grid=k.grid_fn,
                   name=f"{k.name}_{tag}", max_vgprs=k.max_vgprs,
                   min_occupancy=k.min_occupancy,
                   defines={**k.defines, **defs})


T64 = hk.RT(hk.fp32, 16, 64)


@hk.kernel(warps=4, grid=lambda p: (hk.cdiv(p.o.cols, 256), hk.cdiv(p.o.rows, 16), 1))
def residual_add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    """new-op's template stub, as generated: c = a + b on bf16 rows."""
    r0 = hk.arch.block_idx.y * 16
    c0 = (hk.arch.block_idx.x * 4 + hk.arch.warp_id()) * 64
    x = a.at[r0, c0].load(T64, bounded=True)
    y = b.at[r0, c0].load(T64, bounded=True)
    x += y
    o.at[r0, c0].store(x, bounded=True)


def case_add(rows, cols):
    g = torch.Generator(device="cuda").manual_seed(rows + cols)
    a = torch.randn(rows, cols, device="cuda", dtype=BF16, generator=g)
    b = torch.randn(rows, cols, device="cuda", dtype=BF16, generator=g)
    want = a + b
    arms = {}
    for tag, defs in MODES:
        k = variant(residual_add, defs, tag)
        o = torch.empty_like(a)
        arms[tag] = (lambda k=k, o=o: k(a, b, o), o)
    return f"add {rows}x{cols}", want, arms, 3 * a.numel() * 2


def case_strip(kind, rows, cols):
    from hk.ops import norm

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=BF16, generator=g)
    k0, chunks, run, exact, reserve = norm._resolve_strip(kind, BF16, cols, rows)
    xf = x.float()
    if kind == "rms":
        want = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(BF16)
    elif kind == "layer":
        want = torch.nn.functional.layer_norm(xf, (cols,), eps=1e-5).to(BF16)
    else:
        want = torch.softmax(xf, -1).to(BF16)
    arms = {}
    for tag, defs in MODES:
        k = variant(k0, defs, tag)
        o = torch.empty_like(x)
        if kind == "softmax":
            fn = (lambda k=k, o=o: k(x, o, CHUNKS=chunks, RUN=run, EXACT=exact,
                                     RESERVE=reserve))
        else:
            eps = 1e-6 if kind == "rms" else 1e-5
            fn = (lambda k=k, o=o, eps=eps: k(x, o, CHUNKS=chunks, RUN=run,
                                              EXACT=exact, RESERVE=reserve,
                                              EPS=eps))
        arms[tag] = (fn, o)
    return f"{kind} {rows}x{cols}", want, arms, 2 * x.numel() * 2


def case_addrms(rows, cols):
    """vLLM's fused_add_rms_norm on a strip: x + r written back (a residual
    add's ties) and normalised by a weight."""
    from hk.ops import norm

    g = torch.Generator(device="cuda").manual_seed(rows * 11 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=BF16, generator=g)
    r = torch.randn(rows, cols, device="cuda", dtype=BF16, generator=g)
    w = (1 + 0.1 * torch.randn(cols, device="cuda", generator=g)).to(BF16)
    k0, chunks, run, exact, reserve = norm._resolve_strip("addrmsw", BF16, cols, rows)
    want = x + r
    arms = {}
    for tag, defs in MODES:
        k = variant(k0, defs, tag)
        o, r_out = torch.empty_like(x), torch.empty_like(x)
        arms[tag] = (lambda k=k, o=o, r_out=r_out: k(
            x, r, w, o, r_out, CHUNKS=chunks, RUN=run, EXACT=exact,
            RESERVE=reserve, EPS=1e-6), r_out)
    return f"fused add {rows}x{cols}", want, arms, 0


def case_gemm(m, n, k):
    from hk.ops import gemm

    kern = gemm.TUNER.kernel(f"{m}x{n}x{k}")
    g = torch.Generator(device="cuda").manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=BF16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=BF16, generator=g)  # (N, K): hk's B
    want = a @ b.T
    arms = {}
    for tag, defs in MODES:
        kk = variant(kern, defs, tag)
        c = torch.empty(m, n, device="cuda", dtype=BF16)
        arms[tag] = (lambda kk=kk, c=c: kk(a, b, c), c)
    return f"gemm {m}x{n}x{k}", want, arms, 0


def case_attn(n, d=128, heads=16):
    from hk.ops import attn

    k0 = attn._kernel_for(d, False, None, n)
    g = torch.Generator(device="cuda").manual_seed(n + d)
    q, kk, v = (torch.randn(1, heads, n, d, device="cuda", dtype=BF16,
                            generator=g) for _ in range(3))
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), kk.float(), v.float()).to(BF16)
    arms = {}
    for tag, defs in MODES:
        k = variant(k0, defs, tag)
        o = torch.empty_like(q)
        arms[tag] = (lambda k=k, o=o: k(q, kk, v, o), o)
    return f"attn d{d} N={n}", want, arms, 0


def timed(arms, rounds, block=10):
    fns = {t: fn for t, (fn, _) in arms.items()}
    for fn in fns.values():
        fn()
    torch.cuda.synchronize()
    per = {t: [] for t in fns}
    names = list(fns)
    for r in range(rounds):
        k = r % len(names)
        for t in names[k:] + names[:k]:
            fn = fns[t]
            best = float("inf")
            for _ in range(block):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                fn()
                e.record()
                e.synchronize()
                best = min(best, s.elapsed_time(e))
            per[t].append(best)
    return {t: sorted(v)[len(v) // 2] for t, v in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    burn_in()
    makers = [lambda: case_add(16, 4096), lambda: case_add(2048, 4096),
              lambda: case_add(8192, 4096),
              lambda: case_strip("rms", 1, 4096), lambda: case_strip("rms", 32, 4096),
              lambda: case_addrms(1, 4096), lambda: case_addrms(32, 4096),
              lambda: case_addrms(2048, 4096),
              lambda: case_strip("rms", 2048, 4096),
              lambda: case_strip("layer", 2048, 4096),
              lambda: case_strip("softmax", 2048, 4096),
              lambda: case_gemm(4096, 4096, 4096),
              lambda: case_attn(4096)]
    rows = []
    print(f"{'case':<22}{'= torch old':>12}{'new':>9}{'ties':>9}"
          f"{'old us':>10}{'new us':>9}{'nan us':>9}{'new':>8}{'nan':>8}")
    for mk in makers:
        try:
            name, want, arms, _ = mk()
        except Exception as e:                    # noqa: BLE001
            print(f"(case skipped: {type(e).__name__}: {str(e).splitlines()[0][:150]})")
            continue
        for fn, _ in arms.values():
            fn()
        torch.cuda.synchronize()
        eq = {t: (o.view(torch.int16) == want.view(torch.int16)).float().mean().item()
              for t, (_, o) in arms.items()}
        ties = (arms["old"][1].view(torch.int16)
                != arms["new"][1].view(torch.int16)).float().mean().item()
        t = timed(arms, args.rounds)
        cn = 100 * (t["new"] / t["old"] - 1)
        cx = 100 * (t["nan"] / t["old"] - 1)
        print(f"{name:<22}{100 * eq['old']:>11.2f}%{100 * eq['new']:>8.2f}%"
              f"{100 * ties:>8.2f}%{1e3 * t['old']:>10.1f}{1e3 * t['new']:>9.1f}"
              f"{1e3 * t['nan']:>9.1f}{cn:>+7.1f}%{cx:>+7.1f}%", flush=True)
        rows.append(dict(case=name, eq_old=eq["old"], eq_new=eq["new"],
                         eq_nan=eq["nan"], ties=ties, us_old=1e3 * t["old"],
                         us_new=1e3 * t["new"], us_nan=1e3 * t["nan"]))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
