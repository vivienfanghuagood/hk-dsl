#!/usr/bin/env python3
"""What does rounding fp32 to bf16 correctly cost?

`convertor<bf16, float>` used to truncate. Truncation is one shift; round to
nearest even is a compare, an add and a shift, and it sits in attention's
inner loop (the probability tile, every KV block) and in GEMM's epilogue
(every output element). So the fix has a price, and this measures it instead
of asserting it is small.

Both variants are compiled in *this* process and timed in alternating order,
because on a W7900D a cross-process comparison manufactures 13-16% out of
clock and power state alone. `-DHK_BF16_TRUNCATE=1` is part of the compile
cache key, so the two are genuinely different modules.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/bf16_round_cost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch                                       # noqa: E402

import hk                                          # noqa: E402
from hk.runtime import module as rtm               # noqa: E402
from hk.runtime.compile import build as rt_build   # noqa: E402
from _timing import interleave                     # noqa: E402

#: The settings `convertor<bf16, float>` offers, cheapest first.
MODES = {
    "trunc": ("-DHK_BF16_ROUND=0",),
    "ties-away": ("-DHK_BF16_ROUND=1",),
    "ties-even": ("-DHK_BF16_ROUND=2",),
    "even+nan": ("-DHK_BF16_ROUND=3",),
}


def load(kernel, extra=()):
    """Compile and load one kernel with extra flags, bypassing Kernel.build.

    Kernel.build has no extra_flags argument on purpose -- a kernel's flags
    are part of what it is -- so the A/B reaches past it rather than widening
    the API for a benchmark.
    """
    b = rt_build(kernel.source(), kernel.arch, name=kernel.name,
                 max_vgprs=kernel.max_vgprs, min_occupancy=kernel.min_occupancy,
                 extra_flags=extra)
    return rtm.load(b.so_path, b.module_name), b


def attn_case(n, head_dim=128, causal=False, heads=16):
    from hk.ops import attn

    k = attn._kernel_for(head_dim, causal, None, n)
    g = torch.Generator(device="cuda").manual_seed(0)
    q, kk, v = (torch.randn(1, heads, n, head_dim, device="cuda",
                            dtype=torch.bfloat16, generator=g)
                for _ in range(3))
    variants, builds = {}, {}
    for mode, flags in MODES.items():
        mod, b = load(k, flags)
        fn, out = getattr(mod, k.name), torch.empty_like(q)
        variants[mode] = (lambda f=fn, o=out: f(q, kk, v, o))
        builds[mode] = b
    flops = 4.0 * heads * n * n * head_dim / (2.0 if causal else 1.0)
    name = f"attn d{head_dim}{' causal' if causal else ''} N={n}"
    return name, flops, variants, builds


def gemm_case(m, n, k):
    from hk.ops import gemm

    kern = gemm.TUNER.kernel(f"{m}x{n}x{k}")
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16, generator=g)
    variants, builds = {}, {}
    for mode, flags in MODES.items():
        # Constexprs are baked at compile time; the binding takes tensors only.
        mod, bd = load(kern, flags)
        fn = getattr(mod, kern.name)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        variants[mode] = (lambda f=fn, o=c: f(a, b, o))
        builds[mode] = bd
    return f"gemm {m}x{n}x{k}", 2.0 * m * n * k, variants, builds


def report(cases):
    head = "".join(f"{m:>12}" for m in MODES)
    print(f"{'case':<26}{head}   vs trunc")
    for name, flops, variants, builds in cases:
        best = interleave(variants, rounds=5)
        base = best["trunc"]
        row = "".join(f"{best[m]:>12.3f}" for m in MODES)
        cost = "  ".join(f"{m} {100 * (best[m] - base) / base:+.1f}%"
                         for m in MODES if m != "trunc")
        print(f"{name:<26}{row}   {cost}")
        vg = {m: (builds[m].kernels[0].vgpr if builds[m].kernels else None)
              for m in MODES}
        scr = {m: (builds[m].kernels[0].scratch if builds[m].kernels else None)
               for m in MODES}
        print(f"{'':<26}{'':>12}vgpr {vg}  scratch {set(scr.values())}")


def main() -> int:
    cases = [attn_case(4096), attn_case(16384), attn_case(8192, causal=True),
             attn_case(4096, head_dim=64)]
    try:
        cases.append(gemm_case(4096, 4096, 4096))
    except Exception as e:  # noqa: BLE001
        # One line: the exception from a pybind arity mismatch carries every
        # tensor it was handed, and that is pages of output in a bench log.
        print(f"(gemm case skipped: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]})")
    report(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
