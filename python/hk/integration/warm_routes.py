"""`python3 -m hk.integration warm-routes`: build the kernels of hk's DeepSeek-V4.1 vLLM routes
before the engine starts, so no request pays a JIT compile (docs/design/serving-readiness.md, A3).

The routes (vllm_mxfp8, vllm_moe_a8w4) compile their kernels on first use, in every TP worker.
`--warm` builds the attention ops only. This runs each route once, on synthetic data of the
model's shapes and on this GPU, which builds exactly the modules the engine will load into
`HK_CACHE_DIR`. Run it once per image or cache, with the same HK_CACHE_DIR the server will use,
then start vLLM. The kernels do not depend on the weights' values, nor, past the route's
choice of schedule, on M: one M per kernel combination the route can pick for a shape
(`k2_plans`) covers the set.

    HK_CACHE_DIR=/cache/hk python3 -m hk.integration warm-routes --tp 2

Needs the vLLM image (aiter for the MoE route's sort and quant) and a gfx950 GPU.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

#: DeepSeek-V4.1-Flash: hidden size, the routed experts' intermediate size before TP, top-k.
HIDDEN, MOE_INTER, TOPK = 5120, 2304, 6
#: The largest M K2 is warmed for: a linear sees at most max_num_batched_tokens rows a step (16384
#: by default in the measured deployment); twice that covers a larger setting.
K2_MAX_M = 32768


def k2_plans(n: int, k: int, max_m: int = K2_MAX_M) -> Dict[tuple, int]:
    """Every kernel combination K2 can launch for one (N, K), with the smallest M that launches it:
    {(kind, schedule, tail): m}. `kind` and `schedule` are vllm_mxfp8.schedule's answer (its
    measured RANGES below MIN_M included: the four-warp stages=2 schedule lives only there); `tail`
    is whether gemm_8w_tail recuts the last wave onto a second, four-warp kernel at this M
    (mxfp8.tail_split). A call at each M builds exactly the kernels the route can pick for this
    shape at serving M up to `max_m`. No GPU, no torch: the warm-routes test checks it."""
    from ..ops import mxfp8  # noqa: PLC0415
    from . import vllm_mxfp8 as P  # noqa: PLC0415

    plans: Dict[tuple, int] = {}
    tails: Dict[int, bool] = {}
    for m in range(1, max_m + 1):
        plan = P.schedule(m, n, k)
        if plan is None:
            continue
        kind, sched = plan
        tail = False
        if kind == "8w":
            mb = -(-m // 256)                          # tail_split depends on M only through this
            if mb not in tails:
                n1 = mxfp8.tail_split(m, n)
                tails[mb] = bool(n1) and n1 < n
            tail = tails[mb]
        plans.setdefault((kind, tuple(sorted(sched.items())), tail), m)
    return plans


def k2_shapes(tp: int) -> List[Tuple[int, int]]:
    """The (N, K) K2 routes at this TP degree (the measured TP2 shapes, rescaled for another TP)."""
    from . import vllm_mxfp8 as P  # noqa: PLC0415

    return sorted(P.MEASURED) if tp == 2 else sorted({(n * 2 // tp, k) for n, k in P.MEASURED})


def _k2(tp: int) -> List[Tuple[str, float]]:
    import torch  # noqa: PLC0415

    from ..ops import mxfp8  # noqa: PLC0415
    from . import vllm_mxfp8 as P  # noqa: PLC0415

    if tp != 2:
        P.ANY = True                    # the measured (N, K) are TP2's
    out = []
    for n, k in k2_shapes(tp):
        t0 = time.perf_counter()
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        ws = torch.full((n // 32, k // 32), 127, device="cuda", dtype=torch.uint8)
        rows = mxfp8.expand_block_scales(ws)[:n]
        wa, wp = mxfp8.aiter_scales(rows), mxfp8.pack_scales(rows)
        for m in sorted(k2_plans(n, k).values()):     # one call per kernel combination it can pick
            x = torch.randn(m, k, device="cuda").bfloat16()
            P.mxfp8_linear(x, w, ws, wa, wp, torch.bfloat16)
        torch.cuda.synchronize()
        out.append((f"mxfp8 N{n} K{k}", time.perf_counter() - t0))
    return out


def _k1(tp: int) -> List[Tuple[str, float]]:
    import torch  # noqa: PLC0415
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight  # noqa: PLC0415

    from . import vllm_moe_a8w4 as PM  # noqa: PLC0415

    inter, k, e = MOE_INTER // tp, HIDDEN, 8      # E is a run-time count: 8 experts build the same
    t0 = time.perf_counter()
    f4 = torch.float4_e2m1fn_x2
    w1 = torch.randint(0, 256, (e, 2 * inter, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, inter // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.full((e * 2 * inter, k // 32), 127, device="cuda", dtype=torch.uint8)
    s2 = torch.full((e * k, inter // 32), 127, device="cuda", dtype=torch.uint8)
    w1s = shuffle_weight(w1.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=True)
    w2s = shuffle_weight(w2.view(f4), layout=(16, 16), is_guinterleave=True, gate_up=False)
    sc1 = shuffle_scale(s1, experts_cnt=e, is_guinterleave=True, gate_up=True)
    sc2 = shuffle_scale(s2, experts_cnt=e, is_guinterleave=True, gate_up=False)
    m = 512
    x = torch.randn(m, k, device="cuda").bfloat16()
    ids = torch.rand(m, e, device="cuda").topk(TOPK, 1).indices.to(torch.int32)
    wt = torch.full((m, TOPK), 1.0 / TOPK, device="cuda")
    PM.moe(x, w1s, w2s, wt, ids, sc1, sc2, 10.0)
    torch.cuda.synchronize()
    return [(f"moe_a8w4 inter {inter} hidden {k}", time.perf_counter() - t0)]


def warm(tp: int = 2, routes=("mxfp8", "moe_a8w4")) -> Dict[str, float]:
    """Run each route once; returns seconds per (route, shape)."""
    from ..runtime.compile import cache_dir  # noqa: PLC0415

    done: Dict[str, float] = {}
    print(f"warming hk's DeepSeek-V4.1 routes for TP{tp} into {cache_dir()}", flush=True)
    for r in routes:
        for label, s in (_k2(tp) if r == "mxfp8" else _k1(tp)):
            done[label] = s
            print(f"  {label:32s} {s:7.1f} s", flush=True)
    print(f"{len(done)} route shapes, {sum(done.values()):.1f} s", flush=True)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m hk.integration warm-routes",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel degree of the deployment")
    ap.add_argument("--routes", nargs="*", default=["mxfp8", "moe_a8w4"],
                    choices=["mxfp8", "moe_a8w4"])
    a = ap.parse_args(argv)
    warm(a.tp, tuple(a.routes))
    return 0
