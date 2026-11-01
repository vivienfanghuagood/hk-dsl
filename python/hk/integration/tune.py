"""`python -m hk.integration tune`: measure a model's linear shapes before a
server starts, into this machine's routing records.

    python -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct
    python -m hk.integration tune --model meta-llama/Llama-3.1-8B --tp 2
    python -m hk.integration tune --model ./config.json --engine sglang

The shapes come from the model's config.json (a path, the local HF cache,
or the hub): qkv, o, gate_up, down and the LM head of a dense decoder,
sharded for `--tp`. A shape the shipped table or an earlier run already has
is skipped (`--force` measures it again); so is one every call of which
carries a bias, which the linear op sends to the engine. Each shape is
written as soon as it is measured (`hk.autotune.routes`), under the same
lock a server tuning at load takes, and the next engine start routes by it.

This is the same measurement the vLLM patch makes at load within
`HK_AUTOTUNE_BUDGET` (`hk.integration.vllm_autotune`); run here, it has no
budget unless given one, and a quiet card.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    from ..autotune import routes as R  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m hk.integration tune",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    help="HF name, a directory holding config.json, or the file")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    ap.add_argument("--engine", choices=("vllm", "sglang"), default="vllm",
                    help="whose linear the kernels must beat")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds; decode-sized Ms first (default: no limit)")
    ap.add_argument("--shapes", default="",
                    help="comma-separated subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--force", action="store_true",
                    help="measure shapes a table already has")
    ap.add_argument("--flydsl", action="store_true",
                    help="time FlyDSL's GEMM too, where it is installed")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be measured, and stop")
    args = ap.parse_args(argv)

    cfg = R.model_config(args.model)
    shapes = R.linear_shapes(cfg, args.tp)
    model = [s[:3] for s in shapes]          # every layer, biased or not: the context they run in
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    bias = R.biased(cfg)
    for name, n, k, _ in shapes:
        if name in bias:
            print(f"{name} {n}x{k}: carries a bias in this model; stays on the engine's path")
    shapes = [s for s in shapes if s[0] not in bias]

    import torch  # noqa: PLC0415

    from ..target import current_arch  # noqa: PLC0415
    from . import engine as E  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("no GPU: nothing can be measured here")
        return 2
    arch = current_arch()
    kind = f"{args.engine}_linear"
    env = "HK_VLLM_ROUTES" if args.engine == "vllm" else "HK_SGLANG_ROUTES"
    fn, ename = R.engine_path(args.engine)
    table = E.shipped_table(kind, arch, env, engine=ename)
    have = table.complete() if table is not None else set()
    todo = [s for s in shapes if args.force or (s[1], s[2]) not in have]
    print(f"{args.model} on {arch}, tp {args.tp}, against {ename}: "
          f"{len(todo)} of {len(shapes)} shapes to measure"
          + (f" ({', '.join(f'{s[0]} {s[1]}x{s[2]}' for s in todo)})" if todo else ""),
          flush=True)
    if args.dry_run or not todo:
        return 0
    rep = R.tune_missing(kind, arch, todo, engine=fn, engine_name=ename,
                         complete=set() if args.force else have, budget=args.budget,
                         force=args.force, flydsl=args.flydsl, by="hk.integration tune",
                         log=lambda s: print(s, flush=True), model=model,
                         route_of=lambda m, n, k: table.lookup(m, n, k) if (n, k) in have
                         else "engine")
    print(rep.line())
    return 0 if not rep.left else 1


if __name__ == "__main__":
    sys.exit(main())
