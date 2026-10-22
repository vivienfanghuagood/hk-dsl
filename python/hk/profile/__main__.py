"""python -m hk.profile TARGET [--inputs SPEC] : hk.profile from a shell.

TARGET   what to run (as python -m hk.check takes it):
           path/to/file.py:name    an @hk.kernel or a function in a file
           path/to/file.py         the file's CHECK dict, else its `kernel` or `fn`
           package.module:name     an importable one: hk.ops:silu_mul,
                                   torch.nn.functional:linear (hipBLASLt)
SPEC     its inputs: "bf16[1024,4096], bf16[28672,4096]" (python -m hk.check
         --help has the syntax), or FILE.py:name / module:name, a generator
         called as gen(device). Default: the file's `inputs`.

Each pass is a run of the call in a child process under rocprofv3: the
kernel trace (timing), the counter groups, the thread trace of the slowest
kernel. A pass that cannot run is a note; the timing and the ISA analysis
remain. --vs NAME=TARGET profiles another target on the same inputs and
lines the two up. Exit status: 0 profiled, 2 an error (it did not run).
docs/profile.md says how to read every field.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hk.profile",
        description="Profile the kernels a call launches under rocprofv3: per kernel "
                    "the roofline, matrix core, occupancy, stalls, LDS, caches, the main "
                    "loop's instruction mix and a verdict. See docs/profile.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("target", help="FILE.py[:name], module:name or package.module.name")
    p.add_argument("--inputs", help="the inputs (a spec, or FILE.py:name)")
    p.add_argument("--shape", help="for a generator that takes a shape: comma list")
    p.add_argument("--const", action="append", default=[], metavar="K=V",
                   help="a constexpr for a raw kernel (repeatable)")
    p.add_argument("--vs", metavar="NAME=TARGET",
                   help="also profile TARGET on the same inputs and compare")
    p.add_argument("--no-counters", action="store_true", help="skip the counter passes")
    p.add_argument("--no-att", action="store_true", help="skip the thread trace")
    p.add_argument("--att-kernels", type=int, default=1,
                   help="thread-trace this many of the slowest kernels (default 1)")
    p.add_argument("--iters", type=int, default=20, help="calls timed in the kernel trace")
    p.add_argument("--gap-ms", type=float, default=2.0,
                   help="host sleep between the timed calls (0: back to back)")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds a pass may take")
    p.add_argument("--flops", type=float, help="FLOPs of the call's main kernel")
    p.add_argument("--bytes", type=float, help="bytes of the call's main kernel")
    p.add_argument("--keep", metavar="DIR", help="keep the profiler's output in DIR")
    p.add_argument("--record", metavar="PATH",
                   help="write the raw measurements (no trace) as JSON, for replaying the "
                        "verdict logic without a GPU")
    p.add_argument("--build-only", action="store_true",
                   help="no GPU work: build and disassemble only")
    p.add_argument("--json", metavar="PATH", help="write the report as JSON ('-' for stdout)")
    p.add_argument("-q", "--quiet", action="store_true", help="no summary")
    p.add_argument("-v", "--verbose", action="store_true", help="say each pass as it runs")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    from hk.check.__main__ import _consts  # noqa: PLC0415

    import hk.profile as hp  # noqa: PLC0415

    vs = None
    if args.vs:
        k, _, v = args.vs.partition("=")
        vs = {k: v} if v else args.vs
    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else None
    p = hp.profile(args.target, args.inputs, consts=_consts(args.const), vs=vs,
                   counters=not args.no_counters, att=not args.no_att,
                   att_kernels=args.att_kernels, iters=args.iters, gap_ms=args.gap_ms,
                   timeout=args.timeout,
                   flops=args.flops, bytes=args.bytes,
                   shape=[int(x) for x in args.shape.split(",")] if args.shape else None,
                   workdir=args.keep, keep=bool(args.keep), record=args.record,
                   gpu=False if args.build_only else None, log=log)
    if args.json == "-":
        print(p.dumps())
        if not args.quiet:
            print(p, file=sys.stderr)
    else:
        if args.json:
            Path(args.json).write_text(p.dumps())
        if not args.quiet:
            print(p)
    return 0 if p.ok else 2


if __name__ == "__main__":
    sys.exit(main())
