"""python -m hk.check TARGET [--ref REF] [--inputs SPEC]... : hk.check from a shell.

TARGET   what to check:
           path/to/file.py:name    an @hk.kernel or a function in a file
           path/to/file.py         the file's CHECK dict, else its `kernel` or `fn`
           package.module:name     an importable one, e.g. hk.ops:silu_mul
           package.module.name     the same, dotted
REF      the torch reference: FILE.py:name, module:name, a lambda, or an
         expression over the reference's argument names, which are the
         target's own parameter names (a raw kernel's minus its outputs):
           --ref "F.silu(a) * b"                    # torch, F, math, hk are in scope
         Default: the target file's `reference` (or `ref`).
SPEC     the inputs, one --inputs per case (shapes to sweep):
           "bf16[4096,14336], bf16[4096,14336]"   randn of that dtype and shape
           "i32[64]@randint(0,128), fp32[64]@rand, bf16[16,64]@zeros, 1e-6"
         dtypes bf16 fp16 fp32 fp64 i8 i16 i32 i64 u8 bool fp8e4m3 fp8e5m2;
         fills @randn (floats' default) @rand @zeros @ones @full(v) @arange
         @randint(lo,hi) (ints' default, 0..100) @empty, then optionally *s
         to scale (`bf16[64,64]@randn*0.1`); a bare number is a scalar
         argument. Or FILE.py:name / module:name, a generator called as
         gen(device) (or gen(device=..., shape=...)). Default: the target
         file's `inputs`, once per entry of its SHAPES if it has them.

Exit status: 0 pass (or built, with no GPU), 1 a wrong answer, 2 an error
(it did not trace, build or run). `--json -` prints the report as JSON on
stdout and the summary on stderr. docs/check.md has the rest.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------- loading


_FILES: Dict[str, Any] = {}


def load_file(path: str):
    """Import a .py file by path, as a real module (its kernels' source must
    be readable by inspect, so not exec)."""
    p = str(Path(path).resolve())
    if p in _FILES:
        return _FILES[p]
    name = "hk_check_" + re.sub(r"\W", "_", Path(p).stem)
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"hk.check: cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(Path(p).parent))
    spec.loader.exec_module(mod)
    _FILES[p] = mod
    return mod


def resolve(ref: str):
    """FILE.py[:name], module:name or a.b.c -> (object, the module it came from)."""
    path, _, attr = ref.partition(":") if ".py:" in ref or ref.endswith(".py") \
        else (None, None, None)
    if path is not None:
        mod = load_file(path)
        if not attr:
            return None, mod
        return _getattr_path(mod, attr, ref), mod
    if ":" in ref:
        modname, attr = ref.split(":", 1)
        mod = importlib.import_module(modname)
        return _getattr_path(mod, attr, ref), mod
    parts = ref.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        return _getattr_path(mod, ".".join(parts[i:]), ref), mod
    raise SystemExit(f"hk.check: cannot resolve {ref!r}: not FILE.py:name, "
                     f"module:name or package.module.name")


def _getattr_path(mod, attr: str, what: str):
    obj = mod
    for a in attr.split("."):
        if not hasattr(obj, a):
            raise SystemExit(f"hk.check: {what}: {getattr(obj, '__name__', obj)!r} "
                             f"has no {a!r}")
        obj = getattr(obj, a)
    return obj


def _looks_like_ref(s: str) -> bool:
    return bool(re.match(r"^[\w./\\-]+\.py(:[\w.]+)?$", s)) or \
        bool(re.match(r"^[A-Za-z_][\w.]*:[\w.]+$", s))


def param_names(fn, outputs: List[int]) -> Optional[List[str]]:
    """The reference's argument names: the target's positional parameters,
    a raw kernel's minus the tensors it writes."""
    from hk.dsl.kernel import Kernel  # noqa: PLC0415

    if isinstance(fn, Kernel):
        names = [n for n, _ in fn._tensors]
        try:
            written = set(fn.trace().written_tensors())
        except Exception:  # noqa: BLE001
            written = set()
        return [n for i, n in enumerate(names) if n not in written and i not in outputs]
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    names = [p.name for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
             and p.default is p.empty]
    return [n for i, n in enumerate(names) if i not in outputs]


def namespace() -> Dict[str, Any]:
    ns: Dict[str, Any] = {}
    import math  # noqa: PLC0415

    ns["math"] = math
    try:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415, N812

        ns.update(torch=torch, F=F)
    except ImportError:
        pass
    import hk  # noqa: PLC0415

    ns["hk"] = hk
    return ns


def make_ref(text: str, fn, outputs: List[int]):
    """--ref as a callable."""
    if _looks_like_ref(text):
        obj, _ = resolve(text)
        return obj
    body = text.strip()
    if body.startswith("lambda"):
        return eval(body, namespace())  # noqa: S307 -- the user's own command line
    names = param_names(fn, outputs)
    if not names:
        raise SystemExit("hk.check: --ref is an expression but the target's argument "
                         "names are unknown; write it as a lambda")
    return eval(f"lambda {', '.join(names)}: ({body})", namespace())  # noqa: S307


# --------------------------------------------------------------- input specs


DTYPES = {
    "bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "f16": "float16",
    "half": "float16", "float16": "float16", "fp32": "float32", "f32": "float32",
    "float": "float32", "float32": "float32", "fp64": "float64", "f64": "float64",
    "float64": "float64", "i8": "int8", "int8": "int8", "i16": "int16",
    "int16": "int16", "i32": "int32", "int32": "int32", "i64": "int64",
    "int64": "int64", "u8": "uint8", "uint8": "uint8", "bool": "bool",
    "fp8e4m3": "float8_e4m3fn", "fp8e5m2": "float8_e5m2",
}

_ITEM = re.compile(
    r"^(?P<dt>[A-Za-z0-9_]+)\[(?P<shape>[\d,\s]*)\]"
    r"(?:@(?P<fill>[a-z]+)(?:\((?P<args>[^)]*)\))?)?"
    r"(?:\*(?P<scale>[-+\d.eE]+))?$")


def split_top(s: str) -> List[str]:
    """Split on the commas outside brackets and parentheses."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_spec(spec: str) -> List[Dict[str, Any]]:
    """"bf16[16,64]@randn*0.1, 1e-6" -> item dicts; raises ValueError with
    what was expected."""
    items = []
    for raw in split_top(spec):
        m = _ITEM.match(raw.replace(" ", ""))
        if m is None:
            try:
                items.append({"scalar": ast.literal_eval(raw)})
                continue
            except (ValueError, SyntaxError):
                raise ValueError(
                    f"input {raw!r}: expected dtype[d0,d1,...] with an optional "
                    f"@fill and *scale (e.g. bf16[4096,14336]@randn*0.5), or a number"
                ) from None
        dt = DTYPES.get(m["dt"].lower())
        if dt is None:
            raise ValueError(f"input {raw!r}: unknown dtype {m['dt']!r}; one of "
                             f"{sorted(set(DTYPES))}")
        shape = [int(x) for x in m["shape"].replace(" ", "").split(",") if x]
        args = [ast.literal_eval(a) for a in (m["args"] or "").split(",") if a.strip()]
        items.append({"dtype": dt, "shape": shape, "fill": m["fill"], "args": args,
                      "scale": float(m["scale"]) if m["scale"] else None})
    return items


def materialize(items: List[Dict[str, Any]], device: str):
    import torch  # noqa: PLC0415

    out = []
    for it in items:
        if "scalar" in it:
            out.append(it["scalar"])
            continue
        dt = getattr(torch, it["dtype"])
        shape, fill, args = it["shape"], it["fill"], it["args"]
        floating = dt.is_floating_point
        if fill is None:
            fill = "randn" if floating else ("randint" if dt != torch.bool else "rand")
        lowp = dt in (torch.bfloat16, torch.float16) or "float8" in it["dtype"]
        work = torch.float32 if lowp else dt
        if fill == "randn":
            t = torch.randn(shape, device=device, dtype=work)
        elif fill == "rand":
            t = torch.rand(shape, device=device, dtype=work if floating else torch.float32)
            if dt == torch.bool:
                t = t > 0.5
        elif fill == "zeros":
            t = torch.zeros(shape, device=device, dtype=work)
        elif fill == "ones":
            t = torch.ones(shape, device=device, dtype=work)
        elif fill == "full":
            t = torch.full(shape, float(args[0]) if args else 0.0, device=device, dtype=work)
        elif fill == "arange":
            n = 1
            for d in shape:
                n *= d
            t = torch.arange(n, device=device, dtype=torch.float32 if floating
                             else torch.int64).reshape(shape)
        elif fill == "randint":
            lo, hi = {0: (0, 100), 1: (0, args[0] if args else 100)}.get(
                len(args), tuple(args[:2]))
            t = torch.randint(int(lo), int(hi), shape, device=device, dtype=torch.int64)
        elif fill == "empty":
            t = torch.empty(shape, device=device, dtype=work)
        else:
            raise ValueError(f"unknown fill @{fill}; one of randn rand zeros ones "
                             f"full(v) arange randint(lo,hi) empty")
        if it["scale"] is not None:
            t = t * it["scale"]
        out.append(t.to(dt).contiguous())
    return tuple(out)


def make_inputs(text: str):
    """--inputs as a generator."""
    if _looks_like_ref(text):
        obj, _ = resolve(text)
        return obj
    items = parse_spec(text)

    def gen(device):
        return materialize(items, device)

    gen.spec = text
    return gen


# --------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hk.check",
        description="Check an hk kernel: resources (no GPU), correctness against a "
                    "torch reference and timing against torch eager / torch.compile "
                    "(on a GPU). See docs/check.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("target", help="FILE.py[:name], module:name or package.module.name")
    p.add_argument("--ref", help="the torch reference (FILE.py:name, module:name, a "
                                 "lambda or an expression over the argument names)")
    p.add_argument("--inputs", action="append", default=[],
                   help="a case's inputs (a spec, or FILE.py:name); repeat for more cases")
    p.add_argument("--const", action="append", default=[], metavar="K=V",
                   help="a constexpr for a raw kernel (repeatable)")
    p.add_argument("--sweep", action="append", default=[], metavar="K=V1,V2,...",
                   help="time a raw kernel at each value of a constexpr (repeatable: "
                        "every combination), each checked first")
    p.add_argument("--outputs", help="comma list of output arguments (indices or a raw "
                                     "kernel's parameter names)")
    p.add_argument("--inplace", help="comma list of arguments updated in place (read, "
                                     "then written): given to the reference, compared after")
    p.add_argument("--baseline", action="append", default=[], metavar="NAME=REF",
                   help="another contender, e.g. ops=hk.ops:silu_mul (repeatable)")
    p.add_argument("--no-eager", action="store_true", help="do not time torch eager")
    p.add_argument("--no-compile", action="store_true", help="do not time torch.compile")
    p.add_argument("--compile-mode", help="torch.compile mode, e.g. max-autotune-no-cudagraphs")
    p.add_argument("--rtol", type=float, help="relative tolerance (default: the "
                                              "output dtype's, hk.check.TOLERANCES)")
    p.add_argument("--atol", type=float, help="absolute atol (default: relative to max|ref|)")
    p.add_argument("--exact", action="store_true", help="require bit equality")
    p.add_argument("--ref-precision", choices=("fp32", "native"), default="fp32",
                   help="run the reference on fp32 copies of 16-bit inputs, or as given")
    p.add_argument("--no-time", action="store_true", help="correctness only")
    p.add_argument("--rounds", type=int, default=15, help="timing rounds, contenders rotated")
    p.add_argument("--warmup", type=int, default=3, help="untimed calls of each first")
    p.add_argument("--batch-ms", type=float, default=1.0,
                   help="a round's burst of back-to-back calls fills this long")
    p.add_argument("--burn", type=float, default=0.5, help="seconds of clock warm-up")
    p.add_argument("--bytes", type=float, help="bytes a call moves (for GB/s)")
    p.add_argument("--flops", type=float, help="FLOPs a call does (for TFLOP/s)")
    p.add_argument("--build-only", action="store_true", help="no GPU work even if one is there")
    p.add_argument("--trace-only", action="store_true", help="no hipcc either")
    p.add_argument("--arch", help="build for this arch (comma list: one report each, "
                                  "built only where the GPU is another arch)")
    p.add_argument("--dump", choices=("ir", "cpp"), help="print a raw kernel's IR or C++ and exit")
    p.add_argument("--json", metavar="PATH", help="write the report as JSON ('-' for stdout)")
    p.add_argument("--seed", type=int, default=0, help="torch.manual_seed for the inputs")
    p.add_argument("-q", "--quiet", action="store_true", help="no summary")
    return p


def _consts(pairs: List[str]) -> Dict[str, Any]:
    out = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            raise SystemExit(f"hk.check: --const {kv!r}: the value must be a number") from None
    return out


def _multi_arch(args, argv: List[str]) -> int:
    """One child per arch, HK_ARCH set; their reports printed and joined."""
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]
    rest, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--arch":
            skip = True
            continue
        if a.startswith("--arch="):
            continue
        if a == "--json":
            skip = True
            continue
        if a.startswith("--json="):
            continue
        rest.append(a)
    reports, code = [], 0
    for arch in archs:
        env = dict(os.environ, HK_ARCH=arch)
        proc = subprocess.run([sys.executable, "-m", "hk.check", *rest, "--json", "-",
                               "--quiet"], env=env, capture_output=True, text=True)
        try:
            rep = json.loads(proc.stdout)
        except json.JSONDecodeError:
            rep = {"schema": 1, "arch": arch, "status": "error",
                   "errors": [{"stage": "cli", "type": "ChildFailed",
                               "message": (proc.stderr or proc.stdout)[-4000:]}]}
        reports.append(rep)
        code = max(code, {"pass": 0, "built": 0, "fail": 1}.get(rep.get("status"), 2))
        if not args.quiet:
            from hk.check import Report, summary  # noqa: PLC0415

            r = Report(name=rep.get("name", "?"), arch=rep.get("arch", arch),
                       mode=rep.get("mode", "?"), device=rep.get("device"),
                       versions=rep.get("versions", {}), builds=rep.get("builds", []),
                       cases=rep.get("cases", []), errors=rep.get("errors", []),
                       notes=rep.get("notes", []))
            print(summary(r), file=sys.stderr if args.json == "-" else sys.stdout)
    out = reports[0] if len(reports) == 1 else {"schema": 1, "reports": reports}
    if args.json:
        text = json.dumps(out, indent=1)
        if args.json == "-":
            print(text)
        else:
            Path(args.json).write_text(text)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    import hk  # noqa: PLC0415
    from hk import check as hkc  # noqa: PLC0415
    from hk.target import current_arch, get_target  # noqa: PLC0415

    if args.arch:
        # `import hk` has already settled the arch (hk.ops builds its tables
        # for it), so another arch is another process, HK_ARCH set; the GPU
        # check in hk.check makes it build-only where the GPU differs.
        archs = [get_target(a.strip()).arch for a in args.arch.split(",") if a.strip()]
        if archs != [current_arch()]:
            args.arch = ",".join(archs)
            return _multi_arch(args, argv)

    fn, mod = resolve(args.target)
    cfg: Dict[str, Any] = {}
    if fn is None:                       # a bare FILE.py
        cfg = dict(getattr(mod, "CHECK", {}) or {})
        fn = cfg.pop("fn", None) or getattr(mod, "kernel", None) or getattr(mod, "fn", None)
        if fn is None:
            raise SystemExit(f"hk.check: {args.target} has no CHECK dict, `kernel` or `fn`; "
                             f"name one: {args.target}:NAME")
    consts = {**cfg.pop("consts", {}), **_consts(args.const)}
    if args.dump:
        if not isinstance(fn, hk.Kernel):
            raise SystemExit("hk.check: --dump needs an @hk.kernel target")
        print(fn.trace(**consts).dump() if args.dump == "ir" else fn.source(**consts))
        return 0

    outputs = cfg.pop("outputs", None)
    if args.outputs:
        outputs = [int(o) if o.strip().isdigit() else o.strip()
                   for o in args.outputs.split(",")]
    out_idx = [o for o in (outputs or []) if isinstance(o, int)]
    inplace = cfg.pop("inplace", None)
    if args.inplace:
        inplace = [int(o) if o.strip().isdigit() else o.strip()
                   for o in args.inplace.split(",")]

    ref = cfg.pop("reference", None)
    if args.ref:
        ref = make_ref(args.ref, fn, out_idx)
    elif ref is None and mod is not None:
        ref = getattr(mod, "reference", None) or getattr(mod, "ref", None)

    cases = cfg.pop("cases", None)
    if args.inputs:
        cases = [make_inputs(s) for s in args.inputs]
    elif cases is None and mod is not None:
        gen = cfg.pop("inputs", None) or getattr(mod, "inputs", None)
        shapes = getattr(mod, "SHAPES", None)
        if gen is not None and shapes:
            cases = [(lambda device, s=s, g=gen: hkc._call_maker(g, device, s))
                     for s in shapes]
        elif gen is not None:
            cases = [gen]
    if cases is None:
        cases = [None]

    extra = dict(cfg.pop("baselines", {}) or getattr(mod, "BASELINES", {}) or {})
    for b in args.baseline:
        k, _, v = b.partition("=")
        if not v:
            raise SystemExit(f"hk.check: --baseline {b!r}: NAME=REF")
        extra[k] = resolve(v)[0]

    baselines = [b for b, off in (("eager", args.no_eager), ("compile", args.no_compile))
                 if not off]
    sweep = dict(cfg.pop("sweep", {}) or {})
    for kv in args.sweep:
        k, _, v = kv.partition("=")
        try:
            sweep[k.strip()] = [ast.literal_eval(x.strip()) for x in v.split(",") if x.strip()]
        except (ValueError, SyntaxError):
            raise SystemExit(f"hk.check: --sweep {kv!r}: NAME=v1,v2,... with numbers") from None
    gpu = False if (args.build_only or args.trace_only) else None
    report = hkc.check(
        fn, ref, cases=cases, consts=consts, outputs=outputs, inplace=inplace, sweep=sweep,
        baselines=baselines,
        extra=extra, rtol=args.rtol, atol=args.atol, exact=args.exact,
        ref_precision=args.ref_precision, time=not args.no_time, rounds=args.rounds,
        warmup=args.warmup, batch_ms=args.batch_ms, burn=args.burn,
        compile_mode=args.compile_mode, bytes=args.bytes, flops=args.flops,
        gpu=gpu, build=not args.trace_only, seed=args.seed, **cfg)
    if args.json == "-":
        print(report.dumps())
        if not args.quiet:
            print(report, file=sys.stderr)
    else:
        if args.json:
            Path(args.json).write_text(report.dumps())
        if not args.quiet:
            print(report)
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
