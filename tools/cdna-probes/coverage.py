"""Which hk kernels build for a CDNA target today, and what stops the rest.

Traces every example kernel (examples/_catalog.py) and, with --ops, every
hk.ops kernel the MLIR round-trip test covers, for $HK_ARCH, builds each as a
device-only code object through hk's own build (the spill gate included), and
prints one line per kernel: its resources, or the first distinct errors.
Compile-only; no GPU.

    HK_ARCH=gfx950 JOBS=16 python3 tools/cdna-probes/coverage.py [--ops]
"""

import importlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples"))

import _catalog  # noqa: E402

from hk.codegen import scaffold  # noqa: E402
from hk.runtime import compile as hc  # noqa: E402


def ops_kernels():
    jobs = []
    for mod in ("norm", "fused", "quant", "elementwise", "attn", "sdpa", "paged"):
        m = importlib.import_module(f"hk.ops.{mod}")
        jobs += [(f"{mod}.{n}", k) for n, k in sorted(getattr(m, "KERNELS", {}).items())]
    from hk.ops import decode_gemm, gemm  # noqa: PLC0415

    jobs.append(("gemm.default", gemm.TUNER._kernel_for(gemm.DEFAULT)))
    for t, s in sorted(decode_gemm.SCHEDULES.items()):
        jobs.append((f"decode_gemm.t{t}", decode_gemm.kernel_for(tokens=t, **s)))
    return jobs


def first_errors(text: str, n: int = 3):
    out = []
    for line in text.splitlines():
        if "error:" not in line:
            continue
        m = re.sub(r"^.*?error: ", "", line).strip()
        if m not in out:
            out.append(m)
    return out[:n]


def build_one(name, src, arch, traced_name):
    try:
        b = hc.build(src, arch, name=traced_name, kind="co")
        return True, f"{name:28s} OK      " + "; ".join(r.line() for r in b.kernels)
    except Exception as e:  # noqa: BLE001
        errs = first_errors(str(e)) or [str(e).splitlines()[0]]
        return False, f"{name:28s} FAIL    " + " | ".join(x[:110] for x in errs)


def main():
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    arch = os.environ.get("HK_ARCH", "gfx950")
    workers = int(os.environ.get("JOBS", "16"))
    jobs = list(_catalog.kernels())
    if "--ops" in sys.argv:
        jobs += ops_kernels()
    if os.environ.get("FILTER"):
        jobs = [(n, k) for n, k in jobs if re.search(os.environ["FILTER"], n)]
    lines, todo = {}, []
    for name, k in jobs:
        try:
            traced = k.trace()
        except Exception as e:  # noqa: BLE001
            lines[name] = (False, f"{name:28s} TRACE   {str(e).splitlines()[0][:160]}")
            continue
        try:
            # The device scaffold: what the code-object launcher builds.
            src = scaffold.device(traced)
        except Exception as e:  # noqa: BLE001
            lines[name] = (False, f"{name:28s} EMIT    {str(e).splitlines()[0][:160]}")
            continue
        todo.append((name, src, traced.name))
    with ThreadPoolExecutor(workers) as ex:
        futs = {name: ex.submit(build_one, name, src, arch, tn) for name, src, tn in todo}
        for name, f in futs.items():
            lines[name] = f.result()
    ok = 0
    for name, _ in jobs:
        good, line = lines[name]
        ok += good
        print(line)
    print(f"{arch}: {ok}/{len(jobs)} build")


if __name__ == "__main__":
    main()
