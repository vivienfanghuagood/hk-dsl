"""Which hk kernels build for a CDNA target today, and what stops the rest.

Traces every example kernel (examples/_catalog.py) and, with --ops, every
hk.ops kernel the MLIR round-trip test covers, for $HK_ARCH, builds each as a
device-only code object through hk's own build (the spill gate included), and
prints one line per kernel: its resources, or the first distinct errors.
Compile-only; no GPU.

    HK_ARCH=gfx950 python3 tools/cdna-probes/coverage.py [--ops]
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


def main():
    arch = os.environ.get("HK_ARCH", "gfx950")
    jobs = list(_catalog.kernels())
    if "--ops" in sys.argv:
        jobs += ops_kernels()
    ok = 0
    for name, k in jobs:
        try:
            traced = k.trace()
        except Exception as e:  # noqa: BLE001
            print(f"{name:28s} TRACE   {str(e).splitlines()[0][:160]}")
            continue
        try:
            # bare, not device: device's layout table names gl::row_stride_
            # internal, which only the rdna3 gl has (see docs/design/cdna.md).
            src = scaffold.bare(traced)
        except Exception as e:  # noqa: BLE001
            print(f"{name:28s} EMIT    {str(e).splitlines()[0][:160]}")
            continue
        try:
            b = hc.build(src, arch, name=traced.name, kind="co")
            print(f"{name:28s} OK      " + "; ".join(r.line() for r in b.kernels))
            ok += 1
        except Exception as e:  # noqa: BLE001
            errs = first_errors(str(e)) or [str(e).splitlines()[0]]
            print(f"{name:28s} FAIL    " + " | ".join(x[:110] for x in errs))
    print(f"{arch}: {ok}/{len(jobs)} build")


if __name__ == "__main__":
    main()
