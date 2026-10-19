"""Hash the generated source (every scaffold) of every example and hk.ops kernel
for $HK_ARCH, using the hk checkout at argv[1]. Prints one line per kernel.

The RDNA byte-identity check: run it for gfx1100 and gfx1201 against a
checkout of main and one of the branch, each with its own hk._mlir, and diff.

    HK_ARCH=gfx1100 python3 tools/cdna-probes/dump_sources.py ../main > main.txt
    HK_ARCH=gfx1100 python3 tools/cdna-probes/dump_sources.py . > branch.txt
    diff main.txt branch.txt
"""
import hashlib, importlib, os, sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "python"))
sys.path.insert(0, str(repo / "examples"))
import _catalog  # noqa
from hk.codegen import scaffold  # noqa

jobs = list(_catalog.kernels())
for mod in ("norm", "fused", "quant", "elementwise", "attn", "sdpa", "paged"):
    m = importlib.import_module(f"hk.ops.{mod}")
    for table in ("KERNELS", "STRIP_KERNELS", "ROTARY_KERNELS"):
        jobs += [(f"{mod}.{n}", k) for n, k in sorted(getattr(m, table, {}).items())]
from hk.ops import decode_gemm, gemm, paged  # noqa
jobs.append(("gemm.default", gemm.TUNER._kernel_for(gemm.DEFAULT)))
jobs.append(("gemm.thin", gemm.TUNER._kernel_for(gemm.THIN)))
jobs.append(("paged.decode", paged.paged_decode_kernel("rd_paged_decode")))
jobs.append(("paged.split", paged.paged_decode_split_kernel("rd_paged_split")))
for t, s in sorted(decode_gemm.SCHEDULES.items()):
    jobs.append((f"decode_gemm.t{t}", decode_gemm.kernel_for(tokens=t, **s)))
for name, k in jobs:
    try:
        tr = k.trace()
    except Exception as e:  # noqa: BLE001 -- a refusal is a line to diff too
        print(f"{name} refused {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        continue
    parts = []
    for sc in ("pybind", "bare", "device"):
        parts.append(scaffold.render(tr, sc))
    try:
        parts.append(scaffold.render(tr, "torch"))
    except ValueError:
        pass
    parts.append(tr.dump())
    h = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]
    print(f"{name} {tr.threads} {h}")
