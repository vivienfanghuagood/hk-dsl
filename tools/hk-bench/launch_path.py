#!/usr/bin/env python3
"""hk's one-row RMSNorm and fused add, the same kernel launched four ways, in
one process, timed per call (small_rows.durations), with Triton beside them:

  pybind    the pybind entry point as generated: hipFuncSetAttribute(max
            dynamic LDS) then <<<>>>, on every launch
  once      the same source with the attribute set on the first launch only
  noattr    the same source without it (the kernels' LDS is under the 64 KB
            a launch may have without it)
  codeobj   the device-only code object through hipModuleLaunchKernel, as
            HK_LAUNCH=codeobj and the engines' torch ops launch it

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/launch_path.py [--json f]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoreboard as sb                            # noqa: E402
import torch                                       # noqa: E402
import triton_ref                                  # noqa: E402
from small_rows import durations                   # noqa: E402

from hk.ops import norm                            # noqa: E402
from hk.runtime import codeobj                     # noqa: E402
from hk.runtime import compile as hk_compile       # noqa: E402
from hk.runtime import module as rtm               # noqa: E402

ATTR = re.compile(r"^(\s*)(hipFuncSetAttribute\([^;]*\);)$", re.M)


def variants(k, consts, tensors):
    """{way: thunk} launching kernel `k` at `consts` on `tensors`."""
    b = k.build(**consts)
    src = Path(b.source_path).read_text()
    assert ATTR.search(src), "the kernel has no dynamic LDS attribute to move"
    sources = {
        "once": ATTR.sub(r"\1static const hipError_t hk_attr_once = \2\n"
                         r"\1(void)hk_attr_once;", src),
        "noattr": ATTR.sub("", src),
    }
    out = {"pybind": lambda: getattr(k.compile(**consts), k.name)(*tensors)}
    arch = k.trace(**consts).arch
    for way, s in sources.items():
        nb = hk_compile.build(s, arch, name=f"{k.name}_{way}")
        fn = getattr(rtm.load(nb.so_path, nb.module_name), k.name)
        out[way] = (lambda fn=fn: fn(*tensors))
    kc = codeobj.load(k, **consts)
    shapes, ptrs, rows = codeobj.tensor_args(kc, tensors, k.name)
    dev = tensors[0].device.index

    def co():
        kc.launch(dev, torch.cuda.current_stream().cuda_stream, shapes, ptrs, rows)

    out["codeobj"] = co
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=21)
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    sb.burn_in(3.0)
    n, bf = 4096, torch.bfloat16
    res = {}
    x = torch.randn(1, n, device="cuda", dtype=bf)
    w = torch.rand(n, device="cuda", dtype=bf) + 0.5
    o = torch.empty_like(x)
    k, chunks, run, exact, reserve = norm._weighted_strip("rmsw", x, 1, n)
    consts = dict(CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve, EPS=1e-6)
    fns = variants(k, consts, (x, w.view(1, n), o))
    t_out = torch.empty_like(x)
    fns["triton"] = lambda: triton_ref.rms_norm(x, w, out=t_out)
    cases = {"rms_norm 1x4096": fns}
    xr, rr = torch.randn(1, n, device="cuda", dtype=bf), torch.randn(1, n, device="cuda", dtype=bf)
    k2, chunks, run, exact, reserve = norm._weighted_strip("addrmsw", xr, 1, n)
    consts2 = dict(CHUNKS=chunks, RUN=run, EXACT=exact, RESERVE=reserve, EPS=1e-6)
    fns2 = variants(k2, consts2, (xr, rr, w.view(1, n), xr, rr))
    xt, rt_ = xr.clone(), rr.clone()
    fns2["triton"] = lambda: triton_ref.fused_add_rms_norm(xt, rt_, w)
    cases["fused_add_rms_norm 1x4096"] = fns2
    for name, f in cases.items():
        for a in f:
            f[a]()
        torch.cuda.synchronize()
        d = durations(f, args.rounds, args.calls)
        res[name] = d
        print(f"\n{name}")
        for a, rounds in d.items():
            allc = [v for r in rounds for v in r]
            med = statistics.median(allc)
            slow = sum(v > med + 5 for v in allc) / len(allc)
            means = [sum(r) / len(r) for r in rounds]
            print(f"  {a:<8} per call: median {med:.2f} us, {slow:.1%} over median + 5 us, "
                  f"mean {statistics.mean(allc):.2f}; round means {min(means):.2f}-{max(means):.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
