"""python -m hk.profile._child SPEC.json: run a call under rocprofv3.

hk.profile starts this process under the profiler once per pass (the
timing, each counter group, the thread trace). It resolves the target and
its inputs the way `python -m hk.check` does, runs the call `warmup` times
(building or loading its kernels) with hk.check's launch recorder on, then
launches a marker kernel, runs the call `iters` times, launches the marker
again and synchronizes: the dispatches between the two markers are the
profiled ones. It writes what it saw -- the hk kernels the call launched,
with their build (the code object's path, the compiler's resources), the
inputs, the device -- to SPEC's `out`.

SPEC keys: target (FILE.py:name, module:name), consts, gap_ms (host sleep between
the profiled calls, default 2), inputs (a spec string
or FILE.py:name / module:name generator), inputs_pt (a torch.save'd list of
CPU tensors and scalars, from the Python API), shape (for a generator that
takes one), warmup, iters, out, gpu (false: find and build the kernels only).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any, Dict, List

#: The kernel launched before and after the profiled calls; its name in a
#: trace contains MARKER.
MARKER = "spin_kernel"


def _marker() -> None:
    import torch  # noqa: PLC0415

    torch.cuda._sleep(16)


def _resolve(spec: Dict[str, Any]):
    from hk.check.__main__ import resolve  # noqa: PLC0415

    fn, mod = resolve(spec["target"])
    if fn is None:
        cfg = dict(getattr(mod, "CHECK", {}) or {})
        fn = cfg.get("fn") or getattr(mod, "kernel", None) or getattr(mod, "fn", None)
        if fn is None:
            raise SystemExit(f"hk.profile: {spec['target']} has no CHECK dict, `kernel` or "
                             f"`fn`; name one: {spec['target']}:NAME")
    return fn, mod


def _inputs(spec: Dict[str, Any], mod, device: str):
    from hk import check as hkc  # noqa: PLC0415
    from hk.check.__main__ import make_inputs  # noqa: PLC0415

    if spec.get("inputs_pt"):
        import torch  # noqa: PLC0415

        vals = torch.load(spec["inputs_pt"], weights_only=False)
        return tuple(v.to(device) if hkc._is_tensor(v) else v for v in vals)
    gen = None
    if spec.get("inputs"):
        gen = make_inputs(spec["inputs"])
    elif mod is not None:
        cfg = dict(getattr(mod, "CHECK", {}) or {})
        gen = cfg.get("inputs") or getattr(mod, "inputs", None)
        if gen is None and cfg.get("cases"):
            gen = cfg["cases"][0]
    if gen is None:
        raise SystemExit("hk.profile: no inputs: pass --inputs (a spec like "
                         "\"bf16[4096,14336], bf16[4096,14336]\") or give the file an "
                         "`inputs(device)` generator")
    shape = spec.get("shape")
    if shape is None and mod is not None and spec.get("inputs") is None:
        shapes = getattr(mod, "SHAPES", None)
        if shapes:
            shape = shapes[-1]
    return hkc._as_tuple(hkc._call_maker(gen, device, tuple(shape) if shape else None))


def _build_rows(calls) -> List[Dict[str, Any]]:
    from hk.target import get_target  # noqa: PLC0415

    rows = []
    for k, ov in calls:
        row: Dict[str, Any] = {"kernel": k.name, "consts": {a: (v if isinstance(v, (int, float, str, bool))
                                                             else repr(v)) for a, v in ov.items()}}
        try:
            tr = k.trace(**ov)
            row.update(arch=tr.arch, warps=tr.warps, threads=tr.threads, lds_bytes=tr.lds_bytes)
            b = k.build(**ov)
            row["binary"] = str(b.so_path)
            row["source"] = str(b.source_path)
            t = get_target(tr.arch)
            insts = []
            for r in b.kernels:
                charged = t.charged_vgprs(r.vgpr, r.agpr)
                occ_reg = r.occ_by_reg if r.occ_by_reg is not None else r.occupancy
                occ_lds = t.occupancy_by_lds(r.lds or tr.lds_bytes, tr.warps)
                insts.append({"name": r.name, "vgpr": r.vgpr, "agpr": r.agpr, "sgpr": r.sgpr,
                              "vgpr_charged": t.vgpr_rounded(charged) if charged else 0,
                              "lds": r.lds, "scratch": r.scratch,
                              "spill": r.vgpr_spill + r.sgpr_spill,
                              "occupancy_by_regs": occ_reg, "occupancy_by_lds": occ_lds,
                              "occupancy": min(x for x in (occ_reg, occ_lds) if x is not None)
                              if occ_reg is not None else occ_lds,
                              "max_waves_per_simd": t.max_waves_per_simd})
            row["instantiations"] = insts
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        rows.append(row)
    return rows


def main(path: str) -> int:
    spec = json.loads(open(path).read())
    out: Dict[str, Any] = {"errors": [], "notes": []}
    try:
        import hk  # noqa: F401, PLC0415
        from hk import check as hkc  # noqa: PLC0415

        fn, mod = _resolve(spec)
        consts = dict(spec.get("consts") or {})
        if not consts and mod is not None:
            consts = dict((getattr(mod, "CHECK", {}) or {}).get("consts", {}) or {})
        is_kernel = hkc._is_kernel(fn)
        gpu = spec.get("gpu", True)
        if spec.get("mode") == "ab":
            # Two targets on the same inputs, timed as hk.check times:
            # interleaved rounds of bursts in this one process.
            import statistics  # noqa: PLC0415

            import torch  # noqa: PLC0415

            fb, _ = _resolve({"target": spec["vs_target"]})
            torch.manual_seed(spec.get("seed", 0))
            args = list(_inputs(spec, mod, "cuda"))
            with torch.no_grad():
                hkc.burn_in(0.5)
                times, calls = hkc.interleave(
                    {"a": lambda: fn(*args, **consts), "b": lambda: fb(*args, **consts)},
                    rounds=int(spec.get("rounds", 15)), warmup=3, batch_ms=1.0)
            out["ab"] = {"ms": {k: statistics.median(v) for k, v in times.items()},
                         "noise": {k: hkc._spread(v) for k, v in times.items()},
                         "calls_per_round": calls, "rounds": len(times["a"]),
                         "vs": hkc._ratios(times, "a")}
            with open(spec["out"], "w") as f:
                json.dump(out, f, indent=1, default=str)
            return 0
        if not gpu:
            # Find and build only: a raw kernel needs its constexprs; a
            # function runs on CPU tensors with launching off.
            rec = hkc._Recorder(launch=False)
            if is_kernel:
                rec.calls.append((fn, consts))
            else:
                args = _inputs(spec, mod, "cpu")
                with rec.active():
                    fn(*args, **consts)
            out["kernels"] = _build_rows(rec.calls)
            out["inputs"] = None
        else:
            import torch  # noqa: PLC0415

            torch.manual_seed(spec.get("seed", 0))
            args = list(_inputs(spec, mod, "cuda"))
            out["inputs"] = hkc.describe(args)
            out["unique_bytes"] = sum(a.numel() * a.element_size() for a in args
                                      if hkc._is_tensor(a))
            p = torch.cuda.get_device_properties(torch.cuda.current_device())
            out["device"] = {"name": p.name, "cus": p.multi_processor_count,
                             "arch": (getattr(p, "gcnArchName", "") or "").split(":")[0]}
            rec = hkc._Recorder(launch=True)
            with torch.no_grad():
                with rec.active():
                    for _ in range(max(1, int(spec.get("warmup", 2)))):
                        res = fn(*args, **consts)
                torch.cuda.synchronize()
                # Outputs the call returns that are not its arguments are
                # bytes it writes too (a raw kernel's are arguments).
                ptrs = {a.data_ptr() for a in args if hkc._is_tensor(a)}
                out["output_bytes"] = sum(o.numel() * o.element_size()
                                          for o in hkc._outputs_of(res)
                                          if o.data_ptr() not in ptrs)
                _marker()
                torch.cuda.synchronize()
                # Calls apart, by default: back to back, an MI355X's GEMM slowed
                # 40% within 4 ms (power), which would time the limit, not
                # the kernel. gap_ms 0 times them back to back.
                gap = float(spec.get("gap_ms", 2.0)) / 1e3
                for _ in range(int(spec.get("iters", 5))):
                    fn(*args, **consts)
                    if gap > 0:
                        torch.cuda.synchronize()
                        time.sleep(gap)
                torch.cuda.synchronize()
                _marker()
                torch.cuda.synchronize()
                if spec.get("time_call"):
                    # No profiler: the call's time as hk.check takes it.
                    import statistics  # noqa: PLC0415

                    times, _ = hkc.interleave({"call": lambda: fn(*args, **consts)},
                                              rounds=7, warmup=2, batch_ms=1.0)
                    out["call_ms"] = statistics.median(times["call"])
                    out["call_noise"] = hkc._spread(times["call"])
            out["kernels"] = _build_rows(rec.calls)
        out["iters"] = int(spec.get("iters", 5))
        out["target"] = spec.get("target")
    except SystemExit as e:
        out["errors"].append({"stage": "child", "type": "SystemExit", "message": str(e)})
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"stage": "child", "type": type(e).__name__,
                              "message": str(e).strip() or type(e).__name__,
                              "traceback": traceback.format_exc()[-3000:]})
    with open(spec["out"], "w") as f:
        json.dump(out, f, indent=1, default=str)
    return 0 if not out["errors"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
