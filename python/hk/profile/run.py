"""Running a call under rocprofv3, pass by pass, and reading what it wrote.

`collect(spec, ...)` runs `python -m hk.profile._child` under rocprofv3 once
for the kernel timing (a kernel trace, nothing else attached), once per
counter group (`counters.PASSES`) and once with the thread trace (ATT) on
the slowest kernels, each with a timeout, and returns the raw material: per
kernel (in launch order within a call) its durations, grid, resources,
counters and trace, plus the agent's facts and what the child saw.

A pass that fails is a note, not an error: counters that cannot be
collected (a container without the access, a group the part rejects) leave
the timing and the ISA analysis, and the report says what is missing.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import counters as _counters
from ._child import MARKER

#: Text in rocprofv3's log that means the run is over (it can hang after).
_FATAL = ("exceeds the capabilities", "rocprofv3 caught signal", "Could not construct profile",
          "Permission denied", "Unable to open")


def rocprofv3() -> Optional[str]:
    """The rocprofv3 executable, or None."""
    p = shutil.which("rocprofv3")
    if p:
        return p
    try:
        from ..runtime.compile import rocm_path  # noqa: PLC0415

        c = Path(rocm_path()) / "bin" / "rocprofv3"
        if c.exists():
            return str(c)
    except Exception:  # noqa: BLE001
        pass
    return None


def decoder_dir() -> Optional[str]:
    """The directory holding librocprof-trace-decoder.so (the thread trace's
    decoder, a separate download: docs/profile.md#the-thread-trace), or None.
    Looked for in $HK_ATT_DECODER, ~/.local/lib, then ROCm's lib."""
    cands = []
    if os.environ.get("HK_ATT_DECODER"):
        cands.append(os.environ["HK_ATT_DECODER"])
    cands.append(os.path.expanduser("~/.local/lib"))
    try:
        from ..runtime.compile import rocm_path  # noqa: PLC0415

        cands.append(str(Path(rocm_path()) / "lib"))
    except Exception:  # noqa: BLE001
        pass
    cands.append("/opt/rocm/lib")
    for c in cands:
        p = Path(c)
        if p.is_file() and p.name.startswith("librocprof-trace-decoder"):
            return str(p.parent)
        if (p / "librocprof-trace-decoder.so").exists():
            return str(p)
    return None


def child_env() -> Dict[str, str]:
    """The environment of a profiled child: this process's, with the hk
    package this process imported first on PYTHONPATH (a test or a script
    may have put it on sys.path only) and the working directory after it
    (the child runs in a pass's directory)."""
    import hk  # noqa: PLC0415

    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    here = str(Path(hk.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [here, os.getcwd(), env.get("PYTHONPATH", "")] if p)
    return env


def _run(cmd: List[str], log: Path, timeout: float, env: Dict[str, str]) -> Tuple[bool, str]:
    """Run under a timeout in its own process group; kill the group on a
    fatal message in the log (rocprofv3 can hang after aborting the app).
    The working directory is the log's: rocprofv3 leaves a .rocprofv3/ of
    temporary files where it runs."""
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, cwd=str(log.parent))
    t0 = time.time()
    fatal_at = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if time.time() - t0 > timeout:
            _kill(proc)
            return False, f"timed out after {timeout:.0f} s"
        try:
            text = log.read_text(errors="replace")
        except OSError:
            text = ""
        if fatal_at is None and any(m in text for m in _FATAL):
            fatal_at = time.time()
        if fatal_at is not None and time.time() - fatal_at > 3:
            _kill(proc)
            break
        time.sleep(0.1)
    text = log.read_text(errors="replace")
    for m in _FATAL:
        if m in text:
            line = next((ln for ln in text.splitlines() if m in ln), m)
            idx = text.splitlines().index(line) if line in text.splitlines() else 0
            nxt = text.splitlines()[idx + 1] if idx + 1 < len(text.splitlines()) else ""
            return False, (line.strip() + (" " + nxt.strip() if "Could not" in nxt or
                                             "exceeds" in nxt else ""))[-300:]
    if proc.returncode not in (0, None):
        tail = [ln for ln in text.splitlines() if "simple_timer" not in ln][-6:]
        return False, f"exit {proc.returncode}: " + " | ".join(tail)[-400:]
    return True, ""


def _kill(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _agent(rows: List[Dict[str, str]], agent_id: Optional[str]) -> Dict[str, Any]:
    gpus = [r for r in rows if r.get("Agent_Type", "").upper() != "CPU"
            and "gfx" in r.get("Name", "")]
    pick = None
    if agent_id:
        num = re.sub(r"\D", "", agent_id)
        pick = next((r for r in gpus if r.get("Logical_Node_Id") == num), None)
    pick = pick or (gpus[0] if gpus else None)
    if not pick:
        return {}

    def num(k):
        try:
            return int(float(pick.get(k, "") or 0))
        except ValueError:
            return 0
    se = num("Num_Shader_Banks") or 0
    return {"name": pick.get("Name"), "product": pick.get("Product_Name"),
            "cus": num("Cu_Count"), "simds": num("Simd_Count"), "xcc": num("Num_Xcc") or 1,
            "ses": se or None, "simd_per_cu": num("Simd_Per_Cu"),
            "max_waves_per_simd": num("Max_Waves_Per_Simd"),
            "max_waves_per_cu": num("Max_Waves_Per_Cu"), "lds_kb": num("Lds_Size_In_Kb"),
            "max_clock_mhz": num("Max_Engine_Clk_Fcompute"), "wave": num("Wave_Front_Size")}


def _region(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """The dispatches between the two marker kernels, in dispatch order;
    the marker itself dropped."""
    rows = sorted(rows, key=lambda r: int(r["Dispatch_Id"]))
    marks = [i for i, r in enumerate(rows) if MARKER in r.get("Kernel_Name", "")]
    if len(marks) >= 2:
        return rows[marks[-2] + 1:marks[-1]]
    return []


def _per_call(region: List[Dict[str, str]], iters: int) -> List[Tuple[str, List[Dict[str, str]]]]:
    """[(key, the dispatches of that kernel launch, one per call)] in a
    call's launch order. A call that launches the same sequence each time
    is split by position; otherwise by kernel name."""
    out: Dict[str, List[Dict[str, str]]] = {}
    order: List[str] = []
    n = len(region)
    if iters > 0 and n and n % iters == 0:
        per = n // iters
        names = [r["Kernel_Name"] for r in region[:per]]
        same = all(region[c * per + i]["Kernel_Name"] == names[i]
                   for c in range(iters) for i in range(per))
        if same:
            for i in range(per):
                key = f"{i}:{names[i]}"
                order.append(key)
                out[key] = [region[c * per + i] for c in range(iters)]
            return [(k, out[k]) for k in order]
    for r in region:
        key = r["Kernel_Name"]
        if key not in out:
            out[key] = []
            order.append(key)
        out[key].append(r)
    return [(k, out[k]) for k in order]


def _median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _spread(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return statistics.median(abs(x - m) for x in xs) / m if m else 0.0


def collect(spec: Dict[str, Any], *, arch: str, workdir: Optional[str] = None,
            counters: bool = True, att: bool = True, att_kernels: int = 1,
            iters: int = 20, counter_iters: int = 3, timeout: float = 180.0,
            passes: Optional[List[Tuple[str, List[str]]]] = None,
            keep: bool = False, log=None) -> Dict[str, Any]:
    """Run the passes; return {"child", "agent", "kernels", "passes",
    "notes", "errors"}. `kernels` is [{key, name, durations_ns, grid,
    counters, att_dir}] in a call's launch order."""
    tool = rocprofv3()
    res: Dict[str, Any] = {"passes": {}, "notes": [], "errors": [], "kernels": []}
    if tool is None:
        res["errors"].append({"stage": "profile", "type": "NoProfiler",
                              "message": "rocprofv3 not found (ROCm's bin): nothing can be "
                                         "timed or counted; the ISA analysis is all there is"})
        return res
    own = workdir is None
    work = Path(workdir or tempfile.mkdtemp(prefix="hk-profile-"))
    work.mkdir(parents=True, exist_ok=True)
    env = child_env()
    py = sys.executable

    def say(msg):
        if log:
            log(msg)

    def child_run(name: str, extra: List[str], n_iters: int, warmup: int = 2):
        d = work / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        s = dict(spec, iters=n_iters, warmup=warmup, out=str(d / "child.json"))
        (d / "spec.json").write_text(json.dumps(s))
        cmd = [tool, *extra, "--output-format", "csv", "-d", str(d), "-o", "run", "--",
               py, "-m", "hk.profile._child", str(d / "spec.json")]
        t0 = time.time()
        ok, why = _run(cmd, d / "log.txt", timeout, env)
        child = {}
        if (d / "child.json").exists():
            try:
                child = json.loads((d / "child.json").read_text())
            except json.JSONDecodeError:
                child = {}
        res["passes"][name] = {"ok": ok and not child.get("errors"), "seconds":
                               round(time.time() - t0, 1), **({"why": why} if why else {})}
        say(f"  pass {name}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.1f} s)"
            + (f" {why}" if why else ""))
        return d, ok, child

    # 1. timing: the kernel trace alone.
    d, ok, child = child_run("timing", ["--kernel-trace"], iters)
    res["child"] = child
    if child.get("errors"):
        res["errors"].extend(child["errors"])
        return _finish(res, work, own, keep)
    if not ok:
        res["errors"].append({"stage": "profile", "type": "ProfilerFailed",
                              "message": f"rocprofv3 --kernel-trace failed: "
                                         f"{res['passes']['timing'].get('why', '')}; log "
                                         f"{d / 'log.txt'}"})
        return _finish(res, work, own, keep)
    traces = glob.glob(str(d / "**" / "*kernel_trace.csv"), recursive=True)
    agents = glob.glob(str(d / "**" / "*agent_info.csv"), recursive=True)
    if not traces:
        res["errors"].append({"stage": "profile", "type": "NoTrace",
                              "message": f"rocprofv3 wrote no kernel trace in {d}"})
        return _finish(res, work, own, keep)
    rows = _read_csv(traces[0])
    region = _region(rows)
    if not region:
        res["errors"].append({"stage": "profile", "type": "NoRegion",
                              "message": "the marker kernels were not found in the trace "
                                         f"(looked for {MARKER!r}); is torch.cuda._sleep "
                                         "available?"})
        return _finish(res, work, own, keep)
    res["agent"] = _agent(_read_csv(agents[0]) if agents else [], region[0].get("Agent_Id"))
    groups = _per_call(region, iters)
    for key, ds in groups:
        r0 = ds[0]
        durs = [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in ds]

        def num(k, r=r0):
            try:
                return int(float(r.get(k) or 0))
            except ValueError:
                return 0
        wg = num("Workgroup_Size_X") * max(1, num("Workgroup_Size_Y")) * \
            max(1, num("Workgroup_Size_Z"))
        grid = num("Grid_Size_X") * max(1, num("Grid_Size_Y")) * max(1, num("Grid_Size_Z"))
        res["kernels"].append({
            "key": key, "name": r0["Kernel_Name"], "launches": len(ds),
            "durations_ns": durs, "median_ns": _median(durs), "noise": round(_spread(durs), 4),
            "grid": {"threads": grid, "workgroup": wg,
                     "workgroups": -(-grid // wg) if wg else None,
                     "lds_bytes": num("LDS_Block_Size"), "scratch": num("Scratch_Size"),
                     "vgpr": num("VGPR_Count"), "agpr": num("Accum_VGPR_Count"),
                     "sgpr": num("SGPR_Count")},
            "counters": {}, "counter_spread": {}})
    by_key = {k["key"]: k for k in res["kernels"]}

    # 2. counters, one pass per group.
    if counters:
        groups_ = passes if passes is not None else _counters.passes(arch)
        if not groups_:
            res["notes"].append(f"no counter groups are known for {arch}: timing and the "
                                f"ISA only")
        for name, cs in groups_:
            d, ok, child = child_run(f"pmc-{name}", ["--pmc", *cs], counter_iters, warmup=1)
            files = glob.glob(str(d / "**" / "*counter_collection.csv"), recursive=True)
            if not ok or not files:
                why = res["passes"][f"pmc-{name}"].get("why", "no counter file")
                res["notes"].append(f"counter pass {name} ({' '.join(cs)}) failed: {why}")
                continue
            crow = _read_csv(files[0])
            # One row per (dispatch, counter): regroup by dispatch.
            disp: Dict[str, Dict[str, Any]] = {}
            for r in crow:
                e = disp.setdefault(r["Dispatch_Id"], {"Dispatch_Id": r["Dispatch_Id"],
                                                       "Kernel_Name": r["Kernel_Name"],
                                                       "Start_Timestamp": r.get("Start_Timestamp"),
                                                       "End_Timestamp": r.get("End_Timestamp"),
                                                       "_c": {}})
                try:
                    e["_c"][r["Counter_Name"]] = float(r["Counter_Value"])
                except ValueError:
                    pass
            creg = _region(list(disp.values()))
            for key, ds in _per_call(creg, counter_iters):
                k = by_key.get(key)
                if k is None:
                    continue
                here = k.setdefault("counters_by_pass", {}).setdefault(name, {})
                for cname in {c for dd in ds for c in dd["_c"]}:
                    vals = [dd["_c"][cname] for dd in ds if cname in dd["_c"]]
                    here[cname] = _median(vals)
                    # The first pass to count a counter (GRBM_GUI_ACTIVE is in
                    # every one) gives the merged value.
                    k["counters"].setdefault(cname, here[cname])
                    if len(vals) > 1:
                        k["counter_spread"].setdefault(cname, round(_spread(vals), 4))
                durs = [int(dd["End_Timestamp"]) - int(dd["Start_Timestamp"]) for dd in ds
                        if dd.get("End_Timestamp") and dd.get("Start_Timestamp")]
                if durs:
                    k.setdefault("counter_pass_ns", {})[name] = _median(durs)
            # Counters that read zero for kernels that ran (a W7900D pod):
            # the other passes would read zero too.
            grbm = [k["counters"].get("GRBM_GUI_ACTIVE") for k in res["kernels"]
                    if "GRBM_GUI_ACTIVE" in k["counters"]]
            if grbm and not any(grbm):
                res["notes"].append("the first counter pass read GRBM_GUI_ACTIVE = 0 for "
                                    "kernels that ran: the remaining counter passes were "
                                    "skipped")
                break

    # 3. the thread trace, on the slowest kernels.
    if att:
        dec = decoder_dir()
        if dec is None:
            res["notes"].append("no thread trace: librocprof-trace-decoder.so not found "
                                "(set HK_ATT_DECODER; docs/profile.md#the-thread-trace); "
                                "stall reasons come from the counters only")
        else:
            top = sorted(res["kernels"], key=lambda k: -(k["median_ns"] or 0))[:att_kernels]
            for k in top:
                base = re.escape(k["name"].split("(")[0].strip())
                d, ok, child = child_run(
                    f"att-{res['kernels'].index(k)}",
                    ["--att", "--att-library-path", dec, "--kernel-include-regex", base,
                     "--kernel-trace"], 1, warmup=1)
                outs = sorted(glob.glob(str(d / "**" / "ui_output_agent_*_dispatch_*"),
                                        recursive=True),
                              key=lambda p: int(re.search(r"dispatch_(\d+)", p).group(1)))
                if not ok or not outs:
                    why = res["passes"][f"att-{res['kernels'].index(k)}"].get(
                        "why", "no trace written")
                    res["notes"].append(f"thread trace of {k['name'][:60]} failed: {why}")
                    continue
                k["att_dir"] = outs[-1]
                if not keep and not own:
                    pass
                k["att"] = _read_att(outs[-1])
    return _finish(res, work, own, keep)


def _read_att(path: str) -> Dict[str, Any]:
    """A decoded thread trace directory -> {code, waves, wave_spans}."""
    p = Path(path)
    out: Dict[str, Any] = {}
    try:
        out["code"] = json.loads((p / "code.json").read_text())["code"]
    except (OSError, KeyError, json.JSONDecodeError) as e:
        out["error"] = f"code.json: {e}"
        return out
    spans = []
    try:
        fn = json.loads((p / "filenames.json").read_text())
        wf = fn.get("wave_filenames", {})

        def walk(x):
            if isinstance(x, list) and len(x) == 3 and isinstance(x[0], str):
                spans.append((int(x[1]), int(x[2])))
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
        walk(wf)
    except (OSError, json.JSONDecodeError):
        pass
    out["waves"] = len(spans) or len(list(p.glob("se*_wv*.json")))
    if spans:
        out["wave_cycles"] = sum(b - a for a, b in spans)
        out["span"] = [min(a for a, _ in spans), max(b for _, b in spans)]
    # Each wave's instruction timeline: [cycle, type, stall, duration,
    # code.json row], for hk.profile.timeline.
    waves = []
    for f in sorted(p.glob("se*_wv*.json")):
        try:
            w = json.loads(f.read_text()).get("wave") or {}
        except (OSError, json.JSONDecodeError):
            continue
        if w.get("instructions"):
            waves.append({"cu": w.get("cu", 0), "simd": w.get("simd", 0),
                          "slot": w.get("slot", 0), "begin": w.get("begin", 0),
                          "end": w.get("end", 0), "instructions": w["instructions"]})
    out["wave_data"] = waves
    return out


def _finish(res, work: Path, own: bool, keep: bool):
    res["workdir"] = str(work)
    if own and not keep:
        shutil.rmtree(work, ignore_errors=True)
        res["workdir"] = None
    return res


__all__ = ["collect", "rocprofv3", "decoder_dir", "child_env"]
