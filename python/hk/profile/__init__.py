"""hk.profile: why a kernel is slow, not just that it is.

    import hk
    p = hk.profile("my_kernel.py:gelu_mul", inputs="bf16[4096,14336], bf16[4096,14336], "
                                                    "bf16[4096,14336]")
    print(p)            # per kernel: time, roofline, matrix core, occupancy, stalls,
                        # LDS, caches, the main loop's instruction mix, and a verdict
    p.json()            # the same for an agent (SCHEMA; docs/profile.md#the-json)

or from a shell (docs/profile.md):

    python -m hk.profile my_kernel.py:gelu_mul --inputs "bf16[4096,14336], ..." --json out.json

It runs the call under rocprofv3 in a child process, once per pass: a kernel
trace (the timing, nothing else attached), one pass per group of hardware
counters, and a thread trace (ATT) of the slowest kernel, which says on
which instruction each cycle of a wave went. The code object of every hk
kernel is disassembled (no GPU needed) and its main loop found and counted.
`hk.profile.analyze` turns the numbers into findings -- memory- or
compute-bound against the per-arch roofline, a grid's tail, an epilogue
nothing overlaps, the stall the main loop waits on -- and a one-line
verdict. Where a pass cannot run (no counter access, no trace decoder) the
report says so and keeps what it has: timing and the ISA at the least.

What is profiled: anything `python -m hk.check` takes -- an @hk.kernel, an
hk.ops op, a template's product, or any function that launches kernels
(torch's included: `torch.nn.functional:linear` profiles hipBLASLt) -- named
as FILE.py:name or module:name, or passed as an object that can be named so
(defined at a module's top level).
"""

from __future__ import annotations

import json as _json
import os
import sys
import tempfile
import time as _time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import analyze as _analyze
from . import counters as _counters
from . import isa as _isa

#: Version of the JSON report's layout. Bumped when a key changes meaning.
SCHEMA = 1


# ------------------------------------------------------------------ the report


@dataclass
class Profile:
    """What `hk.profile` found. `str(p)` is the human summary, `p.json()`
    the dict (SCHEMA), `p.verdict` the call's one-line verdict."""

    name: str
    arch: str
    device: Optional[str] = None
    agent: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    call: Dict[str, Any] = field(default_factory=dict)
    kernels: List[Dict[str, Any]] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)
    passes: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    compare: Optional[Dict[str, Any]] = None
    seconds: Optional[float] = None

    @property
    def ok(self) -> bool:
        """True when it ran: no errors (a pass that could not run is a note)."""
        return not self.errors

    @property
    def verdict(self) -> Dict[str, Any]:
        """The call's verdict: its slowest kernel's line, bound and losses
        (`call_verdict`)."""
        return call_verdict(self.kernels)

    def json(self) -> Dict[str, Any]:
        """The report as a dict of JSON types (layout version `SCHEMA`;
        docs/profile.md#the-json)."""
        out = {"schema": SCHEMA, "tool": "hk.profile", "name": self.name, "arch": self.arch,
               "device": self.device, "agent": self.agent, "versions": self.versions,
               "call": self.call, "verdict": self.verdict, "kernels": self.kernels,
               "sources": self.sources, "cannot": _counters.CANNOT.get(self.arch, []),
               "ceilings": {k: v for k, v in _analyze.ceilings(self.arch).items()},
               "passes": self.passes, "seconds": self.seconds,
               "notes": self.notes, "errors": self.errors}
        if self.compare is not None:
            out["compare"] = self.compare
        return out

    def dumps(self, **kw) -> str:
        """`json()` as text."""
        return _json.dumps(self.json(), indent=kw.pop("indent", 1), default=str, **kw)

    def __str__(self) -> str:
        return summary(self)


def call_verdict(kernels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The call's verdict: its slowest kernel's, with that kernel's share of
    the call's kernel time."""
    if not kernels:
        return {"line": "nothing was measured", "kernel": None}
    tot = sum((k.get("time") or {}).get("us") or 0 for k in kernels) or 1
    top = max(kernels, key=lambda k: (k.get("time") or {}).get("us") or 0)
    share = ((top.get("time") or {}).get("us") or 0) / tot
    v = top.get("verdict") or {}
    head = _short(top.get("name") or "?")
    line = (f"{head} ({share * 100:.0f}% of the call): " if len(kernels) > 1 else "") + \
        v.get("line", "")
    return {"line": line, "kernel": top.get("name"), "share": round(share, 4),
            "bound": v.get("bound"), "losses": v.get("losses", [])}


def _short(name: str, n: int = 60) -> str:
    s = name.split("(")[0]
    return s if len(s) <= n else s[:n - 3] + "..."


# ------------------------------------------------------------------ naming targets


def _target_ref(fn) -> str:
    """A string the child process can resolve: FILE.py:name or module:name."""
    if isinstance(fn, str):
        return fn
    from ..dsl.kernel import Kernel  # noqa: PLC0415

    obj = fn.fn if isinstance(fn, Kernel) else fn
    mod = getattr(obj, "__module__", None)
    qual = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if isinstance(fn, Kernel):
        # The module attribute is the Kernel, under the function's name.
        qual = getattr(obj, "__name__", qual)
    if not mod or not qual or "<" in qual:
        raise TypeError(f"hk.profile: {fn!r} cannot be named for the profiled process "
                        f"(a lambda or a closure); define it at a module's top level or "
                        f"pass 'FILE.py:name'")
    m = sys.modules.get(mod)
    path = getattr(m, "__file__", None)
    if mod == "__main__" or mod.startswith("hk_check_") or mod.startswith("hk_profile_"):
        if not path:
            raise TypeError(f"hk.profile: {qual} is defined in {mod}, which has no file; "
                            f"pass 'FILE.py:name'")
        return f"{path}:{qual}"
    return f"{mod}:{qual}"


def _absolute(ref: str) -> str:
    """FILE.py[:name] with FILE made absolute (the passes run elsewhere)."""
    path, sep, attr = ref.partition(".py")
    if sep and (not attr or attr.startswith(":")) and os.path.exists(path + ".py"):
        return os.path.abspath(path + ".py") + attr
    return ref


def _inputs_ref(inputs, work: Path) -> Dict[str, Any]:
    if inputs is None:
        return {}
    if isinstance(inputs, str):
        return {"inputs": _absolute(inputs)}
    if callable(inputs):
        return {"inputs": _absolute(_target_ref(inputs))}
    import torch  # noqa: PLC0415

    vals = [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in inputs]
    p = work / "inputs.pt"
    torch.save(vals, p)
    return {"inputs_pt": str(p)}


# ------------------------------------------------------------------ profile


def _hk_rows(child: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in child.get("kernels") or [] if not r.get("error")]


def _match_build(name: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    base = name.split("(")[0].strip()
    for r in rows:
        for i in r.get("instantiations") or []:
            if i.get("name", "").split("(")[0].strip() == base:
                return r
    for r in rows:
        if base == f"{r['kernel']}_kernel" or base.startswith(f"{r['kernel']}_kernel"):
            return r
    return None


_DIS_CACHE: Dict[str, List[Any]] = {}


def _functions_of(build: Dict[str, Any], arch: str) -> List[Any]:
    path = build.get("binary")
    if not path:
        return []
    if path not in _DIS_CACHE:
        _DIS_CACHE[path] = _isa.disassemble(_isa.code_object(path, arch), arch)
    return _DIS_CACHE[path]


def _pick_function(funcs: List[Any], name: str):
    if not funcs:
        return None
    base = name.split("(")[0].strip()
    for f in funcs:
        if f.name.split("(")[0].strip() == base or base in f.name:
            return f
    return funcs[0] if len(funcs) == 1 else None


def _matrix_dtype(f) -> Optional[str]:
    names = [i.mnemonic for i in f.insts if i.cls in ("mfma", "wmma")]
    if not names:
        return None
    m = max(set(names), key=names.count)
    if any(s in m for s in ("fp8", "bf8", "f8f6f4", "_f8")):
        return "fp8"
    if "bf16" in m:
        return "bf16"
    if "f16" in m:
        return "fp16"
    return "bf16"


def _isa_for(rec: Dict[str, Any], build: Optional[Dict[str, Any]], arch: str,
             notes: List[str]) -> Optional[Dict[str, Any]]:
    att = rec.get("att") or {}
    stats = None
    att_funcs: List[Any] = []
    if att.get("code"):
        att_funcs, stats = _isa.from_att(att["code"])
    f = None
    if build is not None:
        try:
            f = _pick_function(_functions_of(build, arch), rec["name"])
        except Exception as e:  # noqa: BLE001
            notes.append(f"{_short(rec['name'])}: no disassembly ({type(e).__name__}: {e}); "
                         f"the thread trace's is used where there is one")
    if f is None and att_funcs:
        f = _pick_function(att_funcs, rec["name"])
    if f is None:
        return None
    waves = att.get("waves") or 0
    res = _isa.analyze(f, att=stats, waves_traced=waves or None)
    res["source"] = "code object (llvm-objdump)" if build is not None and f not in att_funcs \
        else "thread trace's disassembly"
    res["matrix_dtype"] = _matrix_dtype(f)
    if stats and waves:
        n, fl = _isa.matrix_ops_per_wave(f, stats, waves)
        res["att_matrix_insts_per_wave"] = round(n, 2)
        res["att_matrix_flops_per_wave"] = fl
    if att.get("wave_data") and res.get("matrix_dtype"):
        try:
            res["matrix_idle"] = _matrix_idle(att, res, arch)
        except Exception as e:  # noqa: BLE001
            notes.append(f"{_short(rec['name'])}: matrix-idle attribution failed: "
                         f"{type(e).__name__}: {e}")
    return res


def _matrix_idle(att: Dict[str, Any], res: Dict[str, Any], arch: str):
    """hk.profile.timeline.matrix_idle over a thread trace, labels from the
    ISA analysis."""
    import bisect  # noqa: PLC0415

    from .timeline import SIMD_FLOPS_PER_CYCLE, matrix_idle  # noqa: PLC0415

    code = att["code"]
    regs = _isa.regions(res)
    starts = [a for a, _ in regs]
    rate = SIMD_FLOPS_PER_CYCLE.get(arch, {})
    lab: Dict[int, Any] = {}
    pipe: Dict[int, int] = {}
    mat: Dict[int, bool] = {}
    for i, row in enumerate(code):
        if not row or str(row[0]).startswith(";") or len(row) < 6:
            continue
        text = str(row[0])
        parts = text.split(None, 1)
        mn, ops = parts[0], (parts[1] if len(parts) > 1 else "")
        cls = _isa.classify(mn, ops)
        key = "waitcnt:" + _isa.wait_kind(mn, ops) if cls == "waitcnt" else cls
        addr = int(row[5])
        region = regs[max(0, bisect.bisect_right(starts, addr) - 1)][1]
        lab[i] = (_analyze.REASONS.get(key, "other"), region)
        if cls in ("mfma", "wmma"):
            mat[i] = True
            dt = "fp8" if any(s in mn for s in ("fp8", "bf8", "f8f6f4")) else \
                ("bf16" if "bf16" in mn else "fp16")
            per = rate.get(dt) or rate.get("bf16") or 0
            pipe[i] = int(_isa.mma_flops(mn) / per) if per else 16
    return matrix_idle(att["wave_data"], arch=arch,
                       label=lambda i: lab.get(i, ("other", "kernel")),
                       pipe_cycles=lambda i: pipe.get(i, 16),
                       is_matrix=lambda i: mat.get(i, False))


def profile(fn, inputs=None, *, consts: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None, vs: Optional[Union[str, Dict[str, Any]]] = None,
            counters: bool = True, att: bool = True, att_kernels: int = 1,
            iters: int = 20, gap_ms: float = 2.0, timeout: float = 180.0,
            flops: Optional[float] = None,
            bytes: Optional[float] = None, shape: Optional[Sequence[int]] = None,  # noqa: A002
            workdir: Optional[str] = None, keep: bool = False, gpu: Optional[bool] = None,
            record: Optional[str] = None,
            log: Optional[Callable[[str], None]] = None) -> Profile:
    """Profile the kernels `fn(*inputs)` launches; see the module docstring
    and docs/profile.md.

    fn        what to run: 'FILE.py:name', 'module:name', or an object that
              can be named so (an @hk.kernel or function at a module's top
              level)
    inputs    a spec string ("bf16[4096,4096], bf16[4096,4096]"), a list of
              tensors (saved and loaded in the profiled process), a
              generator named as fn is, or None for the file's own `inputs`
    consts    constexprs for a raw kernel
    vs        another target on the same inputs, profiled the same way and
              compared kernel by kernel: 'module:name' or {label: target}
    counters, att   the counter passes and the thread trace (each skipped
              with a note where it cannot run)
    iters     calls timed in the kernel trace (the median is the time)
    gap_ms    host sleep between those calls (0: back to back, where a
              power limit can slow a long run of GEMMs by tens of percent)
    flops, bytes    the work of the call's main kernel, where the caller
              knows it better than the counters
    gpu       False: find, build and disassemble only (no profiler)
    record    write the raw material (per kernel: the profiler's record
              without the trace, the build row, the ISA analysis) to this
              JSON path: what tests/hk/ir/test_profile.py replays the
              verdict logic on
    """
    from .. import target as _target  # noqa: PLC0415
    from ..check import versions as _versions  # noqa: PLC0415
    from . import run as _run  # noqa: PLC0415

    t0 = _time.time()
    arch = _target.current_arch()
    work = Path(workdir or tempfile.mkdtemp(prefix="hk-profile-"))
    work.mkdir(parents=True, exist_ok=True)
    ref = _absolute(_target_ref(fn))
    spec: Dict[str, Any] = {"target": ref, "consts": dict(consts or {}), "gap_ms": gap_ms,
                            **_inputs_ref(inputs, work)}
    if shape is not None:
        spec["shape"] = list(shape)
    prof = Profile(name=name or (ref.split(":")[-1] if ":" in ref else Path(ref).stem),
                   arch=arch, versions=_versions())
    prof.sources = {"timing": "rocprofv3 --kernel-trace", "counters": "off",
                    "att": "off", "isa": "llvm-objdump of the code object"}
    if gpu is None:
        gpu = _gpu_present()
    if not gpu:
        _isa_only(prof, spec, work)
        prof.seconds = round(_time.time() - t0, 1)
        return prof
    raw = _run.collect(spec, arch=arch, workdir=str(work / "passes"), counters=counters,
                       att=att, att_kernels=att_kernels, iters=iters, timeout=timeout,
                       keep=keep or bool(workdir), log=log)
    prof.passes = raw.get("passes", {})
    prof.notes.extend(raw.get("notes", []))
    prof.errors.extend(raw.get("errors", []))
    child = raw.get("child") or {}
    prof.agent = raw.get("agent") or {}
    prof.device = (child.get("device") or {}).get("name") or prof.agent.get("product")
    rv = _rocprof_version()
    if rv:
        prof.versions["rocprofv3"] = rv
    counter_ok = any(k.startswith("pmc-") and v.get("ok") for k, v in prof.passes.items())
    if counter_ok and _counters_read_zero(raw.get("kernels") or []):
        # Seen on a W7900D pod (ROCm 7.2.4): every counter but SQ_WAVES reads
        # 0. A zero cycle count for a kernel that ran is not a measurement.
        for r in raw.get("kernels") or []:
            r["counters"] = {}
        counter_ok = False
        prof.notes.append("the hardware counters read zero here (GRBM_GUI_ACTIVE = 0 for "
                          "kernels that ran): counter collection does not work on this "
                          "machine, so they are ignored; timing, the thread trace and the "
                          "ISA remain")
    for r in raw.get("kernels") or []:
        d = r.get("durations_ns") or []
        if len(d) >= 8:
            h = len(d) // 2
            a, b = sorted(d[:h])[h // 2], sorted(d[h:])[(len(d) - h) // 2]
            if b > a * 1.05:
                prof.notes.append(
                    f"{_short(r['name'])}: the timed calls slowed by {(b / a - 1) * 100:.0f}% "
                    f"from the first half to the second ({a / 1e3:.1f} -> {b / 1e3:.1f} us): "
                    f"a clock dropping under a power or thermal limit, or another job on "
                    f"the GPU; min {min(d) / 1e3:.1f} us")
    att_ok = any(k.startswith("att-") and v.get("ok") for k, v in prof.passes.items())
    prof.sources["counters"] = ("rocprofv3 --pmc, " + str(sum(
        1 for k in prof.passes if k.startswith("pmc-"))) + " passes") if counter_ok else \
        ("off" if not counters else "unavailable (see notes)")
    prof.sources["att"] = "rocprofv3 --att (thread trace, one CU)" if att_ok else \
        ("off" if not att else "unavailable (see notes)")
    unique = (child.get("unique_bytes") or 0) + (child.get("output_bytes") or 0)
    recs = raw.get("kernels") or []
    total_ns = sum(r.get("median_ns") or 0 for r in recs)
    prof.call = {"inputs": child.get("inputs"), "footprint_bytes": unique or None,
                 "kernels": len(recs), "us": round(total_ns / 1e3, 2),
                 "iters": child.get("iters")}
    rows = _hk_rows(child)
    for r in child.get("kernels") or []:
        if r.get("error"):
            prof.notes.append(f"hk kernel {r['kernel']}: {r['error']}")
    biggest = max(recs, key=lambda r: r.get("median_ns") or 0) if recs else None
    recorded: List[Dict[str, Any]] = []
    for rec in recs:
        rec["iters"] = child.get("iters")
        b = _match_build(rec["name"], rows)
        isa = None
        try:
            isa = _isa_for(rec, b, arch, prof.notes)
        except Exception as e:  # noqa: BLE001
            prof.notes.append(f"{_short(rec['name'])}: ISA analysis failed: "
                              f"{type(e).__name__}: {e}")
        main = rec is biggest
        k = _analyze.kernel_report(arch, prof.agent, rec, build=b, isa=isa,
                                   call_bytes=unique or None,
                                   flops=flops if main else None,
                                   bytes_=bytes if main else None, single=len(recs) == 1)
        k["share"] = round((rec.get("median_ns") or 0) / total_ns, 4) if total_ns else None
        if rec.get("att_dir"):
            k["att_dir"] = rec["att_dir"]
        prof.kernels.append(k)
        if record:
            recorded.append({"rec": {kk: v for kk, v in rec.items() if kk != "att"},
                             "build": b, "isa": isa, "main": main,
                             "flops": flops if main else None,
                             "bytes": bytes if main else None})
    if record:
        Path(record).write_text(_json.dumps(
            {"schema": SCHEMA, "arch": arch, "agent": prof.agent,
             "call_bytes": unique or None, "single": len(recs) == 1,
             "versions": prof.versions, "device": prof.device, "kernels": recorded},
            indent=1, default=str))
    if vs is not None:
        label, other = (next(iter(vs.items())) if isinstance(vs, dict)
                        else (str(vs).split(":")[-1], vs))
        sub = profile(other, inputs, consts=consts, name=label, counters=counters, att=att,
                      record=(record[:-5] + f".vs-{label}.json") if record else None,
                      att_kernels=att_kernels, iters=iters, gap_ms=gap_ms, timeout=timeout,
                      shape=shape,
                      workdir=str(work / "vs"), keep=keep, gpu=gpu, log=log)
        prof.compare = compare(prof, sub)
        ab = _interleaved(spec, _absolute(_target_ref(other)), work, timeout)
        if ab.get("ab"):
            r = ab["ab"]
            v = r["vs"]["b"]
            prof.compare["interleaved"] = {
                "a_us": round(r["ms"]["a"] * 1e3, 2), "b_us": round(r["ms"]["b"] * 1e3, 2),
                "b_over_a_time": v.get("x"),
                "band": v["band"], "verdict_for_a": v["verdict"], "rounds": r["rounds"],
                "how": "hk.check.interleave in one process: rounds of ~1 ms bursts, the "
                       "contenders rotated; medians"}
            line = prof.compare["line"]
            if line.startswith(f"{label} "):
                line = "profiled calls: " + line
            prof.compare["line"] = (
                f"{label} takes {v['x']:.3f}x the time of {prof.name} (interleaved in one "
                f"process: {r['ms']['b'] * 1e3:.1f} vs {r['ms']['a'] * 1e3:.1f} us, band "
                f"{v['band'] * 100:.0f}%); " + line)
        elif ab.get("errors"):
            prof.notes.append(f"interleaved timing of --vs failed: "
                              f"{ab['errors'][0].get('message', '')[:200]}")
    prof.seconds = round(_time.time() - t0, 1)
    if not workdir and not keep:
        import shutil  # noqa: PLC0415

        shutil.rmtree(work, ignore_errors=True)
    return prof


def replay(path) -> List[Dict[str, Any]]:
    """The per-kernel reports of a `record=` file, recomputed from its raw
    measurements -- no GPU, no profiler: how the verdict logic is tested
    and how a change to it is checked against past runs."""
    doc = _json.loads(Path(path).read_text())
    arch, agent = doc["arch"], doc.get("agent") or {}
    out = []
    for k in doc.get("kernels") or []:
        out.append(_analyze.kernel_report(
            arch, agent, k["rec"], build=k.get("build"), isa=k.get("isa"),
            call_bytes=doc.get("call_bytes"), flops=k.get("flops"), bytes_=k.get("bytes"),
            single=doc.get("single", False)))
    return out


def _interleaved(spec: Dict[str, Any], other: str, work: Path, timeout: float) -> Dict[str, Any]:
    """Both targets timed in one process (no profiler), as hk.check times."""
    import subprocess  # noqa: PLC0415

    d = work / "ab"
    d.mkdir(parents=True, exist_ok=True)
    s = dict(spec, mode="ab", vs_target=other, out=str(d / "child.json"))
    (d / "spec.json").write_text(_json.dumps(s))
    try:
        subprocess.run([sys.executable, "-m", "hk.profile._child", str(d / "spec.json")],
                       capture_output=True, text=True, timeout=timeout)
        return _json.loads((d / "child.json").read_text())
    except Exception as e:  # noqa: BLE001
        return {"errors": [{"message": f"{type(e).__name__}: {e}"}]}


def _counters_read_zero(recs: List[Dict[str, Any]]) -> bool:
    """True when the counters were collected but measure nothing: a kernel
    that ran for microseconds with zero GPU-active cycles."""
    seen = False
    for r in recs:
        c = r.get("counters") or {}
        if "GRBM_GUI_ACTIVE" not in c:
            continue
        seen = True
        if c["GRBM_GUI_ACTIVE"] and (r.get("median_ns") or 0) > 0:
            return False
    return seen


def _gpu_present() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _rocprof_version() -> Optional[str]:
    import subprocess  # noqa: PLC0415

    from .run import rocprofv3  # noqa: PLC0415

    tool = rocprofv3()
    if not tool:
        return None
    try:
        out = subprocess.run([tool, "--version"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "version:" in line:
            return line.split("version:", 1)[1].strip()
    return None


def _isa_only(prof: Profile, spec: Dict[str, Any], work: Path) -> None:
    """No GPU: find and build the kernels (as hk.check does), disassemble."""
    import subprocess  # noqa: PLC0415

    prof.sources = {"timing": "off (no GPU)", "counters": "off (no GPU)", "att": "off (no GPU)",
                    "isa": "llvm-objdump of the code object"}
    s = dict(spec, gpu=False, out=str(work / "child.json"))
    (work / "spec.json").write_text(_json.dumps(s))
    proc = subprocess.run([sys.executable, "-m", "hk.profile._child", str(work / "spec.json")],
                          capture_output=True, text=True)
    try:
        child = _json.loads((work / "child.json").read_text())
    except (OSError, ValueError):
        prof.errors.append({"stage": "child", "type": "ChildFailed",
                            "message": (proc.stderr or proc.stdout)[-2000:]})
        return
    prof.errors.extend(child.get("errors", []))
    prof.notes.append("no GPU: the kernels were built and disassembled; timing, counters and "
                      "the thread trace need one")
    for b in _hk_rows(child):
        name = (b.get("instantiations") or [{}])[0].get("name") or f"{b['kernel']}_kernel"
        k: Dict[str, Any] = {"name": name, "hk": True, "build": {
            kk: b.get(kk) for kk in ("kernel", "consts", "warps", "lds_bytes", "source",
                                     "binary")}}
        if b.get("instantiations"):
            k["build"]["resources"] = b["instantiations"][0]
        try:
            f = _pick_function(_functions_of(b, prof.arch), name)
            if f is not None:
                k["isa"] = _isa.analyze(f)
                k["isa"]["matrix_dtype"] = _matrix_dtype(f)
        except Exception as e:  # noqa: BLE001
            prof.notes.append(f"{b['kernel']}: no disassembly: {type(e).__name__}: {e}")
        k["findings"] = []
        k["verdict"] = {"line": "built only (no GPU): see the ISA", "bound": None,
                        "losses": []}
        prof.kernels.append(k)


# ------------------------------------------------------------------ compare


def _get(d: Dict[str, Any], path: str):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


#: The metrics a comparison lines up, as dotted paths into a kernel's JSON.
COMPARED = ("time.us", "roofline.of_compute_peak", "roofline.of_memory_peak",
            "roofline.clock_mhz", "matrix_core.busy", "occupancy.achieved_waves_per_simd",
            "occupancy.theoretical_waves_per_simd", "occupancy.cu_busy", "grid.workgroups",
            "grid.waves_per_workgroup", "grid.workgroups_per_slot", "grid.rounds",
            "grid.lds_bytes", "isa.main_loop.instructions",
            "isa.main_loop.mix.per_matrix_instruction", "isa.main_loop.cycle_share",
            "isa.epilogue.cycle_share", "isa.drain.cycle_share", "isa.prologue.cycle_share",
            "instructions.valu", "instructions.mfma", "instructions.salu", "instructions.lds",
            "instructions.vmem_load", "instructions.vmem_store", "lds.bank_conflict",
            "cache.l2_hit", "cache.fetch_bytes", "latency_cycles.vmem",
            "stalls.counters.dependency_wait", "stalls.counters.issue_wait",
            "stalls.matrix_idle.busy", "stalls.matrix_idle.idle_by_region.main_loop",
            "stalls.matrix_idle.idle_by_region.drain",
            "stalls.matrix_idle.idle_by_region.epilogue",
            "stalls.matrix_idle.idle_by_region.prologue")


def compare(a: Profile, b: Profile) -> Dict[str, Any]:
    """`b` against `a`: the calls' times, and their slowest kernels' metrics
    side by side with a sentence on what differs."""
    ka = max(a.kernels, key=lambda k: (k.get("time") or {}).get("us") or 0) if a.kernels else {}
    kb = max(b.kernels, key=lambda k: (k.get("time") or {}).get("us") or 0) if b.kernels else {}
    ua, ub = a.call.get("us") or 0, b.call.get("us") or 0
    rows = {}
    for p in COMPARED:
        va, vb = _get(ka, p), _get(kb, p)
        if va is None and vb is None:
            continue
        rows[p] = [va, vb]
    for src in ("matrix_idle.idle", "att.by_reason"):
        sa = _get(ka, "stalls." + src) or {}
        sb = _get(kb, "stalls." + src) or {}
        for r in sorted(set(sa) | set(sb)):
            va, vb = sa.get(r), sb.get(r)
            if isinstance(va, dict) or isinstance(vb, dict):
                va, vb = (va or {}).get("share"), (vb or {}).get("share")
            rows[f"stalls.{src}.{r}"] = [va, vb]
    words = []
    if ua and ub:
        words.append(f"{b.name} {ub:.1f} us vs {a.name} {ua:.1f} us ({ub / ua:.3f}x the time)")

    def pair(p):
        return rows.get(p, [None, None])

    da, db = (pair("stalls.matrix_idle.idle_by_region.drain"),
              pair("stalls.matrix_idle.idle_by_region.epilogue"))
    if None not in (da[0], da[1], db[0], db[1]):
        words.append(f"matrix core idle after the main loop (drain + epilogue) "
                     f"{(da[0] + db[0]) * 100:.1f}% -> {(da[1] + db[1]) * 100:.1f}% of the kernel")
    for p, label, pct in (("matrix_core.busy", "matrix core busy", True),
                          ("stalls.matrix_idle.idle.valu", "matrix idle at VALU", True),
                          ("instructions.valu", "VALU instructions", False),
                          ("occupancy.theoretical_waves_per_simd", "waves a SIMD", False),
                          ("grid.workgroups", "workgroups", False),
                          ("isa.main_loop.mix.per_matrix_instruction",
                           "loop instructions per matrix one", False),
                          ("occupancy.cu_busy", "CUs busy", True)):
        va, vb = pair(p)
        if va is None or vb is None or va == vb:
            continue
        if pct:
            if abs(va - vb) >= 0.01:
                words.append(f"{label} {va * 100:.0f}% -> {vb * 100:.0f}%")
        elif isinstance(va, (int, float)) and va and abs(vb / va - 1) >= 0.02:
            words.append(f"{label} {va:,.4g} -> {vb:,.4g}")
    return {"a": a.name, "b": b.name, "us": [ua, ub],
            "ratio": round(ub / ua, 4) if ua else None,
            "kernels": [ka.get("name"), kb.get("name")], "metrics": rows,
            "line": "; ".join(words), "b_report": b.json()}


# ------------------------------------------------------------------ summary


def _fmt_rate(x: Optional[float], unit: str) -> str:
    if x is None:
        return "?"
    for div, p in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M")):
        if x >= div:
            return f"{x / div:.2f} {p}{unit}"
    return f"{x:.0f} {unit}"


def _pct(x) -> str:
    return "?" if x is None else f"{x * 100:.0f}%"


def _mix_text(mx: Dict[str, Any]) -> str:
    cls = mx.get("classes") or {}
    order = [c for c in _isa.CLASSES if cls.get(c)]
    parts = [f"{cls[c]} {c}" for c in order if c != "waitcnt"]
    if cls.get("waitcnt"):
        w = mx.get("waits") or {}
        parts.append(f"{cls['waitcnt']} waitcnt (" + ", ".join(f"{v} {k}" for k, v in
                                                               w.items()) + ")")
    return ", ".join(parts)


def kernel_summary(k: Dict[str, Any]) -> List[str]:
    """The human lines for one kernel's report."""
    L: List[str] = []
    t = k.get("time") or {}
    share = k.get("share")
    L.append(f"  kernel {_short(k.get('name') or '?', 90)}"
             + (f"  {t.get('us')} us" if t.get("us") is not None else "")
             + (f" ±{t['noise'] * 100:.1f}%" if t.get("noise") else "")
             + (f"  ({share * 100:.0f}% of the call)" if share is not None and share < 0.995
                else ""))
    g = k.get("grid") or {}
    if g.get("workgroups"):
        s = (f"    grid      {g['workgroups']} workgroups x {g.get('threads_per_workgroup')} "
             f"threads ({g.get('waves_per_workgroup')} waves)")
        if g.get("lds_bytes") is not None:
            s += f", LDS {g['lds_bytes']} B"
        if g.get("vgpr"):
            s += f", {g['vgpr']} VGPR" + (f" + {g['agpr']} AGPR" if g.get("agpr") else "")
        if g.get("workgroups_per_slot") is not None:
            s += (f" -> {g['workgroups_per_slot']} a {g.get('slot')} (bound by "
                  f"{g.get('bound_by')}); {g.get('rounds')} rounds of {g.get('slots')} "
                  f"{g.get('slot')}s")
        L.append(s)
    r = k.get("roofline") or {}
    if r:
        parts = []
        if r.get("flops"):
            parts.append(f"{r['flops'] / 1e9:.1f} GFLOP -> {_fmt_rate(r.get('flop_s'), 'FLOP/s')}"
                         + (f" = {_pct(r.get('of_compute_peak'))} of the {r.get('matrix_dtype')}"
                            f" matrix peak" if r.get("of_compute_peak") is not None else ""))
        if r.get("bytes"):
            past = "past the L2 " if "past the L2" in (r.get("bytes_source") or "") else ""
            parts.append(f"{r['bytes'] / 1e6:.0f} MB {past}-> {_fmt_rate(r.get('bytes_s'), 'B/s')}"
                         + (f" = {_pct(r.get('of_cache'))} of the cache-resident rate"
                            if r.get("of_cache") is not None else "")
                         + (f" ({r['refetch']:.1f}x the call's bytes)" if r.get("refetch")
                            else ""))
        if r.get("of_dram_min") is not None:
            parts.append(f"DRAM at least {_pct(r['of_dram_min'])} (the call's "
                         f"{r['dram_bytes_min'] / 1e6:.0f} MB once)")
        elif r.get("of_dram") is not None and not past:
            parts.append(f"{_pct(r['of_dram'])} of DRAM")
        if r.get("intensity") is not None:
            parts.append(f"{r['intensity']:.0f} FLOP/B{' past the L2' if past else ''} "
                         f"(DRAM ridge {r.get('ridge')})")
        L.append("    roofline  " + "; ".join(parts))
    mc = k.get("matrix_core") or {}
    occ = k.get("occupancy") or {}
    line = []
    if mc.get("busy") is not None:
        line.append(f"matrix core {_pct(mc['busy'])} busy")
    if occ.get("achieved_waves_per_simd") is not None:
        line.append(f"{occ['achieved_waves_per_simd']:.2f} of "
                    f"{occ.get('theoretical_waves_per_simd')} waves/SIMD achieved")
    elif occ.get("theoretical_waves_per_simd") is not None:
        line.append(f"occupancy {occ['theoretical_waves_per_simd']} waves/SIMD (theoretical)")
    if occ.get("cu_busy") is not None:
        line.append(f"CUs busy {_pct(occ['cu_busy'])}")
    if k.get("clock_mhz"):
        line.append(f"clock {k['clock_mhz'] / 1e3:.2f} GHz")
    if line:
        L.append("    busy      " + ", ".join(line))
    st = k.get("stalls") or {}
    if st.get("counters"):
        c = st["counters"]
        L.append("    waves     " + ", ".join(
            f"{lab} {_pct(c[key])}" for key, lab in
            (("dependency_wait", "waiting on a dependency"), ("issue_wait", "waiting to issue"),
             ("active", "issuing"), ("lds_issue_wait", "waiting to issue LDS"))
            if c.get(key) is not None) + " of wave time (counters)")
    mem = []
    lds = k.get("lds") or {}
    if lds.get("bank_conflict") is not None:
        mem.append(f"LDS bank conflicts {lds['bank_conflict'] * 100:.1f}% of LDS cycles")
    cache = k.get("cache") or {}
    if cache.get("l2_hit") is not None:
        mem.append(f"L2 hit {_pct(cache['l2_hit'])}")
    lat = k.get("latency_cycles") or {}
    if lat:
        mem.append("latency " + ", ".join(f"{a} {v:.0f}" for a, v in lat.items()) + " cycles")
    if mem:
        L.append("    memory    " + "; ".join(mem))
    isa = k.get("isa") or {}
    ml = isa.get("main_loop")
    if ml:
        L.append(f"    loop      {ml['start']}-{ml['end']}, {ml['instructions']} instructions: "
                 f"{_mix_text(ml['mix'])}"
                 + (f"; {_pct(ml.get('cycle_share'))} of a wave's cycles"
                    if ml.get("cycle_share") is not None else ""))
    elif isa:
        L.append(f"    code      no loop; {isa.get('instructions')} instructions: "
                 f"{_mix_text(isa.get('total') or {})}")
    a = st.get("att")
    if a:
        L.append(f"    stalls    ({a['region']}, thread trace) " + ", ".join(
            f"{r} {v['share'] * 100:.0f}%" for r, v in list(a["by_reason"].items())[:6]))
    mi = st.get("matrix_idle")
    if mi:
        L.append(f"    matrix    (thread trace, {mi['simds']} SIMDs) busy {_pct(mi['busy'])}; "
                 f"idle with the waves at " + ", ".join(
                     f"{r} {v * 100:.0f}%" for r, v in list(mi["idle"].items())[:6] if v >= 0.005)
                 + "; idle by part " + ", ".join(
                     f"{r} {v * 100:.0f}%" for r, v in mi["idle_by_region"].items() if v >= 0.005)
                 + (f"; no wave {mi['no_wave'] * 100:.0f}%" if mi.get("no_wave") else ""))
    for part in ("prologue", "drain", "epilogue"):
        p = isa.get(part)
        if p and ml:
            L.append(f"    {part:<9} {p['instructions']} instructions: {_mix_text(p['mix'])}"
                     + (f"; {_pct(p.get('cycle_share'))} of a wave's cycles"
                        if p.get("cycle_share") is not None else ""))
    v = k.get("verdict") or {}
    if v.get("line"):
        L.append(f"    verdict   {v['line']}")
    return L


def summary(p: Profile) -> str:
    v = p.versions
    head = (f"hk.profile {p.name}  {p.arch}" + (f" ({p.device}" if p.device else "")
            + (f", {p.agent.get('cus')} CUs)" if p.device and p.agent.get("cus") else
               (")" if p.device else ""))
            + f"  hk {v.get('hk', '?')}" + (f"+{v['hk_commit']}" if v.get("hk_commit") else "")
            + (f"  rocprofv3 {v['rocprofv3']}" if v.get("rocprofv3") else ""))
    L = [head]
    c = p.call
    if c:
        L.append(f"  call      {c.get('inputs') or ''}  {c.get('us')} us of kernels a call, "
                 f"{c.get('kernels')} kernel(s)")
    L.append("  sources   " + "; ".join(f"{k}: {s}" for k, s in p.sources.items()))
    for k in p.kernels:
        L.extend(kernel_summary(k))
    if p.compare:
        L.append(f"  vs        {p.compare['line']}")
        bv = (p.compare.get("b_report") or {}).get("verdict") or {}
        if bv.get("line"):
            L.append(f"  {p.compare['b']}: {bv['line']}")
    for e in p.errors:
        L.append(f"  ERROR [{e.get('stage')}] {e.get('type')}: {e.get('message')}")
    for n in p.notes:
        L.append(f"  note: {n}")
    cv = p.verdict
    L.append(f"VERDICT {cv['line']}")
    if p.seconds is not None:
        L.append(f"  ({p.seconds:.0f} s)")
    return "\n".join(L)


# ------------------------------------------------------------------ hk.profile(...)


class _CallableModule(types.ModuleType):
    """So that `hk.profile(fn, inputs)` is the call, while `hk.profile.isa`,
    `hk.profile.analyze` and `python -m hk.profile` stay where a module's are."""

    def __call__(self, *args, **kwargs) -> Profile:
        return profile(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule

__all__ = ["profile", "Profile", "compare", "summary", "kernel_summary", "call_verdict",
           "SCHEMA", "COMPARED"]
