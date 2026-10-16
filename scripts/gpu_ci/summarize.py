#!/usr/bin/env python3
"""One GPU CI run, summarised: scripts/gpu_suite.sh's output directory becomes
result.json (what orchestrate.py reads) and report.txt (the log it uploads).

    summarize.py SUITE_OUT --sha SHA --tier gfx1100 --device "W7900D, GPU 0" --out RUN
    summarize.py --error "the dialect did not build" --sha SHA --tier gfx1100 --out RUN

Standard library only: it runs on the workstation, in the pod and in the
rocm/pytorch container, each with its own Python.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

_COUNT = re.compile(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)")
_TAIL = 60


def _last_summary(text: str) -> str:
    for line in reversed(text.splitlines()):
        if re.search(r"\b\d+ (passed|failed|skipped|errors?|deselected)\b", line) and " in " in line:
            return line.strip().strip("=").strip()
    return ""


def _counts(summary: str) -> dict:
    out: dict = {}
    for n, kind in _COUNT.findall(summary):
        kind = "error" if kind.startswith("error") else kind
        out[kind] = out.get(kind, 0) + int(n)
    return out


def _groups(suite: Path):
    status = suite / "status"
    lines = status.read_text().splitlines() if status.exists() else []
    done = "done" in lines
    groups = []
    for line in lines:
        m = re.fullmatch(r"(\S+) exit (\d+)", line.strip())
        if not m:
            continue
        name, code = m.group(1), int(m.group(2))
        log = suite / f"{name}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        summary = _last_summary(text)
        groups.append({
            "name": name,
            "exit": code,
            "summary": summary,
            "counts": _counts(summary),
            "failures": [l.strip() for l in text.splitlines()
                         if l.startswith(("FAILED ", "ERROR "))],
            "tail": text.splitlines()[-_TAIL:] if code else [],
        })
    return done, groups


def _state(done: bool, groups: list) -> str:
    if not done or not groups:
        return "error"
    if all(g["exit"] == 0 for g in groups):
        return "success"
    # A group with no pytest summary was killed (the suite's timeout) or never
    # collected: that is the harness failing, not a test.
    if any(g["exit"] and not g["summary"] for g in groups):
        return "error"
    return "failure"


def _describe(state: str, counts: dict, minutes: float, error: str = "") -> str:
    if error:
        return f"error: {error}"
    parts = [f"{counts[k]} {k}" for k in ("failed", "error", "passed", "skipped", "xfailed")
             if counts.get(k)]
    return f"{state}: {', '.join(parts) or 'nothing ran'} ({minutes:.0f} min)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite", nargs="?", help="scripts/gpu_suite.sh's output directory")
    ap.add_argument("--sha", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", required=True, help="the run directory")
    ap.add_argument("--error", default="", help="the run failed before the suite")
    a = ap.parse_args()

    run = Path(a.out)
    started = run / "started"
    t0 = os.path.getmtime(started) if started.exists() else time.time()
    minutes = (time.time() - t0) / 60

    if a.error:
        done, groups, state = False, [], "error"
    else:
        done, groups = _groups(Path(a.suite))
        state = _state(done, groups)
    counts: dict = {}
    for g in groups:
        for k, v in g["counts"].items():
            counts[k] = counts.get(k, 0) + v
    if not a.error and not done:
        a.error = "the suite did not finish"
    dead = [f"{g['name']} (exit {g['exit']})" for g in groups if g["exit"] and not g["summary"]]
    if not a.error and dead:
        a.error = f"{', '.join(dead)} ended without a pytest summary"
    description = _describe(state, counts, minutes, a.error)

    result = {
        "sha": a.sha, "tier": a.tier, "device": a.device, "state": state,
        "description": description, "counts": counts, "minutes": round(minutes, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "groups": [{k: g[k] for k in ("name", "exit", "summary")} for g in groups],
        "failures": [f for g in groups for f in g["failures"]][:50],
    }

    report = [f"hk-dsl GPU CI -- {a.tier}{' on ' + a.device if a.device else ''}",
              f"commit   {a.sha}",
              f"result   {description}",
              f"finished {result['finished']}", ""]
    if groups:
        report.append("groups")
        report += [f"  {g['name']:<8} exit {g['exit']:<4} {g['summary']}" for g in groups]
        report.append("")
    if result["failures"]:
        report.append("failures")
        report += [f"  {f}" for f in result["failures"]]
        report.append("")
    for g in groups:
        if g["tail"]:
            report.append(f"--- last {_TAIL} lines of {g['name']}.log ---")
            report += g["tail"]
            report.append("")
    for name in ("build.log", "driver.log"):
        log = run / name
        if a.error and log.exists():
            report.append(f"--- last {_TAIL} lines of {name} ---")
            report += log.read_text(errors="replace").splitlines()[-_TAIL:]
            report.append("")

    (run / "report.txt").write_text("\n".join(report) + "\n")
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    tmp.replace(run / "result.json")
    print(description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
