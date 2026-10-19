#!/usr/bin/env python3
"""summarize.py's result.json -> the job's log line, an annotation with the
counts on the check run, the step summary, and the job's exit code.

    report.py TIER RESULT_JSON
"""
import json
import os
import sys

tier, path = sys.argv[1], sys.argv[2]
r = json.load(open(path))
ok = r["state"] == "success"
print(r["description"])
print(f"::{'notice' if ok else 'error'} title={tier}::{r['description']}")
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write(f"### {tier}: {r['description']}\n\n{r.get('device', '')}, commit `{r.get('sha', '')[:10]}`\n\n")
        f.write("| group | exit | summary |\n|---|---|---|\n")
        for g in r.get("groups", []):
            f.write(f"| {g['name']} | {g['exit']} | {g.get('summary', '')} |\n")
        if r.get("failures"):
            f.write("\nfailures:\n\n" + "".join(f"- `{x}`\n" for x in r["failures"]))
sys.exit(0 if ok else 1)
