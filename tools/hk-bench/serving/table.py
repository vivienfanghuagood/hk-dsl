"""table.py -- a serving scorecard row from its server starts (serve_bench.py's OUT directories).

    python3 table.py --ref fair --arms fair,hk_fair --cols "TPOT 1k c4, ms:1024/128/4/16:median_tpot_ms" \
        RUN_DIR...

RUN_DIR holds one directory per start (each with result.json). Per arm and metric: every start's value,
the median over starts and the min-max. A ratio against --ref is called a difference only when the two
arms' start ranges do not overlap and it is at least 1%; otherwise the cell says "=". Discarded: a bench
with failed requests, a non-zero exit, or another process holding more than 64 MiB on a GPU of the run.
Quality: each start's accuracy, and McNemar's exact p (paired by item) against every start of --ref.
"""
import glob, itertools, json, math, os, statistics as st

METRICS = [  # (spec, field, label, lower_is_better)
    ("TPOT", "median_tpot_ms", "TPOT p50 ms", True),
    ("TTFT", "median_ttft_ms", "TTFT p50 ms", True),
    ("OUT", "output_throughput", "output tok/s", False),
]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def load(run_dirs):
    starts = []
    for d in run_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*", "result.json"))):
            r = json.load(open(p))
            if r.get("mode") != "measure":
                continue
            r["_dir"] = os.path.dirname(p)
            r["_id"] = os.path.basename(d.rstrip("/")) + "/" + os.path.basename(r["_dir"])
            q = os.path.join(r["_dir"], "quality.json")
            r["_quality"] = json.load(open(q)) if os.path.exists(q) else None
            starts.append(r)
    return starts


def summarize(run_dirs, ref="ship"):
    starts = load(run_dirs)
    dropped = []
    vals = {}   # (arm, spec, field) -> [(start id, value)]
    for r in starts:
        for tag, b in r["benches"].items():
            if tag.startswith("warm"):
                continue
            bad = []
            if b.get("failed"):
                bad.append(f"{b['failed']} failed requests")
            if b.get("foreign"):
                bad.append(f"foreign GPU holder {b['foreign']}")
            if b.get("rc"):
                bad.append(f"bench rc {b['rc']}")
            if bad:
                dropped.append(f"{r['_id']} {b['spec']}: " + "; ".join(bad))
                continue
            for _, f, _, _ in METRICS:
                if b.get(f) is not None:
                    vals.setdefault((r["arm"], b["spec"], f), []).append((r["_id"], b[f]))
    agg = {}
    for (arm, spec, f), v in vals.items():
        xs = [x for _, x in v]
        agg[f"{arm}|{spec}|{f}"] = {"starts": v, "median": st.median(xs), "min": min(xs), "max": max(xs)}
    qual = {}
    by_arm = {}
    for r in starts:
        if r["_quality"]:
            by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in by_arm.items():
        for kind in ("gsm8k", "chartqa"):
            accs = []
            for r in rs:
                it = [x for x in r["_quality"]["items"] if x["kind"] == kind]
                if it:
                    accs.append(sum(x["ok"] for x in it) / len(it))
            if not accs:
                continue
            ps = []
            for r in rs:
                for s in by_arm.get(ref, []):
                    if r is s:
                        continue
                    a = [x for x in r["_quality"]["items"] if x["kind"] == kind]
                    b = [x for x in s["_quality"]["items"] if x["kind"] == kind]
                    if len(a) != len(b) or not a:
                        continue
                    oa = sum(x["ok"] and not y["ok"] for x, y in zip(a, b))
                    ob = sum(y["ok"] and not x["ok"] for x, y in zip(a, b))
                    ps.append(mcnemar(oa, ob))
            qual[f"{arm}|{kind}"] = {"acc": accs, "n": len([x for x in rs[0]["_quality"]["items"] if x["kind"] == kind]),
                                     "mcnemar_p_vs_ref": ps}
    return {"starts": [{k: r.get(k) for k in ("_id", "arm", "startup_s", "lock_wait_s", "t_start", "t_end")} for r in starts],
            "agg": agg, "quality": qual, "dropped": dropped}


def ratio(agg, a, b, spec, f, lower):
    """b relative to a: speedup (>1 = b better) and whether the start ranges are disjoint."""
    A = agg.get(f"{a}|{spec}|{f}"); B = agg.get(f"{b}|{spec}|{f}")
    if not A or not B:
        return None
    sp = A["median"] / B["median"] if lower else B["median"] / A["median"]
    disjoint = A["max"] < B["min"] or B["max"] < A["min"]
    return sp, disjoint



def table(run_dirs, ref, arms, cols):
    """Markdown lines: one row per arm; cols = [(label, spec, field)]."""
    S = summarize(run_dirs, ref=ref)
    lower = {f: lo for _, f, _, lo in METRICS}
    out = ["| arm | " + " | ".join(c[0] for c in cols) + " | quality |", "|---|" + "---:|" * len(cols) + "---|"]
    for arm in arms:
        cells = [arm]
        for _, spec, f in cols:
            v = S["agg"].get(f"{arm}|{spec}|{f}")
            if not v:
                cells.append("--")
                continue
            p = 2 if v["median"] < 100 else 0
            txt = (f"{v['median']:.{p}f} ({v['min']:.{p}f}-{v['max']:.{p}f})" if len(v["starts"]) > 1
                   else f"{v['median']:.{p}f} (1 start)")
            if arm != ref:
                rr = ratio(S["agg"], ref, arm, spec, f, lower[f])
                if rr:
                    txt += (f"<br>**{rr[0]:.3f}x**" if rr[1] and abs(rr[0] - 1) >= 0.01
                            else f"<br>{rr[0]:.3f}x (=)")
            cells.append(txt)
        qs = []
        for kind in ("gsm8k", "chartqa"):
            q = S["quality"].get(f"{arm}|{kind}")
            if q:
                ps = q["mcnemar_p_vs_ref"]
                qs.append(f"{kind} {' / '.join(f'{x * 100:.1f}' for x in q['acc'])}% (n {q['n']}"
                          + (f", McNemar p vs {ref} min {min(ps):.2f}" if ps else "") + ")")
        cells.append("<br>".join(qs) or "--")
        out.append("| " + " | ".join(cells) + " |")
    if S["dropped"]:
        out += ["", "Discarded: " + "; ".join(S["dropped"])]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="the arm the others are compared with")
    ap.add_argument("--arms", required=True, help="comma-separated arms, in table order")
    ap.add_argument("--cols", action="append", required=True, help="LABEL:SPEC:FIELD, e.g. 'TTFT 16k, ms:16384/16/1/4:median_ttft_ms'")
    ap.add_argument("--json", action="store_true", help="print the whole summary as JSON instead")
    ap.add_argument("run_dirs", nargs="+")
    a = ap.parse_args()
    if a.json:
        print(json.dumps(summarize(a.run_dirs, ref=a.ref), indent=1, default=str))
    else:
        cols = [tuple(c.rsplit(":", 2)) for c in a.cols]
        print("\n".join(table(a.run_dirs, a.ref, a.arms.split(","), cols)))
