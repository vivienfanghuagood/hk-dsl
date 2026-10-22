"""The tuning benchmark's scoreboard: what bench/tune/run.py measured, per
shape and in total, before (the current routes) and after (hk.tune's records).

    python3 -m bench.tune.scoreboard DIR [--arch gfx950] [--split acceptance]
                                     [--write bench/tune/results]

The acceptance (docs/design/tuning-goal.md, T3), per GPU, on the held-out
split: the share of shapes at >= 0.95x the best available baseline after
tuning (target >= 80%), the median and worst ratio, GPU-minutes per shape.
A shape's ratio is the median over the scoring sets of best/hk, the band the
sets' range; "best" is the fastest correct baseline at that shape.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

TARGET = 0.95
SHARE = 0.80


def load(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[(r["m"], r["n"], r["k"])] = r          # the last measurement of a shape wins
    return [rows[k] for k in sorted(rows)]


def summarize(rows, arch: str, split: str) -> dict:
    shapes = []
    for r in rows:
        sc = r.get("score", {})
        t = r.get("tune") or {}
        dec = t.get("decision") or {}
        b = sc.get("hk_before") or [None, None, None]
        a = sc.get("hk_after") or [None, None, None]
        best = r.get("best_baseline")
        shapes.append(dict(
            m=r["m"], n=r["n"], k=r["k"], best_baseline=best,
            best_us=round(r["ms"].get(f"base:{best}", 0) * 1e3, 2) if best else None,
            before=b, after=a,
            before_us=round(r["ms"].get("hk_before", 0) * 1e3, 2),
            after_us=round(r["ms"].get("hk_after", 0) * 1e3, 2),
            route_before=r.get("route_before"), route_after=r.get("route_after"),
            verdict=t.get("verdict"), gain=dec.get("gain"), bar=dec.get("bar"),
            tune_gpu_s=t.get("gpu_seconds"), tune_wall_s=t.get("wall_seconds"),
            baselines_us={k[5:]: round(v * 1e3, 2) for k, v in r["ms"].items()
                          if k.startswith("base:")},
            wrong={k: v for k, v in r.get("errors", {}).items()
                   if not (isinstance(v, (int, float)) and v < 2e-2)},
            reached=bool(a[0] and a[0] >= TARGET)))

    def stats_of(key):
        xs = [s[key][0] for s in shapes if s[key][0]]
        if not xs:
            return {}
        return dict(share=round(sum(x >= TARGET for x in xs) / len(xs), 3),
                    reached=sum(x >= TARGET for x in xs), of=len(xs),
                    median=round(statistics.median(xs), 3), worst=round(min(xs), 3),
                    geomean=round(statistics.geometric_mean(xs), 3))

    gpu = [s["tune_gpu_s"] for s in shapes if s["tune_gpu_s"] is not None]
    wall = [s["tune_wall_s"] for s in shapes if s["tune_wall_s"] is not None]
    meta = rows[0].get("meta", {}) if rows else {}
    return dict(
        arch=arch, split=split, date=time.strftime("%Y-%m-%d"), target=TARGET, share_target=SHARE,
        before=stats_of("before"), after=stats_of("after"),
        gpu_minutes_per_shape=dict(mean=round(statistics.mean(gpu) / 60, 2) if gpu else None,
                                   max=round(max(gpu) / 60, 2) if gpu else None,
                                   total=round(sum(gpu) / 60, 1) if gpu else None),
        wall_minutes_per_shape=dict(mean=round(statistics.mean(wall) / 60, 2) if wall else None,
                                    max=round(max(wall) / 60, 2) if wall else None),
        wins=sum(s["verdict"] == "win" for s in shapes),
        meta=meta, shapes=shapes,
        misses=[s for s in shapes if not s["reached"]])


def markdown(d: dict) -> str:
    b, a = d["before"], d["after"]
    meta = d.get("meta", {})
    v = meta.get("versions", {})
    lines = [f"# hk.tune scoreboard: {d['arch']}, {d['split']} split, {d['date']}", "",
             f"Device {meta.get('device', '?')}; torch {v.get('torch', '?')}, HIP {v.get('hip', '?')}, "
             f"hipcc `{meta.get('hipcc', '?')}`; "
             + ", ".join(f"{k} {x}" for k, x in v.items() if k not in ("torch", "hip")) + ".",
             f"Baselines missing: {meta.get('missing') or 'none'}. hk.tune budget "
             f"{meta.get('budget', '?')} GPU-s a shape; scoring {meta.get('sets', '?')} sets x "
             f"{meta.get('rounds', '?')} rounds, interleaved, queued, weights rotated past the "
             f"Infinity Cache.", "",
             "| | before (routes) | after (hk.tune) |", "|---|---|---|",
             f"| shapes >= {d['target']}x best | {b.get('reached')}/{b.get('of')} "
             f"({b.get('share', 0):.0%}) | **{a.get('reached')}/{a.get('of')} "
             f"({a.get('share', 0):.0%})** |",
             f"| median ratio | {b.get('median')} | {a.get('median')} |",
             f"| worst ratio | {b.get('worst')} | {a.get('worst')} |",
             f"| geomean ratio | {b.get('geomean')} | {a.get('geomean')} |", "",
             f"Tuning cost: {d['gpu_minutes_per_shape']['mean']} GPU-min a shape on average, "
             f"{d['gpu_minutes_per_shape']['max']} at most, {d['gpu_minutes_per_shape']['total']} "
             f"in all; {d['wall_minutes_per_shape']['mean']} min of wall clock a shape (the "
             f"compiles). {d['wins']} of {len(d['shapes'])} shapes got a record; the rest kept "
             f"their route.", "",
             "Ratio = best baseline time / hk time (>1: hk faster), median over the scoring "
             "sets [their range].", "",
             "| M x N x K | best baseline | best us | before | after | route before -> after "
             "| tune | GPU-s |", "|---|---|---|---|---|---|---|---|"]
    for s in d["shapes"]:
        bb, aa = s["before"], s["after"]

        def f(x):
            return f"{x[0]:.3f} [{x[1]:.3f}-{x[2]:.3f}]" if x and x[0] else "-"
        tune = s["verdict"] or "-"
        if s["gain"] is not None and s["verdict"] == "win":
            tune += f" +{s['gain'] * 100:.1f}%"
        mark = "" if s["reached"] else " **miss**"
        lines.append(f"| {s['m']}x{s['n']}x{s['k']} | {s['best_baseline']} | {s['best_us']} | "
                     f"{f(bb)} | {f(aa)}{mark} | {s['route_before']} -> {s['route_after']} | "
                     f"{tune} | {s['tune_gpu_s']} |")
    lines += ["", "Every baseline's time (us):", "",
              "| M x N x K | " + " | ".join(sorted({k for s in d["shapes"] for k in s["baselines_us"]}))
              + " |"]
    names = sorted({k for s in d["shapes"] for k in s["baselines_us"]})
    lines.append("|---|" + "---|" * len(names))
    for s in d["shapes"]:
        lines.append(f"| {s['m']}x{s['n']}x{s['k']} | "
                     + " | ".join(str(s["baselines_us"].get(x, "-")) for x in names) + " |")
    wrong = [(s, s["wrong"]) for s in d["shapes"] if s["wrong"]]
    if wrong:
        lines += ["", "Arms dropped at a shape (wrong against fp32, or refused the shape):", ""]
        for s, w in wrong:
            lines.append(f"- {s['m']}x{s['n']}x{s['k']}: "
                         + "; ".join(f"{k}: {v}" for k, v in w.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--arch", default="")
    ap.add_argument("--split", default="acceptance")
    ap.add_argument("--write", default="", help="directory for <arch>-<date>.{json,md}")
    ap.add_argument("--name", default="", help="file stem (default <arch>-<date>)")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    files = sorted(d.glob(f"{args.arch or '*'}-{args.split}.jsonl"))
    if not files:
        print(f"no {args.split} results in {d}")
        return 2
    for f in files:
        arch = f.name.split("-")[0]
        s = summarize(load(f), arch, args.split)
        md = markdown(s)
        print(md)
        if args.write:
            out = Path(args.write)
            out.mkdir(parents=True, exist_ok=True)
            stem = args.name or f"{arch}-{s['date']}"
            (out / f"{stem}.json").write_text(json.dumps(s, indent=1) + "\n")
            (out / f"{stem}.md").write_text(md)
            print(f"wrote {out / stem}.{{json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
