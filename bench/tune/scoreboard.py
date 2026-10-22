"""The tuning benchmark's scoreboard: what bench/tune/run.py measured, per
shape and in total, before (the current routes) and after (hk.tune's records).

    python3 -m bench.tune.scoreboard DIR [DIR...] [--arch gfx950] [--split acceptance]
                                     [--explain] [--write bench/tune/results]

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


#: Dense bf16 peak FLOP/s, HBM bytes/s (datasheets), CUs.
CARDS = {"gfx950": (2.5e15, 8.0e12, 256), "gfx1100": (122.6e12, 864e9, 96)}


def explain(d: dict, records=None) -> None:
    """A sentence per miss: how close the best baseline and hk are to the
    card's compute and memory limits, and -- where hk imports -- the grid of
    the schedule hk ran (workgroups against CUs: the wave quantization)."""
    peak, bw, cus = CARDS.get(d["arch"], (None, None, None))
    sched_of = None
    if records:
        import os  # noqa: PLC0415

        os.environ["HK_ARCH"] = d["arch"]
        try:
            import hk.schedules as S  # noqa: PLC0415
            from hk.tune import records as R  # noqa: PLC0415

            def sched_of(m, n, k):
                # The records of the run that measured the shape.
                os.environ["HK_TUNE_DIR"] = records[(m, n, k)]
                R.forget()
                return S.lookup(m, n, k, d["arch"])
        except Exception:  # noqa: BLE001 -- explanation without the grid
            sched_of = None
    for s in d["misses"]:
        m, n, k = s["m"], s["n"], s["k"]
        flops, byts = 2.0 * m * n * k, 2.0 * (m * k + n * k + m * n)
        tb, th = s["best_us"] * 1e-6, s["after_us"] * 1e-6
        if not (peak and tb and th):
            continue
        parts = [f"best {flops / tb / 1e12:.0f} TFLOP/s, {byts / tb / 1e12:.2f} TB/s; "
                 f"hk {flops / th / 1e12:.0f} TFLOP/s, {byts / th / 1e12:.2f} TB/s"]
        if byts / tb > 0.6 * bw:
            parts.append(f"memory-bound: the best baseline streams at {byts / tb / bw:.0%} of "
                         f"HBM, hk at {byts / th / bw:.0%}")
        elif flops / tb > 0.4 * peak:
            parts.append(f"compute-bound: the best baseline at {flops / tb / peak:.0%} of the "
                         f"bf16 peak, hk at {flops / th / peak:.0%}")
        else:
            parts.append("neither limit is near: latency, launch or occupancy")
        if sched_of is not None:
            try:
                sc = sched_of(m, n, k)
                grids = []
                for st in sc.steps(m, n, k):
                    bm, bn = st.sched.get("block_m"), st.sched.get("block_n")
                    if st.kernel in ("pp", "dp", "rdna") and bm and bn:
                        lo, hi = st.cols
                        wgs = -(-m // bm) * -(-(hi - lo) // bn) * st.sched.get("splits", 1)
                        grids.append(f"{st.kernel} {bm}x{bn}"
                                     + (f" x{st.sched['splits']} K shares"
                                        if st.sched.get("splits", 1) > 1 else "")
                                     + f": {wgs} workgroups, {wgs / cus:.2f} of {cus} CUs")
                    elif st.kernel == "decode":
                        grids.append(f"decode GEMV ({n // 16 * st.sched.get('splits', 1)} "
                                     f"workgroups)")
                if grids:
                    parts.append("hk ran " + "; ".join(grids))
            except Exception:  # noqa: BLE001
                pass
        s["why"] = "; ".join(parts)


def attach_profiles(d: dict, pdir: str) -> None:
    """hk.profile's verdicts (K2; `python -m hk.profile hk.ops:matmul --vs
    hipblaslt=torch.nn.functional:linear --json DIR/<M>x<N>x<K>.json`) on
    the misses: its one line, and its interleaved comparison."""
    for s in d["misses"]:
        f = Path(pdir) / f"{s['m']}x{s['n']}x{s['k']}.json"
        if not f.exists():
            continue
        try:
            p = json.loads(f.read_text())
        except ValueError:
            continue
        line = (p.get("verdict") or {}).get("line", "")
        cmp_ = p.get("compare") or {}
        other = ""
        if cmp_.get("ratio"):
            ks = cmp_.get("kernels") or []
            other = (f" -- against {cmp_.get('b')}: {cmp_['ratio']:.3f}x the time"
                     + (f", its kernel {ks[1].split('_UserArgs')[0][:30]}..."
                        f"{'MT' + ks[1].split('_MT')[1].split('_')[0] if '_MT' in ks[1] else ''}"
                        f"{' stream-K' if '_SK' in ks[1] and '_SK0_' not in ks[1] else ''}"
                        if len(ks) > 1 else ""))
        s["profile"] = line + other


def markdown(d: dict) -> str:
    b, a = d["before"], d["after"]
    meta = d.get("meta", {})
    v = meta.get("versions", {})
    lines = [f"# hk.tune scoreboard: {d['arch']}, {d['split']} split, {d['date']}", "",
             f"Device {meta.get('device', '?')}; torch {v.get('torch', '?')}, HIP {v.get('hip', '?')}, "
             f"hipcc `{meta.get('hipcc', '?')}`; "
             + ", ".join(f"{k} {x}" for k, x in v.items()
                         if k not in ("torch", "hip") and x != "?") + ".",
             f"hk-dsl at `{meta.get('commit') or '?'}` (claude/tune). "
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
    if d["misses"]:
        lines += ["", f"Misses (under {d['target']}x the best baseline after tuning), and what "
                      "the numbers say about why:", ""]
        for s in d["misses"]:
            lines.append(f"- **{s['m']}x{s['n']}x{s['k']}** ({s['after'][0]:.3f}x "
                         f"{s['best_baseline']}): {s.get('why') or '-'}")
            if s.get("profile"):
                lines.append(f"  - hk.profile: {s['profile']}")
    wrong = [(s, s["wrong"]) for s in d["shapes"] if s["wrong"]]
    if wrong:
        lines += ["", "Arms dropped at a shape (wrong against fp32, or refused the shape):", ""]
        for s, w in wrong:
            lines.append(f"- {s['m']}x{s['n']}x{s['k']}: "
                         + "; ".join(f"{k}: {v}" for k, v in w.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="+", help="result directories; a shape in a later one wins")
    ap.add_argument("--profiles", default="",
                    help="a directory of hk.profile JSON reports, <M>x<N>x<K>.json, for the misses")
    ap.add_argument("--explain", action="store_true",
                    help="say why each miss misses (with hk importable: the grid hk ran)")
    ap.add_argument("--arch", default="")
    ap.add_argument("--split", default="acceptance")
    ap.add_argument("--write", default="", help="directory for <arch>-<date>.{json,md}")
    ap.add_argument("--name", default="", help="file stem (default <arch>-<date>)")
    args = ap.parse_args(argv)
    by_arch = {}
    for dd in args.dir:
        for f in sorted(Path(dd).glob(f"{args.arch or '*'}-{args.split}.jsonl")):
            by_arch.setdefault(f.name.split("-")[0], []).append(f)
    if not by_arch:
        print(f"no {args.split} results in {args.dir}")
        return 2
    for arch, files in by_arch.items():
        rows, where = {}, {}
        for f in files:
            for r in load(f):
                rows[(r["m"], r["n"], r["k"])] = r
                where[(r["m"], r["n"], r["k"])] = str(Path(f).resolve().parent / "records")
        s = summarize([rows[x] for x in sorted(rows)], arch, args.split)
        if args.explain:
            explain(s, where)
        if args.profiles:
            attach_profiles(s, args.profiles)
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
