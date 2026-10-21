#!/usr/bin/env python3
"""The README's results chart: the templates against torch.compile, per task, on both GPUs.

    python3 tools/docs/make_charts.py            # writes docs/assets/templates-vs-torch-compile{,-dark}.svg

Every value is read from a committed file, none is typed in here:

- the DX suite rows of the templates (T1, T2, T3, T8): the table "task (template lines)" in
  docs/templates/elementwise-rowwise.md (geomean over the task's shapes, and min-max);
- the GEMM epilogue rows: the per-layer tables in docs/templates/gemm.md, column "template vs
  compile" (the MI355X's device-time table and the W7900D's table); the dot is the geometric
  mean of the table's cases for that epilogue, computed here, the bar their min-max;
- the "16x64 tiles" row: bench/dx/results/<arch>-*.json, task t1, solution docs_only (the
  kernel the first page used to teach).

Needs matplotlib, no GPU. The SVGs are deterministic (no date, fixed ids), so re-running it on
unchanged inputs changes nothing.
"""

import glob
import json
import math
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "assets")

GPUS = ("gfx1100", "gfx950")
GPU_NAME = {"gfx1100": "Radeon PRO W7900D (gfx1100)", "gfx950": "Instinct MI355X (gfx950)"}

# The validated two-slot categorical palette (dataviz reference palette, slots 1-2), light and
# dark steps, and the text/grid roles for each surface.
THEMES = {
    "light": {"gfx1100": "#eb6834", "gfx950": "#2a78d6", "text": "#0b0b0b", "text2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "ref": "#52514e"},
    "dark": {"gfx1100": "#d95926", "gfx950": "#3987e5", "text": "#ffffff", "text2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "ref": "#c3c2b7"},
}


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def tables(text):
    """Every markdown table in `text` as (heading above it, header cells, rows of cells)."""
    out, heading, lines = [], "", text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            cells = lambda s: [c.strip() for c in s.strip().strip("|").split("|")]
            head, rows, i = cells(line), [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(cells(lines[i]))
                i += 1
            out.append((heading, head, rows))
            continue
        i += 1
    return out


def geo_range(cell):
    """'1.11 (1.03-1.63)' -> (1.11, 1.03, 1.63)."""
    m = re.match(r"([\d.]+)\s*\(([\d.]+)-([\d.]+)\)", cell)
    if not m:
        raise ValueError(f"not 'geomean (min-max)': {cell!r}")
    return tuple(float(x) for x in m.groups())


def dx_template_rows():
    """{task label: {arch: (geomean, min, max)}} from the templates' DX suite table."""
    for _, head, rows in tables(read("docs/templates/elementwise-rowwise.md")):
        if head[0].startswith("task (template lines)"):
            c950 = head.index("MI355X x torch.compile")
            c1100 = next(i for i, h in enumerate(head) if h.startswith("W7900D") and "torch.compile" in h)
            out = {}
            for r in rows:
                m = re.match(r"T(\d) (.*) \((\d+)\)", r[0])
                out[(int(m.group(1)), m.group(2), int(m.group(3)))] = {
                    "gfx950": geo_range(r[c950]), "gfx1100": geo_range(r[c1100])}
            return out
    raise SystemExit("docs/templates/elementwise-rowwise.md: the DX suite table is missing")


def gemm_rows():
    """{epilogue: {arch: [template vs compile, per case]}} from docs/templates/gemm.md."""
    want = {"MI355X (gfx950)": "gfx950", "W7900D (gfx1100)": "gfx1100"}
    out, seen = {}, set()
    for heading, head, rows in tables(read("docs/templates/gemm.md")):
        arch = want.get(heading)
        if arch is None or arch in seen or "template vs compile" not in head:
            continue                      # the first table under each heading: device time on the MI355X
        seen.add(arch)
        ce, cx = head.index("epilogue"), head.index("template vs compile")
        for r in rows:
            out.setdefault(r[ce], {}).setdefault(arch, []).append(float(r[cx].rstrip("x")))
    if seen != set(GPUS):
        raise SystemExit(f"docs/templates/gemm.md: found tables for {sorted(seen)} only")
    return out


def dx_docs_only_tiles():
    """{arch: (geomean, min, max)} of t1's docs_only solution (16x64 register tiles)."""
    out = {}
    for arch in GPUS:
        path = sorted(glob.glob(os.path.join(ROOT, "bench", "dx", "results", f"{arch}-*.json")))[-1]
        with open(path, encoding="utf-8") as f:
            for s in json.load(f)["summary"]:
                if s["task"].startswith("t1_") and s["solution"] == "docs_only":
                    out[arch] = (s["x_tc_geomean"], s["x_tc_min"], s["x_tc_max"])
    return out


def rows():
    """The chart's rows, top to bottom: (label, sublabel, {arch: (dot, lo, hi)}, muted)."""
    dx, gemm, tiles = dx_template_rows(), gemm_rows(), dx_docs_only_tiles()
    by_task = {k[0]: (k, v) for k, v in dx.items()}

    def dx_row(n, label):
        (_, _, lines), v = by_task[n]
        return (label, f"rowwise template, {lines} lines" if n != 1 else
                f"elementwise template, {lines} lines", v, False)

    def gemm_row(ep, label):
        v = {}
        for arch, xs in gemm[ep].items():
            v[arch] = (math.exp(sum(map(math.log, xs)) / len(xs)), min(xs), max(xs))
        return (label, f"GEMM template + epilogue, {len(gemm[ep]['gfx950'])} layer x M cases", v, False)

    return [
        dx_row(1, "GELU(tanh)-and-mul"),
        ("GELU(tanh)-and-mul, by hand", "16x64 register tiles, as the old first page taught", tiles, True),
        dx_row(2, "add + RMSNorm + int8 quant"),
        dx_row(3, "cross-entropy, vocab to 151936"),
        dx_row(8, "MoE router: softmax + top-k"),
        gemm_row("bias_gelu", "linear + bias + GELU(tanh)"),
        gemm_row("residual", "linear + residual add"),
    ]


def draw(data, theme, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    t = THEMES[theme]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "hk-dsl-docs",
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10})
    n = len(data)
    fig, ax = plt.subplots(figsize=(8.6, 0.56 * n + 1.55), dpi=100)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0.33, right=0.97, top=1 - 0.62 / (0.56 * n + 1.55),
                        bottom=0.75 / (0.56 * n + 1.55))

    lo_x, hi_x = 0.4, 16
    ax.set_xscale("log")
    ax.set_xlim(lo_x, hi_x)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{x:g}x" for x in ticks], color=t["muted"], fontsize=9)
    for x in ticks:
        ax.axvline(x, color=t["grid"], lw=0.8, zorder=0)
    ax.axvline(1.0, color=t["ref"], lw=1.2, zorder=1)
    ax.text(1.0, n - 0.35, " torch.compile = 1x", color=t["text2"], fontsize=8.5, va="bottom", ha="left")
    ax.text(hi_x, n - 0.35, "faster →", color=t["muted"], fontsize=8.5,
            va="bottom", ha="right")

    off = {"gfx1100": 0.14, "gfx950": -0.14}
    for i, (label, sub, vals, muted) in enumerate(data):
        y = n - 1 - i
        ax.text(-0.02, y + 0.08, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=t["text2"] if muted else t["text"], fontsize=10,
                style="italic" if muted else "normal")
        ax.text(-0.02, y - 0.24, sub, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=t["muted"], fontsize=8)
        for arch in GPUS:
            g, lo, hi = vals[arch]
            yy = y + off[arch]
            c = t[arch]
            ax.plot([lo, hi], [yy, yy], color=c, lw=2, solid_capstyle="round", zorder=2,
                    alpha=0.7 if muted else 1)
            ax.plot([g], [yy], marker="o", ms=8, color=c, mec=c, zorder=3,
                    alpha=0.7 if muted else 1, mfc="none" if muted else c, mew=2)
            ax.text(hi * 1.06, yy, f"{g:.2f}x", va="center", ha="left", fontsize=8.5, color=t["text2"])
    ax.set_ylim(-0.6, n - 0.1)
    ax.yaxis.set_major_locator(NullLocator())
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(axis="x", length=0, pad=4)

    # legend, top left, in the label column
    fig.text(0.015, 1 - 0.18 / (0.56 * n + 1.55), "hk.templates against torch.compile: speedup (dot: geomean over shapes, bar: min-max)",
             color=t["text"], fontsize=11, weight="bold", va="top")
    x = 0.015
    for arch in GPUS:
        fig.text(x, 1 - 0.42 / (0.56 * n + 1.55), "●", color=t[arch], fontsize=11, va="top")
        fig.text(x + 0.022, 1 - 0.44 / (0.56 * n + 1.55), GPU_NAME[arch], color=t["text2"],
                 fontsize=9, va="top")
        x += 0.30
    fig.text(0.015, 0.012,
             "MI355X: device time (CUDA-graph replay). W7900D: DX tasks at 3 shapes (--quick), GEMM rows "
             "eager.\nSources: docs/templates/elementwise-rowwise.md, docs/templates/gemm.md, "
             "bench/dx/results; made by tools/docs/make_charts.py.",
             color=t["muted"], fontsize=7.5, va="bottom")
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, transparent=True)
    plt.close(fig)


def main():
    data = rows()
    os.makedirs(OUT, exist_ok=True)
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        path = os.path.join(OUT, f"templates-vs-torch-compile{suffix}.svg")
        draw(data, theme, path)
        print(os.path.relpath(path, ROOT))
    for label, sub, vals, _ in data:
        print(f"  {label:34s} " + "  ".join(
            f"{a} {g:.2f} ({lo:.2f}-{hi:.2f})" for a, (g, lo, hi) in vals.items()))


if __name__ == "__main__":
    sys.exit(main())
