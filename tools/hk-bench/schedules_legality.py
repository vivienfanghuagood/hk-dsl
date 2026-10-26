#!/usr/bin/env python3
"""hk.schedules' legality, measured: sample each GEMM family's space, build
what check() accepts and refuses, and fit the register model check() uses.

    python3 tools/hk-bench/schedules_legality.py sample --out s.json --n 300 --seed 7
    python3 tools/hk-bench/schedules_legality.py eval s.json [more.json ...]
    python3 tools/hk-bench/schedules_legality.py fit s0.json s1.json ... --write
    python3 tools/hk-bench/schedules_legality.py sample-attn --out a.json --all

`sample` draws points uniformly from each family's full value product (every
knob's `values`) on each arch it runs on, keeps the ones whose knob
arithmetic is legal (Schedule._check_struct: the resource prediction is what
is being measured, so it is not applied), builds them all -- hipcc, no GPU,
`workers` at a time -- and records what the build said: registers (VGPRs,
AGPRs, SGPRs), spills, or the pass that refused it. `--seeds` adds every
point of the families' seed spaces.

`eval` scores the current check_kernel() against such records: a false
accept is a point it accepts that did not build, a false reject one it
refuses that did. `fit` refits hk/schedules/_regs.py's MODEL: per (family,
arch), the tightest linear upper envelope of the builds' registers (min
sum(prediction - actual) subject to prediction >= actual on every build that
fit, prediction over the limit on every one that spilled, and under it on
every schedule hk ships), by scipy's linprog. A new hipcc moves registers;
re-sample and refit then, and evaluate on a sample the fit did not see.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

RES_RE = re.compile(r"All instantiations:\s+\S+: vgpr=(\d+)(?:\+(\d+)a)?\S* occ=\S+ "
                    r"lds=\d+ scratch=(\d+) spill=(\d+)")
SGPR_SPILL_RE = re.compile(r"spill vgpr=(\d+) sgpr=(\d+)")


def families():
    import hk.schedules as S

    return (S.Rdna, S.Decode, S.PingPong, S.Dp)


def draw(cls, arch, n, rng, seeds=False):
    """n structurally legal, distinct points of cls's full value product."""
    import hk.schedules as S

    got, seen = [], set()
    if seeds:
        for t in (1, 2, 4) if cls is S.Decode else (None,):
            base = cls() if t is None else cls(tokens=t)
            for s in cls.space(1, 1, 1, arch, base=base, legal=False):
                if s not in seen and not s._check_struct(arch):
                    seen.add(s)
                    got.append(s)
    target, tries = len(got) + n, 0
    ax = cls.axes()
    while len(got) < target and tries < 400000:
        tries += 1
        s = cls(**{a.name: rng.choice(a.values) for a in ax.values() if a.values}).canonical()
        if s in seen:
            continue
        seen.add(s)
        if not s._check_struct(arch):
            got.append(s)
    return got


def sample(args) -> int:
    from hk.runtime.warm import warm_cache

    rng = random.Random(args.seed)
    points = []
    for cls in families():
        for arch in cls.arches:
            if args.families and cls.family not in args.families:
                continue
            if args.arches and arch not in args.arches:
                continue
            pts = draw(cls, arch, args.n, rng, args.seeds) if args.n else []
            if args.around_shipped:
                have, near = set(pts), []
                for s, a in shipped():
                    if a != arch or type(s) is not cls:
                        continue
                    for t in s.neighbors(args.around_shipped):
                        t = t.canonical()
                        if t not in have and not t._check_struct(arch):
                            have.add(t)
                            near.append(t)
                if args.max_around and len(near) > args.max_around:
                    near = rng.sample(near, args.max_around)
                pts += near
            print(f"{cls.family}@{arch}: {len(pts)} points", flush=True)
            points += [(cls.family, arch, s) for s in pts]
    jobs, meta = [], {}
    for i, (fam, arch, s) in enumerate(points):
        meta[str(i)] = dict(family=fam, arch=arch, rec=s.to_record(full=True))
        try:
            jobs.append((str(i), s._make(None, arch), {}))
        except Exception as e:  # noqa: BLE001 -- a factory's refusal is a result
            meta[str(i)].update(status="factory", reason=f"{type(e).__name__}: {e}")
    t0 = time.time()
    res = warm_cache(jobs, workers=args.workers)
    print(f"built {len(res.built)}, refused {len(res.errors)}, {time.time() - t0:.0f} s",
          flush=True)
    for label, b in res.built.items():
        r = b.kernels[0]
        meta[label].update(status="ok", vgpr=r.vgpr, agpr=r.agpr, sgpr=r.sgpr, lds=r.lds,
                           occ=r.occ_by_reg or r.occupancy, scratch=r.scratch)
    for label, e in res.errors.items():
        msg = str(e)
        m = RES_RE.search(msg)
        sp = SGPR_SPILL_RE.search(msg)
        meta[label].update(status=type(e).__name__, reason=msg.strip()[:400],
                           vgpr=int(m.group(1)) if m else None,
                           agpr=int(m.group(2)) if m and m.group(2) else 0,
                           scratch=int(m.group(3)) if m else None,
                           spilled=int(m.group(4)) if m else None,
                           sgpr_spill=int(sp.group(2)) if sp else 0)
    Path(args.out).write_text(json.dumps(list(meta.values()), indent=0) + "\n")
    print(f"wrote {args.out}")
    return 0


def sample_attention(args) -> int:
    """sample, for the attention families: every point of each family's
    full value product (they are small), at head_dim 64 and 128, causal or
    not, each structurally legal one built."""
    import itertools

    import hk.schedules as S
    from hk.runtime.warm import warm_cache

    jobs, meta = [], {}
    for cls in (S.AttnRdna, S.AttnCdna):
        for arch in cls.arches:
            ax = [a for a in cls.axes().values() if a.values]
            for combo in itertools.product(*(a.values for a in ax)):
                s = cls(**dict(zip((a.name for a in ax), combo)))
                for d, causal in itertools.product((64, 128), (False, True)):
                    if s._check_struct(arch, d) and not args.all:
                        continue
                    label = str(len(meta))
                    meta[label] = dict(family=cls.family, arch=arch, head_dim=d,
                                       causal=causal, rec=s.to_record(full=True))
                    try:
                        jobs.append((label, s._make(d, causal, None, arch), {}))
                    except Exception as e:  # noqa: BLE001
                        meta[label].update(status="factory", reason=f"{type(e).__name__}: {e}")
    print(f"{len(jobs)} attention kernels to build", flush=True)
    t0 = time.time()
    res = warm_cache(jobs, workers=args.workers)
    print(f"built {len(res.built)}, refused {len(res.errors)}, {time.time() - t0:.0f} s",
          flush=True)
    for label, b in res.built.items():
        r = b.kernels[0]
        meta[label].update(status="ok", vgpr=r.vgpr, agpr=r.agpr, sgpr=r.sgpr,
                           occ=r.occ_by_reg or r.occupancy, scratch=r.scratch)
    for label, e in res.errors.items():
        m = RES_RE.search(str(e))
        meta[label].update(status=type(e).__name__, reason=str(e).strip()[:400],
                           vgpr=int(m.group(1)) if m else None)
    Path(args.out).write_text(json.dumps(list(meta.values()), indent=0) + "\n")
    print(f"wrote {args.out}")
    return 0


def _load(paths):
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())
    return rows


def evaluate(args) -> int:
    import hk.schedules as S

    st = defaultdict(lambda: defaultdict(int))
    fa = []
    for p in _load(args.records):
        s = S.from_record(p["rec"])
        key = f"{p['family']}@{p['arch']}"
        if "head_dim" in p:
            key += f" d{p['head_dim']}{' causal' if p['causal'] else ''}"
            why = s.check_kernel(p["arch"], head_dim=p["head_dim"], causal=p["causal"])
        else:
            why = s.check_kernel(p["arch"])
        built = p["status"] == "ok"
        st[key]["n"] += 1
        if not why:
            st[key]["accepted"] += 1
            if not built:
                st[key]["false_accept"] += 1
                fa.append((key, str(s), p["status"], p.get("reason", "")[:200]))
        else:
            st[key]["refused"] += 1
            st[key]["false_reject"] += built
            st[key]["no_reason"] += not all(isinstance(r, str) and r for r in why)
    tot = defaultdict(int)
    for k, v in sorted(st.items()):
        print(f"{k:16} sampled {v['n']:5}  accepted {v['accepted']:5}  false accepts "
              f"{v['false_accept']:3} ({100 * v['false_accept'] / max(v['accepted'], 1):.2f}%)"
              f"  refused {v['refused']:5} of which built {v['false_reject']:4}"
              f"  without a reason {v['no_reason']}")
        for kk, vv in v.items():
            tot[kk] += vv
    print(f"{'all':16} sampled {tot['n']:5}  accepted {tot['accepted']:5}  false accepts "
          f"{tot['false_accept']:3} ({100 * tot['false_accept'] / max(tot['accepted'], 1):.2f}%)"
          f"  refused {tot['refused']:5} of which built {tot['false_reject']:4}"
          f"  without a reason {tot['no_reason']}")
    for x in fa:
        print("  false accept:", *x)
    return 0


def shipped():
    """Every kernel schedule hk ships (routes, candidates, records, decode)."""
    import hk.schedules as S
    from hk.ops import decode_gemm, gemm_cdna
    from hk.ops import gemm as rdna

    nk = [(4096, 4096), (6144, 4096), (12288, 4096), (14336, 4096), (24576, 4096),
          (28672, 4096), (4096, 12288), (4096, 14336), (128256, 4096), (151936, 4096)]
    ms = [1, 4, 16, 32, 64, 77, 128, 256, 300, 512, 1024, 2048, 4096, 8192]
    out = set()

    def walk(s, arch):
        if isinstance(s, S.Columns):
            walk(s.head, arch)
            walk(s.tail, arch)
        else:
            out.add((s, arch))

    for arch in ("gfx1100", "gfx1201", "gfx950"):
        for n, k in nk:
            for m in ms:
                for op in ("matmul", "template"):
                    try:
                        walk(S.lookup(m, n, k, arch, op=op), arch)
                    except (ValueError, KeyError):
                        pass
    for name in gemm_cdna.CANDIDATES:
        for m in (16, 256, 2048):
            for n, k in nk:
                if gemm_cdna.fits(name, m, n, k):
                    walk(S.from_candidate(name, m, n, k), "gfx950")
    for sch in [rdna.DEFAULT, rdna.THIN] + rdna.SPACE:
        for arch in ("gfx1100", "gfx1201"):
            out.add((S.Rdna(**sch), arch))
    for t, sc in decode_gemm.SCHEDULES.items():
        for arch in ("gfx1100", "gfx1201", "gfx950"):
            out.add((S.Decode(tokens=t, **sc), arch))
    return out


def fit(args) -> int:
    import numpy as np
    from scipy.optimize import linprog

    import hk.schedules as S
    from hk.schedules import _regs
    from hk.target import get_target

    def demand(p):
        if p["arch"] == "gfx950":
            return -(-(p.get("vgpr") or 0) // 4) * 4 + (p.get("agpr") or 0)
        return p.get("vgpr") or 0

    by = defaultdict(list)
    for p in _load(args.records):
        if p["status"] not in ("ok", "ResourceError"):
            continue
        s = S.from_record(p["rec"])
        arch = p["arch"]
        if s._check_struct(arch) or _regs.lds(s, arch) > get_target(arch).lds_per_workgroup:
            continue
        spill = p["status"] == "ResourceError"
        by[(p["family"], arch)].append((_regs.features(s, arch), demand(p), spill,
                                        _regs.limit(s, arch)))
    must = defaultdict(list)
    for s, arch in shipped():
        must[(s.family, arch)].append(s)
    model = {}
    for key, rows in sorted(by.items()):
        names = sorted({n for f, *_ in rows for n in f})

        def vec(f):
            return [f.get(n, 0.0) for n in names] + [1.0]
        ok = [r for r in rows if not r[2]]
        sp = [r for r in rows if r[2]]
        X = np.array([vec(r[0]) for r in ok])
        y = np.array([r[1] for r in ok], dtype=float)
        A, b = [-X], [-y]
        if sp:
            A.append(-np.array([vec(r[0]) for r in sp]))
            b.append(-np.array([r[3] + 1.0 for r in sp]))
        if must[key]:
            A.append(np.array([vec(_regs.features(s, key[1])) for s in must[key]]))
            b.append(np.array([_regs.limit(s, key[1]) - 0.5 for s in must[key]]))
        r = linprog(X.sum(axis=0), A_ub=np.vstack(A), b_ub=np.concatenate(b),
                    bounds=[(None, None)] * X.shape[1], method="highs")
        if not r.success and must[key]:
            # A schedule that ships at the very edge (gfx1201's 4-token decode GEMV builds at
            # 255 of 256) next to a point with its features that built at 256: hold the
            # shipped ones to the limit itself rather than half a register under it, and
            # take the solver's rounding off the intercept, so ceil() still says the limit.
            Mf = np.array([vec(_regs.features(s, key[1])) for s in must[key]])
            lim = np.array([_regs.limit(s, key[1]) for s in must[key]], dtype=float)
            b[-1] = lim
            r = linprog(X.sum(axis=0), A_ub=np.vstack(A), b_ub=np.concatenate(b),
                        bounds=[(None, None)] * X.shape[1], method="highs")
            if r.success:
                over = float((Mf @ r.x - lim).max())
                if over > -1e-6:
                    r.x[-1] -= over + 1e-6
                print(f"{key}: shipped schedules held to the limit, not half a register under")
        if not r.success:
            # Built points at the very edge of the register file can share their features
            # with a spill, or with a schedule that ships (a knob the features do not see),
            # and make the hard problem infeasible. Then the spill and the shipped-schedule
            # constraints each get a slack, priced far above any register -- a shipped
            # schedule's a thousand times a spill's, so a spill is given up first -- and the
            # build's spill gate refuses what the model misses.
            soft = [np.array([vec(r_[0]) for r_ in sp]) * -1.0] if sp else []
            soft_b = [-np.array([r_[3] + 1.0 for r_ in sp])] if sp else []
            if must[key]:
                soft.append(np.array([vec(_regs.features(s, key[1])) for s in must[key]]))
                soft_b.append(np.array([float(_regs.limit(s, key[1])) for s in must[key]]))
            S_ = np.vstack(soft)
            ns, nv = S_.shape[0], X.shape[1]
            A2 = np.vstack([np.hstack([-X, np.zeros((len(ok), ns))]),
                            np.hstack([S_, -np.eye(ns)])])
            b2 = np.concatenate([-y] + soft_b)
            cost = np.concatenate([X.sum(axis=0), np.full(len(sp), 1e4),
                                   np.full(ns - len(sp), 1e7)])
            r = linprog(cost, A_ub=A2, b_ub=b2,
                        bounds=[(None, None)] * nv + [(0, None)] * ns, method="highs")
            if r.success:
                given = r.x[nv:] > 1e-6
                print(f"{key}: infeasible as stated; {int(given[:len(sp)].sum())} of {len(sp)} "
                      f"spills and {int(given[len(sp):].sum())} of {len(must[key])} shipped "
                      f"schedules given up")
                r.x = r.x[:nv]
                if must[key]:
                    Mf = np.array([vec(_regs.features(s, key[1])) for s in must[key]])
                    lim = np.array([_regs.limit(s, key[1]) for s in must[key]], dtype=float)
                    over = Mf @ r.x - lim
                    over = float(over[over < 1.0].max()) if (over < 1.0).any() else -1.0
                    if over > -1e-6:          # the solver's rounding, not a schedule given up
                        r.x[-1] -= over + 1e-6
        if not r.success:
            print(f"{key}: {r.message}; MODEL left as it is")
            continue
        c = {n: round(float(v), 4) for n, v in zip(names + ["1"], r.x)}
        if must[key]:
            # The rounding to four places can put a shipped schedule held to the limit a
            # hair over it, which ceil() turns into a register: take it off the intercept.
            cv = np.array([c[n] for n in names + ["1"]])
            Mf = np.array([vec(_regs.features(s, key[1])) for s in must[key]])
            lim = np.array([_regs.limit(s, key[1]) for s in must[key]], dtype=float)
            over = Mf @ cv - lim
            near = over[(over > 0) & (over < 1.0)]
            if near.size:
                c["1"] = float(np.floor((c["1"] - near.max()) * 1e4) / 1e4)
        fr = sum(1 for f, _, _, lim in ok if np.dot(vec(f), r.x) > lim)
        print(f"{key[0]}@{key[1]}: {len(ok)} built, {len(sp)} spilled; refuses {fr} "
              f"that built")
        model[f"{key[0]}@{key[1]}"] = c
    print(json.dumps(model))
    if args.write:
        p = REPO / "python" / "hk" / "schedules" / "_regs.py"
        src = p.read_text()
        lines = ["MODEL: Dict[str, Dict[str, Any]] = {"]
        for k, c in sorted(model.items()):
            lines.append(f'    "{k}": {{"offset": MARGIN, "coef": {{')
            row = "        "
            for it in [f'"{n}": {v}' for n, v in c.items()]:
                if len(row) + len(it) + 2 > 92:
                    lines.append(row.rstrip())
                    row = "        "
                row += it + ", "
            lines.append(row.rstrip().rstrip(",") + "}},")
        lines.append("}")
        a = src.index("MODEL: Dict[str, Dict[str, Any]] = {")
        z = src.index("\n}\n", a) + 3
        p.write_text(src[:a] + "\n".join(lines) + "\n" + src[z:])
        print(f"wrote {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=300, help="random points per family and arch")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--seeds", action="store_true", help="add the seed spaces")
    a.add_argument("--around-shipped", type=int, default=0, metavar="R",
                   help="add every shipped schedule's neighbours within R knob changes")
    a.add_argument("--max-around", type=int, default=0, metavar="K",
                   help="with --around-shipped, a random K of them per family and arch")
    a.add_argument("--families", nargs="*")
    a.add_argument("--arches", nargs="*", help="only these archs (default: every arch a family runs on)")
    a.add_argument("--workers", type=int, default=48)
    a = sub.add_parser("sample-attn")
    a.add_argument("--out", required=True)
    a.add_argument("--all", action="store_true", help="build the refused points too")
    a.add_argument("--workers", type=int, default=48)
    a = sub.add_parser("eval")
    a.add_argument("records", nargs="+")
    a = sub.add_parser("fit")
    a.add_argument("records", nargs="+")
    a.add_argument("--write", action="store_true")
    args = ap.parse_args()
    return {"sample": sample, "sample-attn": sample_attention, "eval": evaluate,
            "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
