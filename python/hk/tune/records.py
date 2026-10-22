"""The records `hk.tune` writes, and the lookup hk.ops.matmul and
hk.templates.gemm make in front of their own routes.

One file per arch, `$HK_TUNE_DIR/routes_gemm_<arch>.json` (`hk.autotune.
records_dir`, beside the compile cache), in the format of the shipped gfx950
table (`python/hk/tuned/routes_gemm_gfx950.json`):

    {"arch": "gfx950", "kind": "gemm",
     "candidates": {"u:<label>": [kind, schedule], ...},
     "routes": {"<N>x<K>": {"<M>": "u:<label>", ...}, ...},
     "points": [{"m":..,"n":..,"k":.., "candidate":.., "verdict": "win", "gain":..,
                 "band": [lo, hi], "route": .., "route_ms":.., "ms":.., ...}, ...]}

`candidates` are what `hk.ops.gemm_cdna.CANDIDATES` holds on gfx950 -- a
kernel kind ("pp", "dp", "decode", "ppw"; on gfx1100 "rdna", "decode") and
the schedule the module that owns it spells -- so a record is run by the same
code a shipped route is.

**Which M a record answers.** A tuned shape answers the Ms of its power-of-two
bucket (`bucket`, as hk.ops.gemm's tuning key does), and only those: the
schedule that wins at 2048 rows says nothing about 16. Outside its bucket the
shipped route (or its rule) decides, as before.

**Only wins are written.** `hk.tune` writes a route only where its candidate
beat the current route by its acceptance rule (hk.tune.stats.decide), so a
record is never a tie; the measurement is kept beside it (`points`).
`HK_AUTOTUNE=0` ignores the records, as it ignores every tuning record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

__all__ = ["path", "load", "user_route", "save", "bucket", "forget"]

_CACHE: Dict[str, Tuple[Tuple[float, int], dict]] = {}


def bucket(m: int) -> int:
    """M rounded up to a power of two: the Ms one record answers."""
    b = 1
    while b < m:
        b *= 2
    return b


def path(arch: str) -> Path:
    from ..autotune import records_dir  # noqa: PLC0415

    return records_dir() / f"routes_gemm_{arch}.json"


def _empty(arch: str) -> dict:
    return {"arch": arch, "kind": "gemm", "candidates": {}, "routes": {}, "points": []}


def load(arch: str) -> dict:
    """This machine's records for `arch` (an empty table if none). Read once
    per process and again only when the file changes or a tuner wrote."""
    from ..autotune import generation  # noqa: PLC0415

    try:
        p = path(arch)
        st = p.stat()
        stamp = (st.st_mtime, generation())
    except OSError:
        return _empty(arch)
    hit = _CACHE.get(arch)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        d = json.loads(p.read_text())
        if not isinstance(d, dict) or not isinstance(d.get("routes"), dict):
            d = _empty(arch)
    except (OSError, ValueError):
        # A corrupt record costs speed, never a launch.
        d = _empty(arch)
    _CACHE[arch] = (stamp, d)
    return d


def user_route(arch: str, m: int, n: int, k: int) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """(name, kind, schedule) of this machine's tuned candidate for the shape,
    or None: no record for (N, K) in M's bucket, or HK_AUTOTUNE=0."""
    from ..autotune import enabled  # noqa: PLC0415

    if not enabled() or os.environ.get("HK_TUNE_RECORDS", "1") in ("0", "off", "no"):
        return None
    d = load(arch)
    by_m = d.get("routes", {}).get(f"{n}x{k}")
    if not by_m:
        return None
    b = bucket(m)
    for ms, name in by_m.items():
        if bucket(int(ms)) == b:
            spec = d.get("candidates", {}).get(name)
            if spec:
                return name, spec[0], dict(spec[1])
    return None


def save(arch: str, m: int, n: int, k: int, name: str, spec, point: Dict[str, Any]) -> Path:
    """Write one tuned shape: its candidate, its route and its measurement.
    Atomic; replaces an earlier route for the same (N, K) and M bucket."""
    from ..autotune import forget_records  # noqa: PLC0415

    p = path(arch)
    try:
        cur = json.loads(p.read_text())
        if not isinstance(cur, dict) or "routes" not in cur:
            cur = _empty(arch)
    except (OSError, ValueError):
        cur = _empty(arch)
    cur.setdefault("candidates", {})[name] = [spec[0], dict(spec[1])]
    by_m = cur.setdefault("routes", {}).setdefault(f"{n}x{k}", {})
    for ms in [x for x in by_m if bucket(int(x)) == bucket(m)]:
        del by_m[ms]
    by_m[str(m)] = name
    pts = [q for q in cur.setdefault("points", [])
           if not (q.get("n") == n and q.get("k") == k and bucket(q.get("m", 0)) == bucket(m))]
    cur["points"] = pts + [point]
    used = {nm for r in cur["routes"].values() for nm in r.values()}
    cur["candidates"] = {nm: s for nm, s in cur["candidates"].items() if nm in used}
    tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
    tmp.replace(p)
    forget()
    forget_records()
    return p


def forget() -> None:
    _CACHE.clear()
