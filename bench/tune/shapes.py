"""The hidden-shape set the tuner is judged on, per GPU, and how it was drawn.

    python3 -m bench.tune.shapes --write        # regenerate shapes_<arch>.json (deterministic)
    python3 -m bench.tune.shapes --show gfx950  # the splits, as a table

**What a shape is.** A bf16 GEMM C = A @ B.T, A (M, K), B (N, K) -- F.linear's
layout and hk.ops.matmul's -- from the linear layers of a model hk has not
tuned, sharded as vLLM shards them at a tensor-parallel size such a model is
served at. The configs below are the public ones (config.json on the Hub); the
pods have no network, so they are written out here.

**What is left out, and why.**
  * Any (N, K) a shipped table holds -- python/hk/tuned/*.json, every route
    table and record, for either GPU. Such a shape is not "unseen": its M grid
    was measured and routed. (Ms of an unseen (N, K) are all unseen: a record
    of one M is not used at another, see hk.tune.records.)
  * An (N, K) hk's GEMMs cannot take at all: N a multiple of 128 and K of 64
    on both GPUs (hk.ops.gemm_cdna.matmul's contract, and the RDNA default
    tiling's). These are listed in the file with the reason -- Qwen2.5-72B at
    TP 8 (N = 7392) is one -- because "hk has no kernel" is a result, but a
    tuner cannot be judged on them.
  * Calls longer than the arch's FLOP cap (a few ms of the fastest kernel), so
    that a tuning budget of minutes per shape means something.

**The splits.** Shapes are grouped by (N, K), the groups shuffled with SEED,
and dealt alternately to "tuning" and "acceptance", so every M of an (N, K)
lands in the same split: the tuner is developed on the first and judged on the
second, and nothing learned about one (N, K) at one M is judged at another.
Within a split, Ms are drawn per class (decode 1-16, 128-512, 1024-2048,
4096-8192) so that each class is represented. 2048x16384x4096 is named by the
goal (docs/design/tuning-goal.md, T3) and goes into gfx950's acceptance split
by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
TUNED = HERE.parents[1] / "python" / "hk" / "tuned"
SEED = 20261022
ARCHES = ("gfx950", "gfx1100")

#: hidden, heads, kv heads, head dim, intermediate (per expert for MoE), TP sizes.
MODELS: Dict[str, dict] = {
    "meta-llama/Meta-Llama-3-70B": dict(hidden=8192, heads=64, kv=8, hd=128, inter=28672, tp=(4, 8)),
    "mistralai/Mixtral-8x7B-v0.1": dict(hidden=4096, heads=32, kv=8, hd=128, inter=14336, tp=(2, 4),
                                        note="experts as dense GEMMs, M = tokens an expert gets"),
    "Qwen/Qwen2.5-32B": dict(hidden=5120, heads=40, kv=8, hd=128, inter=27648, tp=(1, 2, 4)),
    "Qwen/Qwen2.5-72B": dict(hidden=8192, heads=64, kv=8, hd=128, inter=29568, tp=(4, 8)),
    "google/gemma-2-27b": dict(hidden=4608, heads=32, kv=16, hd=128, inter=36864, tp=(1, 2, 4)),
    "microsoft/Phi-3-medium-4k-instruct": dict(hidden=5120, heads=40, kv=10, hd=128, inter=17920, tp=(1, 2)),
    "microsoft/Phi-3-mini-4k-instruct": dict(hidden=3072, heads=32, kv=32, hd=96, inter=8192, tp=(1,)),
    "mistralai/Mistral-Nemo-Base-2407": dict(hidden=5120, heads=32, kv=8, hd=128, inter=14336, tp=(1, 2)),
    "meta-llama/Llama-3.2-3B": dict(hidden=3072, heads=24, kv=8, hd=128, inter=8192, tp=(1,)),
}

#: DeepSeek-V2-Lite's dense layers (MLA attention, the first layer's dense MLP
#: and the shared experts), TP 1: (name, N, K).
DEEPSEEK_V2_LITE = [("q_proj", 3072, 2048), ("kv_b_proj", 4096, 512), ("o_proj", 2048, 2048),
                    ("kv_a_proj", 576, 2048), ("dense_gate_up", 21888, 2048),
                    ("dense_down", 2048, 10944), ("shared_gate_up", 5632, 2048),
                    ("shared_down", 2048, 2816)]

M_CLASSES = {"decode": (1, 8, 16), "small": (128, 512), "mid": (1024, 2048),
             "large": (4096, 8192)}

#: Most FLOPs a call may have, per arch: about 10 ms of the fastest kernel.
FLOP_CAP = {"gfx950": 1.6e13, "gfx1100": 1.0e12}
#: Shapes per split, per arch.
PER_SPLIT = 24
#: Named by the goal; forced into the acceptance split.
NAMED = {"gfx950": [(2048, 16384, 4096, "named target (tuning-goal T3): route() rule, 0.76x hipBLASLt")]}


def linear_shapes() -> List[Tuple[str, str, int, int]]:
    """(model, layer@tp, N, K) for every model and TP size."""
    out = []
    for name, c in MODELS.items():
        for tp in c["tp"]:
            hd = c["hd"]
            out += [(name, f"qkv@tp{tp}", (c["heads"] + 2 * c["kv"]) * hd // tp, c["hidden"]),
                    (name, f"o@tp{tp}", c["hidden"], c["heads"] * hd // tp),
                    (name, f"gate_up@tp{tp}", 2 * c["inter"] // tp, c["hidden"]),
                    (name, f"down@tp{tp}", c["hidden"], c["inter"] // tp)]
    out += [("deepseek-ai/DeepSeek-V2-Lite", f"{lay}@tp1", n, k) for lay, n, k in DEEPSEEK_V2_LITE]
    return out


def shipped_nk() -> Dict[Tuple[int, int], List[str]]:
    """{(N, K): [files]} for every (N, K) any shipped table or record holds."""
    held: Dict[Tuple[int, int], List[str]] = {}
    for p in sorted(TUNED.glob("*.json")):
        d = json.loads(p.read_text())
        keys = list(d.get("routes", {})) + list(d.get("shapes", {}))
        for key in d if not keys and isinstance(d, dict) else []:   # schedule records: MxNxK
            parts = key.split("x")
            if len(parts) == 3 and all(x.isdigit() for x in parts):
                keys.append(f"{parts[1]}x{parts[2]}")
        for key in keys:
            n, k = (int(x) for x in key.split("x"))
            held.setdefault((n, k), []).append(p.name)
    return held


def unsupported(n: int, k: int) -> str:
    if n % 128:
        return f"N={n} is not a multiple of 128"
    if k % 64:
        return f"K={k} is not a multiple of 64"
    return ""


def draw(arch: str, seed: int = SEED) -> dict:
    held = shipped_nk()
    groups: Dict[Tuple[int, int], List[str]] = {}
    excluded = []
    for model, layer, n, k in linear_shapes():
        who = f"{model} {layer}"
        if (n, k) in held:
            excluded.append(dict(n=n, k=k, source=who, why=f"held by {', '.join(held[(n, k)])}"))
            continue
        why = unsupported(n, k)
        if why:
            excluded.append(dict(n=n, k=k, source=who, why=why))
            continue
        groups.setdefault((n, k), []).append(who)
    for n, k, _ in [(n, k, w) for _, n, k, w in NAMED.get(arch, [])]:
        groups.pop((n, k), None)
    rng = random.Random(f"{seed}-{arch}")
    keys = sorted(groups)
    rng.shuffle(keys)
    split_of = {nk: ("tuning", "acceptance")[i % 2] for i, nk in enumerate(keys)}
    out = {"tuning": [], "acceptance": []}
    for split in out:
        nks = [nk for nk in keys if split_of[nk] == split]
        # Per M class, an equal share of the split, each (N, K) at most once a
        # class; drawn in the shuffled order with a random M of the class.
        per_class = PER_SPLIT // len(M_CLASSES)
        for cls, ms in M_CLASSES.items():
            got = 0
            order = nks[:]
            rng.shuffle(order)
            for n, k in order:
                if got == per_class:
                    break
                cands = [m for m in ms if 2.0 * m * n * k <= FLOP_CAP[arch]]
                if not cands:
                    continue
                m = rng.choice(cands)
                out[split].append(dict(m=m, n=n, k=k, cls=cls, source=groups[(n, k)]))
                got += 1
    for m, n, k, why in NAMED.get(arch, []):
        out["acceptance"].append(dict(m=m, n=n, k=k, cls="mid", source=[why], named=True))
    for split in out:
        out[split].sort(key=lambda s: (s["m"], s["n"], s["k"]))
    return dict(arch=arch, seed=seed, per_split=PER_SPLIT, flop_cap=FLOP_CAP[arch],
                m_classes=M_CLASSES, by="bench/tune/shapes.py", splits=out,
                excluded=sorted(excluded, key=lambda e: (e["n"], e["k"], e["source"])))


def path_for(arch: str) -> Path:
    return HERE / f"shapes_{arch}.json"


def load(arch: str, split: str = "acceptance") -> List[Tuple[int, int, int]]:
    """[(M, N, K)] of a split, from the committed file."""
    d = json.loads(path_for(arch).read_text())
    return [(s["m"], s["n"], s["k"]) for s in d["splits"][split]]


def dump(d: dict) -> str:
    head = {k: v for k, v in d.items() if k not in ("splits", "excluded")}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append(' "splits": {')
    for i, (split, shapes) in enumerate(d["splits"].items()):
        lines.append(f"  {json.dumps(split)}: [")
        lines += [f"   {json.dumps(s)}," for s in shapes]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("  ]" + ("," if i == 0 else ""))
    lines.append(" },")
    lines.append(' "excluded": [')
    lines += [f"  {json.dumps(e)}," for e in d["excluded"]]
    lines[-1] = lines[-1].rstrip(",")
    lines += [" ]", "}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="regenerate shapes_<arch>.json")
    ap.add_argument("--show", default="", help="print an arch's splits")
    args = ap.parse_args(argv)
    if args.write:
        for arch in ARCHES:
            path_for(arch).write_text(dump(draw(arch)))
            print(f"wrote {path_for(arch)}")
    if args.show:
        d = json.loads(path_for(args.show).read_text())
        for split, shapes in d["splits"].items():
            print(f"{split} ({len(shapes)}):")
            for s in shapes:
                print(f"  {s['m']:>5} x {s['n']:>6} x {s['k']:>6}  {s['cls']:<7} {'; '.join(s['source'])}")
        print(f"excluded: {len(d['excluded'])} (N, K)s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
