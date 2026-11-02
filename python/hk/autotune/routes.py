"""Which kernel each linear shape should run on, at every M a step can have --
measured, and kept as records the routing tables read.

This is the measurement `tools/hk-bench/route_tune.py` made, as a library:
`tune_shape(n, k, ...)` measures one (N, K) and returns its table entry, and
the CLI, `python -m hk.integration tune` and the vLLM patch's tuning at load
(`hk.integration.vllm_autotune`) are thin shells over it.

**Why a grid, not a rule.** The GEMM's speed is not monotone in M: the last
128-row block is computed whole however little of it is kept, so between
multiples of 128 the kernel wastes work in a sawtooth, and at small grids two
workgroups share a WGP. A rule written as thresholds, or bucketed by powers of
two, either routes the teeth or gives up the gaps between them. So the grid is
every M up to 64, every 16 rows to 1024, coarser above, where the wasted rows
are at most an eighth of the work.

**What a point is.** At each M: the engine's own path (the installed vLLM's
`rocm_unquantized_gemm`, or `F.linear`; SGLang's linear reaches `F.linear` on
ROCm), the generated GEMM, the decode GEMV where it takes the shape, and
optionally FlyDSL's GEMM or HipKittens' CDNA4 GEMM. Each is checked against
fp32 before it is timed; they interleave in one process, the weights rotated
through more than three times the Infinity Cache. A kernel takes an M only
if it beat the engine there by `min_gain`; a tie stays the engine's.

**How a point is timed** (docs/history/32). A set is `rounds` rounds of every
path on activations of its own; a path's time in a set is its fastest round.
The first `warmup` sets are thrown away (straight after the correctness check
FlyDSL ran a median 1.5% slow). Then `sets` sets are timed, and a point with a
decision within `close_k` x its noise of the bar gets `remeasure` sets more;
its time is then the median of the sets. Each path's noise -- its rounds'
spread in a set, or the sets' about their median, whichever is larger -- is
written beside its time: a point is `[M, times, tiles, noise, sets]`.

**Records.** A shape measured on this machine is a user record, in
`records_dir()/routes_<kind>_<arch>.json` (`hk.autotune.records_dir`), in the
format of a shipped table, each entry with its own measurement's metadata.
`hk.integration.engine.shipped_table` overlays it on the shipped table, user
entries winning per (N, K) -- whoever measured this machine measured it --
and `HK_AUTOTUNE=0` ignores them, as it ignores schedule records. Writers
take `records_lock`, so tensor-parallel ranks, or two servers, tune a shape
once and read each other's results.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

GRID = (list(range(1, 65)) + list(range(80, 1025, 16)) +
        list(range(1088, 2049, 64)) + list(range(2176, 4097, 128)) +
        list(range(4608, 8193, 512)) + [12288, 16384])

#: The Ms the decode GEMV can take (16 x decode_gemm.MAX_TOKENS rows), which
#: is where a decode step's linear layers are and where hk's wins are largest.
DECODE_MS = tuple(m for m in GRID if m <= 64)

#: Candidates from libraries hk does not depend on: timed, and routed by an
#: overlay rather than the runs (engine.RouteTable).
OPTIONAL = ("flydsl",)
#: A decision within CLOSE_K x its pair's measured noise of its bar is
#: re-measured on fresh activations before it is taken.
CLOSE_K = 2.0
#: A margin must clear its gain and NOISE_K x the pair's noise. Off: at 1 and
#: 2 it flipped as many points between runs as the bare gains, and at 2 it
#: dropped FlyDSL rows that win end to end (docs/history/32).
NOISE_K = 0.0
#: What a kernel must beat the engine by to take an M.
MIN_GAIN = 0.03
#: What an optional path must beat the runs' own answer by, on top.
OPT_GAIN = 0.02
#: Weight bytes one timed pass streams: three times a W7900's 96 MB
#: Infinity Cache. An MI355X's is 256 MB.
ROTATE_BYTES = 3 * 96 * 2**20
#: The clock `torch.cuda._sleep`'s cycles are counted at, high enough that a
#: queued burst's spin outlasts its enqueue on any card here.
SPIN_HZ = 2.4e9
#: vLLM computes logits for the last token of each sequence only, so the LM
#: head's M is a sequence count, never a prefill's token count.
MAX_SEQS = 256
#: The paths a table's runs may name, of hk's and HipKittens' kernels.
PATHS = ("hk", "hk_decode", "hipkittens")


# ------------------------------------------------------------------ decisions


def _noise(point):
    """{path: relative noise} of a measured point; {} for tables from before
    route_tune measured it, whose decisions are then the bare gains'."""
    return point[3] if len(point) > 3 else {}


def _tiles(point):
    return point[2] if len(point) > 2 else {}


def margin(t, noise, fast, slow, gain, k=NOISE_K):
    """(how much faster `fast` is than `slow`, the bar it must clear, the
    pair's noise): the bar is the gain or k x noise, whichever is larger."""
    sigma = noise.get(fast, 0.0) + noise.get(slow, 0.0)
    return t[slow] / t[fast] - 1, max(gain, k * sigma), sigma


def base_path(t, min_gain, noise=None, k=NOISE_K):
    """The runs' answer at one measured M: the fastest of hk's kernels if it
    beat the engine by `min_gain` and by k x their noise, else the engine's."""
    cand = {p: v for p, v in t.items() if p != "engine" and p not in OPTIONAL}
    best = min(cand, key=cand.get) if cand else None
    if not best:
        return "fallback"
    r, bar, _ = margin(t, noise or {}, best, "engine", min_gain, k)
    return best if r >= bar else "fallback"


def optional_wins(t, noise, name, min_gain, opt_gain, k=NOISE_K):
    """Whether optional path `name` beat the engine by `min_gain` and the
    runs' answer by `opt_gain`, each by k x the pair's noise as well."""
    base = base_path(t, min_gain, noise, k)
    base = "engine" if base == "fallback" else base
    r_e, bar_e, _ = margin(t, noise, name, "engine", min_gain, k)
    r_b, bar_b, _ = margin(t, noise, name, base, opt_gain, k)
    return r_e >= bar_e and r_b >= bar_b


def close_call(t, noise, min_gain, opt_gain, k=CLOSE_K, noise_k=NOISE_K):
    """True if a decision at this point would go the other way were one of
    its margins k x its noise away: hk's best path against the engine, hk's
    two paths against each other, an optional path against both (each
    against the bar `noise_k` sets)."""
    cand = sorted((v, p) for p, v in t.items() if p != "engine" and p not in OPTIONAL)
    pairs = []
    if cand:
        pairs.append((cand[0][1], "engine", min_gain))
    if len(cand) > 1:
        pairs.append((cand[0][1], cand[1][1], None))      # which of hk's two
    base = base_path(t, min_gain, noise, noise_k)
    for name in OPTIONAL:
        if name in t:
            pairs += [(name, "engine", min_gain),
                      (name, "engine" if base == "fallback" else base, opt_gain)]
    for fast, slow, gain in pairs:
        r, bar, sigma = margin(t, noise, fast, slow, gain or 0.0, noise_k)
        if (r <= k * sigma) if gain is None else (abs(r - bar) <= k * sigma):
            return True
    return False


def estimate(sets):
    """(times, noise) of one point from its sets of rounds, each set on
    activations of its own: a path's time is the median over the sets of its
    fastest round in each, and its noise the larger of the rounds' spread
    within a set (the median set's: its median round over its fastest) and
    the sets' spread about their median (half their range)."""
    times, noise = {}, {}
    for p in sets[0]:
        mins = [min(s[p]) for s in sets]
        t = statistics.median(mins)
        within = statistics.median(statistics.median(s[p]) / min(s[p]) - 1 for s in sets)
        across = (max(mins) - min(mins)) / 2 / t if len(sets) > 1 else 0.0
        times[p], noise[p] = t, max(within, across)
    return times, noise


def decisions(table):
    """{(shape, M): (the runs' path, whether FlyDSL's overlay takes it)} for
    every measured point of a table, by the table's own rule. A point where
    FlyDSL was not timed is one it does not take."""
    k = table.get("noise_k", 0.0)
    out = {}
    for key, s in table["shapes"].items():
        pref = s.get("prefer", {}).get("flydsl", [])
        gain = s.get("meta", {}).get("min_gain", table.get("min_gain", MIN_GAIN))
        for p in s["points"]:
            m, t = p[0], p[1]
            tile = _tiles(p).get("flydsl")
            fly = tile is not None and any(
                lo <= m and (hi is None or m <= hi) and var == tile for lo, hi, var in pref)
            out[(key, m)] = (base_path(t, gain, _noise(p), k), fly)
    return out


def compare(a_path, b_path) -> int:
    """Print the points two tables decide differently."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    da, db = decisions(a), decisions(b)
    common = sorted(set(da) & set(db), key=lambda x: (x[0], x[1]))
    names = {key: s["name"] for key, s in a["shapes"].items()}
    flips = [x for x in common if da[x] != db[x]]
    for key, m in flips:
        (pa, fa), (pb, fb) = da[(key, m)], db[(key, m)]
        print(f"  {names.get(key, key):<8} M={m:<6} {pa:>9}{' +flydsl' if fa else '':<8} -> "
              f"{pb:>9}{' +flydsl' if fb else ''}")
    print(f"{len(flips)} of {len(common)} points decided differently "
          f"({sum(da[x][0] != db[x][0] for x in common)} in the runs, "
          f"{sum(da[x][1] != db[x][1] for x in common)} in FlyDSL's overlay)")
    return 0


def runs(points, min_gain, k=NOISE_K):
    """[[lo, hi, path], ...] over the measured Ms, consecutive equal winners
    merged, and the path at the largest M (what the table answers above it)."""
    out = []
    for p in points:
        m, t = p[0], p[1]
        path = base_path(t, min_gain, _noise(p), k)
        if out and out[-1][2] == path:
            out[-1][1] = m
        else:
            out.append([m, m, path])
    return out, (out[-1][2] if out else "fallback")


def prefer(points, name, min_gain, opt_gain, k=NOISE_K, confirm=None):
    """[[lo, hi, variant], ...]: for each variant (tile) of optional path
    `name`, the measured Ms where it beat the engine by `min_gain` and the
    runs' answer by `opt_gain` (each by k x the pair's noise as well),
    consecutive Ms of that variant merged. `hi` is None where the variant's
    largest measured M was a win. With `confirm`, a set of Ms, only those
    Ms can win: the ones an earlier run's overlay took."""
    seqs = {}
    for p in points:
        m, t = p[0], p[1]
        var = _tiles(p).get(name)
        if var is None or name not in t:
            continue
        win = (optional_wins(t, _noise(p), name, min_gain, opt_gain, k)
               and (confirm is None or m in confirm))
        seqs.setdefault(var, []).append((m, win))
    out = []
    for var, seq in seqs.items():
        cur = None
        for m, win in seq:
            if not win:
                cur = None
            elif cur is None:
                cur = [m, m, var]
                out.append(cur)
            else:
                cur[1] = m
        if seq[-1][1]:
            cur[1] = None
    return sorted(out, key=lambda r: (r[0], r[2]))


def entry(name: str, points: List[list], *, min_gain: float = MIN_GAIN,
          noise_k: float = NOISE_K, opt_gain: float = OPT_GAIN,
          confirm: Optional[set] = None, partial: bool = False) -> Dict[str, Any]:
    """A table's entry for one shape, from its measured points (sorted by M
    here). A `partial` entry -- its grid cut short, by a budget -- answers
    nothing above its largest measured M: what was measured there says
    nothing about the Ms that were not."""
    points = sorted(points, key=lambda p: p[0])
    rs, above = runs(points, min_gain, noise_k)
    ent: Dict[str, Any] = dict(name=name, runs=rs,
                               above="fallback" if partial else above, points=points)
    pref = {opt: prefer(points, opt, min_gain, opt_gain, noise_k, confirm)
            for opt in OPTIONAL if any(opt in p[1] for p in points)}
    if any(pref.values()):
        ent["prefer"] = pref
    if partial:
        ent["partial"] = True
    return ent


#: How measure_points times a burst (see its docstring).
TIMINGS = ("burst", "queued", "graph", "context")


def default_timing(arch: Optional[str] = None) -> str:
    """measure_points' timing when none is given: "queued" on gfx950, "context" on RDNA."""
    if arch is None:
        from ..target import current_arch  # noqa: PLC0415

        arch = current_arch()
    return "queued" if arch == "gfx950" else "context"


def tunableop_state() -> Any:
    """PyTorch's TunableOp as this process runs it, as a record says it: "off", or its results
    file's name and a hash of its contents. Which GEMM the engine's path runs at an M is what
    that file says: with MiniCPM5's file on an R9700, vLLM's o_proj at M = 8 runs a tuned
    hipBLASLt kernel at 17 us in a burst, and the untuned one at 29."""
    import hashlib  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415

        t = torch.cuda.tunable
        if not t.is_enabled():
            return "off"
        name = t.get_filename()
        dev = torch.cuda.current_device()
    except Exception:  # noqa: BLE001 -- no torch, no TunableOp: metadata, not a failure
        return "off"
    out: Dict[str, Any] = {"file": os.path.basename(name), "tuning": bool(t.tuning_is_enabled())}
    for path in (name.replace("%d", str(dev)), name):
        try:
            out["sha"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
            break
        except OSError:
            continue
    return out


def _record_timing(ent: dict, arch: str) -> Optional[str]:
    """How a record was timed: its meta's; for a record from before timing was recorded (a meta
    without it), what records were then (queued on gfx950, an unqueued burst elsewhere); None for
    a row with no meta at all, which says nothing about how it was measured (as with its engine)."""
    meta = ent.get("meta")
    if meta is None:
        return None
    return meta.get("timing") or ("queued" if arch == "gfx950" else "burst")


def route_at(points: Sequence[list], m: int, min_gain: float = MIN_GAIN,
             noise_k: float = NOISE_K) -> str:
    """The path measured points take at M by the runs' rule ("engine" where the engine's path
    won, or M was not measured)."""
    for p in points:
        if p[0] == m:
            path = base_path(p[1], min_gain, _noise(p), noise_k)
            return "engine" if path == "fallback" else path
    return "engine"


def dump(table: dict) -> str:
    """JSON with one line per run and per measured M, so a diff of two tables
    reads as a diff of decisions."""
    head = {k: v for k, v in table.items() if k != "shapes"}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append(' "shapes": {')
    keys = list(table["shapes"])
    for i, key in enumerate(keys):
        s = table["shapes"][key]
        lines.append(f'  {json.dumps(key)}: {{"name": {json.dumps(s["name"])}, '
                     f'"above": {json.dumps(s["above"])},')
        for extra in ("partial", "meta"):
            if extra in s:
                lines.append(f'   {json.dumps(extra)}: {json.dumps(s[extra])},')
        pref = {name: rs for name, rs in s.get("prefer", {}).items() if rs}
        if pref:
            lines.append('   "prefer": {')
            for j, (name, ranges) in enumerate(pref.items()):
                lines.append(f"    {json.dumps(name)}: [")
                lines += [f"     {json.dumps(r)}," for r in ranges]
                lines[-1] = lines[-1].rstrip(",")
                lines.append("    ]" + ("," if j < len(pref) - 1 else ""))
            lines.append("   },")
        lines.append('   "runs": [')
        lines += [f"    {json.dumps(r)}," for r in s["runs"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append('   ], "points": [')
        lines += [f"    {json.dumps(p)}," for p in s["points"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("   ]}" + ("," if i < len(keys) - 1 else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ models


def model_config(name: str) -> dict:
    """A model's config.json: a path to one, the local HF cache (the pods have
    no network), or the hub through huggingface_hub where it can reach it."""
    import glob  # noqa: PLC0415

    if os.path.isfile(name):
        return json.loads(Path(name).read_text())
    if os.path.isdir(name) and os.path.isfile(os.path.join(name, "config.json")):
        return json.loads(Path(name, "config.json").read_text())
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = os.path.join(root, "hub", f"models--{name.replace('/', '--')}",
                       "snapshots", "*", "config.json")
    hits = sorted(glob.glob(pat))
    if hits:
        return json.loads(Path(hits[-1]).read_text())
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        return json.loads(Path(hf_hub_download(name, "config.json")).read_text())
    except Exception as e:  # noqa: BLE001 -- no hub, no network, no such repo
        raise FileNotFoundError(f"no config.json for {name}: not a file, not "
                                f"under {pat}, and the hub said "
                                f"{type(e).__name__}: {e}") from e


#: Architectures whose attention projection carries a bias. The linear op
#: sends a biased call to the engine (a second pass over the output costs
#: more than it saves), so those shapes are not worth tuning.
QKV_BIAS = {"Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration"}


def linear_shapes(cfg: dict, tp: int = 1) -> List[Tuple[str, int, int, int]]:
    """(name, N, K, calls per step) for every unquantized linear of a decoder,
    sharded the way vLLM shards them at tensor-parallel size `tp`; the LM
    head's rows padded to a multiple of 64, as vLLM pads them.

    A hybrid decoder (Qwen3.5 / Qwen3.6 / Qwen3.8, Qwen3-Next: `layer_types`
    mixing "full_attention" with "linear_attention") has the attention
    projections only on its full-attention layers, and on the others the
    Gated-DeltaNet projections: `qkvz` (q and k of `linear_num_key_heads`, v
    and the gate z of `linear_num_value_heads`), `ba` (one b and one a per
    value head) and `out` (back to the hidden size). `attn_output_gate`
    doubles q for the gate it carries. Shapes two names share are listed once
    each by name; a table keys on (N, K)."""
    cfg = cfg.get("text_config", cfg)
    hid = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or hid // heads
    inter = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    vocab = -(-cfg["vocab_size"] // 64) * 64
    types = cfg.get("layer_types") or []
    full = sum(t == "full_attention" for t in types) if types else layers
    lin = sum(t == "linear_attention" for t in types)
    q = heads * hd * (2 if cfg.get("attn_output_gate") else 1)
    out = [
        ("qkv", (q + 2 * kv * hd) // tp, hid, full),
        ("o", hid, heads * hd // tp, full),
    ]
    if lin:
        nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        out += [
            ("qkvz", (2 * nk * dk + 2 * nv * dv) // tp, hid, lin),
            ("ba", 2 * nv // tp, hid, lin),
            ("out", hid, nv * dv // tp, lin),
        ]
    return out + [
        ("gate_up", 2 * inter // tp, hid, layers),
        ("down", hid, inter // tp, layers),
        ("lm_head", vocab // tp, hid, 1),
    ]


def biased(cfg: dict) -> set:
    """The linear_shapes names whose calls carry a bias in this model."""
    arch = set(cfg.get("architectures") or ())
    text = cfg.get("text_config", cfg)
    out = set()
    if arch & QKV_BIAS or text.get("attention_bias"):
        out.add("qkv")
    if text.get("attention_bias") and text.get("o_proj_bias", True):
        out.add("o")
    if text.get("mlp_bias"):
        out |= {"gate_up", "down"}
    return out


# ------------------------------------------------------------------ records


def records_name(kind: str, arch: str) -> str:
    return f"routes_{kind}_{arch}"


def user_path(kind: str, arch: str) -> Path:
    """Where this machine's measured shapes for `kind` on `arch` are kept."""
    from . import records_dir  # noqa: PLC0415

    return records_dir() / f"{records_name(kind, arch)}.json"


def load_user(kind: str, arch: str) -> Optional[dict]:
    """This machine's table for `kind` on `arch`, or None. A corrupt file is
    None: a missing record costs speed, never a launch."""
    try:
        d = json.loads(user_path(kind, arch).read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError):
        return None
    return d if isinstance(d, dict) and isinstance(d.get("shapes"), dict) else None


@contextlib.contextmanager
def records_lock(kind: str, arch: str):
    """Exclusive, across processes, for as long as the block runs: whoever
    holds it is the one measuring. Others wait, then read what it wrote."""
    import fcntl  # noqa: PLC0415

    p = user_path(kind, arch).with_suffix(".lock")
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def save_user(kind: str, arch: str, shapes: Dict[str, dict], meta: Dict[str, Any]) -> Path:
    """Write measured shape entries into this machine's table, replacing any
    earlier entry for the same (N, K). Atomic; call under `records_lock`."""
    from . import forget_records  # noqa: PLC0415

    p = user_path(kind, arch)
    cur = load_user(kind, arch) or {"kind": kind, "arch": arch, "shapes": {}}
    cur.update({k: v for k, v in meta.items() if k not in ("shapes",)})
    cur["shapes"].update(shapes)
    tmp = p.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(dump(cur))
    tmp.replace(p)
    forget_records()
    return p


# ------------------------------------------------------------------ measuring


def engine_path(name: str = "vllm") -> Tuple[Callable, str]:
    """What the engine's unquantized linear reaches, and its name. vLLM: the
    installed vLLM's own op where it registers one -- since 0.2x it runs
    skinny GEMMs (wvSplitK, LLMM1) at M <= 5 on gfx11 -- else F.linear.
    SGLang: F.linear, which its UnquantizedLinearMethod calls on ROCm without
    aiter, whether or not a vLLM is importable beside it. A table measured
    against one is a table for that engine."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: N812,PLC0415

    if name == "sglang":
        return F.linear, "sglang F.linear"
    try:
        import vllm  # noqa: PLC0415
        import vllm.model_executor.layers.utils  # noqa: F401,PLC0415 -- registers the op
        return (torch.ops.vllm.rocm_unquantized_gemm,
                f"vllm {vllm.__version__} rocm_unquantized_gemm")
    except (ImportError, AttributeError, RuntimeError):
        return F.linear, "F.linear"


_OPS: Dict[Any, Any] = {}


def _gemm_op(m, n, k):
    from ..ops import gemm as hk_gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415
    from ..target import current_arch  # noqa: PLC0415

    if current_arch() == "gfx950":
        # hk's gfx950 GEMMs, by their own measured table (hk.ops.gemm_cdna).
        from ..ops import gemm_cdna  # noqa: PLC0415

        if k % gemm_cdna.K_STEP or n % gemm_cdna.BLOCK_N_MIN:
            return None
        return gemm_cdna.matmul
    s = hk_gemm.schedule_for(m, n, k)
    if k % s["k_step"] or n % s["block_n"]:
        return None
    key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
    if key not in _OPS:
        kernel = hk_gemm.TUNER._kernel_for(s)
        if not _builds(kernel):
            return None
        _OPS[key] = torch_op(kernel)
    return _OPS[key]


def _decode_op(m, n, k):
    from ..ops import decode_gemm as hk_decode  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    if not hk_decode.fits(m, n, k):
        return None
    s = hk_decode.schedule_for(m, k)
    key = ("decode",) + tuple(sorted(s.items()))
    if key not in _OPS:
        kernel = hk_decode.kernel_for(**s)
        if not _builds(kernel):
            return None
        _OPS[key] = torch_op(kernel)
    return _OPS[key]


def _builds(kernel) -> bool:
    """Whether this compiler builds the kernel within hk's gates. One it
    refuses (a spill) is not a path to measure: the point is measured without
    it, as vLLM's linear route would run it, instead of the tune failing."""
    from ..runtime.resources import ResourceError  # noqa: PLC0415

    try:
        kernel.build()
        return True
    except ResourceError as e:
        print(f"[hk] tune: {kernel.name} does not build with this compiler, not measured "
              f"({str(e).splitlines()[0]})", flush=True)
        return False


class OutOfTime(Exception):
    """A deadline passed between two points."""


def measure_points(n: int, k: int, ms: Sequence[int], *, engine: Callable,
                   paths: Iterable[str] = ("hk", "hk_decode"), flydsl: bool = False,
                   rounds: int = 3, warmup: int = 1, sets: int = 1, remeasure: int = 2,
                   min_gain: float = MIN_GAIN, opt_gain: float = OPT_GAIN,
                   close_k: float = CLOSE_K, noise_k: float = NOISE_K,
                   rotate_bytes: int = ROTATE_BYTES, raw: bool = False,
                   deadline: Optional[float] = None, device=None,
                   log: Optional[Callable[[str], None]] = None,
                   tag: str = "", queued: Optional[bool] = None, timing: Optional[str] = None,
                   context: Sequence[Tuple[int, int, Callable[[int], str]]] = ()
                   ) -> Tuple[List[list], int, bool]:
    """Measure y = x @ W.T for W (n, k) at each M of `ms`, in order.

    Returns (points, how many were re-measured as close calls, whether every
    M was measured). With `deadline` (a time.monotonic() value), stops at the
    first M that starts after it; the points measured so far are kept.

    `queued` holds each timed burst behind a GPU spin longer than its host
    enqueue, so the calls run back to back as a captured step runs them and
    no path is charged for its launch's host cost. Default: on for gfx950,
    where vLLM's path is aiter's tuned GEMM, ~60 us of Python a call at
    M = 16 against a 12 us kernel, which an unqueued burst times as the
    kernel's (docs/history/42).

    `timing` says how a burst is timed: "burst" (unqueued, launches as the
    host makes them), "queued" (as above) or "graph": each path's burst over
    the rotated weights captured once as a CUDA graph and its replays timed --
    how a captured decode step runs it, with no host cost in either arm and
    cold weights -- or "context": as "graph", each call among the model's
    other linear layers, `context`, as a decode step runs them. Default: "queued" on gfx950 (or `queued`'s value when given), "context"
    on RDNA. An unqueued burst on gfx1100 charged vLLM's ~8 us of Python
    dispatch at M <= 5 (wvSplitK) to the engine: 1536x1536 at M = 1, 17.2 vs
    hk's 11.5 us unqueued, 8.6 vs 11.3 queued, 9.4 vs 11.0 in a graph, and a
    table measured so sent small M to hk where serving, in graphs, was slower.

    `context` is (N, K, path at M) of each other linear layer, in the model's
    order from the one measured. Each call of a path runs after a kernel that
    writes its input (`_glue`) and before one call of every layer of the
    context, on its route at that M ("engine", "hk", "hk_decode"), each with
    an input kernel and weights of its own; the context's own graph, with the
    measured shape's input kernel in it, is timed in the same rounds and
    taken off. What is left is what routing the shape so adds to a step,
    including what it does to the kernels around it. A burst of one shape
    alone flatters a tuned hipBLASLt kernel: on an R9700 with MiniCPM5's
    TunableOp results, vLLM's o_proj at M = 8 ran 15.6 us (kernel time) after
    itself, 17.6-19.0 after another GEMM and 20.5-21.6 in the served step,
    while hk's GEMV, one small kernel for every shape, ran 17.3-17.6
    throughout. Timed alone, M = 8 stayed on the engine; served, the routes
    context timing picks there took 10.2 us a layer (5.8%) off the step's
    linear time, as it predicted. Without a context, "context" times as
    "graph".
    """
    if timing is None:
        timing = default_timing() if queued is None else ("queued" if queued else "burst")
    if timing not in TIMINGS:
        raise ValueError(f"timing={timing!r}: one of {TIMINGS}")
    ctx = list(context) if timing == "context" else []
    if timing == "context":
        timing = "graph"
    import torch  # noqa: PLC0415

    paths = set(paths)
    log = log or (lambda s: None)
    dev = torch.device("cuda") if device is None else torch.device(device)
    hkk = None
    if "hipkittens" in paths:
        from ..ops import hipkittens as hkk  # noqa: PLC0415
    fly = None
    if flydsl:
        from ..integration import flydsl_gemm  # noqa: PLC0415

        fly = flydsl_gemm if flydsl_gemm.available() else None

    # Calls a timed graph makes: with a context, CONTEXT_CALLS layers or more (at most
    # CONTEXT_CALLS_MAX), each shape -- this one and every layer of the context -- going through
    # CONTEXT_ROTATE x rotate_bytes of copies of its own. The context's streaming does not keep a
    # shape's weights out of the Infinity Cache: on an R9700, qkv's four copies (40 MB) among
    # MiniCPM5's other layers gave hk's GEMV 9.8 us a call at M = 1, read from the cache, and 16
    # copies 21.5, its time alone with cold weights; on a W7900D, down 2048x6144 at M = 1 took
    # 34.8 us over 288 MB of copies and 38.4-39.2 over 576 MB to 2.3 GB.
    w_bytes = n * k * 2
    rot = rotate_bytes * (CONTEXT_ROTATE if ctx else 1)
    if ctx:
        # Within the free memory, the context's copies counted: halved down to rotate_bytes.
        need = sum(min(-(-rot // b), CONTEXT_CALLS_MAX) * b
                   for b in [w_bytes] + [cn * ck * 2 for cn, ck, _ in ctx])
        free = torch.cuda.mem_get_info(dev)[0]
        while rot > rotate_bytes and need > free // 2:
            rot //= 2
            need //= 2
            log(f"  {tag}: context copies over {rot >> 20} MB a shape, for the free memory")

    def copies_of(nbytes):
        return max(2, -(-rot // nbytes))

    copies = copies_of(w_bytes)
    if ctx:
        calls_n = min(max(copies, CONTEXT_CALLS), CONTEXT_CALLS_MAX)
        copies = min(copies, calls_n)
    else:
        calls_n = copies
    g = torch.Generator(device=dev).manual_seed(0)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
          * k ** -0.5 for _ in range(copies)]
    cws = [[torch.randn(cn, ck, device=dev, dtype=torch.bfloat16, generator=g) * ck ** -0.5
            for _ in range(min(copies_of(cn * ck * 2), calls_n))] for cn, ck, _ in ctx]
    points: List[list] = []
    remeasured = 0

    def variants_for(m, x, out):
        variants = {"engine": lambda w: engine(x, w, None)}
        op = _gemm_op(m, n, k) if "hk" in paths else None
        if op is not None:
            variants["hk"] = lambda w, op=op: op(x, w, out) or out
        dop = _decode_op(m, n, k) if "hk_decode" in paths else None
        if dop is not None:
            variants["hk_decode"] = lambda w, dop=dop: dop(x, w, out) or out
        if hkk is not None and hkk.gemm_fits(m, n, k):
            variants["hipkittens"] = lambda w: hkk.gemm_bf16(x, w, out=out)
        if fly is not None and fly.variant(m, n, k) is not None:
            variants["flydsl"] = lambda w: fly.run(x, w, out) or out
        return variants

    queued = timing == "queued"

    def one_set(m, names):
        """{path: [ms per round]} on activations of their own."""
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ok = {p: c for p, c in variants_for(m, x, out).items() if p in names}
        per = {p: [] for p in ok}
        order = list(ok)
        host = {}
        if timing == "graph":
            # Each call: the path's on a copy, then the context's layers on theirs ([] without one).
            around = [[] for _ in range(calls_n)]
            for (cn, ck, at), cw in zip(ctx, cws):
                glue, run = _context_call(engine, m, cn, ck, at(m), dev, g)
                for i in range(calls_n):
                    around[i] += [glue, lambda w=cw[i % len(cw)], run=run: run(w)]
            # With a context, each layer's input is written by a kernel of its own just before
            # it, as a norm, attention or activation writes it in a step -- this shape's too, in
            # the context's own graph as well, so that what is taken off leaves its GEMM alone.
            before = [_glue(x, dev, g)] if ctx else []
            bursts = {name: [c for i in range(calls_n) for c in
                             before + [lambda w=ws[i % copies], call=call: call(w)] + around[i]]
                      for name, call in ok.items()}
            if ctx:
                bursts[_CONTEXT] = [c for i in range(calls_n) for c in before + around[i]]
            # Captured on a stream of its own, warmed on that stream first: a library that keeps
            # per-stream state (hipBLASLt's handle and workspace, which TunableOp's tuned GEMMs
            # reach) makes it on first use, which a stream being captured does not permit.
            graphs = {}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            for name, calls in bursts.items():
                with torch.cuda.stream(side):
                    for c in calls[:len(calls) // calls_n]:
                        c()
                side.synchronize()
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph, stream=side):
                    for c in calls:
                        c()
                graphs[name] = gph
            torch.cuda.synchronize()
            names = list(graphs)
            took: Dict[str, List[float]] = {name: [] for name in names}
            for r in range(rounds):
                for name in (names if r % 2 == 0 else names[::-1]):
                    graphs[name].replay()
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    graphs[name].replay()
                    e.record()
                    torch.cuda.synchronize()
                    took[name].append(s.elapsed_time(e))
            del graphs
            # With a context: what the path's calls add to it, the effect on its other layers'
            # kernels included -- what routing the shape so changes in a step.
            base = took.pop(_CONTEXT, [0.0] * rounds)
            for name in order:
                per[name] = [max((t - b) / calls_n, 1e-4) for t, b in zip(took[name], base)]
            return per
        if queued:
            for name, call in ok.items():
                t0 = time.perf_counter()
                for w in ws:
                    call(w)
                host[name] = time.perf_counter() - t0
                torch.cuda.synchronize()
        for r in range(rounds):
            for name in (order if r % 2 == 0 else order[::-1]):
                call = ok[name]
                call(ws[-1])
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                if queued:
                    torch.cuda._sleep(int((2 * host[name] + 5e-4) * SPIN_HZ))
                s.record()
                for w in ws:
                    call(w)
                e.record()
                torch.cuda.synchronize()
                per[name].append(s.elapsed_time(e) / copies)
        return per

    complete = True
    try:
        for m in ms:
            if deadline is not None and time.monotonic() > deadline:
                complete = False
                break
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            w32 = ws[0].float().t()
            # In row blocks: at 16384 rows of a 37888-wide layer one fp32
            # product is 2.5 GB, and a tuner inside a server that has loaded
            # its weights must not need four of them at once.
            step = max(1, (256 << 20) // (4 * n))
            ref = [x[i:i + step].float() @ w32 for i in range(0, m, step)]
            scale = max(r.abs().max().item() for r in ref)
            tiles = {}
            if fly is not None and fly.variant(m, n, k) is not None:
                tiles["flydsl"] = fly.variant(m, n, k)
            ok = []
            for name, call in variants_for(m, x, out).items():
                y = call(ws[0])
                err = max((y[i:i + step].float() - r).abs().max().item()
                          for i, r in zip(range(0, m, step), ref)) / max(scale, 1e-30)
                del y
                if name != "engine" and not err < 2e-2:
                    log(f"  {tag} m={m}: {name} FAILED the gate "
                        f"(rel err {err:.3g}); not a candidate")
                    continue
                ok.append(name)
            del x, out, ref, w32
            for _ in range(warmup):
                one_set(m, ok)
            got = [one_set(m, ok) for _ in range(sets)]
            times, noise = estimate(got)
            if remeasure and close_call(times, noise, min_gain, opt_gain, close_k, noise_k):
                got += [one_set(m, ok) for _ in range(remeasure)]
                times, noise = estimate(got)
                remeasured += 1
            tiles = {p: v for p, v in tiles.items() if p in times}
            point = [m, {p: round(v, 5) for p, v in times.items()}, tiles,
                     {p: round(v, 4) for p, v in noise.items()}, len(got)]
            if raw:
                point.append(got)
            points.append(point)
    finally:
        del ws, cws
        torch.cuda.empty_cache()
    return points, remeasured, complete


#: measure_points' name for the context's own graph, timed beside the paths' and taken off them.
_CONTEXT = "(context)"
#: Layers a context-timed graph runs: at least CONTEXT_CALLS, so that what a graph pays once is
#: a small part of a call; at most CONTEXT_CALLS_MAX, where a small shape's copies stop short of
#: rotate_bytes rather than make a graph of thousands of layers.
CONTEXT_CALLS = 16
CONTEXT_CALLS_MAX = 64
#: Passes over the decode Ms, after the first, re-timing each M whose context has moved: the
#: routes one shape takes change the others' contexts. On an R9700 (MiniCPM5, TunableOp) the
#: first such pass decided 0-16 of 25-54 re-timed Ms a shape differently, the second 0-1, the
#: third none; on a W7900D the first decided 0-1.
CONTEXT_PASSES = 3
#: How many times rotate_bytes each shape of a context-timed graph rotates through (see
#: measure_points: a W7900D's 96 MB Infinity Cache still served part of 288 MB).
CONTEXT_ROTATE = 2


def _context_call(engine, m, n, k, path, dev, g) -> Tuple[Callable, Callable]:
    """One layer of a context: (the call that writes its input, y = x @ W.T for W (n, k) at M on
    `path` as a call of W). The engine's path where hk's would not take the shape."""
    import torch  # noqa: PLC0415

    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
    glue = _glue(x, dev, g)
    op = _decode_op(m, n, k) if path == "hk_decode" else _gemm_op(m, n, k) if path == "hk" else None
    if op is None:
        return glue, lambda w: engine(x, w, None)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    return glue, lambda w: op(x, w, out)


def _glue(x, dev, g) -> Callable:
    """A call that writes x as a step's norm would: an RMS norm of another (M, K), copied in.
    Between the GEMMs of a step run other kernels, and a tuned hipBLASLt GEMM runs slower after
    them than after a GEMM: on an R9700 (MiniCPM5 at M = 8, TunableOp), hk's GEMV on o_proj
    saved 1.1 us a layer over vLLM's among the other GEMMs alone, 1.5 with this before each,
    3.4 with the step's own norm, attention and activation; served, 3.2."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: N812,PLC0415

    src = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=g)
    w = torch.ones(x.shape[-1], device=dev, dtype=x.dtype)
    return lambda: x.copy_(F.rms_norm(src, (x.shape[-1],), w))


def context_of(order: Sequence[Tuple[int, int]], n: int, k: int,
               path_at: Callable[[int, int], Callable[[int], str]]) -> list:
    """measure_points' `context` for (n, k) in a model whose linear layers are `order`, (N, K)
    in model order, each once: the others, starting after (n, k) and wrapping round, each
    with `path_at(N, K)`, its path at an M."""
    keys = list(dict.fromkeys(order))
    i = keys.index((n, k)) if (n, k) in keys else len(keys)
    return [(cn, ck, path_at(cn, ck)) for cn, ck in keys[i + 1:] + keys[:i]]


def tune_shape(n: int, k: int, *, engine: Optional[Callable] = None,
               ms: Optional[Sequence[int]] = None, min_gain: float = MIN_GAIN,
               name: str = "", lm_head: bool = False, confirm: Optional[set] = None,
               **how) -> Dict[str, Any]:
    """Measure one (N, K) over the grid (or `ms`) and return its table entry:
    `{name, runs, above, points[, prefer][, partial]}`, which
    `engine.RouteTable` reads. `lm_head` stops at MAX_SEQS rows. Keyword
    arguments are `measure_points`'s; `engine` defaults to the installed
    vLLM's path. The entry's `remeasured` and `seconds` say what it cost."""
    if engine is None:
        engine = engine_path()[0]
    grid = list(GRID if ms is None else ms)
    if lm_head:
        grid = [m for m in grid if m <= MAX_SEQS]
    t0 = time.monotonic()
    points, remeasured, complete = measure_points(
        n, k, grid, engine=engine, min_gain=min_gain, tag=name or f"{n}x{k}", **how)
    ent = entry(name or f"{n}x{k}", points, min_gain=min_gain,
                noise_k=how.get("noise_k", NOISE_K), opt_gain=how.get("opt_gain", OPT_GAIN),
                confirm=confirm, partial=not complete)
    ent["remeasured"] = remeasured
    ent["seconds"] = round(time.monotonic() - t0, 1)
    return ent


# ------------------------------------------------------------------ a model's missing shapes


class Report:
    """What `tune_missing` did, shape by shape, and what it cost."""

    def __init__(self):
        self.tuned: List[Tuple[str, int, int, float]] = []      # name, n, k, seconds
        self.partial: List[Tuple[str, int, int, int]] = []      # ..., Ms measured
        self.left: List[Tuple[str, int, int]] = []              # out of time
        self.known: List[Tuple[str, int, int]] = []             # already in a table
        self.seconds = 0.0
        self.waited = 0.0
        self.path: Optional[Path] = None

    def as_dict(self) -> Dict[str, Any]:
        return dict(tuned=self.tuned, partial=self.partial, left=self.left,
                    known=self.known, seconds=round(self.seconds, 1),
                    waited=round(self.waited, 1), path=str(self.path or ""))

    def line(self) -> str:
        def names(xs):
            return ", ".join(f"{x[0]} {x[1]}x{x[2]}" for x in xs)

        parts = []
        if self.tuned:
            parts.append(f"measured {len(self.tuned)} shape{'s' * (len(self.tuned) != 1)} "
                         f"({names(self.tuned)})")
        if self.partial:
            parts.append(f"{len(self.partial)} only up to M="
                         + ", ".join(f"{p[3]} ({p[0]})" for p in self.partial))
        if self.left:
            parts.append(f"out of time for {names(self.left)}, which stay on the "
                         f"engine's path")
        if not parts:
            parts.append("nothing to measure")
        if self.known:
            parts.append(f"{len(self.known)} already measured")
        waited = f", {self.waited:.0f} s of it waiting for another process" \
            if self.waited >= 1 else ""
        return "; ".join(parts) + f" -- {self.seconds:.0f} s{waited}" + (
            f"; records {self.path}" if self.path and (self.tuned or self.partial) else "")


def measured_ms(ent: Optional[dict]) -> set:
    return {p[0] for p in (ent or {}).get("points", ())}


def _where(device=None) -> Dict[str, Any]:
    """What a record says it was measured on: the toolchain and, where torch
    can say, the device and runtime."""
    out: Dict[str, Any] = {}
    try:
        from ..runtime.compile import toolchain_fingerprint  # noqa: PLC0415

        out["toolchain"] = toolchain_fingerprint()
    except Exception:  # noqa: BLE001 -- metadata, not a reason to stop
        pass
    try:
        import torch  # noqa: PLC0415

        out.update(device=torch.cuda.get_device_name(device), torch=torch.__version__,
                   hip=torch.version.hip)
    except Exception:  # noqa: BLE001
        pass
    return out


def tune_missing(kind: str, arch: str, shapes: Sequence[Tuple[str, int, int, int]], *,
                 engine: Callable, engine_name: str, complete: Iterable[Tuple[int, int]] = (),
                 budget: Optional[float] = None, force: bool = False,
                 grid: Sequence[int] = GRID, by: str = "",
                 log: Optional[Callable[[str], None]] = None,
                 model: Optional[Sequence[Tuple[str, int, int]]] = None,
                 route_of: Optional[Callable[[int, int, int], str]] = None, **how) -> Report:
    """Measure every (N, K) of `shapes` -- (name, N, K, calls per step) --
    that no table has measured, and write each into this machine's records
    as soon as it is done.

    `complete` is the shapes the shipped table has; this machine's records
    are read here, under the lock, so a shape another process measured while
    this one waited is not measured again. A record of this machine's
    measured against another engine (`engine_name`), timed another way, or
    under another TunableOp state (`tunableop_state`: which GEMM the engine's
    path runs is what its results file says) is measured again, and so is
    everything with `force`.

    `budget` seconds, from the call, bound the whole thing. Decode-sized Ms
    go first, for every shape -- they are where a decode step's time is, and
    each costs a fraction of a prefill-sized point -- largest weight stream
    first; then the rest of the grid, shape by shape. A shape whose grid was
    cut short is written `partial`: its runs answer only up to the largest M
    measured, and the next call carries on from there.

    Context timing (`measure_points`) runs each call among the model's other
    linear layers: `model`, (name, N, K) of every one in model order
    (default: `shapes`), each on its route at the M -- a shape measured here
    on its points so far, any other on `route_of(M, N, K)` (the table's; the
    engine's without it). A shape measured first is timed among layers still
    on the engine's path, and its routes change the others' context; so once
    every shape's decode Ms are in, each decode M whose context has changed
    since it was measured is measured again, in the routes chosen so far, for
    up to CONTEXT_PASSES passes or until none has moved. A record keeps each
    point's context (meta.context_routes), so a later call re-checks what an
    earlier one measured. The LM head, one call a step, is timed alone and is
    in no context.
    """
    log = log or (lambda s: None)
    t0 = time.monotonic()
    deadline = None if budget is None else t0 + budget
    rep = Report()
    complete = set(complete)
    lm_names = ("lm_head",)
    timing = how.pop("timing", None) or default_timing(arch)
    meta = dict(engine=engine_name, **_where(how.get("device")),
                min_gain=how.get("min_gain", MIN_GAIN),
                rounds=how.get("rounds", 3), warmup=how.get("warmup", 1),
                sets=how.get("sets", 1), remeasure=how.get("remeasure", 2),
                close_k=how.get("close_k", CLOSE_K), noise_k=how.get("noise_k", NOISE_K),
                tunableop=tunableop_state(), by=by or "hk.autotune.routes")
    order = [(n, k) for name, n, k in (model or [s[:3] for s in shapes]) if name not in lm_names]

    def timing_of(n, k):
        """How (n, k) is timed: in context where the model has other layers -- but the LM head,
        one call a step of ~1 ms, in a graph of its own: what a context changes is microseconds."""
        if timing != "context":
            return timing
        return "context" if (n, k) in order and len(set(order)) > 1 else "graph"

    with records_lock(kind, arch):
        rep.waited = time.monotonic() - t0
        user = (load_user(kind, arch) or {}).get("shapes", {})
        todo = []
        for name, n, k, calls in shapes:
            key = f"{n}x{k}"
            ent = user.get(key)
            # A record measured against another engine, timed another way (gfx1100's unqueued
            # bursts before "graph", single-shape graphs before "context") or under another
            # TunableOp state is measured again.
            fresh = (ent is not None and ent.get("meta", {}).get("engine") in (None, engine_name)
                     and _record_timing(ent, arch) in (None, timing_of(n, k))
                     and ent.get("meta", {}).get("tunableop", meta["tunableop"])
                     == meta["tunableop"])
            if not force and ((fresh and not ent.get("partial"))
                              or (ent is None and (n, k) in complete)):
                rep.known.append((name, n, k))
                continue
            want = [m for m in grid if not (name in lm_names and m > MAX_SEQS)]
            have = [p for p in ent["points"]] if (fresh and not force and ent) else []
            todo.append(dict(name=name, n=n, k=k, calls=calls, want=want, points=have,
                             seconds=float((ent or {}).get("meta", {}).get("seconds", 0.0))
                             if have else 0.0, remeasured=0,
                             seen={int(m): sig for m, sig in (ent or {}).get("meta", {})
                                   .get("context_routes", {}).items()} if have else {}))
        todo.sort(key=lambda s: -s["n"] * s["k"] * s["calls"])
        mine = {(s["n"], s["k"]): s for s in todo}

        def path_at(n, k):
            s = mine.get((n, k))
            if s is not None:
                return lambda m: route_at(s["points"], m, meta["min_gain"], meta["noise_k"])
            if route_of is not None:
                return lambda m: _engine_or(route_of(m, n, k))
            return lambda m: "engine"

        def save(s, complete_now):
            ent = entry(s["name"], s["points"], min_gain=meta["min_gain"],
                        noise_k=meta["noise_k"], partial=not complete_now)
            ent["meta"] = {**meta, "timing": timing_of(s["n"], s["k"]),
                           "date": time.strftime("%Y-%m-%d"),
                           "seconds": round(s["seconds"], 1), "remeasured": s["remeasured"]}
            if s["seen"]:
                ent["meta"]["context_routes"] = {str(m): sig for m, sig in sorted(s["seen"].items())}
            rep.path = save_user(kind, arch, {f"{s['n']}x{s['k']}": ent},
                                 dict(kind=kind, arch=arch))

        def measure(s, ms, phase):
            how_s = timing_of(s["n"], s["k"])
            ctx = context_of(order, s["n"], s["k"], path_at) if how_s == "context" else []
            t1 = time.monotonic()
            pts, rm, _ = measure_points(s["n"], s["k"], ms, engine=engine, deadline=deadline,
                                        log=log, tag=s["name"], timing=how_s, context=ctx,
                                        **how)
            for p in pts:
                s["seen"][p[0]] = _routes_sig(ctx, p[0])
            got = {p[0] for p in pts}
            s["points"] = [p for p in s["points"] if p[0] not in got] + pts
            s["remeasured"] += rm
            s["seconds"] += time.monotonic() - t1
            if pts:
                save(s, set(s["want"]) <= measured_ms({"points": s["points"]}))
            log(f"  {s['name']} {s['n']}x{s['k']}: {phase} Ms, {len(pts)} of "
                f"{len(ms)} measured, {time.monotonic() - t1:.1f} s")
            return pts

        settled = False
        for phase in ("decode",) + ("decode again",) * CONTEXT_PASSES + ("rest",):
            if phase == "decode again":
                # Only once every shape's decode Ms are in (before, a context is still moving),
                # and until a pass finds no decode M whose context moved since it was timed.
                if settled or timing != "context" or any(
                        set(DECODE_MS) & set(t["want"]) - measured_ms(t) for t in todo):
                    continue
                settled = True
            for s in todo:
                if deadline is not None and time.monotonic() > deadline:
                    break
                if phase == "decode again":
                    if timing_of(s["n"], s["k"]) != "context":
                        continue
                    ctx = context_of(order, s["n"], s["k"], path_at)
                    ms = [m for m, was in s["seen"].items() if m in DECODE_MS
                          and was != _routes_sig(ctx, m)]
                    before = {m: route_at(s["points"], m, meta["min_gain"], meta["noise_k"])
                              for m in ms}
                    if ms and measure(s, sorted(ms), phase):
                        settled = False
                        flips = sum(route_at(s["points"], m, meta["min_gain"], meta["noise_k"])
                                    != was for m, was in before.items())
                        log(f"  {s['name']}: {flips} of {len(ms)} re-timed decode Ms decided "
                            f"differently in the chosen routes' context")
                    continue
                done = measured_ms({"points": s["points"]})
                ms = [m for m in s["want"] if m not in done
                      and ((m in DECODE_MS) == (phase == "decode"))]
                if ms:
                    measure(s, ms, phase)
        for s in todo:
            got = measured_ms({"points": s["points"]})
            if set(s["want"]) <= got:
                rep.tuned.append((s["name"], s["n"], s["k"], round(s["seconds"], 1)))
            elif got:
                rep.partial.append((s["name"], s["n"], s["k"], max(got)))
            else:
                rep.left.append((s["name"], s["n"], s["k"]))
    rep.seconds = time.monotonic() - t0
    return rep


def _routes_sig(ctx, m: int) -> str:
    """A context's routes at M, a letter a layer (e engine, d hk's GEMV, g hk's GEMM), as a
    record keeps them: the context each point was measured in."""
    return "".join({"hk_decode": "d", "hk": "g"}.get(at(m), "e") for _, _, at in ctx)


def _engine_or(path: Optional[str]) -> str:
    """A table's answer as a context's path: hk's kernels by name, anything else the engine's."""
    return path if path in ("hk", "hk_decode") else "engine"
