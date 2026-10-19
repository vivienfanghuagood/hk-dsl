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
    """(name, N, K, calls per step) for every unquantized linear of a dense
    decoder, sharded the way vLLM shards them at tensor-parallel size `tp`;
    the LM head's rows padded to a multiple of 64, as vLLM pads them."""
    cfg = cfg.get("text_config", cfg)
    hid = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or hid // heads
    inter = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    vocab = -(-cfg["vocab_size"] // 64) * 64
    return [
        ("qkv", (heads + 2 * kv) * hd // tp, hid, layers),
        ("o", hid, heads * hd // tp, layers),
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

    s = hk_gemm.schedule_for(m, n, k)
    if k % s["k_step"] or n % s["block_n"]:
        return None
    key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
    if key not in _OPS:
        _OPS[key] = torch_op(hk_gemm.TUNER._kernel_for(s))
    return _OPS[key]


def _decode_op(m, n, k):
    from ..ops import decode_gemm as hk_decode  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    if not hk_decode.fits(m, n, k):
        return None
    s = hk_decode.schedule_for(m, k)
    key = ("decode",) + tuple(sorted(s.items()))
    if key not in _OPS:
        _OPS[key] = torch_op(hk_decode.kernel_for(**s))
    return _OPS[key]


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
                   tag: str = "") -> Tuple[List[list], int, bool]:
    """Measure y = x @ W.T for W (n, k) at each M of `ms`, in order.

    Returns (points, how many were re-measured as close calls, whether every
    M was measured). With `deadline` (a time.monotonic() value), stops at the
    first M that starts after it; the points measured so far are kept.
    """
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

    w_bytes = n * k * 2
    copies = max(2, -(-rotate_bytes // w_bytes))
    g = torch.Generator(device=dev).manual_seed(0)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
          * k ** -0.5 for _ in range(copies)]
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

    def one_set(m, names):
        """{path: [ms per round]} on activations of their own."""
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ok = {p: c for p, c in variants_for(m, x, out).items() if p in names}
        per = {p: [] for p in ok}
        order = list(ok)
        for r in range(rounds):
            for name in (order if r % 2 == 0 else order[::-1]):
                call = ok[name]
                call(ws[-1])
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
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
        del ws
        torch.cuda.empty_cache()
    return points, remeasured, complete


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
                 log: Optional[Callable[[str], None]] = None, **how) -> Report:
    """Measure every (N, K) of `shapes` -- (name, N, K, calls per step) --
    that no table has measured, and write each into this machine's records
    as soon as it is done.

    `complete` is the shapes the shipped table has; this machine's records
    are read here, under the lock, so a shape another process measured while
    this one waited is not measured again. A record of this machine's
    measured against another engine (`engine_name`) is measured again, and
    so is everything with `force`.

    `budget` seconds, from the call, bound the whole thing. Decode-sized Ms
    go first, for every shape -- they are where a decode step's time is, and
    each costs a fraction of a prefill-sized point -- largest weight stream
    first; then the rest of the grid, shape by shape. A shape whose grid was
    cut short is written `partial`: its runs answer only up to the largest M
    measured, and the next call carries on from there.
    """
    log = log or (lambda s: None)
    t0 = time.monotonic()
    deadline = None if budget is None else t0 + budget
    rep = Report()
    complete = set(complete)
    lm_names = ("lm_head",)
    meta = dict(engine=engine_name, **_where(how.get("device")),
                min_gain=how.get("min_gain", MIN_GAIN),
                rounds=how.get("rounds", 3), warmup=how.get("warmup", 1),
                sets=how.get("sets", 1), remeasure=how.get("remeasure", 2),
                close_k=how.get("close_k", CLOSE_K), noise_k=how.get("noise_k", NOISE_K),
                by=by or "hk.autotune.routes")
    with records_lock(kind, arch):
        rep.waited = time.monotonic() - t0
        user = (load_user(kind, arch) or {}).get("shapes", {})
        todo = []
        for name, n, k, calls in shapes:
            key = f"{n}x{k}"
            ent = user.get(key)
            fresh = ent is not None and ent.get("meta", {}).get("engine") in (None, engine_name)
            if not force and ((fresh and not ent.get("partial"))
                              or (ent is None and (n, k) in complete)):
                rep.known.append((name, n, k))
                continue
            want = [m for m in grid if not (name in lm_names and m > MAX_SEQS)]
            have = [p for p in ent["points"]] if (fresh and not force and ent) else []
            todo.append(dict(name=name, n=n, k=k, calls=calls, want=want, points=have,
                             seconds=float((ent or {}).get("meta", {}).get("seconds", 0.0))
                             if have else 0.0, remeasured=0))
        todo.sort(key=lambda s: -s["n"] * s["k"] * s["calls"])

        def save(s, complete_now):
            ent = entry(s["name"], s["points"], min_gain=meta["min_gain"],
                        noise_k=meta["noise_k"], partial=not complete_now)
            ent["meta"] = {**meta, "date": time.strftime("%Y-%m-%d"),
                           "seconds": round(s["seconds"], 1), "remeasured": s["remeasured"]}
            rep.path = save_user(kind, arch, {f"{s['n']}x{s['k']}": ent},
                                 dict(kind=kind, arch=arch))

        for phase in ("decode", "rest"):
            for s in todo:
                done = measured_ms({"points": s["points"]})
                ms = [m for m in s["want"] if m not in done
                      and ((m in DECODE_MS) == (phase == "decode"))]
                if not ms:
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    break
                t1 = time.monotonic()
                pts, rm, _ = measure_points(s["n"], s["k"], ms, engine=engine,
                                            deadline=deadline, log=log, tag=s["name"], **how)
                s["points"] += pts
                s["remeasured"] += rm
                s["seconds"] += time.monotonic() - t1
                if pts:
                    save(s, set(s["want"]) <= measured_ms({"points": s["points"]}))
                log(f"  {s['name']} {s['n']}x{s['k']}: {phase} Ms, {len(pts)} of "
                    f"{len(ms)} measured, {time.monotonic() - t1:.1f} s")
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
