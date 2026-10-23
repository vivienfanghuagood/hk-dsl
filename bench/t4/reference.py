"""T4's operation: the grouped GEMM of a mixture-of-experts layer.

    out[offsets[g]:offsets[g+1]] = x[offsets[g]:offsets[g+1]] @ w[g].T     for g in 0..E-1

`x` (rows, K) bf16 holds a batch's (token, expert) pairs already permuted so
that each expert's rows are contiguous, expert 0's first; `w` (E, N, K) bf16
is the experts' weights, each in F.linear's (N, K) layout -- the layout vLLM
stores `w13` (E, 2I, H) and `w2` (E, H, I) in; `offsets` (E + 1,) int32 on the
GPU is 0, the running row counts, rows. A group may be empty. `out` (rows, N)
is bf16, accumulated in fp32.

This module is the task's semantics and its inputs, shared by score.py, the
baselines and the CPU tests: shapes, the routing that sizes the groups, the
fp32 reference and the check. The routing and the shape files are torch-free
(numpy), so the tests run without a GPU or torch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "shapes_public.json"

#: Relative max error against fp32 (max |out - ref| / max |ref|) a correct
#: solution stays under -- the project's GEMM tolerance (bench/tune, hk.tune).
TOL = 2e-2

#: Keys every shape has.
KEYS = ("name", "model", "proj", "E", "N", "K", "tokens", "top_k", "skew", "seed")


# ---------------------------------------------------------------- shapes


def rows(shape: Dict) -> int:
    """(token, expert) pairs: the rows of x and out."""
    return int(shape["tokens"]) * int(shape["top_k"])


def flops(shape: Dict) -> float:
    return 2.0 * rows(shape) * shape["N"] * shape["K"]


def weight_bytes(shape: Dict) -> int:
    return 2 * shape["E"] * shape["N"] * shape["K"]


def validate(shape: Dict) -> None:
    """Raises ValueError for a shape outside the task's guarantees (task.md)."""
    missing = [k for k in KEYS if k not in shape]
    if missing:
        raise ValueError(f"shape {shape.get('name')}: missing {missing}")
    if shape["proj"] not in ("gate_up", "down"):
        raise ValueError(f"{shape['name']}: proj must be gate_up or down")
    for k in ("N", "K"):
        if shape[k] % 64 or shape[k] <= 0:
            raise ValueError(f"{shape['name']}: {k}={shape[k]} is not a positive multiple of 64")
    if not 1 <= shape["top_k"] <= shape["E"] <= 512:
        raise ValueError(f"{shape['name']}: need 1 <= top_k <= E <= 512")
    if shape["tokens"] < 1:
        raise ValueError(f"{shape['name']}: tokens must be >= 1")


def load_shapes(path, arch: Optional[str] = None) -> List[Dict]:
    """The shapes of a shape file ({"shapes": [...]}, or a list), those for
    `arch` when the file gives per-arch lists ({"gfx950": [...], ...})."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "shapes" in data:
        data = data["shapes"]
    if isinstance(data, dict):
        if arch is None:
            raise ValueError(f"{path}: per-arch shape lists; say which arch")
        data = data.get(arch, [])
    for s in data:
        validate(s)
    names = [s["name"] for s in data]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate shape names")
    return list(data)


# ---------------------------------------------------------------- routing


def popularity(E: int, skew: float, seed: int):
    """Per-expert log-popularity: a router's learned imbalance. skew 0 is
    uniform; 1 makes the most popular expert take several times its share."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng([int(seed), 7, E])
    return rng.normal(0.0, float(skew), size=E)


def group_sizes(shape: Dict, seed: int) -> List[int]:
    """Rows per expert for one batch: each token picks top_k distinct experts,
    sampled without replacement with probability ~ exp(popularity) (the
    Gumbel top-k trick), as a softmax router's top-k does on varied tokens.
    Deterministic in (shape, seed); sums to rows(shape)."""
    import numpy as np  # noqa: PLC0415

    E, k, t = int(shape["E"]), int(shape["top_k"]), int(shape["tokens"])
    bias = popularity(E, shape["skew"], shape["seed"])
    rng = np.random.default_rng([int(shape["seed"]), int(seed), 11])
    counts = np.zeros(E, dtype=np.int64)
    step = max(1, (1 << 22) // E)
    for lo in range(0, t, step):
        n = min(step, t - lo)
        g = bias[None, :] + rng.gumbel(size=(n, E))
        if k == E:
            top = np.broadcast_to(np.arange(E), (n, E))
        else:
            top = np.argpartition(-g, k - 1, axis=1)[:, :k]
        counts += np.bincount(top.reshape(-1), minlength=E)
    assert int(counts.sum()) == t * k
    return [int(c) for c in counts]


def edge_sizes(shape: Dict, seed: int = 0) -> List[int]:
    """Rows per expert for the correctness-only routing: empty experts (the
    first and the last among them), a one-row expert, one taking about half
    the rows, odd sizes, the rest spread at random. Same total as the shape's
    routing."""
    import numpy as np  # noqa: PLC0415

    E, total = int(shape["E"]), rows(shape)
    rng = np.random.default_rng([int(shape["seed"]), int(seed), 13])
    sizes = [0] * E
    left = total
    fixed = {0: 0, 1: 1, 2: total // 2, 3: 0, 4: 17, 5: 129, E - 1: 0}
    for e, c in fixed.items():
        if e < E and c <= left:
            sizes[e] = c
            left -= c
    rest = [e for e in range(E) if e not in fixed]
    if not rest:
        rest = [2]
    split = rng.multinomial(left, np.ones(len(rest)) / len(rest))
    for e, c in zip(rest, split):
        sizes[e] += int(c)
    assert sum(sizes) == total
    return sizes


def offsets_of(sizes: Iterable[int]) -> List[int]:
    out = [0]
    for s in sizes:
        out.append(out[-1] + int(s))
    return out


# ---------------------------------------------------------------- tensors


def make_weights(shape: Dict, copies: int = 1, device="cuda"):
    """`copies` independent (E, N, K) bf16 weight tensors, ~N(0, 1/K)."""
    import torch  # noqa: PLC0415

    gen = torch.Generator(device=device).manual_seed(int(shape["seed"]) * 7919 + 1)
    E, N, K = shape["E"], shape["N"], shape["K"]
    out = []
    for _ in range(copies):
        w = torch.empty(E, N, K, device=device, dtype=torch.bfloat16)
        for e in range(E):        # one expert at a time: no fp32 (E, N, K) temporary
            w[e] = torch.randn(N, K, device=device, generator=gen) * K ** -0.5
        out.append(w)
    return out


def make_x(shape: Dict, seed: int, device="cuda"):
    """(rows, K) bf16 activations, ~N(0, 1)."""
    import torch  # noqa: PLC0415

    gen = torch.Generator(device=device).manual_seed(int(shape["seed"]) * 104729 + int(seed))
    return torch.randn(rows(shape), shape["K"], device=device, generator=gen).to(torch.bfloat16)


def offsets_tensor(sizes: Iterable[int], device="cuda"):
    import torch  # noqa: PLC0415

    return torch.tensor(offsets_of(sizes), dtype=torch.int32, device=device)


def make_inputs(shape: Dict, seed: int = 0, device="cuda") -> Dict:
    """One call's inputs: x, w, offsets (the shape's routing for `seed`) and
    a preallocated out."""
    import torch  # noqa: PLC0415

    w = make_weights(shape, 1, device)[0]
    x = make_x(shape, seed, device)
    offs = offsets_tensor(group_sizes(shape, seed), device)
    out = torch.empty(rows(shape), shape["N"], device=device, dtype=torch.bfloat16)
    return dict(x=x, w=w, offsets=offs, out=out)


def torch_impl(x, w, offsets, out=None):
    """The natural PyTorch code: a GEMM per expert (hipBLASLt on ROCm); the
    group boundaries come to the host, one synchronisation a call."""
    import torch  # noqa: PLC0415

    out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=x.dtype) if out is None else out
    o = offsets.tolist()
    for g in range(w.shape[0]):
        if o[g + 1] > o[g]:
            torch.mm(x[o[g]:o[g + 1]], w[g].t(), out=out[o[g]:o[g + 1]])
    return out


def reference(x, w, offsets):
    """The golden output, fp32: (rows, N)."""
    import torch  # noqa: PLC0415

    o = offsets.tolist()
    ref = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=torch.float32)
    for g in range(w.shape[0]):
        if o[g + 1] > o[g]:
            ref[o[g]:o[g + 1]] = x[o[g]:o[g + 1]].float() @ w[g].float().t()
    return ref


def error(out, ref) -> float:
    """max |out - ref| / max |ref| over the whole output; inf if out has a
    non-finite value (score.py fills out with NaN before a check, so an
    unwritten row is one) or the wrong shape."""
    if tuple(out.shape) != tuple(ref.shape):
        return math.inf
    scale = ref.abs().max().item()
    err = 0.0
    step = max(1, (256 << 20) // (4 * max(1, ref.shape[1])))
    for i in range(0, ref.shape[0], step):
        d = (out[i:i + step].float() - ref[i:i + step]).abs().max().item()
        if d != d or d == math.inf:
            return math.inf
        err = max(err, d)
    return err / max(scale, 1e-30)


def check(out, ref) -> str:
    """Raises AssertionError on a wrong answer; returns a one-line summary."""
    e = error(out, ref)
    assert e < TOL, f"relative max error {e:.3g} >= {TOL}"
    return f"relative max error {e:.2e}"
