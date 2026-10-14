"""The shipped kernels (hk.ops) at typical LLM shapes, against torch:
correctness first, then speed.

Each case runs the hk op and its torch equivalent on the same inputs,
checks the hk result against an fp32 reference, and times both in one
process, interleaved round by round. The shapes are the ones a 7-8B decoder
produces: a 4096-wide hidden state, 12288-wide MLP projections, 32 heads of
128, a 4096-token context, and the single-token rows of decode.

    python3 examples/validate.py                 # everything
    python3 examples/validate.py --only gemm,attention
    python3 examples/validate.py --quick         # smaller shapes, for a fast check
    python3 examples/validate.py --list

A case that fails its check is reported and the run continues; the exit
status is the number of failures.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, List

import _common

import hk


@dataclass
class Case:
    name: str
    group: str
    shape: str
    setup: Callable[[], dict]          # -> tensors and anything else the calls need
    hk_fn: Callable[[dict], object]    # the hk op; returns its output
    torch_fn: Callable[[dict], object]  # the torch call it replaces, for timing
    ref: Callable[[dict], object]      # fp32 reference for the check
    rel: float                         # allowed error, relative to the reference's magnitude
    work: float = 0.0                  # bytes or flops per call, for the throughput column
    unit: str = ""


def _cases(quick: bool) -> List[Case]:
    import torch
    import torch.nn.functional as F

    dev, bf16 = "cuda", torch.bfloat16
    rows, hidden, mlp = (2048, 4096, 4096) if quick else (8192, 4096, 12288)
    nseq = 1024 if quick else 4096

    def randn(*s, scale=1.0):
        return scale * torch.randn(*s, device=dev, dtype=bf16)

    def rope_ref(t):
        x, cos, sin = t["x"].float(), t["cos"], t["sin"]
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def rope_inputs():
        heads, seq, d = 32, (512 if quick else 2048), 128
        ang = torch.rand(seq, d // 2, device=dev) * 6.28
        return {"x": randn(heads, seq, d), "cos": ang.cos(), "sin": ang.sin()}

    def quant_ref(t):
        x = t["x"].float()
        scale = x.abs().amax(-1, keepdim=True) / 127
        return (x / scale).round().clamp(-127, 127) * scale

    def quant_torch(t):
        x = t["x"].float()
        scale = x.abs().amax(-1, keepdim=True) / 127
        return (x / scale).round().clamp(-127, 127).to(torch.int8), scale

    def dequant_hk(t):
        q, s = hk.ops.quant.quantize(t["x"])
        return q.float() * s[..., None]

    def attn_inputs(causal):
        return lambda: {"q": randn(1, 32, nseq, 128), "k": randn(1, 32, nseq, 128),
                        "v": randn(1, 32, nseq, 128), "causal": causal}

    def attn_ref(t):
        return F.scaled_dot_product_attention(t["q"].float(), t["k"].float(),
                                              t["v"].float(), is_causal=t["causal"])

    def mm_inputs(m, n, k):
        return lambda: {"a": randn(m, k), "b": randn(n, k)}

    elem_bytes = 3 * rows * hidden * 2
    cases = [
        Case("add", "elementwise", f"{rows}x{hidden}",
             lambda: {"a": randn(rows, hidden), "b": randn(rows, hidden)},
             lambda t: hk.ops.elementwise.add(t["a"], t["b"]),
             lambda t: t["a"] + t["b"],
             lambda t: (t["a"] + t["b"]).float(), 0.0, elem_bytes, "TB/s"),
        Case("exp", "elementwise", f"{rows}x{hidden}",
             lambda: {"a": randn(rows, hidden)},
             lambda t: hk.ops.elementwise.exp(t["a"]),
             lambda t: torch.exp(t["a"]),
             lambda t: torch.exp(t["a"].float()), 1e-2, 2 * rows * hidden * 2, "TB/s"),
        Case("rmsnorm", "norm", f"{rows}x{hidden}",
             lambda: {"x": randn(rows, hidden)},
             lambda t: hk.ops.norm.rmsnorm(t["x"]),
             lambda t: F.rms_norm(t["x"], (hidden,), eps=1e-6),
             lambda t: t["x"].float() * torch.rsqrt(t["x"].float().pow(2).mean(-1, keepdim=True) + 1e-6),
             2e-2, 2 * rows * hidden * 2, "TB/s"),
        Case("layernorm", "norm", f"{rows}x{hidden}",
             lambda: {"x": randn(rows, hidden)},
             lambda t: hk.ops.norm.layernorm(t["x"]),
             lambda t: F.layer_norm(t["x"], (hidden,), eps=1e-5),
             lambda t: F.layer_norm(t["x"].float(), (hidden,), eps=1e-5),
             2e-2, 2 * rows * hidden * 2, "TB/s"),
        Case("softmax", "norm", f"{rows}x{hidden}",
             lambda: {"x": randn(rows, hidden, scale=4.0)},
             lambda t: hk.ops.norm.softmax(t["x"]),
             lambda t: torch.softmax(t["x"], -1),
             lambda t: torch.softmax(t["x"].float(), -1),
             2e-2, 2 * rows * hidden * 2, "TB/s"),
        Case("silu_mul", "fused", f"{rows}x{hidden}",
             lambda: {"a": randn(rows, hidden), "b": randn(rows, hidden)},
             lambda t: hk.ops.fused.silu_mul(t["a"], t["b"]),
             lambda t: F.silu(t["a"]) * t["b"],
             lambda t: F.silu(t["a"].float()) * t["b"].float(),
             2e-2, 3 * rows * hidden * 2, "TB/s"),
        Case("rope", "fused", "32 heads x seq x 128",
             rope_inputs,
             lambda t: hk.ops.fused.rope(t["x"], t["cos"], t["sin"]),
             lambda t: rope_ref({"x": t["x"], "cos": t["cos"], "sin": t["sin"]}).to(bf16),
             rope_ref, 2e-2),
        Case("quantize int8", "quant", f"{rows // 2}x{hidden}",
             lambda: {"x": randn(rows // 2, hidden)},
             dequant_hk, quant_torch, quant_ref, 1e-2,
             (rows // 2) * hidden * 3, "TB/s"),
        Case("gemm", "gemm", f"{hidden}^3",
             mm_inputs(hidden, hidden, hidden),
             lambda t: hk.ops.gemm.matmul(t["a"], t["b"]),
             lambda t: torch.matmul(t["a"], t["b"].T),
             lambda t: t["a"].float() @ t["b"].float().T,
             1e-2, 2.0 * hidden ** 3, "TFLOP/s"),
        Case("gemm (MLP up)", "gemm", f"M={rows // 4} N={mlp} K={hidden}",
             mm_inputs(rows // 4, mlp, hidden),
             lambda t: hk.ops.gemm.matmul(t["a"], t["b"]),
             lambda t: torch.matmul(t["a"], t["b"].T),
             lambda t: t["a"].float() @ t["b"].float().T,
             1e-2, 2.0 * (rows // 4) * mlp * hidden, "TFLOP/s"),
        Case("decode GEMV", "decode", f"M=1 N={hidden} K={hidden}",
             mm_inputs(1, hidden, hidden),
             lambda t: hk.ops.decode_gemm.matmul(t["a"], t["b"]),
             lambda t: F.linear(t["a"], t["b"]),
             lambda t: t["a"].float() @ t["b"].float().T,
             1e-2, 2.0 * hidden * hidden, "TB/s"),
        Case("decode GEMV (MLP)", "decode", f"M=1 N={mlp} K={hidden}",
             mm_inputs(1, mlp, hidden),
             lambda t: hk.ops.decode_gemm.matmul(t["a"], t["b"]),
             lambda t: F.linear(t["a"], t["b"]),
             lambda t: t["a"].float() @ t["b"].float().T,
             1e-2, 2.0 * mlp * hidden, "TB/s"),
    ]
    for causal in (False, True):
        cases.append(Case(
            "attention" + (" causal" if causal else ""), "attention",
            f"1x32x{nseq}x128", attn_inputs(causal),
            lambda t: hk.attention(t["q"], t["k"], t["v"], causal=t["causal"]),
            lambda t: F.scaled_dot_product_attention(t["q"], t["k"], t["v"],
                                                     is_causal=t["causal"]),
            attn_ref, 2e-2,
            4.0 * 32 * nseq * nseq * 128 * (0.5 if causal else 1.0), "TFLOP/s"))
    return cases


def main():
    def extra(p):
        p.add_argument("--only", help="comma-separated case names or groups")
        p.add_argument("--quick", action="store_true", help="smaller shapes")
        p.add_argument("--list", action="store_true", help="list the cases and exit")

    args = _common.parse(__doc__, extra)
    if args.dump or args.build:
        sys.exit("validate.py runs the shipped ops on a GPU; --dump / --build "
                 "belong to the single-kernel examples")
    if args.list:
        import torch  # noqa: F401  -- the case table builds torch lambdas
    else:
        _common.require_gpu()
    cases = _cases(args.quick)
    if args.only:
        want = {w.strip() for w in args.only.split(",")}
        cases = [c for c in cases if c.name in want or c.group in want]
    if args.list:
        for c in cases:
            print(f"{c.group:<12} {c.name:<20} {c.shape}")
        return 0

    rows, failures = [], 0
    for c in cases:
        print(f"{c.group}: {c.name}, {c.shape}")
        t = c.setup()
        try:
            _common.check(f"{c.name} vs fp32 torch", c.hk_fn(t), c.ref(t), rel=c.rel)
        except Exception as e:  # noqa: BLE001 -- report and keep going
            failures += 1
            print(f"  FAILED: {e}")
            rows.append((c, False, None))
            continue
        best = None
        if not args.no_bench:
            best = _common.compare({"hk": lambda: c.hk_fn(t), "torch": lambda: c.torch_fn(t)},
                                   args.rounds)
        rows.append((c, True, best))

    print(f"\n{'case':<22}{'shape':<26}{'check':>7}{'hk ms':>10}{'torch ms':>10}{'hk vs torch':>13}")
    for c, ok, best in rows:
        line = f"{c.name:<22}{c.shape:<26}{'ok' if ok else 'FAIL':>7}"
        if best is not None:
            line += (f"{best['hk']:>10.3f}{best['torch']:>10.3f}"
                     f"{best['torch'] / best['hk']:>12.2f}x")
        print(line)
    return failures


if __name__ == "__main__":
    sys.exit(main())
