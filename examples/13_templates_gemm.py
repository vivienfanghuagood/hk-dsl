"""Linear + bias + GELU as one kernel: hk.templates.gemm with an epilogue.

    y = gelu(x @ w.T + b)        # F.gelu(F.linear(x, w, b), approximate="tanh")

The template is the GEMM hk.ops.matmul runs on this GPU, shape for shape --
the tuned WMMA kernel on a Radeon, the routed pingpong / four-warp / split-K
kernels on an MI355X -- and the epilogue below is traced into its store: the
bias and the GELU are applied to the fp32 accumulator in registers, so they
cost a few percent of the GEMM instead of a second pass over C. On a split-K
route the epilogue runs once, on the sum, in the reduction.

All the user code is the epilogue and the call (between the rules below);
docs/templates/gemm.md has the rest of the API and the measurements.

    python3 examples/13_templates_gemm.py
    python3 examples/13_templates_gemm.py --dump cpp | grep -n "exp2\\|rsqrt"
    python3 examples/13_templates_gemm.py --build --arch gfx950
"""

import _common

import hk


# ---------------------------------------------------------------- your code
def bias_gelu(acc, ctx):
    """acc: an fp32 tile of x @ w.T; ctx.bias: b, loaded for its columns."""
    acc += ctx.bias
    return hk.templates.gemm.gelu_tanh(acc)    # acc.gelu() is the same, slower


def linear_bias_gelu(x, w, b):
    return hk.templates.gemm(x, w, epilogue=bias_gelu, bias=b)
# ---------------------------------------------------------------------------


M, N, K = 2048, 16384, 4096


def main():
    args = _common.parse(__doc__)
    if args.dump or args.build:
        import torch

        meta = torch.empty(N, dtype=torch.bfloat16, device="meta")
        launches = hk.templates.gemm.kernels(M, N, K, epilogue=bias_gelu, bias=meta)
        step, kernel, consts = next(x for x in launches if x[0].epilogue)
        _common.static(args, kernel, **consts)
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * K ** -0.5
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    print(f"y = gelu(x @ w.T + b), x {M}x{K}, w {N}x{K}, bf16 in, fp32 accumulate")
    print("  plan: " + "; ".join(map(str, hk.templates.gemm.plan(M, N, K))))
    ref = F.gelu(x.float() @ w.float().T + b.float(), approximate="tanh")
    _common.check("template vs fp32 torch", linear_bias_gelu(x, w, b), ref, rel=1e-2)
    if args.no_bench:
        return

    def eager():
        return F.gelu(F.linear(x, w, b), approximate="tanh")

    compiled = torch.compile(eager)
    contenders = {
        "hk.templates.gemm": lambda: linear_bias_gelu(x, w, b),
        "hk.ops.matmul alone": lambda: hk.ops.matmul(x, w),
        "hk.ops.matmul + torch": lambda: F.gelu(hk.ops.matmul(x, w) + b,
                                                approximate="tanh"),
        "torch.compile": compiled,
        "F.linear + F.gelu": eager,
    }
    best = _common.compare(contenders, args.rounds)
    _common.report("time", best, baseline="F.linear + F.gelu", work=2.0 * M * N * K,
                   unit="TFLOP/s")


if __name__ == "__main__":
    main()
