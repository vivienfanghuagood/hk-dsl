"""Your kernel in vLLM: write it, register it, check it, patch it in, A/B it.

The kernel is a fused SiLU-and-mul. vLLM's `SiluAndMul` is handed the
gate_up projection's (tokens, 2I) output and returns
silu(x[:, :I]) * x[:, I:]; the kernel reads both halves of the one tensor
in place (a global may start at a column offset of another) and writes the
product.

  1. **Register** it as a torch op, `torch.ops.hk_user.silu_mul`, with the
     torch reference it replaces, an output allocator (also its fake, for
     torch.compile) and a route deciding which calls take the kernel:
     `hk.integration.register.op`.
  2. **Check** it against the reference: `op.check(x)`.
  3. **Patch** vLLM: `register.patch(...)` rebinds `SiluAndMul.forward_native`
     (with vLLM's default compile config, custom ops off, that is the method
     its graphs trace) to call the op.
  4. **A/B** it end to end. The plugin imports this file in vLLM's own
     process, which is where the patch has to happen:

    python3 examples/11_vllm_integration.py              # check, then time vs torch
    HK_VLLM_USER=examples/11_vllm_integration.py \\
        python -m hk.integration ab --model Qwen/Qwen3-8B --arms user \\
        --context 1024 --batch 1,16 --short 16 --tokens 48
    HK_VLLM_USER=examples/11_vllm_integration.py vllm serve Qwen/Qwen3-8B

   The served engine also runs hk's default routes for this GPU (hk is on by
   default since 0.7.0; `HK_PATCH_VLLM=1` would add the encoder SDPA patch
   too). `ab` keeps them out of both arms, so it measures this patch alone.

Measure before you ship a patch: vLLM compiles its own SiLU-and-mul with
Inductor, and this kernel is not guaranteed to beat it. The route is where a
kernel that wins only at some shapes keeps only those.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common  # noqa: E402

import hk  # noqa: E402

T = hk.RT(hk.fp32, 16, 64)
WARPS = 4


@hk.kernel(warps=WARPS, grid=lambda p: (hk.cdiv(p.o.cols, 64 * WARPS),
                                        hk.cdiv(p.o.rows, 16), 1))
def silu_mul(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    r0 = hk.arch.block_idx.y * 16
    c0 = (hk.arch.block_idx.x * WARPS + hk.arch.warp_id()) * 64
    g = x.at[r0, c0].load(T, bounded=True)               # the gate half
    u = x.at[r0, o.cols + c0].load(T, bounded=True)      # the up half
    g.silu(into=g)
    g *= u
    o.at[r0, c0].store(g, bounded=True)


_OP = []


def registered():
    """The kernel as `torch.ops.hk_user.silu_mul`, registered on first use."""
    if not _OP:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        from hk.integration import register  # noqa: PLC0415

        _OP.append(register.op(
            "silu_mul", kernel=silu_mul,
            reference=lambda x: F.silu(x[:, :x.shape[1] // 2]) * x[:, x.shape[1] // 2:],
            out=lambda x: x.new_empty(x.shape[0], x.shape[1] // 2),
            # Whole 256-column blocks, bf16: what the kernel was written for.
            route=lambda x: x.dtype == torch.bfloat16 and (x.shape[1] // 2) % 256 == 0))
    return _OP[0]


def install() -> str:
    """Patch vLLM's SiluAndMul to call the op."""
    import inspect  # noqa: PLC0415

    from vllm.model_executor.layers.activation import SiluAndMul  # noqa: PLC0415

    from hk.integration import register  # noqa: PLC0415

    op = registered()

    def forward_native(*args):
        x = args[-1]                       # (self, x), or (x,) if it is static
        d = x.shape[-1] // 2
        return op(x.reshape(-1, 2 * d)).view(*x.shape[:-1], d)

    static = isinstance(inspect.getattr_static(SiluAndMul, "forward_native"), staticmethod)
    return register.patch(
        "vllm.model_executor.layers.activation.SiluAndMul.forward_native",
        staticmethod(forward_native) if static else forward_native)


# Loaded by the vLLM plugin or `ab` from HK_VLLM_USER: install the patch.
if __name__.startswith("hk_user_"):
    print("[hk] " + install(), flush=True)


def main():
    args = _common.parse(__doc__)
    if _common.static(args, silu_mul):
        return
    _common.require_gpu()
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    op = registered()
    for tokens in (16, 256, 4096):
        inter = 12288                          # Qwen3-8B's intermediate size
        x = torch.randn(tokens, 2 * inter, device="cuda", dtype=torch.bfloat16)
        print(f"\nSiLU-and-mul, {tokens} x 2*{inter} bf16")
        err = op.check(x)
        print(f"  this kernel vs torch        ok   max error {err:.2e} (allowed 2e-02)")
        if args.no_bench:
            continue
        compiled = torch.compile(lambda t: F.silu(t[:, :inter]) * t[:, inter:])
        best = _common.compare({
            "hk (this kernel, as the op)": lambda: op(x),
            "torch eager": lambda: F.silu(x[:, :inter]) * x[:, inter:],
            "torch.compile (Inductor)": lambda: compiled(x),
        }, args.rounds)
        _common.report("time", best, baseline="torch.compile (Inductor)",
                       work=3 * tokens * inter * 2, unit="TB/s")
    print(f"\n{op.summary()}")


if __name__ == "__main__":
    main()
