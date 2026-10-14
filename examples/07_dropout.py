"""Dropout from a seed: Triton's low-memory dropout (tutorial 04), drawing
the same random numbers, so keeping exactly the same elements.

Each element keeps with probability 1 - p, scaled by 1 / (1 - p). Its draw
is `hk.rand(T, seed, offset, row_stride)` -- Philox on the element's flat
offset, bit for bit what `triton.language.rand(seed, offsets)` draws -- so
nothing but the seed is stored to replay the mask in a backward pass.
`(r - p).step()` is the keep mask as a factor: 1 where r > p, else 0.

The seed is a one-element int32 tensor, read with `load_scalar`, rather than
an argument: a kernel's non-tensor arguments are constexprs, baked into the
build, and a seed that changed every step would compile every step.

    python3 examples/07_dropout.py
    python3 examples/07_dropout.py --dump cpp
"""

import _common

import hk

T = hk.RT(hk.fp32, 16, 64)


def dropout_kernel(p: float = 0.5):
    @hk.kernel(grid=lambda q: (q.x.cols // 64, hk.cdiv(q.x.rows, 16), 1))
    def dropout(x: hk.Tensor[hk.bf16], seed: hk.Tensor[hk.i32],
                o: hk.Tensor[hk.bf16], P: hk.Constexpr = p):
        r0, c0 = hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 64
        s = seed[hk.elem_coord(0, 0, 0, 0)].load_scalar()
        v = x[hk.elem_coord(0, 0, r0, c0)].load(T, bounded=True)
        keep = hk.rand(T, s, r0 * x.cols + c0, x.cols)
        keep -= P
        v *= keep.step()
        v *= 1.0 / (1.0 - P)
        o[hk.elem_coord(0, 0, r0, c0)].store(v, bounded=True)

    return dropout


def _triton_dropout():
    import triton
    import triton.language as tl

    @triton.jit
    def seeded_dropout(x_ptr, out_ptr, n, p, seed, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        keep = tl.rand(seed, offs) > p
        tl.store(out_ptr + offs, tl.where(keep, x / (1 - p), 0.0), mask=mask)

    def run(x, out, p, seed):
        n = x.numel()
        seeded_dropout[(triton.cdiv(n, 1024),)](x, out, n, p, seed, BLOCK=1024)
    return run


def main():
    args = _common.parse(__doc__)
    kernel = dropout_kernel()
    if _common.static(args, kernel):
        return
    _common.require_gpu()
    import torch

    rows, cols, p, seed = 8192, 4096, 0.5, 1234
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    s = torch.tensor([seed], device="cuda", dtype=torch.int32)
    o = torch.empty_like(x)
    print(f"dropout p={p}, {rows} x {cols} bf16")
    kernel(x, s, o)
    torch.cuda.synchronize()
    kept = (o != 0).float().mean().item()
    print(f"  kept {kept:.4f} of the elements (expected {1 - p})")
    fns = {"hk (this kernel)": lambda: kernel(x, s, o)}
    try:
        tri = _triton_dropout()
        t_out = torch.empty_like(x)
        tri(x, t_out, p, seed)
        torch.cuda.synchronize()
        # Same draws and an exact scale (1 / (1 - 0.5) = 2): the same elements
        # kept, with the same values. A dropped negative input is -0.0 here,
        # x * 0, and +0.0 from Triton's select: equal, not the same bits.
        assert torch.equal(o != 0, t_out != 0), "the masks differ from Triton's"
        assert torch.equal(o, t_out), "the kept values differ from Triton's"
        print("  hk vs Triton tutorial        ok   same mask, same values")
        fns["Triton tutorial"] = lambda: tri(x, t_out, p, seed)
    except ImportError:
        print("  (no triton here; skipping the comparison)")
    fns["torch F.dropout"] = lambda: torch.nn.functional.dropout(x, p)
    if args.no_bench:
        return
    best = _common.compare(fns, args.rounds)
    base = "Triton tutorial" if "Triton tutorial" in fns else "torch F.dropout"
    _common.report("time", best, baseline=base,
                   work=2 * x.numel() * x.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
