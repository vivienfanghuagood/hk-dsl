"""A KV cache write: each new token's K row into its cache slot, a scatter.

`cache.scatter(t, slots, at, col=...)` stores tile row r to cache row
`slots[at + r]`. A slot of -1 -- what a CUDA graph's padding tokens carry --
is outside the cache, and the buffer store drops it: nothing to branch on,
nothing written. The cache here is vLLM's (slots, kv_heads * head_dim) view
of its pages.

    python3 examples/09_kv_scatter.py
    python3 examples/09_kv_scatter.py --dump cpp
"""

import _common

import hk

T = hk.RT(hk.bf16, 16, 64)


@hk.kernel(grid=lambda p: (p.k.cols // 64, hk.cdiv(p.k.rows, 16), 1))
def kv_scatter(k: hk.Tensor[hk.bf16], slots: hk.Tensor[hk.i32],
               cache: hk.Tensor[hk.bf16]):
    at, col = hk.arch.block_idx.y * 16, hk.arch.block_idx.x * 64
    t = k[hk.elem_coord(0, 0, at, col)].load(T, bounded=True)
    cache.scatter(t, slots, at, col=col)


def _triton_scatter():
    import triton
    import triton.language as tl

    @triton.jit
    def scatter_rows(src, slots, cache, d, BLOCK: tl.constexpr):
        slot = tl.load(slots + tl.program_id(0)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(src + tl.program_id(0).to(tl.int64) * d + cols)
        tl.store(cache + slot * d + cols, v, mask=(slot >= 0) & (cols < d))

    def run(src, slots, cache):
        d = src.shape[1]
        scatter_rows[(slots.numel(), triton.cdiv(d, 1024))](src, slots, cache,
                                                           d, BLOCK=1024)
    return run


def main():
    args = _common.parse(__doc__)
    if _common.static(args, kv_scatter):
        return
    _common.require_gpu()
    import torch

    n_slots, d = 32768 * 16, 8 * 128     # 32k pages of 16, Qwen3-8B's K row
    cache = torch.zeros(n_slots, d, device="cuda", dtype=torch.bfloat16)
    for n in (16, 256, 4096):
        k = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
        slots = torch.randperm(n_slots, device="cuda")[:n].to(torch.int32)
        slots[::7] = -1                   # graph padding: must be skipped
        print(f"\nKV cache write, {n} tokens of {d} bf16 into {n_slots} slots")
        cache.zero_()
        kv_scatter(k, slots, cache)
        want = torch.zeros_like(cache)
        live = slots >= 0
        want[slots[live].long()] = k[live]
        _common.check("this kernel vs torch", cache, want, rel=0.0)
        fns = {"hk (this kernel)": lambda: kv_scatter(k, slots, cache),
               "torch index_copy_": lambda: cache.index_copy_(
                   0, slots[live].long(), k[live])}
        try:
            tri = _triton_scatter()
            fns["Triton"] = lambda: tri(k, slots, cache)
        except ImportError:
            pass
        if args.no_bench:
            continue
        best = _common.compare(fns, args.rounds)
        _common.report("time", best, baseline="Triton" if "Triton" in fns
                       else "torch index_copy_",
                       work=2 * k.numel() * k.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
