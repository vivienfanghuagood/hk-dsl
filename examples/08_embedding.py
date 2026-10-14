"""Embedding lookup: rows of a table by token id, a gather.

`table.gather(T, ids, at, col=...)` loads tile row r from table row
`ids[at + r]`. Each lane's row comes in as the voffset of one buffer load
over the whole table, so an id outside the table -- or a position past the
id list's end -- reads zeros with no branch: the bounded store then writes
only the rows that exist.

    python3 examples/08_embedding.py
    python3 examples/08_embedding.py --dump cpp
"""

import itertools

import _common

import hk

# 32 rows a wave, not 16: each gathered row is a random address, and a wave
# with more of them in flight hides more of their latency.
T = hk.RT(hk.bf16, 32, 64)


@hk.kernel(grid=lambda p: (p.o.cols // 64, hk.cdiv(p.o.rows, 32), 1))
def embedding(table: hk.Tensor[hk.bf16], ids: hk.Tensor[hk.i32],
              o: hk.Tensor[hk.bf16]):
    at, col = hk.arch.block_idx.y * 32, hk.arch.block_idx.x * 64
    t = table.gather(T, ids, at, col=col)
    o[hk.elem_coord(0, 0, at, col)].store(t, bounded=True)


def _triton_embedding():
    import triton
    import triton.language as tl

    @triton.jit
    def gather_rows(table, ids, out, d, BLOCK: tl.constexpr):
        row = tl.load(ids + tl.program_id(0)).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(table + row * d + cols)
        tl.store(out + tl.program_id(0).to(tl.int64) * d + cols, v)

    def run(table, ids, out):
        d = table.shape[1]
        gather_rows[(ids.numel(), d // 1024)](table, ids, out, d, BLOCK=1024)
    return run


def _cycling(fn, sets):
    """fn over each id set in turn: one set, repeated, would leave its table
    rows in the 96 MB Infinity Cache and time a cache, not an embedding."""
    it = itertools.cycle(sets)
    return lambda: fn(next(it))


def main():
    args = _common.parse(__doc__)
    if _common.static(args, embedding):
        return
    _common.require_gpu()
    import torch
    import torch.nn.functional as F

    vocab, d = 151936, 4096        # Qwen3's vocabulary, an 8B hidden size
    table = torch.randn(vocab, d, device="cuda", dtype=torch.bfloat16)
    for n in (1, 32, 4096):
        sets = [torch.randint(0, vocab, (n,), device="cuda", dtype=torch.int32)
                for _ in range(8)]
        ids = sets[0]
        o = torch.empty(n, d, device="cuda", dtype=torch.bfloat16)
        print(f"\nembedding, {n} tokens of {vocab} x {d} bf16")
        embedding(table, ids, o)
        _common.check("this kernel vs F.embedding", o,
                      F.embedding(ids.long(), table), rel=0.0)
        fns = {"hk (this kernel)": _cycling(lambda i: embedding(table, i, o), sets),
               "torch F.embedding": _cycling(lambda i: F.embedding(i, table), sets)}
        try:
            tri, t_out = _triton_embedding(), torch.empty_like(o)
            fns["Triton"] = _cycling(lambda i: tri(table, i, t_out), sets)
        except ImportError:
            pass
        if args.no_bench:
            continue
        best = _common.compare(fns, args.rounds)
        _common.report("time", best, baseline="torch F.embedding",
                       work=2 * o.numel() * o.element_size(), unit="TB/s")


if __name__ == "__main__":
    main()
