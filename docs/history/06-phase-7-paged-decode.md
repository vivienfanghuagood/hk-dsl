# Phase 7: the decoder, as a vLLM attention backend

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Phase 7: the decoder, which the SDPA patch structurally could not reach.**
vLLM V1 keeps KV in a paged cache and dispatches through `AttentionImpl`, so
the way in is to *be* a backend. `hk/ops/paged.py` is the kernel and
`hk/integration/vllm_backend.py` registers it.

The cache layout decides the kernel. `gl` derives its strides from its dims,
so vLLM's default `(block, token, head, dim)` cannot yield a contiguous
`(block_size, head_dim)` tile for one head. A backend declares its own shape,
so this one declares `(2, num_blocks, num_kv_heads, block_size, head_size)` --
exactly `gl`'s `(b, d, r, c)`, so one page of one head is a plain tile load.
The price is that the cache *update* becomes a strided scatter, and that is
the right way round: a decode step reads `seq_len` pages per layer and writes
one.

**It loses, and an earlier version of this record said it won.** That claim came
from Qwen3-0.6B, four short prompts, measured as total wall time. Total wall
time mixes prefill with decode and is neither latency nor throughput, and a
0.6B model spends so little of a step in attention that the number was mostly
measuring something else. Retracted.

The measurement that means something isolates one decode step. Each point runs
the same requests at two output lengths and takes the *slope*,
`(T(72) - T(8)) / 64`, which cancels prefill and every other fixed cost.
Qwen3-8B (36 layers, 32 query heads over 8 KV heads, head_dim 128), bf16,
eager, three reps per endpoint, `tools/hk-bench/vllm_sweep.py`:

```
                 ms / decode step            tokens / s
ctx   batch     hk     triton   ratio      hk    triton
1024      1   54.43    42.01    1.30x    18.4     23.8
1024      4   54.60    42.83    1.27x    73.3     93.4
1024     16   57.65    46.54    1.24x   277.5    343.8
1024     32   59.23    61.53    0.96x   540.3    520.1   <- the only win
4096      1   82.68    42.97    1.92x    12.1     23.3
4096      4   83.09    45.97    1.81x    48.1     87.0
4096      8   83.86    49.13    1.71x    95.4    162.8
4096     16   91.74    56.78    1.62x   174.4    281.8
```

Measurement bands were under 1% of the value at every point except hk at
ctx 1024 batch 1, so the ordering is not in question.

**Why, in one number.** Going from 1024 to 4096 tokens of context at batch 1
costs Triton **+0.97 ms/step** and this kernel **+28.2 ms/step**. That extra
3072 tokens is 12.6 MB of K and V per layer, so per layer:

```
triton   0.027 ms   469 GB/s    54% of the W7900D's 864 GB/s
hk       0.785 ms    16 GB/s     1.9%
```

Decode attention is a bandwidth problem and this kernel is getting 2% of the
bandwidth. The cause is the limitation already written down when it was built:
**no split along the KV axis**. One workgroup per (request, kv head) means
batch 1 launches 8 workgroups of one warp each on a 48-WGP GPU, and they walk
the whole context serially. Triton splits the KV range across the machine.

That also explains the shape of the whole table. hk is nearly flat in batch --
54.4 ms at batch 1, 59.2 at batch 32 -- because at low batch the GPU is idle
and extra requests fill it for free. Triton starts near the bandwidth limit
and degrades as requests compete for it (42.0 -> 61.5). The two cross at
batch 32, which is the only cell hk wins, and it wins it by 4%.

**Splitting the KV axis closes most of it.** `paged_decode_split_kernel` cuts
the page range `splits` ways and writes a partial `(O, m, l)` per split;
`merge_splits` combines them. How many ways was measured rather than reasoned
(`tools/hk-bench/paged_splits.py`, fifteen shapes), and the first guess was
wrong by an order of magnitude: the best split at every point puts
`n_reqs * n_kv_heads * splits` near **a thousand** workgroups, not the 96 that
"two per WGP" suggests -- which is the right instinct for a twelve-warp
workgroup and wrong for a one-warp one. At batch 16 the old rule asked for one
split, i.e. 128 workgroups of a single warp each walking 128 pages in series.

```
                 ms / decode step
ctx   batch   before   split   +planner   triton   ratio
1024      1    54.43   48.60     49.16     42.01   1.17x
1024      4    54.60   51.66     51.10     42.83   1.19x
1024     16    57.65   57.65     54.46     46.54   1.17x
1024     32    59.23   59.68     59.59     61.53   0.97x   <- win
4096      1    82.68   51.10     50.67     42.97   1.18x
4096      4    83.09   61.65     53.51     45.97   1.16x
4096      8    83.86   71.20     58.06     49.13   1.18x
4096     16    91.74   91.29     64.54     56.78   1.14x
```

Worst case 1.92x -> 1.19x. Standalone, the kernel now reaches 56% of the
card's 864 GB/s at ctx 4096 batch 16 and 76% at ctx 16384 batch 32, against
1.9% before.

**And the remaining gap is flat, which says what it is.** 1.14-1.19x at every
shape is not a kernel that is uniformly slower -- a kernel-speed difference
would grow with context, as the 1.92x did. It is a constant ~7-9 ms per step,
or ~0.2 ms per layer, and the backend's `forward` issues about a dozen small
torch ops per layer around the kernel: the query pad copy, the output copy,
two scatters for the cache update, and six or seven elementwise passes in
`merge_splits`. At ROCm's launch cost that is the whole difference. It is the
same class of mistake as the `.tolist()` that made the first version slower
than Triton, one level down.

So: the kernel is no longer the problem and the wrapper is. Folding the merge
and the cache update into kernels, and keeping the pad and the output copy out
of the per-layer path, is what a later round has to do -- and the flatness of
that column is the evidence that it is worth doing.

---

← [Phase 6](05-phase-6-torch-ops.md) · [Development record](README.md) · [Against the engine](07-against-the-engine.md) →
