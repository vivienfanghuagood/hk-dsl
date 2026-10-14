# A reduction across workgroups, and the decode GEMV that did not need it

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The IR had no way for workgroups to combine results inside one launch, which
is what split-K across workgroups needs. `hk.wg_arrive(counter, idx,
expected)` is that primitive, as a counter rather than fp32 atomics:

* Each workgroup stores its partial and arrives. The one that gets back
  `expected - 1` arrived last, and finishes by reading the partials in a
  fixed order. So the answer does not depend on which workgroup that was --
  20 repeats are bit-identical, which a greedy-token comparison against the
  engine needs and fp32 atomics would not give.
* The fences (release before the increment, acquire after) make every store
  issued before the call visible to the last arrival.
* The last arrival resets its counter, so one zeroed int32 tensor serves
  every launch and every replay of a captured graph with no memset.
* Its broadcast slot is 16 B of static LDS outside the allocator, and the
  first kernel that used it asked for 65,552 B and failed to launch
  (`HSA_STATUS_ERROR_INVALID_ALLOCATION`). The IR now counts static LDS and
  the verifier refuses the sum over 64 KB at build time, with that error in
  the message.

`decode_gemm(splits=S)` puts it to work: workgroup (b, s) takes the s-th
slice of K, and the last of the S finishes the block. It is correct, and it
is slower on every shape at every split, M=1 and M=16 alike (one process,
>= 1.5 GB of distinct weights per shape):

```
us at M=1      splits 1    splits 2    splits 3    splits 4
qkv                66.4        76.9          --        81.8
o                  46.5        51.4          --        55.0
gate_up           252.4       271.2          --       294.2
down              139.5       144.3       153.6       146.9
lm_head          1515.5      1663.0          --      1768.4
```

The case for it was `down`: 256 blocks over 96 resident slots leave a
third-full last round. The workspace round trip and the second pass cost
more than that tail. So the decode GEMV ships unsplit, and `wg_arrive`
ships as a capability for kernels whose grid really is too small -- a
negative result, measured, not a guess.

Where the decode GEMV stands now, in GEMM milliseconds per decode step (36
layers plus the LM head, `tools/hk-bench/vllm_gemm.py --upstream-skinny`,
>= 1.5 GB of distinct weights per shape, every variant gated against fp32):

```
tokens   hipBLASLt   rocBLAS   upstream wvSplitK   hk decode   vs hipBLASLt
     1       39.98     37.02               19.11       19.66        2.03x
     2       39.68     37.57               19.87       19.69        2.01x
     4       39.72     39.02               20.40       19.77        2.01x
     8       39.50     40.86                  --       19.89        1.99x
    16       41.00     41.75                  --       20.23        2.03x
    24       39.74     43.93                  --       20.56        1.93x
    32       40.48     46.81                  --       20.84        1.94x
```

Against the ceiling a plain read reaches (`hbm_probe.hip`, one kernel per
read: 676 GB/s for `o`'s 34 MB up to 821 for 302 MB), at one token: `qkv`
762 GB/s, `o` 724, `gate_up` 800, `down` 720, LM head 819. Upstream's
`wvSplitK`, which handles one to five tokens, is level on `qkv`, 1.5-2.6%
ahead on `o`, `gate_up` and the LM head, and 6.9% ahead on `down` (770
GB/s). That last one is where what is left lives. The row predicate is
also worth seeing here: without the pad, two and four tokens now beat
`wvSplitK` per step, and 32 tokens dropped from 22.3 to 20.8 ms.

**End to end.** Qwen3-8B, bf16, one W7900D. The baseline (1.00x) is stock
vLLM 0.16.1.dev0 choosing its own backends; the patched arm is the same
engine with `HK_VLLM_GEMM=1`, so every unquantized linear runs on the decode
GEMV (M <= 64) or the generated GEMM and everything else is stock. Prefix
caching and chunked prefill are off, so a prefill step is whole prompts. The
arms alternate in separate processes, min over rounds (two in graph mode,
one in eager). `total` is the whole batch from prompt to last token,
`prefill` is the same batch generating one token, and `decode` is one step at
the full batch, from the difference of two output lengths
(`tools/hk-bench/vllm_e2e.py`).

Graph mode (torch.compile plus CUDA graphs, vLLM's default), 72 output
tokens (128 at batch 32 and 64):

```
             total ms              prefill ms            decode ms per step
ctx x batch  stock     hk   x      stock     hk   x      stock     hk   x
1024x1      3211.6  1750.2 1.84    273.9   243.0 1.13    41.37  21.23 1.95
1024x4      4029.4  2489.2 1.62   1025.1   922.7 1.11    42.30  22.05 1.92
1024x16     7551.4  5600.6 1.35   4291.5  3706.4 1.16    45.90  26.04 1.76
1024x32    16346.1 12836.7 1.27   8616.1  7403.7 1.16    61.59  42.56 1.45
1024x64    28382.1 24055.6 1.18  17341.0 14881.4 1.17    88.22  71.72 1.23
4096x1      4452.9  2930.6 1.52   1446.3  1364.0 1.06    42.35  22.03 1.92
4096x4      9056.0  7237.1 1.25   5929.3  5439.1 1.09    44.97  24.45 1.84
4096x16    27523.8 24837.4 1.11  23860.2 21779.3 1.10    56.11  36.38 1.54
```

Eager (`enforce_eager=True`), 72 output tokens, a subset:

```
1024x1      3258.2  1792.3 1.82    274.5   243.8 1.13    42.09  21.83 1.93
1024x16     7545.4  5604.0 1.35   4283.1  3699.6 1.16    46.45  26.15 1.78
1024x32    12858.9 10483.0 1.23   8623.4  7423.4 1.16    61.01  42.10 1.45
4096x1      4483.2  2971.2 1.51   1440.2  1365.4 1.06    42.93  22.65 1.90
4096x16    27452.4 24869.2 1.10  23775.4 21748.2 1.09    55.43  37.14 1.49
```

How to read it:

* **Batch 1 is a latency number.** The whole request is 1.52-1.84x faster,
  and nearly all of that is decode, whose GEMMs halve (41-43 ms to 21-23 ms
  per token). Prefill latency improves 6-13%.
* **Batches are throughput numbers.** The batch finishes 1.10-1.35x sooner
  at 16 requests and 1.18-1.27x sooner at 32-64. The gain shrinks with batch
  and with context, as the profile predicts. Attention, which this patch
  does not touch, grows with both. And at 64 tokens the decode GEMV leads
  hipBLASLt by 1.65x, not 2x.
* **Nothing loses:** no shape, no phase, in either mode. On every shape the
  first request's greedy output is token-identical to stock (72/72,
  128/128).
* **Graph and eager agree to within 2.5% for both arms.** On gfx1100 stock
  vLLM's step is GEMM-bound, and capturing it saves launch overhead that is
  small next to 40 ms of GEMMs.
* **Not every call was routed.** At batch 32-64, vLLM captures graphs for
  every batch size up to 512. Under the hand rule, the capture sizes between
  64 and 1024 that are not whole 128-row blocks fell back (2,304 calls). The
  measured routing table [below](14-routing-table.md) routes the ones that measured faster.

---

← [Row-predicated loads and stores](10-row-predicated.md) · [Development record](README.md) · [Strided globals](12-strided-globals.md) →
