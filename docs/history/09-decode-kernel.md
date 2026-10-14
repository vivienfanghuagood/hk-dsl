# The decode kernel

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Why the GEMM cannot do it.** The DSL GEMM at a 16-row tiling already beats
hipBLASLt on every decode shape -- 1.19-1.62x, which shipped first and was
worth 1.36-1.44x on decode end to end -- but it streams only 418-755 GB/s.
Its schedule is the reason. Each workgroup keeps exactly one 18 KB batch of
loads in flight, and the grid is N/128 workgroups: 32 on a 4096-wide layer.
At about a microsecond of latency, Little's law caps that near 32 x 18 KB/us,
which is where `o` and `down` land. Split-K on the same schedule was measured
in the C++ (`thin_config`) and topped out at 372-633 GB/s, so the template was
not worth extending.

`hk/ops/decode_gemm.py` is laid out for the read instead, in about sixty
lines of DSL:

* **W goes straight to registers**, never through LDS -- it is used once.
  W*X^T puts W's rows on the WMMA's 16-row side, the way the paged decode
  kernel puts K there; the accumulator is C^T and its transpose is free.
* **One 16-row block of W per workgroup, K split across its warps**: N/16
  workgroups of 16 waves, 6144 waves on a 6144-wide layer.
* **The split is interleaved**: on round t, warp w reads K-chunk t*16 + w, so
  a workgroup reads 2 KB of each of its rows contiguously at once instead of
  128 B islands 8 KB apart.
* The warps' partial sums meet once, through LDS, at the end.

It was right the first time it ran: 32 cases across 1-64 rows, six warp
splits, strided input. Timed as a step reads the weights, M=16, microseconds:

```
              F.linear   GEMM 16-row   decode w4   w8      w16     w32
qkv             156.7       119.1        105.8    85.4    75.9    76.3
o               105.9        65.5         65.6    59.3    47.5    51.4
gate_up         592.4       354.4        311.5   278.1   257.5   261.0
down            250.6       166.6        168.9   168.1   145.5   150.0
lm_head        1980.3      1649.2       1632.6  1584.2  1551.2  1549.3
per step, ms     41.8        27.1         25.1    22.9    20.5    20.9
```

At 16 warps that is 663-802 GB/s against hipBLASLt's 317-638. More token
tiles cost registers -- each is another X operand and another accumulator per
wave -- so the schedule depends on the batch width (`decode_gemm.SCHEDULES`).
GEMM milliseconds per decode step, 36 layers plus the LM head:

```
tokens    installed vLLM   upstream wvSplitK   hk decode     vs installed
1-16           39.7-41.8      19.1-20.4 (<=5)       20.5      1.94-2.04x
17-32               40.4                  --        22.3           1.81x
33-64               42.2                  --        28.6           1.47x
```

So the band where nobody had a kernel now runs 7% behind upstream's GEMV at
one token, and the kernel covers the batch widths upstream's does not. The
last row is register-bound (four token tiles at 8 warps run at occupancy 5),
and it is the one with the most left.

End to end, vLLM as it ships (torch.compile, CUDA graphs, compile cache off),
arms alternating in separate processes, two rounds, min. The `gemm` arm is
now the prefill route [above](08-linear-layers.md) plus this kernel for M <= 64:

```
                     stock ms     gemm    hk+gemm
total request (72 output tokens)
  1024 ctx x 1         3202.4   1.717x   1.382x
  1024 x 4             4035.5   1.552x   1.309x
  1024 x 16            7625.0   1.354x   1.297x
  4096 x 1             4440.6   1.457x   1.436x
  4096 x 4             9059.0   1.227x   1.464x
  4096 x 16           27588.0   1.104x   1.476x
decode, ms per step at the full batch
  1024 x 1              41.24   1.808x   1.402x
  1024 x 16             46.99   1.807x   1.417x
  4096 x 1              42.09   1.784x   1.371x
  4096 x 16             57.18   1.580x   1.281x
wider batches (1024 ctx, 128 output tokens; decode over T(128) - T(64))
  total, x 32         16512.7   1.267x   1.370x
  total, x 64         28509.3   1.162x   1.362x
  decode, x 32          62.98   1.447x   1.591x
  decode, x 64          89.22   1.195x   1.668x
```

Prefill is unchanged from the table [above](08-linear-layers.md) (1.06-1.15x for `gemm`). Every arm
produced the same greedy tokens as stock at every shape, 72 of 72 and 128 of
128. Where each configuration wins and loses:

* **`gemm` alone is the best patch whenever decode dominates** -- short
  context, small batch: 1.72x on a whole 1024 x 1 request. Its decode gain
  shrinks as attention's share of the step grows, 1.81x at 1024 context to
  1.58x at 4096 x 16, and as the batch widens past one token tile.
* **`hk+gemm` is best on long-context prefill and on wide decode.** At batch
  1-16 the attention backend's decode is slower than stock's, which leaves
  decode at 0.75-0.81x of what `gemm` alone gets. At batch 32 and 64 that
  reverses: the same backend makes decode faster, 1.59x and 1.67x with the
  GEMMs, against 1.45x and 1.20x without it.
* **Not covered:** decode steps of 65 to 383 tokens stay on hipBLASLt (the
  wide runs route 2304 such calls), and batch 128 -- where the prefill GEMM
  is already 1.36x per step -- is still excluded by the prefill rule.

**Residency is a lever, and in the direction that looks wrong.** At 16 warps,
two workgroups per WGP read faster than four: 145 against 163 us on `down`,
47.5 against 50.8 on `o`, the same code either way. More concurrent 16-row
streams cost more than the latency they hide -- the likely cause is DRAM row
conflicts, but with the counters dead that is an inference, not a
measurement. The schedule therefore asks for two (`resident=2`, which sizes
the combine's LDS so a third does not fit).

Finding that took two wrong turns, both worth not repeating:

* **The first residency test measured nothing.** It reserved LDS for 2, 3 and
  4 workgroups per WGP against the 64 KB pool in `hk/target/gfx1100.py`, and
  nothing changed. Reserving 48 KB and 64 KB later both gave the
  two-per-WGP time, identical to each other; 32 KB had given the four-per-WGP
  one. Only a **128 KB pool** produces that pattern, and that is what RDNA3
  has; the target now says so. Every "LDS binds occupancy" estimate the
  verifier printed for gfx1100 before this was off by 2x.
* **The combine was blamed first.** The fp32 combine reads LDS with flat
  loads (the library vectorises only bf16 shared tiles), and batching those
  reads coincided with the slowdown, so a combine moved onto `ds_read_b128`
  -- partials as bf16 hi/lo pairs -- was built and measured. Same speed as
  the fp32 one at the same residency, 21.6 ms a step at four per WGP and
  20.6 at two. The flat loads were never the cost. (The DSL's LDS verifier
  had to learn that a counted wait inside an `hk.if_` may count reads issued
  in the same, still-open branch before that kernel would build; that fix
  stayed.)

Measured and not levers: 32 warps (20.9 ms), batching the combine's reads,
the combine's LDS path. What is left to 800 GB/s -- about 1.5 ms a step -- is
not explained yet.

---

← [The linear layers](08-linear-layers.md) · [Development record](README.md) · [Row-predicated loads and stores](10-row-predicated.md) →
