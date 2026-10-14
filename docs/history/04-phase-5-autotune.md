# Phase 5: `hk.autotune`

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

**Phase 5 is done: `hk.autotune` replaces `sweep.sh`, and most of what it
measured turned out not to be worth acting on.** A `Tuner` is a kernel factory, a schedule
space and a record. Searching it is two stages, and the first one never touches
the GPU:

```
python3 -m hk.autotune --list
python3 -m hk.autotune --tune gemm_bf16 -j 32      # search
python3 -m hk.autotune --ship                      # promote what is real
python3 -m hk.autotune --aot  -j 32                # prebuild it all
```

Stage 1 builds every candidate through `runtime.warm` -- 32 hipcc invocations
at a time -- and three kinds of candidate die there for free: the factory's own
arithmetic refuses a tiling that does not divide, the verifier refuses an
illegal layout, and the resource gate refuses anything that spills. Of
attention's 32 schedules, 26 never reach the GPU. Stage 2 times what is left in
*one process*, interleaved, reversing the order on odd rounds and scoring each
candidate by its minimum, because cross-process A/B on this chip manufactures
13-16% differences out of nothing.

The production path never searches. `Tuner.kernel(key)` is a dict probe and a
memoised factory call, so a record costs nothing per launch; a missing, stale
or foreign record costs performance and never correctness.

`min` over a column of noisy numbers always names a winner, so a tuner that
ships its winner unconditionally ships noise as a finding. `tune()` therefore
also records `default_ms`, and a record changes nothing unless it beat the
default by `MIN_GAIN` (1%). The bar is applied at `--ship` *and* on the lookup
path, so a record measured here is held to the same standard as one that
arrived in a wheel; `HK_AUTOTUNE_MIN_GAIN` moves it and `HK_AUTOTUNE=0` ignores
records entirely.

Out of 16 keys across five tuners, four clear it:

```
tuner                  key                 winner                  gain
gemm_bf16              4096x4096x4096      k_step32 wgm16 pos1    +1.2%
attn_fwd_d128          n16384              qk1 pv2                +1.1%
attn_fwd_d128          n65536              qk1 pv2                +1.1%
attn_fwd_d128_causal   n4096               qk1 pv1                +2.8%
```

The other twelve stay on the default, including every `gemm_bf16` shape but
one and all six `attn_fwd_d64` keys. That is the result and not a gap: a space
whose winners are inside the noise is a space whose default was already right.

The bar had to be checked against itself before any of this was believable.
The first attention sweep reported gains of 0.10-0.21% with four *different*
winners across three lengths -- a coin. A second sweep on the same machine
reported 0.26% to 2.8%. Two runs of one tuner disagreeing by 10x is not a
result, so the three keys above were re-measured three times each at
`--rounds 5`:

```
key                        repeat 1        repeat 2        repeat 3
d128 n16384          qk1 pv2 +1.04%  qk1 pv2 +0.99%  qk1 pv2 +1.00%
d128 n65536          qk1 pv2 +1.12%  qk1 pv2 +1.12%  qk1 pv2 +1.09%
d128_causal n4096    qk2 pv1 +1.81%  qk1 pv4 +2.13%  qk1 pv2 +1.29%
```

The first two reproduce exactly -- same schedule, same gain to within 0.05 --
so `qk1 pv2` is a real if small win at head_dim 128, and the original sweep's
0.1% was the measurement that was wrong. The third says something different
and more useful: at `n4096` causal the gain is reproducible but the *winner* is
not, because the default (`qk2 pv2`) is specifically the slow one there and all
three alternatives beat it by about the same amount. The shipped record names
one of them, and this paragraph is what stops the next reader from concluding
that `qk1 pv1` is special.

`--aot` prebuilds every schedule production can reach -- the recorded ones *and*
the default, since the default is what an unrecorded key gets -- into the
content-addressed cache, which is what takes hipcc's 20-40 s off the first
request. `pip install -e .` works from the repo root; `hk/tuned/*.json` and
`hk/include/**` are package data, because the generated C++ `#include`s the
headers at JIT time.

---

← [Phase 4](03-phase-4-attention.md) · [Development record](README.md) · [Phase 6](05-phase-6-torch-ops.md) →
