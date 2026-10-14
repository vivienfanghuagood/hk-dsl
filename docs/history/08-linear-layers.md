# The other 57.6%: the linear layers

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

On gfx1100 every one of those GEMMs ends in `torch.nn.functional.linear`, i.e.
hipBLASLt. The installed vLLM has a skinny-GEMM path for decode, but it is
gated on gfx9, and correctly: its `wvSplitK` is compiled for gfx9 only, and
called on gfx1100 it fails a device-side assert and takes the process with it.
So the engine brings no GEMM of its own here, and the question is only whether
the DSL GEMM beats the library at the shapes this model runs.

Phase 0 on exactly those shapes against exactly that call
(`tools/hk-bench/vllm_gemm.py`: one process, variants interleaved, weights
rotated through 3x the 96 MB Infinity Cache, every variant checked against
fp32 before it is timed). The kernel is the [Phase 3](02-phase-3-gemm.md) DSL GEMM, unchanged:

```
hk / F.linear              M=1024   M=4096   M=16384
qkv      6144 x 4096        1.15x    1.10x    1.10x
o        4096 x 4096        1.18x    1.09x    1.09x
gate_up 24576 x 4096        1.11x    1.16x    1.23x
down     4096 x 12288       1.17x    1.10x    1.17x
```

All 36 layers' GEMMs for one forward step at M=4096: 838.5 ms on hipBLASLt,
744.4 on hk, 1.13x. rocBLAS, the other library torch can be told to use, is
slower than hipBLASLt at every prefill shape.

Below 1024 rows it is a comb, not a curve:

```
M          128    192    256    384    512    768
qkv       1.38   0.88   1.13   1.43   1.20   1.17
o         1.15   0.75   0.95   1.11   1.24   1.17
gate_up   1.56   0.95   1.21   1.18   1.19   1.09
down      1.16   0.71   0.96   1.15   1.33   1.18
```

The last M block backs up to end at M and recomputes a full 128 rows, and at
small grids two workgroups then share a WGP. The rule that comes out of it:
**route when M >= 1024, or M >= 384 in whole 128-row blocks**; everything else
-- including all of decode, where M is the batch -- stays on vLLM's path.
(Decode got its own kernel afterwards; see "[The decode kernel](09-decode-kernel.md)" below.)
Steps that mix a new prompt with the running requests' decode tokens land just
above a multiple of 128, so those were measured too: M=4097..4111 costs hk
~18 ms per step over M=4096 for the extra block, and it still wins 1.09-1.10x.

`hk/integration/vllm_gemm.py` installs it, behind its own opt-in
(`HK_PATCH_VLLM=1 HK_VLLM_GEMM=1`). It rebinds `dispatch_unquantized_gemm` by
identity in every vLLM module that imported it, and what that returns calls a
single opaque custom op, `hk_vllm::linear`, with a fake implementation. Under
vLLM's torch.compile the token count is symbolic, so branching on it inside
traced code would be a guard and a recompile; inside the op it is neither, and
CUDA-graph capture sees a launch on torch's stream. Below the rule the op calls
vLLM's own `rocm_unquantized_gemm`.

End to end, eager like every earlier number in this record, arms alternating in
separate processes, two rounds, min; the rounds agreed to 0.3%, and every arm
produced the same 72 greedy tokens as stock at every shape:

```
                     stock ms     gemm       hk    hk+gemm
total request
  1024 ctx x 1         3268.5   1.010x   0.870x   0.877x
  1024 x 4             4085.5   1.025x   0.893x   0.912x
  1024 x 16            7533.5   1.070x   0.971x   1.024x
  4096 x 1             4504.8   1.019x   0.979x   1.000x
  4096 x 4             9117.3   1.040x   1.149x   1.205x
  4096 x 16           27542.9   1.044x   1.296x   1.366x
prefill (the same batch, one output token)
  1024 x 1              274.0   1.111x   1.059x   1.190x
  1024 x 4             1019.3   1.093x   1.081x   1.188x
  1024 x 16            4288.3   1.152x   1.104x   1.257x
  4096 x 1             1445.7   1.054x   1.452x   1.584x
  4096 x 4             5920.8   1.083x   1.437x   1.625x
  4096 x 16           23788.4   1.088x   1.438x   1.632x
```

The GEMM patch alone is never slower, and everything it buys is in prefill:
5-15% of the prefill, 1-7% of the request. The two patches compose -- the
prefill gains roughly multiply -- and together they are 1.37x on the
long-context batch.
What they cannot fix is the left half of the table, where the request is
nearly all decode and the attention backend's decode deficit shows through.

**And with vLLM as it ships** -- torch.compile and CUDA graphs on, the compile
cache off so no arm can load another's graph -- the same table to within a
percent. The attention backend runs under piecewise graphs (it declares it
cannot be captured whole; vLLM logs the downgrade), the GEMM patch under
vLLM's default full-and-piecewise. Decode is measured directly here, as the
slope `(T(72) - T(24)) / 48`, which cancels every step that admits a prompt:

```
                     stock ms     gemm       hk    hk+gemm
total request
  1024 ctx x 1         3216.2   1.007x   0.864x   0.870x
  1024 x 4             4035.1   1.024x   0.887x   0.904x
  1024 x 16            7556.9   1.073x   0.965x   1.025x
  4096 x 1             4452.6   1.016x   0.973x   0.993x
  4096 x 4             9063.7   1.040x   1.149x   1.201x
  4096 x 16           27517.9   1.044x   1.293x   1.364x
prefill
  1024 x 1              273.8   1.111x   1.064x   1.192x
  4096 x 16           23829.6   1.090x   1.436x   1.630x
decode, ms per step at the full batch
  1024 x 1              41.46   0.998x   0.849x   0.848x
  1024 x 16             46.02   1.003x   0.834x   0.845x
  4096 x 1              42.33   0.998x   0.840x   0.842x
  4096 x 16             56.23   0.999x   0.837x   0.856x
```

Same tokens as stock, 72 of 72, at every shape in both modes. Two things in
it are worth a sentence:

* **Graphs barely move stock's decode**: 41.5 ms a step against 42.2 eager at
  batch 1. A decode step on this card is not launch-bound, it is GEMV-bound --
  which is the next paragraph -- so the eager numbers above were not flattering
  the patches by more than a percent or two.
* **Steps that admit a prompt: the ~51 ms they seemed to cost was the
  comparison, not the patch.** A first version of this bullet reported that
  such a step (one new prompt plus the running requests' decode tokens,
  M = 4096 + r) cost the patch ~51 ms more end to end than the GEMMs explained,
  inferred by subtracting runs of different lengths. Timed directly instead
  (`tools/hk-bench/vllm_steps.py`, 4096 x 16, the same 87 steps in both arms,
  two rounds), the patch makes every step faster:

  ```
  step                                     stock ms    gemm ms    saved
  first: two prompts, 8192 tokens            2937.8     2728.4    209.4
  admit one prompt + decode (x14, avg)       1467.8     1411.8     56.0
  decode at batch 16 (x57, avg)                57.6       37.3     20.3
  ```

  The decode saving is exactly what the GEMMs predict (19.9 ms of layers plus
  0.4 of LM head). The ~51 ms was a mixed step measured against *half the
  first step*: there the patch's mixed step costs 48.7 ms more and stock's
  1 ms less. The GEMMs, timed in place with a synchronize around each call,
  account for 45 of those 48.7 ms. The patched GEMM's mixed step costs 29 ms
  more than half its M = 8192 step, because the last M block backs up and
  recomputes 128 rows. hipBLASLt's costs 16 ms *less* than half of its own.

  What is **not** explained: hipBLASLt is faster inside vLLM than in any
  standalone run of the same 36 layers -- 837 ms of GEMM per mixed step,
  against 849-870 -- so the patch's lead there is 51 ms where the benchmarks
  say 76-88. The patched GEMM times the same everywhere (785-786 ms, shape by
  shape). Ruled out: the timing method (per call with a synchronize vs back to
  back), the operands (Qwen3-8B's own weights with RMSNorm- and SiLU-shaped
  activations time like random ones), and vLLM's GEMM path (on gfx1100 it ends
  in the same `F.linear`). And it moves: in one process alternating the two
  GEMMs step by step, hipBLASLt's mixed step took 859 ms, its qkv layers 21%
  slower than in a process of its own. So the variation is real and depends
  on context; which part of the context, these runs do not say.

  One tool lesson came out of it: on this ROCm build the torch profiler's
  kernel durations include time queued behind earlier launches -- in an eager
  step they summed to 4.4x the step's wall time. `vllm_profile.py` ran eager,
  so the shares it printed at the top of [Against the engine](07-against-the-engine.md) are queue-weighted. The
  ordering they gave (GEMM, then prefill attention, then the rest) has since
  been borne out end to end; the percentages should not be quoted.

**What is left is decode, and decode is GEMV.** At batch 1 the 36 layers' four
GEMVs plus the LM head take 39.7 ms of a 41.5 ms step on hipBLASLt, at 37-46%
of the card's 864 GB/s for the layers and 75% for the LM head. What a kernel
can reach is measured, not the datasheet: `tools/hk-bench/hbm_probe.hip`
streams 680-820 GB/s depending on size, and upstream vLLM's `wvSplitK` --
which upstream has since enabled on gfx11, for one to five tokens -- reaches
743-835 GB/s on these exact matrices, built standalone for gfx1100. Per decode
step, in GEMM milliseconds (`vllm_gemm.py --upstream-skinny`):

```
tokens     installed vLLM   upstream vLLM   best of everything
             (hipBLASLt)    (wvSplitK <= 5)   incl. rocBLAS, hk
  1             39.7            19.1             19.1
  4             39.7            20.4             20.4
  8             39.5            39.5             34.8
 16             41.4            41.4             36.5
 32             41.1            41.1             36.4
 64             42.0            42.0             38.2
128             55.0            55.0             37.5   <- hk GEMM, 1.36x
```

So the gap has a precise shape. At one to five tokens a near-ceiling GEMV
exists and the installed engine cannot run it. From six tokens up nobody has
one: every library lands at 35-42 ms, about half of what `wvSplitK` shows is
reachable. That band -- the batch sizes a serving engine spends most of its
decode steps in -- is the next kernel ([below](09-decode-kernel.md)); and at batch 128 the existing
GEMM is already 1.36x, which the routing rule, written for prefill, still
leaves on the table.

One measurement lesson cost a run and is worth keeping: timing each call
separately over three rotating weight copies let the Infinity Cache serve most
of a 100 MB matrix to `wvSplitK`, which then read "at" 1917 GB/s, 2.2x the HBM
peak. hipBLASLt, too slow to notice the cache, measured the same either way.
The harness now times what a decode step does: one pass over 36 distinct
weights, back to back.

---

← [Against the engine](07-against-the-engine.md) · [Development record](README.md) · [The decode kernel](09-decode-kernel.md) →
