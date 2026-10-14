# One patch layer, and a routing table that is measured rather than written

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The GEMM patch and the attention backend each wrote the same parts by hand.
`hk.integration.engine` writes them once:

* **`Op`** is an opaque torch custom op with a fake implementation.
* **`Patch`** counts each path a call takes, and holds the patch's CUDA-graph
  contract.
* **`RouteTable`** maps a shape to a kernel, and replaces the transcribed
  rule.

`vllm_gemm` is an `Op` routed by a table. `vllm_backend` is a `Patch`: its
metadata builder's `_cudagraph_support` is derived from its declaration
instead of being a second statement of it.

**The table.** `tools/hk-bench/route_tune.py` measures each (N, K) of the
model at every M up to 64, every 16 rows up to 1024, and coarser above. At
each point it times the engine's path, the prefill GEMM and the decode GEMV,
each gated against fp32 first, interleaved in one process, with weights
rotated through 3x the Infinity Cache. A kernel takes an M only where it
beat the engine by 3%; a tie stays the engine's. The table answers an M as
follows:

* inside a run of measured Ms with the same winner, that winner;
* between two runs, which means between two measured Ms that disagreed, the
  engine's path;
* above the largest measured M, the winner measured there;
* below the smallest, or for a shape never measured, the engine's path, and
  the shape is recorded as a miss.

Tables are per arch and per model. `hk/tuned/routes_vllm_linear_gfx1100.json`
is Qwen3-8B's, raw timings included, and `$HK_VLLM_ROUTES` names another.
On qkv it reads:

```
     1..64     hk_decode
    80..128    hk          one 128-row block, full enough
   144..192    engine      a second block, mostly empty
   208..256    hk
   272..288    engine
   304..512    hk
   528..576    engine
   592..768    hk
   784         engine
   800..16384  hk
```

That is the comb the hand rule could only approximate: the teeth are the Ms
just past a multiple of 128. Scored on the tuning run itself, over its 59 Ms
from 65 to 1023, the table is 1.12x the hand rule in GEMM time per step
(geometric mean). That is in-sample, so the Ms where the two disagree were
re-measured in a fresh process. GEMM milliseconds per step, 36 layers plus
the LM head:

```
   M   engine   hand rule    table    table vs rule
  96    43.57      43.57     37.57    1.16x
 128    53.80      53.80     40.10    1.34x
 208    70.14      70.14     64.26    1.09x
 256    70.97      70.97     64.53    1.10x
 320    90.02      90.02     76.05    1.18x
 448   119.87     119.87    101.36    1.18x
 576   133.74     133.74    125.04    1.07x
 784   180.39     180.39    167.86    1.07x
1216   250.02     235.30    236.51    0.995x
1728   342.98     329.55    336.65    0.979x
```

Out of sample, the table never loses to the engine, and beats the hand rule
by 7-34% wherever the rule fell back. The last two rows are what the 3% bar
costs. There the true gain sits just under it, the fresh run disagreed with
the tuning run, and the blanket rule's "everything above 1024" happened to
be right: 0.5% and 2.1% of a step.

**End to end**, where the difference reaches a request: decode at batch 96
and 128, sizes a busy server runs, where a step's M is the batch. Graph mode,
Qwen3-8B, 512-token prompts, 160 output tokens, one round. The arms are
stock (1.00x), the patch routed by the hand rule written as a table
(`gemm@rule`, which reproduces the rule at every M from 0 to 19,999), and
the patch routed by the measured table:

```
             total ms                           decode ms per step
ctx x batch  stock    hand rule     table       stock  hand rule     table
512x96      27349.9  25561.4 1.07  24523.9 1.12  97.48  97.98 0.99  91.48 1.07
512x128     37156.1  34813.6 1.07  32419.1 1.15 136.11 136.92 0.99 121.14 1.12
```

These are throughput numbers (whole batches):

* Prefill is the same under both patched arms, 1.17-1.18x, because every
  prefill step here has M >= 1024, which both route.
* Decode is where they part. The hand rule sent M=96 and M=128 back to
  hipBLASLt, so its decode is stock's. The table routes them: 1.07x and
  1.12x per step, the 6 and 16 ms a step that the GEMM measurement predicted.
* The batch finishes 4.2% and 7.4% sooner than under the rule.
* Greedy output is identical to stock in every arm (160/160).
* The counters say the same thing. Under the rule, none of the graphs vLLM
  captured for batch sizes 72-128 used the GEMM; under the table, 1,944
  captured calls did.

**The graph check.** A patch declares `graph="safe"` or `graph="never"`, and
"never" must give its reason. Under capture:

* A "never" patch raises `GraphCaptureError` with that reason, instead of
  being frozen at the capture batch's values.
* A "safe" one runs with torch's sync debug mode set to error. A host sync
  in its wrapper then raises at the line that does it, naming the patch,
  instead of failing the capture somewhere else, or succeeding.
* Calls made while capturing are counted apart from eager calls, since a
  replay runs no Python.
* Changing the routes after a capture warns, because graphs captured before
  keep the old routes.

The limit is torch's: its sync debug mode is a prototype that sees torch's
own syncs (`.item()`, `.tolist()`, blocking copies, `synchronize()`), not a
raw HIP call inside an extension.

Not ported: the two SDPA rebinding patches (`vllm.py`, `sglang.py`) still
keep their own counters.

See `.claude/plans/` for the full plan and its gates.

---

← [gfx1201](13-gfx1201.md) · [Development record](README.md) · [The IR as an MLIR dialect](15-mlir-dialect.md) →
