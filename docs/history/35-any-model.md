# Any model: the routes measured where the model runs

> Part of the [development record](README.md), in the order the work happened. Paths, API names and numbers are those of the time.

The linear-layer patch routes each call by a measured table
(`hk/tuned/routes_vllm_linear_gfx1100.json`), and the table has five
shapes, Qwen3-8B's. A (N, K) it never measured takes vLLM's path, so a model
nobody had tuned for got nothing from the patch: switched on, it routed
every call to vLLM and timed like stock. Getting a table meant running
`tools/hk-bench/route_tune.py`, a script with only a `main()`, from a
checkout.

This chapter makes the patch pay off on a model nobody tuned for, with no
step but switching it on. The patch now measures the shapes it lacks when
vLLM loads the model, within a time budget, and keeps what it measured.
Everything was measured in the engines' pod (vLLM 0.31.0, torch
2.13.0+rocm7.2, FlyDSL 0.4.0 installed), on GPU 3, with `hk._mlir` built
there in a directory of its own.

## The measuring, as a library

`route_tune.py`'s measurement moved into the package, as
`hk.autotune.routes`:
- `measure_points(n, k, ms, ...)` measures one (N, K) at a list of Ms, the
  way chapter 32 left it: a warm-up set thrown away, close calls
  re-measured, each path's noise kept beside its time, every candidate
  checked against fp32 first. It stops at the first M that would start
  after an optional deadline.
- `entry(name, points, ...)` turns points into a table entry.
- `tune_shape(n, k, ...)` is the two together.
- `tune_missing(kind, arch, shapes, ...)` measures a model's shapes into
  this machine's records, within a budget (below).

`route_tune.py` is now a thin shell over it. Its options, its output and
the tables it writes are the same, and the tests that read its functions
read them through it. One thing changed: the fp32 check is computed in row
blocks. At 16384 rows of a 37888-wide layer one fp32 product is 2.5 GB, and
the old check held four such tensors at once. That was harmless in a script,
but not in a server that has already loaded its weights.

`tools/hk-bench/vllm_gemm.py` and `route_tune.py` had their own copies of
`linear_shapes` and `model_config`. Both now come from the library, which
also pads the LM head's rows to a multiple of 64, as vLLM does (Granite's
49155-token vocabulary is 49216 rows in vLLM), and reads a config from a
directory or the hub as well as the local HF cache.

## This machine's records

A shape measured on this machine is a record, as schedules already were
(`hk.autotune`). It lives in `$HK_TUNE_DIR` (by default beside the compile
cache), in `routes_<engine>_linear_<arch>.json`, a file in the shipped
table's format. Each entry carries its own measurement's metadata: the
engine's path it was measured against, the toolchain, the device, the
tuner's settings, the seconds it took.

`engine.shipped_table` lays these over the shipped table:
- each record replaces the shipped entry for the same (N, K) whole;
- `RouteTable.origin` says which file each shape came from, and the patch's
  report line counts them;
- a record measured against another engine's path is left out, since the
  path a kernel had to beat is not the one it would be beating (after a
  vLLM upgrade, say), and the shape is measured again;
- `HK_AUTOTUNE=0` ignores the records, as it ignores schedule records;
- `$HK_VLLM_ROUTES` still names a table to use alone.

A shape whose grid was cut short is written `partial`. It answers up to the
largest M it measured, and the engine's path above that, rather than
extending its last run over Ms it never saw. Writers hold a file lock
(`records_lock`), so two processes never measure the same shape at once.

## Measuring at load

`hk.integration.vllm_autotune` wraps vLLM's GPU worker's `load_model`.
When the weights are loaded it walks the model vLLM built, and collects the
unquantized bf16 linear layers and the LM head, as this rank holds them
(`model_shapes`). The shapes the table lacks are measured, written as
records, and the routes reloaded. The place was chosen for three reasons:
- **after the weights**, because the shapes then come from the model vLLM
  actually built, for any architecture it supports and sharded for this
  rank, rather than from a config parser that knows five layer names;
- **before the profile run**, because vLLM sizes the KV cache from the peak
  memory of a dummy forward. The tuner frees its buffers and empties the
  allocator's cache first; the profile run resets the peak itself;
- **before capture**, because a CUDA graph replays the routes it was
  captured with, whatever the table says afterwards.

A shape whose every call carries a bias is not measured: the op sends a
biased call to vLLM. Qwen2.5's qkv projection is one; the report line says
so.

**The budget.** `HK_AUTOTUNE_BUDGET` seconds, 60 by default. The order
puts the time where it pays:
1. every decode M (1-64) of every missing shape, largest weight stream
   first. These are cheap, and they are where a decode step's time goes;
2. the prefill grid, shape by shape in the same order.

Each shape is written as soon as it has new points. A shape cut short is
`partial`, and the next start measures only the Ms it lacks. The deadline
is checked between points, so a start overruns its budget by at most one
point: up to about 2 s, at gate_up's 16384 rows.

**Once per machine.** Every tensor-parallel rank is a process, and so is
every other server on the node. The first process to take the lock
measures. The others wait, then read what it wrote and measure nothing. The
tests run two processes against one lock. More than one GPU was not
available to this chapter, so tensor parallelism was not run end to end.

**Before the server.** `python -m hk.integration tune --model X [--tp N]`
measures the same shapes, from the model's `config.json`, into the same
records, with no budget unless given one. It skips shapes the table or an
earlier run already has (`--force` measures them again) and the shapes a
config says carry a bias.

## What it costs

Measured with the CLI on Qwen2.5-7B-Instruct, whose four unbiased shapes
are in no table:

| | o 3584x3584 | gate_up 37888x3584 | down 3584x18944 | lm_head 152064x3584 | total |
|---|---|---|---|---|---|
| decode Ms (64) | 1.1 s | 8.7 s | 1.6 s | 7.5 s | 19 s |
| the rest of the grid | 12.9 s | 47.0 s | 27.9 s | 2.5 s | 90 s |

The first measurement in a process also pays for loading hk's kernels and
the engine's first calls (gate_up's decode Ms took 8.7 s in the CLI and 2.5
s inside vLLM). The LM head's decode Ms cost 7.5 s because each pass
streams two 1.09 GB weights. So 60 s covers every decode M of a 7B model and
one or two shapes' prefill grids, and a second start finishes. Against a
compile cache that had none of hk's kernels, the whole grid took 7 s longer
(`route_tune.py`, 142 s cold against 135 s warm, process start included).

Inside vLLM the measurements are the same measurements. The routes of
Qwen2.5's four shapes were measured five times: twice at load in the A/B
below (each over two starts), once by the CLI, and twice by
`route_tune.py`. Over 574 points, every pair of them agreed on every point
but one (gate_up at 288 rows, in one of the `route_tune.py` runs).

## What the new shapes measured

Eight shapes in no table, Qwen2.5-7B-Instruct's five and the three of
DeepSeek-R1-Distill-Llama-8B's (Llama-3-8B's) that Qwen3-8B does not share,
gave the same picture as Qwen3-8B's table (`route_tune.py`, against vLLM
0.31's op, engine time over hk's):

| shape | hk's decode GEMV, M 6-64 | M 1-5 | hk's GEMM at every M from | hk's GEMM, M >= 1024, lowest / median |
|---|---|---|---|---|
| Qwen2.5 qkv 4608x3584 | 1.43-1.72x | 0.85-0.91x | 400 | 1.20x / 1.28x |
| Qwen2.5 o 3584x3584 | 1.51-2.03x | 0.82-0.92x | 496 | 1.22x / 1.33x |
| Qwen2.5 gate_up 37888x3584 | 1.20-1.43x | 0.92-0.96x | 288-304 | 1.19x / 1.29x |
| Qwen2.5 down 3584x18944 | 1.28-1.84x | 0.89-0.93x | 496 | 1.20x / 1.32x |
| Qwen2.5 lm_head 152064x3584 | 1.07-1.29x | 0.94-0.95x | (M <= 256) | |
| Llama gate_up 28672x4096 | 1.54-1.95x | 0.94-1.02x | 304 | 1.18x / 1.29x |
| Llama down 4096x14336 | 1.33-1.57x | 0.91-0.96x | 304 | 1.17x / 1.29x |
| Llama lm_head 128256x4096 | 1.13-1.31x | 0.98-0.99x | (M <= 256) | |

So the decode GEMV takes 6-64 rows at every new shape, vLLM's skinny GEMMs
keep 1-5 rows, and hk's GEMM wins above a few hundred rows, with the comb
below that different at each shape. Qwen2.5's qkv projection carries a bias,
so its calls stay on vLLM's path whatever its routes say.

## Qwen2.5-7B-Instruct end to end

**Before.** The patch as it was, on a model no table has, is the patch with
nothing measured: `HK_AUTOTUNE_BUDGET=0`, vLLM's defaults, two rounds. Every
call went to vLLM's path (30,568 a round, 3,472 of them captured), and the
request timed like stock:

| ctx x batch | stock ms | gemm, nothing measured | x | step x |
|---|---|---|---|---|
| 1024 x 16 | 6176.1 | 6173.8 | 1.000x | 1.013x |
| 1024 x 32 | 9990.6 | 9994.7 | 1.000x | 1.011x |
| 4096 x 16 | 22344.3 | 22357.3 | 0.999x | 0.981x |
| 4096 x 32 | 42257.9 | 42298.5 | 0.999x | 0.984x |

**After.** `python -m hk.integration ab --model Qwen/Qwen2.5-7B-Instruct --arms
stock,gemm`, from an empty record directory, in the two modes the earlier
vLLM 0.31 chapters used. The first is [chapter 20](20-vllm-031.md)'s: vLLM's
defaults (CUDA graphs and torch.compile, prefix caching, chunked prefill),
72 tokens, decode as the slope between 72 and 32 tokens. The second is
[chapter 31](31-remeasured.md)'s graph mode at 96 and 128 requests:
512-token prompts, 160 tokens, the slope against 128. Three rounds, arms
alternating, each its own process, best of each. The `gemm` arm is
`HK_VLLM_GEMM` with nothing else set: its first start measured, its second
finished, its third found everything.

vLLM's defaults:

| ctx x batch | stock ms | gemm ms | x | prefill, stock ms | gemm | x | step, stock ms | gemm | x |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 1740.8 | 1675.9 | 1.039x | 244.4 | 185.6 | 1.317x | 21.12 | 21.03 | 1.005x |
| 1024 x 4 | 2429.1 | 2242.4 | 1.083x | 877.3 | 696.5 | 1.260x | 21.93 | 21.84 | 1.004x |
| 1024 x 16 | 6165.6 | 4679.3 | 1.318x | 3609.8 | 2812.1 | 1.284x | 35.63 | 25.76 | 1.383x |
| 1024 x 32 | 10032.1 | 7716.2 | 1.300x | 7293.4 | 5667.5 | 1.287x | 37.96 | 27.60 | 1.376x |
| 4096 x 1 | 3118.4 | 2927.6 | 1.065x | 1000.0 | 813.6 | 1.229x | 29.86 | 29.86 | 1.000x |
| 4096 x 4 | 6325.4 | 5497.8 | 1.151x | 4078.1 | 3258.4 | 1.252x | 30.26 | 30.40 | 0.995x |
| 4096 x 16 | 22332.5 | 18527.9 | 1.205x | 16506.4 | 13026.5 | 1.267x | 47.91 | 37.36 | 1.283x |
| 4096 x 32 | 42259.2 | 35427.7 | 1.193x | 33062.9 | 26091.0 | 1.267x | 53.96 | 44.38 | 1.216x |

Graph mode, 96 and 128 requests:

| ctx x batch | stock ms | gemm ms | x | prefill, stock ms | gemm | x | step, stock ms | gemm | x |
|---|---|---|---|---|---|---|---|---|---|
| 512 x 96 | 17588.6 | 14997.9 | 1.173x | 10710.2 | 8267.5 | 1.295x | 42.75 | 42.10 | 1.015x |
| 512 x 128 | 24594.3 | 18728.8 | 1.313x | 14331.6 | 11070.6 | 1.295x | 66.22 | 48.78 | 1.358x |

Every arm's greedy tokens matched stock's: 72 of 72 at all eight shapes, 160
of 160 at both. Stock's totals agreed within 0.4% from round to round.

- **At 16 and 32 requests the request is 1.19-1.32x faster, and at 96 and
  128, 1.17x and 1.31x.** A decode step there has as many rows as requests:
  hk's GEMV takes 16 and 32, hk's GEMM 128, and at 96 gate_up alone.
- **At 1 and 4 requests the gain is prefill's alone**, 1.04-1.15x a
  request. The decode step is unchanged, because at 1-5 rows vLLM 0.31 runs
  its own skinny GEMMs on gfx11, which hk's GEMV does not beat. That is the
  same for Qwen3-8B (chapter 20).
- **Prefill is 1.23-1.32x faster everywhere**, from hk's GEMM at
  prefill-sized steps, where it measured 1.2-1.33x vLLM's op per call.
- **The decode step at 96 requests is a tie** (1.015x). At 96 rows the
  measured routes keep o, down and the LM head on vLLM's op, where hk's
  128-row GEMM computes a third more rows than it keeps, and qkv carries a
  bias; only gate_up runs hk's GEMM. Qwen3-8B's decode at 96 requests
  gained 12% from FlyDSL's 32- and 64-row tiles (chapter 30), and the
  measuring at load does not time FlyDSL. At 128 rows the GEMM's block is
  full, and the step is 1.36x.

Each round, against the stock round beside it:

| ctx x batch | round 0 (60 s measuring) | round 1 (43 s, the rest) | round 2 (nothing) |
|---|---|---|---|
| 1024 x 16 | 1.249x | 1.318x | 1.321x |
| 1024 x 32 | 1.219x | 1.300x | 1.303x |
| 4096 x 16 | 1.139x | 1.207x | 1.209x |
| 4096 x 32 | 1.129x | 1.192x | 1.194x |
| 512 x 96 | 1.116x | 1.169x | 1.174x |
| 512 x 128 | 1.186x | 1.314x | 1.313x |

The first start's budget covered every decode M and gate_up's prefill grid;
o's, down's and the LM head's prefill Ms took the engine's path until the
second start measured them. That is most of round 0's shortfall: prefill.
The decode step moved after round 0 in two places. At 128 requests a decode
step has 128 rows, a prefill-grid M, which the second start measured (55.57
ms in round 0, then 48.44-48.79). At 4096 x 16 it went from 40.82 to
37.27-37.48 ms. Elsewhere the rounds' steps were within 2.5% of each other,
in no consistent direction.

**The startup**, seconds from `LLM(...)` to a built engine, and of it the
measuring:

| mode | arm | round 0 | round 1 | round 2 |
|---|---|---|---|---|
| defaults | stock | 102.5 | 18.9 | 18.7 |
| defaults | gemm | 88.1 (60.0) | 60.9 (43.4) | 18.1 (0.0) |
| graph, 96/128 | stock | 50.7 | 25.3 | 25.4 |
| graph, 96/128 | gemm | 93.6 (60.1) | 67.3 (43.5) | 24.2 (0.0) |

Round 0's stock start compiled vLLM's graphs into a cold cache, which the
`gemm` arm keeps apart from stock's (`hk-linear`, chapter 20). So the
measuring is what the `gemm` arm's start costs over stock's: 60 s on the
first start, the budget; 43 s on the second; nothing after.

## A second model: DeepSeek-R1-Distill-Llama-8B

Llama-3.1-8B's shapes share qkv (6144 x 4096) and o (4096 x 4096) with
Qwen3-8B, so the table already has those two; gate_up (28672 x 4096), down
(4096 x 14336) and the LM head (128256 x 4096) are new. vLLM's defaults, the
same command, an empty record directory, two rounds:

| ctx x batch | stock ms | gemm ms | x | prefill, stock ms | gemm | x | step, stock ms | gemm | x |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 1897.3 | 1845.9 | 1.028x | 254.7 | 201.4 | 1.264x | 23.21 | 23.19 | 1.001x |
| 1024 x 4 | 2652.5 | 2465.2 | 1.076x | 925.5 | 743.2 | 1.245x | 24.38 | 24.33 | 1.002x |
| 1024 x 16 | 7062.3 | 5004.2 | 1.411x | 3837.3 | 2982.7 | 1.287x | 45.34 | 28.00 | 1.619x |
| 1024 x 32 | 11526.5 | 8723.9 | 1.321x | 7708.4 | 5987.4 | 1.287x | 52.48 | 37.15 | 1.413x |
| 4096 x 1 | 3420.8 | 3234.5 | 1.058x | 1083.9 | 892.0 | 1.215x | 33.00 | 33.04 | 0.999x |
| 4096 x 4 | 6967.5 | 6107.2 | 1.141x | 4426.3 | 3555.2 | 1.245x | 34.19 | 34.67 | 0.986x |
| 4096 x 16 | 25767.0 | 21266.7 | 1.212x | 17954.8 | 14254.6 | 1.260x | 63.90 | 46.33 | 1.379x |
| 4096 x 32 | 49473.8 | 41897.9 | 1.181x | 35934.0 | 28519.1 | 1.260x | 90.84 | 75.67 | 1.201x |

The first start measured gate_up, down up to 2688 rows and the LM head up
to 64, in 60 s; the second the rest, in 17 s (starts of 103.6 and 36.5 s,
against stock's 112.6 and 20.4). Round 0 was already 1.27-1.36x at 1024 x
16-32 and 1.14-1.16x at 4096.

**The greedy tokens did not all match.** Six shapes gave 72 of 72; 1024 x 16
left stock's at the fifth token and 1024 x 32 at the first. The harness
compares round 0's tokens, taken while half the routes were still the
engine's. A separate check with the records complete
(`tiecheck`: the harness's prompts, the first request with its top-5
logprobs, stock and `gemm` twice each, alternating, each its own process)
found:
- each arm gives the same tokens in both of its processes;
- 1024 x 16 matches stock's, 72 of 72;
- at 1024 x 32 the first token's two leading candidates are 0.0625 apart in
  log-probability in both arms (stock -0.872 / -0.935, `gemm` -0.882 /
  -0.944), in opposite order. That is one bf16 step of a logit between 8
  and 16: the two logits are neighbouring bf16 values, and a different
  summation order in the 32 rows' LM head picks the other.

So the patch does not keep greedy outputs bit-identical on every model: it
changes the order of a bf16 GEMM's sums, and an exact tie between the top
two logits goes either way. On the prompt the harness uses (a phrase
repeated 256 times, where the model is unsure what comes next) that
happened at one shape of eight. Qwen2.5-7B's and Qwen3-8B's tokens matched
at every shape.

## Qwen3-8B, which the table has

Qwen3-8B must not lose anything to this. Its five shapes are in the shipped
table, so the measuring at load finds nothing to do. The check is an A/B of
three arms in the same two modes: stock; `gemm`, this chapter's default
path (records over the table, the hook installed, an empty record
directory); and `gemm@shipped`, the shipped table pinned by
`$HK_VLLM_ROUTES`, which is how main routes. FlyDSL is installed, so the
table's FlyDSL rows apply in both. vLLM's defaults, two rounds:

| ctx x batch | stock ms | gemm ms | x | gemm@shipped ms | x | step, stock ms | gemm | gemm@shipped |
|---|---|---|---|---|---|---|---|---|
| 1024 x 1 | 1961.6 | 1906.1 | 1.029x | 1904.7 | 1.030x | 24.04 | 24.06 | 24.03 |
| 1024 x 4 | 2721.4 | 2527.4 | 1.077x | 2528.5 | 1.076x | 25.34 | 25.13 | 25.14 |
| 1024 x 16 | 7386.8 | 5114.5 | 1.444x | 5118.9 | 1.443x | 49.79 | 29.48 | 29.55 |
| 1024 x 32 | 11934.7 | 8898.6 | 1.341x | 8909.0 | 1.340x | 58.55 | 39.22 | 39.36 |
| 4096 x 1 | 3588.1 | 3403.8 | 1.054x | 3404.9 | 1.054x | 35.05 | 35.12 | 35.13 |
| 4096 x 4 | 7202.9 | 6341.1 | 1.136x | 6354.6 | 1.133x | 36.71 | 36.65 | 36.60 |
| 4096 x 16 | 26943.5 | 22349.1 | 1.206x | 22345.2 | 1.206x | 70.88 | 49.95 | 49.59 |
| 4096 x 32 | 51648.8 | 44115.3 | 1.171x | 44041.0 | 1.173x | 101.21 | 84.28 | 81.07 |

- The two `gemm` arms routed every call alike: the same count on each path,
  eager and captured, in every round (`hk` 27,684, `hk_decode` 4,573,
  `flydsl` 8,317, `fallback` 4,526; captured 3,132 / 144 / 1,188).
- Per request they are within 0.4% of each other at every shape, in both
  directions. The one step that differs, 84.28 against 81.07 ms at 4096 x
  32, is the slope's noise there: each arm's two rounds measured 83.1 and
  85.2, and 81.1 and 83.9.
- The hook reported "all 5 shapes in the table" and took 0.0 s; the
  `gemm` arm started in 45.7 s and 19.0 s, against `gemm@shipped`'s 18.9 s
  (round 0 compiled the `hk-linear` cache both arms share).
- Every arm's greedy tokens matched stock's, 72 of 72.

Graph mode at 96 and 128 requests, three rounds:

| ctx x batch | stock ms | gemm ms | x | gemm@shipped ms | x | step, stock ms | gemm | gemm@shipped |
|---|---|---|---|---|---|---|---|---|
| 512 x 96 | 22537.1 | 18025.8 | 1.250x | 18028.1 | 1.250x | 72.71 | 59.07 | 59.36 |
| 512 x 128 | 29261.4 | 22330.7 | 1.310x | 22342.6 | 1.310x | 92.80 | 68.99 | 69.02 |

The same again: identical counts on every path in all three rounds (FlyDSL
took 1,908 captured launches a round), totals within 0.1%, the hook at 0.0 s
and starts of 41.6 / 27.7 / 27.8 s against 27.8 s, and 160 of 160 tokens.

## A rule for the shapes nobody measured: two candidates, both refused

Measuring costs a start a minute. A rule for unmeasured shapes would cost
nothing, and the eight new shapes above all agreed with Qwen3-8B's table:
hk's GEMV from 6 to 64 rows, vLLM's op below. So the plan's third option
was tried: read the table as a rule, and keep it only if shapes nobody had
measured never measured it slower than vLLM's op.

`tools/hk-bench/route_heldout.py` measures the linear shapes of models no
table has, with the tuner's own points: every decode M and nine prefill Ms
from 512 to 16384. `--check` scores rules against the points: how many
(shape, M) a rule sends to hk where hk was slower than the engine (a
misroute), and how many of the Ms the measurement gives hk it covers.

**The first held-out set**: 71 shapes, the linear layers of fifteen models
at tensor-parallel size 1 (Qwen2.5-3B, -14B, -32B; Qwen3-4B, -14B, -32B;
Mistral-7B, Mistral-Nemo; Llama-3.2-3B; Gemma-2-9B; Phi-3-mini; Yi-1.5-9B;
GLM-4-9B; InternLM2.5-7B; Granite-3.1-8B), and four at size 2.

| rule | points it routes | misroutes | shapes | worst |
|---|---|---|---|---|
| decode 6-64 (Qwen3-8B's table) | 4189 | 15 | 3 | -6.2%, Qwen2.5-14B's LM head at 64 rows |
| decode 6-64, and hk's GEMM from 512 rows | 4675 | 16 | 4 | the same; and Qwen2.5-7B's tp2 qkv at 768 rows, -3.3% |

Every misroute of the decode band was a large LM head (131072-152064 rows by
5120) at 59-64 rows, where the GEMV runs its four-tile schedule. On these
shapes a band of 6-56 would have held. A rule chosen after looking at the
held-out set is not validated by it, so a narrower band, 6-48, was fixed
(in `route_heldout.py`, committed) before a second set was measured.

**The second held-out set**: 69 shapes, twelve more models at size 1
(Llama-3.1-70B, Qwen2.5-72B, Phi-4, Gemma-3-12B, Qwen3-1.7B,
Mistral-Small-24B, Falcon3-7B, OLMo-2-7B, DeepSeek-LLM-7B, SmolLM2-1.7B,
Qwen2.5-1.5B, Granite-3.1-2B) and the three largest at 2, 4 and 8.

| rule | points it routes | misroutes | shapes | worst |
|---|---|---|---|---|
| decode 6-48 (fixed beforehand) | 2913 | 36 | 4 | -21.9%, Qwen2.5-72B's tp4 down (8192x7392) at 42 rows |
| decode 6-64 | 4001 | 125 | 7 | -29.8%, the same shape at 64 rows |
| hk's GEMM from 512 rows | 441 | 1 | 1 | -11.5%, Mistral-Small-24B's tp4 o (5120x1024) at 512 rows |

The four shapes the narrower band misroutes say why no band in M can be
right:
- **K that leaves the GEMV few warps.** Its warps must divide K into whole
  rounds, and `schedule_for` halves the count until they do. Qwen2.5-72B's
  down projection has K = 29568 = 2^7 x 231, which leaves 2-4 warps
  (-14% from 15 rows up). At tp4, K = 7392 leaves no schedule below 17
  rows and one warp above; one warp still wins at 17-32 rows (1.2x), and
  loses 20-30% from 33.
- **A long K under a narrow N.** Mistral-Small-24B's down (5120 x 32768,
  and 5120 x 16384 at tp2) loses from 39-44 rows: 320 output blocks, each
  streaming a long K, in the four-tile schedule.

Both are properties of the shape that the measurement sees and a band does
not. A third rule, with conditions on the GEMV's schedule and on N and K,
would need a third held-out set, and the shapes it would serve are the ones
the measuring at load already routes correctly, at a minute's cost. So
**no rule ships**, and a shape nobody measured takes vLLM's path until it is
measured, as before. The finding that does carry over is the GEMV's:
the median shape gains 1.54-1.75x across 6-64 rows (10th percentile
1.17-1.30x), but K's factors, not only its size, decide how many warps it
gets.

## Tests

- `tests/hk/ir/test_route_records.py` (no GPU): records over the shipped
  table, replacing a shape whole; `HK_AUTOTUNE=0`; a pinned table; a record
  measured against another engine left out; records alone making a table
  for an arch with none; a partial entry answering nothing above what it
  measured; config shapes and biases; the budget's order (every shape's
  decode Ms first, largest first), a cut-short shape resuming where it
  stopped, nothing measured twice, a shipped shape known without a record;
  a second process waiting on the lock and then reading the first's
  record; what the hook at load measures, and when it measures nothing.
- `tests/hk/gpu/test_route_tune.py` (gfx1100): a shape measured into an
  entry the table reads, a deadline cutting a grid, and measured shapes
  reaching `vllm_gemm.routes()`.
- `tests/hk/engines/test_vllm_contract.py`: the worker's `load_model`,
  `determine_available_memory` and `compile_or_warm_up_model`, and the layer
  classes and attributes the hook reads, on the installed vLLM.
- `tests/hk/ir/test_flydsl_gemm.py` reads route_tune's functions through
  the shell, unchanged, and passes.

## What is left

1. **Decode at 65-256 rows on a new model.** At 96 requests the measured
   routes keep three of Qwen2.5's four shapes on vLLM's op. Qwen3-8B's 12%
   there came from FlyDSL's 32- and 64-row tiles, which the measuring at
   load does not time: FlyDSL compiles each M, 0.4-0.8 s cold, about 50
   compiles for a model's decode-adjacent Ms. Timing FlyDSL at those Ms
   within the budget, or a 64-row block for hk's GEMM, would close it.
2. **The GEMV where K's factors starve it of warps** (Qwen2.5-72B's
   down projection, 2-4 warps, and one at tp4), and at 33-64 rows on a long
   K under a narrow N. Both route to vLLM correctly today; a K tail, or a
   split of K that does not need whole rounds, would turn them into wins.
3. **Tensor parallelism end to end.** The lock and the waiting are tested
   with two processes, not with two GPUs.
4. **Biased layers.** Qwen2.5's qkv projection, and every model with
   attention bias, stays on vLLM's path. The decode GEMV could add the bias
   in its store.
5. **SGLang.** Its patch reads its own table and has no hook at load; `python
   -m hk.integration tune --engine sglang` writes the records it reads.

## Reproduction

```
# in the engines' pod, GPU 3, PYTHONPATH=python, HK_TUNE_DIR an empty directory
python3 -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct --budget 30
python3 -m hk.integration tune --model Qwen/Qwen2.5-7B-Instruct
python3 tools/hk-bench/route_tune.py --no-flydsl --model Qwen/Qwen2.5-7B-Instruct --out rt.json
python3 tools/hk-bench/route_tune.py --compare rt.json $HK_TUNE_DIR/routes_vllm_linear_gfx1100.json

HK_EAGER=0 HK_VLLM_DEFAULTS=1 python3 -m hk.integration ab --model Qwen/Qwen2.5-7B-Instruct \
    --arms stock,gemm --context 1024,4096 --batch 1,4,16,32 --tokens 72 --short 32 --reps 3
HK_EAGER=0 python3 -m hk.integration ab --model Qwen/Qwen2.5-7B-Instruct \
    --arms stock,gemm --context 512 --batch 96,128 --tokens 160 --short 128 --reps 3
HK_EAGER=0 HK_VLLM_DEFAULTS=1 python3 -m hk.integration ab --model Qwen/Qwen3-8B \
    --arms stock,gemm,gemm@shipped --routes shipped=python/hk/tuned/routes_vllm_linear_gfx1100.json \
    --context 1024,4096 --batch 1,4,16,32 --tokens 72 --short 32 --reps 2

python3 tools/hk-bench/route_heldout.py --models <set 1> --tp 1 --out heldout_tp1.json
python3 tools/hk-bench/route_heldout.py --check heldout_tp1.json heldout_tp2.json
```
