# Performance

Every number on this page comes from the [development record](history/README.md),
and each table names the chapter it was copied from; the chapter has the
harness in `tools/hk-bench/`, the raw output and the caveats. The two GPU
families are kept apart, because nothing measured on one says anything about
the other: [Part A](#part-a-rdna3-gfx1100-one-w7900d) is RDNA3 (one Radeon PRO
W7900D) and [Part B](#part-b-cdna4-gfx950-one-mi355x) is CDNA4 (one Instinct
MI355X). Each part goes level by level: end to end in each engine, kernels
against the tuned field, kernels inside the engines against what the engine
calls, and the losses. Results that newer ones replaced are in
[the appendix](#appendix-superseded-results).

Ratios are hk's speed relative to the baseline named beside them, the
baseline's time over hk's, so above 1 is hk faster, unless a table says it
is a time ratio.

- [How the numbers are taken](#how-the-numbers-are-taken)
  - [The stacks](#the-stacks)
- [Part A: RDNA3 (gfx1100, one W7900D)](#part-a-rdna3-gfx1100-one-w7900d)
  - [A1. End to end, by engine](#a1-end-to-end-by-engine)
  - [A2. Kernels against the tuned field](#a2-kernels-against-the-tuned-field)
  - [A3. Kernels inside the engines](#a3-kernels-inside-the-engines)
  - [A4. Losses and limits](#a4-losses-and-limits)
- [Part B: CDNA4 (gfx950, one MI355X)](#part-b-cdna4-gfx950-one-mi355x)
  - [B1. End to end in vLLM, aiter on](#b1-end-to-end-in-vllm-aiter-on)
  - [B2. Kernels against the tuned field](#b2-kernels-against-the-tuned-field)
  - [B3. Kernels inside the engine](#b3-kernels-inside-the-engine)
  - [B4. Losses and no-headroom findings](#b4-losses-and-no-headroom-findings)
- [Appendix: superseded results](#appendix-superseded-results)

## How the numbers are taken

How a number was taken matters as much as the number. Each rule below exists
because breaking it once produced a plausible result that was wrong
([numbers not to trust](agents.md#numbers-not-to-trust) has the stories).

- **One process, interleaved.** Kernels being compared run in one process,
  round by round, with the order rotated. A kernel A/B across processes is
  invalid on these GPUs: it manufactures 13-16% differences out of nothing
  ([Phase 5](history/04-phase-5-autotune.md)).
- **Checked first.** Every arm is checked against an fp32 torch reference
  before it is timed, and where a case has an exact answer (V all ones for
  attention, small integers for an add) against that too.
- **Rounds queued behind a spin.** A launch that reaches the GPU 5-16 us
  after it went idle costs about 10 us more, so since
  [chapter 32](history/32-measuring-the-measurements.md) the scoreboard
  queues each round's calls behind a GPU spin. The gfx950 GEMM harness and
  the engine-kernel bench do the same ([chapter 41](history/41-cdna-leftovers.md),
  [chapter 42](history/42-mi355x-in-the-engine.md)).
- **The median of rounds, with impossible rounds rejected.** The scoreboard
  rejects rounds that lost kernel records, ran faster than the card can at
  their footprint (GDDR6 past the W7900D's 96 MB cache, 2.43 TB/s inside
  it), or stand alone, and reports the median of the rest (chapter 32).
  Tables that report the best round instead (most chapters before 32, and
  chapters 29 and 34 on gfx950) say so.
- **Verdict bands.** Per call, a tie is within the larger of 2% and the two
  arms' measured noise. End to end, `python -m hk.integration ab` calls a
  result a win or a loss past a band of max(2%, noise), a tie inside a band
  of at most 5%, and noise for a single round, a band past 15%, or a ratio
  inside a band wider than 5% ([chapter 36](history/36-the-agent-loop.md)).
- **End to end, each arm is its own engine.** An engine cannot be
  interleaved inside one process, so arms alternate across rounds in
  separate processes, `ab` pairs the ratio round by round and takes the
  median, and every arm's greedy tokens are compared with stock's. Older
  end-to-end tables took each arm's best round; they say so.
- **The engine as it ships**, unless a table says otherwise: vLLM's defaults
  (CUDA graphs and torch.compile, prefix caching, chunked prefill). SGLang on
  gfx1100 runs eager with Triton attention, as sglang-radeon runs RDNA3,
  because HIP graph capture hangs there.

### The stacks

The software around a number moves it: torch 2.13's SDPA is 3.3x faster than
2.10's on the W7900D, and a Triton 3.8.0 wheel installed over triton-rocm
3.7.1 made vLLM's decode kernel up to 40% slower
([chapter 38](history/38-rdna-losses.md)). Each table below names the stack it
ran on, from these.

**RDNA3: one Radeon PRO W7900D** (gfx1100; 864 GB/s of GDDR6, a 96 MB
Infinity Cache).

| stack | ROCm | torch | Triton | FlyDSL | engine | used by |
|---|---|---|---|---|---|---|
| vLLM 0.31 venv, as built | 7.2.1 | 2.13.0+rocm7.2 | triton-rocm 3.7.1 | -- | vLLM 0.31.0+rocm721, built from source | [chapters 20-22](history/20-vllm-031.md): vLLM 0.31 end to end, MoE, 4-bit |
| the same venv, after October 15 | 7.2.1 | 2.13.0+rocm7.2 | Triton 3.8.0, installed over triton-rocm 3.7.1 | 0.4.0 | vLLM 0.31.0+rocm721 | chapters 30-38: FlyDSL routes, any model, `ab`, chapter 38's end to end; the performance gate's baseline |
| SGLang venv | 7.2.1 | 2.9.1 (HIP 7.2.53211) | triton 3.5.1 | installed in chapter 32 | SGLang 0.5.18, sglang-radeon 0.1.5 | [chapters 24-25](history/24-sglang.md), 32 |
| the scoreboard | 7.2.4 | 2.10.0+rocm7.2.4 | 3.6.0 | 0.4.0.dev970 | -- | [chapters 26-28](history/26-the-scoreboard-retuned.md): kernels against the tuned field |
| vLLM 0.16 (superseded) | 7.2.1 | -- | -- | -- | vLLM 0.16.1.dev0 | chapters 07-17; [appendix](#vllm-016-end-to-end) |

**CDNA4: one Instinct MI355X** (gfx950; 256 CUs, a 256 MB Infinity Cache;
measured peaks 2.48 PFLOP/s of bf16 MFMA and 6.6 TB/s past the cache,
[chapter 42](history/42-mi355x-in-the-engine.md)), one GPU of a shared
8-GPU node.

| stack | ROCm | torch | Triton | FlyDSL | engine | used by |
|---|---|---|---|---|---|---|
| `rocm/pytorch` image | 7.2.3 (hipcc 7.2.3; HipKittens' attention built with 7.1) | 2.10.0+rocm7.2.3 | 3.6.0 | 0.4.0 | -- | [chapters 29](history/29-cdna.md), [33](history/33-cdna-gemm.md), [34](history/34-cdna-attention.md), [37](history/37-cdna-rows.md), [41](history/41-cdna-leftovers.md), [43](history/43-cdna-last-rows.md): kernels against the tuned field |
| vLLM nightly image `vllm/vllm-openai-rocm:nightly-rocm100-0cbac6cd` | 10 (HIP 7.15) | 2.12 | 3.8 | -- | vLLM 0.30.1rc1.dev558, aiter 0.1.23; aiter on (`VLLM_ROCM_USE_AITER=1`) end to end | [chapters 39](history/39-cdna-paged.md), [42](history/42-mi355x-in-the-engine.md): end to end, kernels against what the engine calls |

## Part A: RDNA3 (gfx1100, one W7900D)

Everything in this part was measured on one W7900D. It is hk's primary
target: every kernel and every engine patch runs there.

### A1. End to end, by engine

#### vLLM 0.31 on gfx1100

**Pending re-validation.** Every vLLM table below names its stack. Chapters
20-22 record triton-rocm 3.7.1, the Triton vLLM 0.31's torch 2.13.0+rocm7.2
pins; the later ones (chapters 30-38) ran in a venv that
imported Triton 3.8.0, stock arms included, and [chapter 38](history/38-rdna-losses.md)
found 3.8.0's decode kernel up to 40% slower at 32-64 requests. Their stock
arms may therefore have been slow, and a re-measurement on the matched
triton-rocm 3.7.1 stack is pending; until it lands, read the Triton 3.8.0
ratios as measured against that stock.

Qwen3-8B, bf16, vLLM 0.31.0 under its defaults. Stock 0.31 picks
`ROCM_ATTN` on gfx1100, whose custom paged-attention kernel refuses the card
and falls back to a Triton kernel. hk's attention backend registers over
`TRITON_ATTN`, so there are two baselines: stock, and TRITON_ATTN, the
backend hk replaces. The arms: `hk` is the attention backend (hk's paged
prefill; hk's decode walk from 16 requests, to which vLLM pads 9-15, and
Triton's kernel on hk's cache up to 8); `hk+gemm` adds the linear-layer
patch.

**The patches against stock** ([chapter 20](history/20-vllm-031.md); stack:
vLLM 0.31.0, torch 2.13.0+rocm7.2, triton-rocm 3.7.1, as recorded). 72 output tokens, the best of two rounds,
one process per arm; every arm's greedy tokens were identical to stock's.
Total time per request batch, as a ratio to stock; then the decode step:

| ctx x batch | stock ms | TRITON_ATTN | hk | hk+gemm | step, stock ms | hk ms | hk+gemm ms |
|---|---|---|---|---|---|---|---|
| 1024 x 1 | 1992 | 1.17x | 1.18x | 1.22x | 24.5 | 20.3 | 20.3 |
| 1024 x 4 | 2781 | 1.08x | 1.11x | 1.20x | 26.2 | 22.4 | 22.2 |
| 1024 x 16 | 7493 | 1.02x | 1.05x | 1.52x | 51.1 | 48.3 | 27.8 |
| 1024 x 32 | 12174 | 0.99x | 1.09x | 1.47x | 61.7 | 52.2 | 32.7 |
| 4096 x 1 | 3664 | 1.29x | 1.48x | 1.59x | 36.8 | 21.0 | 21.0 |
| 4096 x 4 | 7180 | 0.98x | 1.25x | 1.44x | 38.5 | 24.6 | 24.8 |
| 4096 x 16 | 26607 | 1.02x | 1.32x | 1.70x | 72.1 | 59.0 | 38.2 |
| 4096 x 32 | 51762 | 1.00x | 1.40x | 1.75x | 112.2 | 71.2 | 51.5 |

hk is 1.01-1.39x TRITON_ATTN, almost all of it prefill: hk's paged prefill
is 3.6-7.9x Triton's ([A3](#paged-prefill)). The linear-layer patch gains
nothing at 1 and 4 requests, where vLLM 0.31 runs its own skinny GEMM
(`wvSplitK`, M ≤ 5) and the routing table leaves it; at 16 and 32 requests
it nearly halves the decode step.

**With chapter 38's decode kernel** ([chapter 38](history/38-rdna-losses.md);
stack: vLLM 0.31.0, torch 2.13.0+rocm7.2, Triton 3.8.0 in both arms). `ab --json`, two
rounds, the tree before chapter 38 and after it each against its own stock
arm; 72 output tokens, 128 at 64 requests. Ratios are stock over the arm:

| ctx x batch | hk, total before / after | hk, decode step before / after | hk+gemm, total before / after | hk+gemm, decode step before / after |
|---|---|---|---|---|
| 1024 x 16 | 1.042x / 1.060x | 1.027x / 1.063x | 1.544x / 1.587x | 1.788x / 1.904x |
| 1024 x 32 | 1.080x / 1.096x | 1.131x / 1.183x | 1.500x / 1.539x | 1.787x / 1.946x |
| 1024 x 64 | 1.108x / 1.126x | 1.215x / 1.246x | 1.498x / 1.530x | 1.715x / 1.786x |
| 4096 x 16 | 1.359x / 1.373x | 1.197x / 1.227x | 1.798x / 1.837x | 1.836x / 1.949x |
| 4096 x 32 | 1.417x / 1.434x | 1.418x / 1.472x | 1.829x / 1.862x | 1.964x / 2.051x |

Every row is a win past its band, before and after. The decode step is
1.9-3.9 ms (4-9%) shorter after at every shape, prefill moved by at most
1.1%, and the greedy outputs were identical to stock's in every run.

#### FlyDSL's GEMM on the routes

[Chapter 30](history/30-flydsl-routes-gate-contracts.md), re-measured in
[chapter 31](history/31-remeasured.md); stack: vLLM 0.31.0, torch 2.13.0+rocm7.2, Triton 3.8.0,
FlyDSL 0.4.0. `hk+gemm` is the attention backend and the linear layers
routed by the table without FlyDSL; `+FlyDSL` uses a table re-measured with
FlyDSL installed, which adds FlyDSL rows where its GEMM, at the tile its
dispatcher picks, beat the table's answer by 2%. Prefix caching and chunked
prefill off, three rounds, arms alternating, best of each; every arm's
greedy tokens matched stock's.

Per call (`route_tune.py`), FlyDSL against the table's own answer at each M
where one of its tiles divides M, as a time ratio (below 1 is FlyDSL
faster); chapter 31's re-measured table:

| M | qkv | o | gate_up | down |
|---|---|---|---|---|
| 64-256, median | 0.84x | 0.87x | 0.88x | 1.00x |
| 272-1024, median | 1.22x | 1.17x | 1.42x | 1.28x |
| above 1024, median | 1.02x | 1.005x | 1.02x | 0.997x |

End to end, graph mode, 512-token prompts, 160 output tokens (chapter 30):

| ctx x batch | total, stock ms | hk+gemm | +FlyDSL | decode, stock ms | hk+gemm | +FlyDSL |
|---|---|---|---|---|---|---|
| 512 x 96 | 22471.29 | 1.266x | 1.343x | 71.72 | 1.224x | 1.394x |
| 512 x 128 | 29184.06 | 1.368x | 1.442x | 92.32 | 1.467x | 1.666x |

FlyDSL pays where a step's M is one its tiles divide and it measured faster:
decode at 96 and 128 requests, 12% a step, where hk's 128-row GEMM leaves the
grid underfilled. Eager at 1-16 requests the two tables are within 0.7% at
every decode step and 1.18-1.57x stock per request either way. After
[chapter 28](history/28-gemm-schedule.md) made hk's GEMM 3% faster on vLLM's
op, the shipped table routes FlyDSL at 41 points instead of 105. Against
the table above, on the same tree: prefill 1.5-1.7% faster at 8192-token
steps (gate_up on FlyDSL's 256x256 tile), 1.2% slower at a 1024-token step
(rows dropped at M = 1024 that measured under the 2% bar per call), and
decode at 96 and 128 requests 1.413x and 1.651x stock (chapter 30: 1.394x
and 1.666x).

#### A model no table had measured

[Chapter 35](history/35-any-model.md); stack: vLLM 0.31.0, torch 2.13.0+rocm7.2, Triton 3.8.0,
FlyDSL 0.4.0 installed. Qwen2.5-7B-Instruct, bf16, whose linear shapes are
in no shipped table. The `gemm` arm is `HK_VLLM_GEMM` with nothing else
set, from an empty record directory: its first start measured the shapes
at load within the default 60 s, its second the rest (43 s), its third
nothing. Three rounds, arms alternating, best of each; greedy tokens
identical to stock's at every shape. Qwen2.5's qkv projection has a bias,
so its calls stay on vLLM's path.

| mode, ctx x batch | total, stock ms | gemm | prefill | decode, stock ms | gemm |
|---|---|---|---|---|---|
| defaults, 1024 x 1 | 1740.8 | 1.039x | 1.317x | 21.12 | 1.005x |
| defaults, 1024 x 4 | 2429.1 | 1.083x | 1.260x | 21.93 | 1.004x |
| defaults, 1024 x 16 | 6165.6 | 1.318x | 1.284x | 35.63 | 1.383x |
| defaults, 1024 x 32 | 10032.1 | 1.300x | 1.287x | 37.96 | 1.376x |
| defaults, 4096 x 1 | 3118.4 | 1.065x | 1.229x | 29.86 | 1.000x |
| defaults, 4096 x 4 | 6325.4 | 1.151x | 1.252x | 30.26 | 0.995x |
| defaults, 4096 x 16 | 22332.5 | 1.205x | 1.267x | 47.91 | 1.283x |
| defaults, 4096 x 32 | 42259.2 | 1.193x | 1.267x | 53.96 | 1.216x |
| graph mode, 512 x 96, 160 tokens | 17588.6 | 1.173x | 1.295x | 42.75 | 1.015x |
| graph mode, 512 x 128, 160 tokens | 24594.3 | 1.313x | 1.295x | 66.22 | 1.358x |

At 1 and 4 requests decode is vLLM's own skinny GEMMs (M ≤ 5), which hk's
GEMV does not beat. At 96 requests the measured routes keep three of the
four shapes on vLLM's op, and the measuring at load does not time FlyDSL.
Qwen3-8B, routed by the same code with nothing to measure, matched the
shipped table within 0.4% per request at every shape.
DeepSeek-R1-Distill-Llama-8B (three new shapes), vLLM's defaults, two
rounds: 1.41x / 1.32x per request at 1024 x 16 / 32, 1.21x / 1.18x at 4096,
1.03-1.14x at 1-4 requests; starts 60 s and 17 s longer, then nothing. Its
greedy tokens matched at six shapes; at 1024 x 32 the first token's two
leading logits are neighbouring bf16 values, and the arms pick opposite
ones.

#### 4-bit and MoE models

**4-bit (AWQ)** ([4-bit weights](history/22-int4.md); stack:
vLLM 0.31.0, torch 2.13.0+rocm7.2, triton-rocm 3.7.1, as recorded). fp16, vLLM 0.31.0 under its defaults, 72 tokens, best
of two rounds, `VLLM_USE_TRITON_AWQ=1` in both arms, so stock runs Triton's
AWQ kernel. `HK_VLLM_INT4=1` gives hk's W4A16 kernel batches of up to 64
tokens; prefill stays on vLLM's path (1.00x). Greedy tokens identical at
every shape.

| model, ctx x batch | total, stock ms | int4 | decode step, stock ms | int4 |
|---|---|---|---|---|
| Qwen3-8B-AWQ, 1024 x 1 | 4965 | 3.80x | 66.37 | 4.46x |
| Qwen3-8B-AWQ, 1024 x 4 | 5710 | 2.74x | 66.49 | 4.33x |
| Qwen3-8B-AWQ, 1024 x 16 | 9252 | 1.66x | 70.08 | 3.70x |
| Qwen3-8B-AWQ, 1024 x 32 | 14421 | 1.30x | 82.48 | 2.27x |
| Qwen3-8B-AWQ, 4096 x 1 | 6706 | 2.20x | 78.34 | 2.92x |
| Qwen3-8B-AWQ, 4096 x 4 | 10408 | 1.56x | 79.72 | 2.87x |
| Qwen3-8B-AWQ, 4096 x 16 | 28885 | 1.15x | 91.58 | 2.34x |
| Qwen3-8B-AWQ, 4096 x 32 | 54765 | 1.07x | 136.04 | 1.57x |
| Qwen3-14B-AWQ, 1024 x 1 | 8857 | 4.43x | 118.06 | 5.52x |
| Qwen3-14B-AWQ, 1024 x 16 | 16853 | 1.69x | 122.00 | 4.71x |
| Qwen3-14B-AWQ, 4096 x 1 | 11497 | 2.50x | 131.95 | 3.80x |
| Qwen3-14B-AWQ, 4096 x 16 | 50210 | 1.16x | 145.80 | 2.87x |

A stock AWQ decode step of 66 ms at one request, against 24.5 ms for the
bf16 model, is the Triton AWQ kernel, which runs this card's 4-bit model
slower than its 16-bit one.

**MoE** ([MoE](history/21-moe.md); stack: vLLM 0.31.0, torch 2.13.0+rocm7.2, triton-rocm 3.7.1, as
recorded). OLMoE-1B-7B, bf16,
vLLM 0.31.0 under its defaults, 72 tokens, best of two rounds. `moe` is the
MoE patch (hk's expert GEMM up to 256 tokens a step); `hk+moe` adds hk's
attention. In that run decode attention was Triton's kernel on hk's cache at
every batch, a routing chapter 20 later reverted above 8 requests; the 16-
and 32-request `hk+moe` rows were not measured again. Greedy tokens
identical to stock's.

| ctx x batch | total, stock ms | moe | hk+moe | decode step, stock ms | moe | hk+moe |
|---|---|---|---|---|---|---|
| 1024 x 1 | 760 | 1.41x | 1.92x | 9.93 | 1.45x | 2.05x |
| 1024 x 4 | 1021 | 1.28x | 1.57x | 12.14 | 1.35x | 1.75x |
| 1024 x 16 | 2238 | 1.11x | 1.19x | 22.70 | 1.16x | 1.24x |
| 1024 x 32 | 3368 | 1.04x | 1.17x | 29.46 | 1.06x | 1.23x |
| 3072 x 1 | 1102 | 1.24x | 2.22x | 13.55 | 1.29x | 2.62x |
| 3072 x 4 | 1876 | 1.18x | 1.67x | 18.62 | 1.27x | 2.03x |
| 3072 x 16 | 5066 | 1.05x | 1.24x | 38.22 | 1.10x | 1.21x |
| 3072 x 32 | 8676 | 1.02x | 1.25x | 55.83 | 1.04x | 1.26x |

**The losing kernels never reach a request** ([chapter 38](history/38-rdna-losses.md);
stack: vLLM 0.31.0, torch 2.13.0+rocm7.2, Triton 3.8.0). `ab`, graphs on, vLLM's defaults, each patch
counting its calls:
- OLMoE-1B-7B at 1024 and 3072 tokens, 1 and 16 requests: 960 MoE calls to
  hk and 2,272 to Triton (every one above 256 tokens), every captured call
  hk's. Prefill ties stock (0.999-1.004x), decode wins 1.09-1.38x, the
  request 1.05-1.34x.
- Qwen3-8B-AWQ at 1, 4 and 16 requests: 3,600 calls to hk (every captured
  one); decode 3.84-4.60x, the request 1.68-3.88x, prefill a tie or a win
  (1.005-1.023x).
- Qwen2.5-7B-Instruct-GPTQ-Int4 at 1 and 4 requests: the int4 patch takes 0
  calls (GPTQ stays on `gptq_gemm_rdna3`), and the request ties stock
  (1.002-1.004x).

#### SGLang 0.5.18 on gfx1100

**Linear layers** ([SGLang](history/24-sglang.md); the SGLang venv).
Qwen3-8B, bf16, eager with Triton attention. `HK_SGLANG_LINEAR=1` routes
every unquantized linear by a table measured against SGLang's `F.linear`.
Served (`launch_server`, then `bench_serving`: random ids, 512 tokens in,
128 out, each server warmed first):

| concurrency | output tok/s, stock | hk | x | TTFT ms, stock | hk | TPOT ms, stock | hk | x |
|---|---|---|---|---|---|---|---|---|
| 1 | 22.75 | 41.27 | 1.81 | 93.5 | 51.9 | 43.53 | 24.00 | 1.81 |
| 8 | 159.52 | 270.65 | 1.70 | 534.6 | 431.7 | 46.08 | 26.14 | 1.76 |
| 32 | 456.02 | 678.08 | 1.49 | 606.2 | 489.7 | 54.41 | 34.26 | 1.59 |

In one process (`bench_one_batch`, 512 tokens in, 64 out; stock's warm runs):
prefill 1.34x, 1.22x and 1.30x and the decode step 1.75x, 1.76x and 1.59x at
batch 1, 8 and 32. With FlyDSL's rows in SGLang's table
([chapter 32](history/32-measuring-the-measurements.md)), the decode step at
32 requests is 0.4-0.6% shorter (34.23-34.25 against 34.37-34.46 ms per
output token) and nothing changes measurably at 1 and 8. hk's linear and
`F.linear` are equally far from fp64; in a greedy check two of three prompts
matched stock over 48 tokens and the third split at a near-tie.

**MoE** ([SGLang MoE](history/25-sglang-moe.md)). Stock SGLang does not run
OLMoE-1B-7B on gfx1100: sgl_kernel's `topk_softmax` fails. So the baseline is
`HK_SGLANG_MOE=topk`, the top-k in torch and SGLang's Triton expert GEMMs;
the patched arm, `HK_SGLANG_MOE=1`, adds hk's GEMMs up to 256 tokens a step.
Warm caches; greedy output identical over SGLang's three correctness prompts.

| concurrency | output tok/s, Triton GEMMs | hk | x | mean TPOT ms, Triton | hk | x |
|---|---|---|---|---|---|---|
| 1 | 61.6 | 65.5 | 1.06 | 16.07 | 15.10 | 1.06 |
| 8 | 345.6 | 457.2 | 1.32 | 22.15 | 16.55 | 1.34 |
| 32 | 836.2 | 1057.6 | 1.26 | 35.82 | 27.92 | 1.28 |

### A2. Kernels against the tuned field

**The scoreboard** ([chapter 26](history/26-the-scoreboard-retuned.md), with
[chapter 27](history/27-strips.md)'s norms and
[chapter 28](history/28-gemm-schedule.md)'s GEMM; the scoreboard stack:
torch 2.10.0+rocm7.2.4, Triton 3.6.0, FlyDSL 0.4.0.dev970).
`tools/hk-bench/scoreboard.py` runs every way to compute each op in one
process, each checked against fp32 first, in a profiler session per arm per
round, rounds rotated, best round. Every DSL runs out of the box and tuned:
hk's default schedule and its tuner's record; Triton's tutorial kernel and
the same kernel autotuned over gfx1100's space (120 matmul configs;
num_warps x waves_per_eu for the row kernels); FlyDSL's dispatcher and its
autotuner forced to search. "Triton" and "FlyDSL" are the faster of each
DSL's two arms. GPU ms; rows marked * are the best of 9 rounds of 50 calls,
the rest the best of 5 of 10.

| case | hk | torch | Inductor | Triton | FlyDSL | hk against the fastest other |
|---|---|---|---|---|---|---|
| gemm 4096x4096x4096 | 1.5229 | 1.9682 | -- | 1.6638 | 1.5423 | 1.01x tie |
| gemm 2048x6144x4096 | 1.1285 | 1.5719 | -- | 1.2523 | 1.1325 | 1.00x tie |
| gemm 2048x4096x4096 | 0.7785 | 0.9988 | -- | 0.8615 | 0.7721 | 0.99x tie |
| gemm 2048x24576x4096 | 4.6041 | 5.9432 | -- | 5.0004 | 4.6292 | 1.00x tie |
| gemm 2048x4096x12288 | 2.3099 | 2.9459 | -- | 2.5596 | 2.2950 | 0.99x tie |
| attention n4096 causal, GQA 4 | 3.0094 | 10.8671 | -- | -- | -- | 3.61x win (torch 2.10) |
| attention n4096, GQA 4 | 4.6924 | 13.2199 | -- | -- | -- | 2.82x win (torch 2.10) |
| rms_norm 1x4096* | 0.0024 | 0.0042 | 0.0026 | 0.0026 | 0.0025 | 1.04x win |
| rms_norm 32x4096* | 0.0023 | 0.0044 | 0.0027 | 0.0027 | 0.0025 | 1.12x, a tie by noise |
| rms_norm 2048x4096 | 0.0184 | 0.0263 | 0.0197 | 0.0193 | 0.0190 | 1.04x win |
| rms_norm 8192x4096 | 0.1699 | 0.1771 | 0.1775 | 0.1744 | 0.1741 | 1.02x win |
| fused add + rms_norm 1x4096* | 0.0032 | 0.0060 | 0.0026 | 0.0027 | 0.0029 | 0.83x loss (0.83-1.08x over five runs) |
| fused add + rms_norm 32x4096* | 0.0025 | 0.0064 | 0.0029 | 0.0028 | 0.0030 | 1.11x, a tie by noise |
| fused add + rms_norm 2048x4096 | 0.0362 | 0.0488 | 0.0396 | 0.0363 | 0.0358 | 0.99x tie |
| fused add + rms_norm 8192x4096 | 0.3390 | 0.4547 | 0.3708 | 0.3280 | 0.3690 | 0.97x loss (1.02x tie in the other run) |
| softmax 2048x4096 | 0.0187 | 0.0304 | 0.0573 | 0.0255 | 0.0191 | 1.02x tie |
| softmax 8192x4096 | 0.1686 | 0.1755 | 0.2240 | 0.1725 | 0.1725 | 1.02x win |

**GEMM.** hk's bf16 GEMM runs 87.9-91.3 TFLOP/s at these shapes and ties
FlyDSL at all five (from 0.95-0.98x in chapter 26), once the `s_setprio`
around each WMMA group stopped fencing the scheduler (chapter 28). Against
Triton tuned over gfx1100's space it is 1.09-1.11x, and against hipBLASLt
1.28-1.39x (chapter 28). FlyDSL's earlier lead was its
`sched_group_barrier` hints (5-6%) and a per-workgroup rotation of the K
start (5-12%); hk can now express both, and without `s_setprio` neither
adds anything to hk's schedule.

**Attention.** FlyDSL's flash attention does not run on gfx11, which has no
global-to-LDS load, and Triton has no tuned arm here, so the comparison is
torch's SDPA (aotriton). torch 2.13's SDPA runs the causal n4096 case in
3.24-3.27 ms where 2.10's took 10.83
([chapter 30](history/30-flydsl-routes-gate-contracts.md)). Against torch
2.13, in the performance gate, hk's attention is 1.24x causal and 1.05x
non-causal since [chapter 38](history/38-rdna-losses.md) packed V's
transpose (1.13x and 0.99x in the gate just before it; chapter 30 measured
1.16x and 0.99x). The scoreboard's 3.61x and 2.82x are against torch 2.10.

**Norms and softmax** run on register strips (chapter 27), each lane owning
16-byte runs of a row. RMSNorm and softmax win or tie at every size, by
4-19% at one and 32 rows, 2-4% at 2048 and 2% at 8192 rows at the HBM
roof. The fused add + RMSNorm ties over the runs, with a loss at each end:
at one row in one run of five (the clock state), and at 8192 rows in one of
two, inside the 2.4-3.6% that buffer placement alone moves an 8192-row
comparison. The small rows were then measured queued behind a spin
(chapter 32; out of the box, the median of 7 rounds of 50 calls, us):

| case | hk | Inductor | Triton | FlyDSL | torch |
|---|---|---|---|---|---|
| rms_norm 1x4096 | 1.95 | 2.04 | 2.06 | 2.17 | 3.62 |
| rms_norm 32x4096 | 2.17 | 2.27 | 2.27 | 2.37 | 3.77 |
| fused add + rms_norm 1x4096 | 2.12 | 2.14 | 2.20 | 2.64 | 7.16 |
| fused add + rms_norm 32x4096 | 2.33 | 2.56 | 2.48 | 2.84 | 7.57 |

Every arm's rounds sit within 1.1% of their median, and hk's kernels are the
fastest at all four: 1.01-1.06x the next over three queued runs. The tuned
arms were not run queued.

**The performance gate** (`tests/perf/baselines/gfx1100.json`) holds the
same scoreboard's out-of-the-box arms on the vLLM 0.31 venv after October 15
(torch 2.13.0+rocm7.2, Triton 3.8.0, FlyDSL 0.4.0). In its baseline as
[chapter 31](history/31-remeasured.md) refreshed it, FlyDSL's dispatcher
takes 1.00-1.03x hk's time on the GEMM, and on the 2048- and 8192-row norms
and softmax each out-of-the-box arm takes 0.99-3.03x hk's time (Triton's
tutorial kernels, not the tuned ones above; the 0.99x is FlyDSL's 2048-row
fused add). [Chapter 38](history/38-rdna-losses.md) refreshed it again after
the bf16 stores and the transpose changed: of 57 pairs the median moved
0.0%, the norms and softmax -2.8% to +2.4%, attention against torch +6.2%
non-causal and +10.1% causal.

**Triton's tutorials and the kernels beside a model**
([rows by index](history/19-rows-by-index.md); LayerNorm backward from
[chapter 38](history/38-rdna-losses.md)). Each checked against fp32 torch
first (the dropout's mask also against Triton's, which it equals), all arms
in one process, best round; the LayerNorm backward row is chapter 38's sweep
of workgroup counts, the best per width, interleaved with the tutorial. ms:

| kernel | shape | hk | Triton | torch | hk against Triton |
|---|---|---|---|---|---|
| dropout, p = 0.5 (tutorial 04) | 8192 x 4096 bf16 | 0.415 | 0.424 | 0.278 | 1.02x |
| KV cache write, -1 slots skipped | 16 / 256 / 4096 tokens, 1024 wide | 0.005 / 0.005 / 0.012 | 0.014 / 0.014 / 0.014 | 0.107 / 0.110 / 0.146 | 3.00x / 2.88x / 1.18x |
| embedding, 151936 x 4096, ids rotated past the cache | 1 / 32 / 4096 tokens | 0.005 / 0.005 / 0.090 | 0.012 / 0.012 / 0.084 | 0.006 / 0.006 / 0.083 | 2.4x / 2.4x / 0.93x |
| LayerNorm backward (tutorial 05) on strips, 4096 rows | 1024 / 2048 / 4096 / 8192 wide | 0.026 / 0.034 / 0.091 / 0.301 | 0.050 / 0.058 / 0.105 / 0.346 | -- | 1.90x / 1.69x / 1.15x / 1.15x |

The dropout is bound by Philox's multiplies in both DSLs; torch's is 1.5x
faster than either, because it uses all four outputs of a Philox call. The
embedding at 4096 tokens is the row layout's 16 random rows per load. The
LayerNorm backward on strips reads a row of any width once; the 1024- and
2048-wide cases fit the 96 MB cache, which is why they run past the card's
HBM rate. The 16-row tile kernel it replaced is in the
[appendix](#pre-strip-norms).

**Elementwise, quantize and RoPE.** The record has no scoreboard run of
these on gfx1100, where the elementwise ops still run the tile kernels they
were tuned on. What it has is against torch: at 8192x4096, against torch
2.10's eager call ([chapter 16](history/16-validation-and-the-cross-warp-race.md)),
add is 0.98x and exp 0.92x, SiLU-mul 1.57x, int8 quantize (4096x4096) 2.78x
and NeoX RoPE (32 heads x 2048 x 128) 9.19x. Against `torch.compile`
([Phases 1 and 2](history/01-phases-1-2.md)), RoPE at 128x1024x128 is 0.076
against 0.107 ms, and SiLU-mul (8192x11008, 0.761 against 0.763) and
quantize (16384x5120, 0.367 against 0.369) are level.

**Rounding.** RDNA's bf16 stores round to nearest even, as torch's do
([chapter 38](history/38-rdna-losses.md)): an add of random bf16 rows is now
torch's bit for bit at every element (89.3% before, every difference an
exact tie), at a cost within about 1%. The GEMM keeps ties away from zero,
which had no tie to round and was 0.7% cheaper at 4096³.

### A3. Kernels inside the engines

Each kernel here is measured against the kernel the engine calls at the same
shapes.

#### Decode GEMV

[A reduction across workgroups](history/11-cross-workgroup-reduction.md).
GEMM milliseconds per decode step of Qwen3-8B, 36 layers plus the LM head,
at least 1.5 GB of distinct weights per shape, every variant checked
against fp32 (`tools/hk-bench/vllm_gemm.py --upstream-skinny`). "wvSplitK"
is upstream vLLM's skinny GEMM, built standalone for gfx1100; it takes one
to five tokens.

| tokens | hipBLASLt | rocBLAS | wvSplitK | hk decode | hk against hipBLASLt |
|---|---|---|---|---|---|
| 1 | 39.98 | 37.02 | 19.11 | 19.66 | 2.03x |
| 2 | 39.68 | 37.57 | 19.87 | 19.69 | 2.01x |
| 4 | 39.72 | 39.02 | 20.40 | 19.77 | 2.01x |
| 8 | 39.50 | 40.86 | -- | 19.89 | 1.99x |
| 16 | 41.00 | 41.75 | -- | 20.23 | 2.03x |
| 24 | 39.74 | 43.93 | -- | 20.56 | 1.93x |
| 32 | 40.48 | 46.81 | -- | 20.84 | 1.94x |

At one token `wvSplitK` is ahead per step (6.9% on `down`); hk reads its
matrices at 720-819 GB/s, against the 676-821 GB/s a plain read reaches. At
33-64 tokens the kernel is register-bound: 28.6 ms against 42.2 installed,
1.47x ([the decode kernel](history/09-decode-kernel.md)). Split-K across
workgroups is correct and slower on every shape at every split, so the GEMV
ships unsplit. vLLM 0.31 runs `wvSplitK` on gfx11 at M ≤ 5, and its table
leaves those calls to it ([chapter 20](history/20-vllm-031.md)).
SGLang's `F.linear` (torch 2.9.1's hipBLASLt) uses 38-44% of the memory
bandwidth at decode; against it hk's GEMV is 1.18-2.30x on Qwen3-8B's
layers at 1-64 tokens ([SGLang](history/24-sglang.md)).

#### Prefill GEMM

[The linear layers](history/08-linear-layers.md); the Phase 3 GEMM, before
chapter 28's schedule. One process, weights rotated through 3x the cache,
each variant checked against fp32 first. hk over `F.linear` (hipBLASLt) on
Qwen3-8B's layers:

| layer | N x K | M=1024 | M=4096 | M=16384 |
|---|---|---|---|---|
| qkv | 6144 x 4096 | 1.15x | 1.10x | 1.10x |
| o | 4096 x 4096 | 1.18x | 1.09x | 1.09x |
| gate_up | 24576 x 4096 | 1.11x | 1.16x | 1.23x |
| down | 4096 x 12288 | 1.17x | 1.10x | 1.17x |

All 36 layers' GEMMs for one step at M = 4096 are 1.13x. Below 1024 rows the
ratio is a comb, not a curve, with losses at M = 192 on every layer
(0.71-0.95x) and at M = 256 on two (0.95x, 0.96x), which is why the patch
routes by a measured table rather than a rule.

#### 4-bit GEMM (W4A16)

[4-bit weights](history/22-int4.md). fp16, µs per call, hk against vLLM
0.31's Triton AWQ kernel (what AWQ runs on ROCm), time over time:

| shape | 1 token | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| qkv 6144 x 4096 | 70.4 / 31.0 (2.27x) | 70.9 / 31.2 (2.27x) | 82.6 / 31.7 (2.61x) | 75.2 / 50.7 (1.48x) | 137.8 / 74.9 (1.84x) |
| o 4096 x 4096 | 53.8 / 24.4 (2.20x) | 53.9 / 24.2 (2.23x) | 54.8 / 24.0 (2.28x) | 56.6 / 38.3 (1.48x) | 95.6 / 53.3 (1.79x) |
| gate_up 24576 x 4096 | 234.0 / 97.9 (2.39x) | 243.9 / 102.0 (2.39x) | 248.1 / 104.7 (2.37x) | 264.1 / 184.9 (1.43x) | 474.5 / 261.6 (1.81x) |
| down 4096 x 12288 | 131.3 / 58.8 (2.23x) | 132.1 / 59.4 (2.22x) | 135.3 / 60.8 (2.23x) | 138.3 / 98.5 (1.40x) | 243.4 / 163.9 (1.49x) |

Against `gptq_gemm_rdna3`, vLLM's HIP W4A16 kernel for gfx1100, which its
GPTQ path selects, hk loses up to 4 tokens (0.69-0.90x; a skinny GEMV
without WMMA's 15 padding rows) and wins from 8 (1.13-2.25x bf16, 1.30-1.82x
fp16 at 64). hk's patch takes only AWQ layers, whose kernel is Triton's.

#### MoE GEMM

[MoE](history/21-moe.md); vLLM 0.31. Both GEMMs of one layer on vLLM's own
sorted lists, vLLM's config and assignment, checked against each other
first, eight routings rotating so the chosen experts are not left in the
cache. µs per layer; hk on a sort redone at 16-row blocks:

| tokens | OLMoE-1B-7B (64 experts, top 8): Triton | hk | x | Qwen3-30B-A3B (128, top 8): Triton | hk | x |
|---|---|---|---|---|---|---|
| 1 | 385.7 | 139.1 | 2.77x | 263.2 | 110.8 | 2.37x |
| 4 | 792.6 | 432.7 | 1.83x | 548.8 | 371.6 | 1.48x |
| 16 | 1226.8 | 896.7 | 1.37x | 1182.5 | 1040.0 | 1.14x |
| 64 | 1454.1 | 1078.1 | 1.35x | 1640.3 | 1516.9 | 1.08x |
| 256 | 1703.1 | 1697.6 | 1.00x | 1955.1 | 1827.2 | 1.07x |
| 512 | 1748.9 | 2608.9 | 0.67x | 2063.0 | 2378.9 | 0.87x |
| 1024 | 2274.5 | 4349.3 | 0.52x | 2357.5 | 3649.3 | 0.65x |
| 4096 | 6312.0 | 15415.0 | 0.41x | 5289.6 | 10938.4 | 0.48x |

The route takes up to 256 tokens, where hk streams the chosen experts at
718-785 GB/s; above, the layer stays on Triton's `fused_moe`. In SGLang,
against its Triton fused-MoE at its default configs, the layer is
1.03-2.51x at 1-256 tokens and 0.61-1.01x at 512-2048
([SGLang MoE](history/25-sglang-moe.md)).

#### Paged prefill

[The decode step captured whole](history/17-the-decode-step-captured-whole.md).
One launch for every prefill request in a step, K/V through the page table,
against vLLM's Triton `unified_attention` on the same batches
(`tools/hk-bench/paged_ab.py --prefill`), ms:

| batch | Triton | hk | x |
|---|---|---|---|
| prompt 2048 | 4.47 | 0.87 | 5.1x |
| prompt 8192 | 82.6 | 10.5 | 7.9x |
| chunk 512 after 4096 | 4.69 | 0.79 | 5.9x |
| 4 x 1024 after 1024 | 12.6 | 2.38 | 5.3x |
| 8 x 3000 + 200 | 10.2 | 2.85 | 3.6x |
| 16 decodes + chunk 512 | 3.03 | 0.72 | 4.2x |

#### Paged decode attention

[Chapter 38](history/38-rdna-losses.md). On vLLM 0.31's cache (one tensor a
layer, K and V side by side in each token's row), every layer's cache read
cold, each arm timed as a CUDA graph's replay, ms a layer. Triton is
`unified_attention` as TRITON_ATTN calls it (its 3-D kernel to 16 requests,
2-D above), on hk's cache; on TRITON_ATTN's own cache it is no faster. Both
Tritons are measured, each in a process of its own with hk's walk beside
it: triton-rocm 3.7.1, which vLLM 0.31 is built for, and Triton 3.8.0,
installed over it in the pod. "Before" is hk's walk before chapter 38;
"now" packs V's transpose (a page's loop from about 3,350 instructions to
1,425) and plans for 256 workgroups instead of 1,024.

| ctx x batch | Triton 3.7.1 | Triton 3.8.0 | hk before | hk now | now against 3.7.1 | now against 3.8.0 |
|---|---|---|---|---|---|---|
| 1024 x 16 | 0.1254 | 0.1246 | 0.1782 | 0.1155 | 1.12x | 1.08x |
| 1024 x 32 | 0.1838 | 0.2582 | 0.2791 | 0.1990 | 0.93x | 1.30x |
| 1024 x 64 | 0.3530 | 0.4413 | 0.4618 | 0.3915 | 0.90x | 1.13x |
| 4096 x 16 | 0.3777 | 0.3988 | 0.4656 | 0.3881 | 0.99x | 1.03x |
| 4096 x 32 | 0.6907 | 0.9462 | 0.8281 | 0.7451 | 0.93x | 1.27x |
| 4096 x 64 | 1.3467 | 1.6118 | 1.5493 | 1.4889 | 0.90x | 1.08x |

At the capture sizes in between, against 3.7.1 / 3.8.0: 0.86-0.87x / 0.90x
at 24 requests, 0.89x / 1.07-1.13x at 40, 1.01-1.02x / 1.02-1.07x at 48,
0.95x / 0.96-1.03x at 56. hk's walk runs at 581-721 GB/s, Triton 3.7.1's
2-D kernel at 730-810. At one and four requests hk's walk is 0.66-0.99x
Triton's 3-D kernel under either Triton, and at eight they tie
(0.97-1.05x), so up to 8 requests stay on Triton's kernel (on hk's cache). In a captured vLLM step Triton's kernel at
17-22 requests took 1.93 ms a call, against 0.517 on the bench and hk's
0.451, so the batches from 16 up stay on hk; why Triton's kernel slows in
the captured step is open (chapter 20 saw the same under triton-rocm 3.7.1).

### A4. Losses and limits

Where hk loses on the W7900D, and what each loss costs a request:

- **Paged decode attention**, kernel for kernel: 0.86-1.12x triton-rocm
  3.7.1's kernel at 16-64 requests, trailing it by 5-14% at 24, 32, 40, 56
  and 64 requests ([A3](#paged-decode-attention)). In the captured step hk's
  walk is the faster, so it keeps those batches. Up to 8 requests Triton's
  3-D kernel runs on hk's cache (hk's walk 0.66-0.99x at 1 and 4 requests).
- **W4A16 at 1-4 tokens**: 0.69-0.90x `gptq_gemm_rdna3`. GPTQ checkpoints
  stay on that kernel, so no request pays it (a tie end to end,
  1.002-1.004x); closing it needs a dot-product GEMV instead of the WMMA
  tile ([chapter 38](history/38-rdna-losses.md)).
- **The MoE GEMM above 256 tokens**: 0.41-0.87x vLLM's `fused_moe` at
  512-4096 tokens, and the layer 0.61-1.01x SGLang's at 512-2048. Those
  calls stay on Triton, and prefill ties stock end to end.
- **The decode GEMV at one token** is behind upstream vLLM's `wvSplitK`, and
  vLLM 0.31 runs `wvSplitK` at M ≤ 5, so the linear patch ties stock's
  decode step at 1-4 requests ([A1](#a-model-no-table-had-measured)).
- **The prefill GEMM below 1024 rows** is a comb with losses (0.71-0.95x at
  M = 192); the routing table sends those Ms to the engine. A 1024-token
  prefill step was 1.2% slower after FlyDSL rows were dropped at M = 1024
  ([FlyDSL's GEMM on the routes](#flydsls-gemm-on-the-routes)).
- **Small kernels against torch**: the dropout is 1.02x Triton's but torch's
  is 1.5x faster than both; the embedding at 4096 tokens is 0.93x Triton;
  elementwise exp is 0.92x and add 0.98x torch 2.10's eager kernels, on the
  tile kernels gfx1100 still runs ([A2](#a2-kernels-against-the-tuned-field)).
- **The fused add + RMSNorm** lost at one row in one run of five (0.83x) and
  at 8192 rows in one of two (0.97x); a tie otherwise.
- **No norm or activation patch.** Inside vLLM's compiled graph the norms and
  SiLU-mul are Inductor's, fused with their neighbours; the tile kernels of
  the time lost to them by 44-64%, and the patches were dropped
  ([appendix](#against-what-vllm-runs-under-torchcompile)). The strips have
  not been measured inside vLLM's graph.
- **The SDPA drop-in** takes head_dim 64 and 128 only; vision towers with
  72 or 80 fall back to torch (0.95x and 0.97x, measured on torch 2.10;
  [appendix](#vision-towers-on-the-call-vllms-vit-path-makes)).
- **One card, one engine version.** Every number is from one W7900D. Since
  October 15 the pod's vLLM venv imported Triton 3.8.0 in place of the
  triton-rocm 3.7.1 vLLM 0.31 is built for; chapters 30-38's end-to-end
  ratios ran against that stock ([the stacks](#the-stacks)). A
  re-validation on the matched triton-rocm 3.7.1 stack is pending.

## Part B: CDNA4 (gfx950, one MI355X)

Everything in this part was measured on one MI355X of a shared node. The
engines are tuned for this card, so the baselines are stronger than on the
W7900D: vLLM with aiter on, aiter's and CK's kernels, hipBLASLt.

### B1. End to end in vLLM, aiter on

[Chapter 42](history/42-mi355x-in-the-engine.md); the vLLM nightly image
(vLLM 0.30.1rc1.dev558, ROCm 10, torch 2.12, aiter 0.1.23). With
`VLLM_ROCM_USE_AITER=1`, aiter replaces the linear layers (its tuned GEMM:
asm `hgemm_bf16`, Triton `gemm_a16w16`, a hipBLASLt solution, or `wvSplitK`
up to 5 tokens), the fused add + RMSNorm and the MoE layers (CK's
`kernel_moe_gemm`); attention stays ROCM_ATTN (Triton's prefill kernel and
ll4mi for decode). `python -m hk.integration ab`: CUDA graphs and
torch.compile, aiter on, block size 32 for every arm, 2048 tokens of
context, 160 output tokens, 3 rounds; ratios are stock over the arm, the
median of rounds. Greedy tokens identical to stock's. SGLang has not been
measured on gfx950.

| Qwen3-8B, bf16 | total x1 | x16 | x64 | prefill x16 | decode x16 | decode x64 |
|---|---|---|---|---|---|---|
| hk (attention backend) | 1.014 tie | **1.126** | **1.209** | 1.147 | 1.037 | 1.150 |
| ROCM_AITER_FA (vLLM's own) | 0.986 tie | 1.093 | 1.147 | 1.142 | 0.997 | 1.016 |
| hk + gemm, shipped table | 1.013 tie | 1.126 | 1.210 | 1.145 | 1.037 | 1.146 |
| hk + gemm, aiter's points | 0.995 tie | **1.142** | **1.217** | 1.200 | 1.048 | 1.161 |

hk's backend beats stock and vLLM's own aiter FA backend, by 1.055x in total
and 1.13x in decode at 64 requests; its prefill matches aiter FA's, and its
decode is chapter 39's kernel. The shipped linear table was measured against
`F.linear`, and with it the gemm arm changes nothing. A second table, sending
hk only the points where it beat aiter by 3% in the kernel bench
([B3](#b3-kernels-inside-the-engine)), lifts prefill from 1.15x to 1.20x and
the total by about 1% over the backend alone.

| Qwen3-8B-AWQ, fp16 | total x1 | x16 | x64 | decode x1 | x16 | x64 |
|---|---|---|---|---|---|---|
| int4, up to 64 tokens | **1.850** | **1.351** | 0.922 loss | 1.890 | 1.529 | 0.856 |
| int4, gfx950 routing (up to 16 tokens) | **1.865** | **1.351** | 1.020 | 1.916 | 1.530 | 1.037 |

Stock vLLM here does not run Triton's `awq_gemm`: torch.compile's symbolic
batch size sends `AutoAWQLinearMethod` down its dequantize branch at every
batch, `awq_dequantize` plus a hipBLASLt GEMM. hk's kernel beats that only at
1-16 tokens, so on gfx950 the patch now takes at most 16 tokens. The hk
backend cannot run beside the int4 arm: it declares bf16 only, and vLLM
serves an AWQ model in fp16.

**Where the step's time goes**, `headroom --graphs`, aiter on, 2048 tokens of
context; each family's share of a step, and in parentheses the share of its
roofline it runs at:

| model, step | ms | linear | attention | MoE | the rest |
|---|---|---|---|---|---|
| Qwen3-8B decode x1 | 3.67 | 72.7% (86%) | 10.1% (12%) | | 1.04 ms |
| Qwen3-8B decode x16 | 4.97 | 58.2% (80%) | 24.4% (60%) | | 1.07 ms |
| Qwen3-8B decode x64 | 8.46 | 40.9% (68%) | 47.6% (73%) | | 1.15 ms |
| Qwen3-8B prefill 8 x 2048 | 216 | 69.1% (62%) | 18.0% (10%) | | 22.6 ms |
| Qwen3-30B-A3B decode x1 | 4.11 | 21.1% | 13.5% | 49.8% | 1.17 ms |
| Qwen3-30B-A3B decode x16 | 5.60 | 21.0% | 18.5% | 45.4% | 1.20 ms |
| Qwen3-30B-A3B decode x64 | 7.87 | 16.1% | 36.1% | 37.3% | 1.25 ms |

The linear layers run near the roofline at decode; the headroom left is
prefill attention, at 10% of peak, and decode attention at one request,
which is too short to fill the card.

### B2. Kernels against the tuned field

The `rocm/pytorch` stack (ROCm 7.2.3, torch 2.10, Triton 3.6.0, FlyDSL
0.4.0). Every arm checked against fp32 first, then interleaved in one
process.

#### GEMM

[Chapter 33](history/33-cdna-gemm.md), routed as in
[chapter 41](history/41-cdna-leftovers.md). On gfx950 `hk.ops.gemm.matmul`
routes every shape to one of hk's own kernels: HipKittens' CDNA4 schedule
written in hk (`hk.ops.gemm_cdna`: copies into LDS left in flight, the
pingpong), a four-warp kernel (`hk.ops.gemm_cdna_dp`), either with two-launch
split-K, or the decode GEMV. bf16, C = A·Bᵀ, eager, rounds queued behind a
spin, the median of 7, in three processes. TFLOP/s:

| M = N = K | hk | HipKittens | hipBLASLt | FlyDSL `gemm_a16w16` | hk / best |
|---|---|---|---|---|---|
| 4096 | 1450-1461 | 1431-1444 | 1493-1506 | 1462-1473 | 0.97-0.98x |
| 8192 | 1592-1594 | 1584-1589 | 1642-1650 | 1609-1618 | 0.97x |

At Llama-3-8B's and Qwen3-8B's linear shapes (N x K of qkv, o, gate_up, down
and lm_head; M from 1 to 8192; 122 points), hk's routed kernel, timed as
one arm, against the best of hipBLASLt, FlyDSL and HipKittens, the median
of three processes:

| N x K | M=1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 | 1.36 | 1.06 | 1.00 | 1.19 | 1.57 | 1.23 | 0.93 | 0.92 | 1.10 | 1.26 | 0.97 | 0.96 |
| 6144 x 4096 | 1.11 | 1.15 | 1.03 | 1.10 | 1.26 | 0.93 | 0.97 | 1.00 | 0.93 | 0.95 | 0.99 | 0.95 |
| 12288 x 4096 | 1.42 | 1.21 | 1.21 | 1.20 | 0.92 | 0.97 | 0.98 | 0.93 | 0.94 | 0.99 | 0.94 | 0.95 |
| 14336 x 4096 | 1.16 | 1.04 | 1.03 | 1.02 | 0.91 | 1.00 | 1.17 | 1.20 | 0.97 | 0.96 | 0.99 | 0.95 |
| 24576 x 4096 | 1.31 | 0.99 | 1.20 | 1.05 | 1.00 | 1.16 | 0.96 | 0.92 | 0.91 | 0.95 | 0.94 | 0.92 |
| 28672 x 4096 | 1.02 | 1.08 | 1.01 | 1.12 | 0.95 | 1.10 | 0.95 | 0.95 | 0.98 | 1.00 | 0.93 | 0.93 |
| 4096 x 12288 | 1.11 | 1.10 | 0.95 | 1.26 | 1.06 | 0.98 | 1.00 | 1.03 | 1.11 | 1.14 | 0.97 | 0.96 |
| 4096 x 14336 | 1.06 | 1.03 | 1.06 | 1.14 | 1.00 | 0.96 | 1.13 | 1.05 | 1.10 | 1.12 | 0.96 | 0.95 |
| 128256 x 4096 | 1.03 | 1.10 | 1.17 | 1.01 | 0.96 | 0.94 | 0.94 | 1.00 | 0.98 | 0.98 | 1.00 | 0.99 |
| 151936 x 4096 | 1.03 | 0.99 | 0.99 | 0.96 | 0.91 | 1.05 | 0.96 | 1.11 | 1.11 | 0.98 | 1.00 | 1.01 |

All 122 points are at 0.9x or better (chapter 33: 108). M = 1-4 runs at
0.99-1.42x, M = 16-128 at 0.91-1.57x, M = 256-512 at 0.92-1.20x and
M ≥ 1024 at 0.91-1.26x. The tightest is the unmerged gate/up at M = 64
(0.911, 0.910, 0.890 over the three processes): 224 blocks of 64 x 64 on 256
CUs. The small-M points stream the weights whole-line and non-temporal. At
8192³, without its copies the kernel runs 15% faster, and with C's store
switched off it is level with hipBLASLt (chapter 33).

#### Attention forward

[Chapter 34](history/34-cdna-attention.md). `hk.ops.attn_cdna` is
HipKittens' CDNA4 GQA schedule written in hk; `hk.attention` and
`hk.ops.sdpa` run it on gfx950. bf16, (B, H, N, D), TFLOP/s counted as
4·B·H·N²·D (halved under causal); hk's output also checked against V = 1
exactly; best of 7 rounds; HipKittens built with ROCm 7.1, hk with 7.2.3:

| B, H, H_KV, N, D | causal | hk | HipKittens | FlyDSL | torch SDPA | hk / best | hk / SDPA |
|---|---|---|---|---|---|---|---|
| 16, 64, 8, 1024, 128 | no / yes | 1056 / 739 | 978 / 681 | 1055 / 746 | 574 / 256 | 1.00x / 0.99x | 1.84x / 2.89x |
| 16, 64, 8, 4096, 128 | no / yes | 1230 / 1107 | 1233 / 1091 | 1247 / 1122 | 690 / 431 | 0.99x / 0.99x | 1.78x / 2.57x |
| 16, 64, 8, 8192, 128 | no / yes | 1259 / 1209 | 1277 / 1202 | 1283 / 1226 | 773 / 497 | 0.98x / 0.99x | 1.63x / 2.43x |
| 1, 32, 8, 4096, 128 | no / yes | 1164 / 1104 | 1157 / 902 | 1178 / 1024 | 653 / 353 | 0.99x / 1.08x | 1.78x / 3.13x |
| 16, 64, 8, 4096, 64 | no / yes | 938 / 857 | 939 / 804 | 942 / 862 | 724 / 539 | 1.00x / 0.99x | 1.30x / 1.59x |

Round-to-round spreads are 0.3-2.5%, and 6-16% for every arm at N = 1024
and at one batch. hk's step is 2% slower than FlyDSL's, and its fixed cost a
workgroup 10% lower.

#### Norms and elementwise

[Chapter 43](history/43-cdna-last-rows.md), after [chapter 37](history/37-cdna-rows.md)
put the row kernels on wave64 strips and [chapter 41](history/41-cdna-leftovers.md)
tuned their plans. `tools/hk-bench/scoreboard.py --only
llm-norm,llm-quant,llm-pointwise --rotate-buffers`: rows 1, 32, 2048 and
8192 x hidden 4096, 5120 and 8192 (intermediate 14336, 17408 and 28672 for
SiLU-mul; heads of 128 for RoPE), bf16. Rounds queued behind a spin, the
median of 7 rounds of 50 calls; each round every arm takes the next arm's
output buffer, because at one row which buffer a kernel writes moves it
0.1 us. The arms are torch 2.10 eager, Inductor (static shapes), Triton 3.6
as written and autotuned, and FlyDSL 0.4.0's dispatcher and autotuner
(norms and softmax; it has no standalone kernel for the rest). Each cell is
hk's speed over the fastest other arm; a tie is within the larger of 2% and
the two arms' noise. The first rotated run; the last column is chapter 41's
run, which did not rotate the buffers.

| op | 1-32 rows | 2048 rows | 8192 rows | verdicts | chapter 41 |
|---|---|---|---|---|---|
| rms_norm (weighted) | 1.00 | 1.02-1.14 | 1.00-1.02 | 7 tie, 5 win | 9 tie, 3 win |
| rmsnorm | 1.00 | 1.03-1.07 | 1.00-1.01 | 9 tie, 3 win | 1 loss, 10 tie, 1 win |
| layernorm | 1.00-1.05 | 1.02-1.22 | 0.99-1.02 | 5 tie, 7 win | 2 loss, 4 tie, 6 win |
| fused add + RMSNorm | 0.99-1.00 | 0.99-1.01 | 1.00-1.01 | 12 tie | 3 loss, 8 tie, 1 win |
| softmax | 0.99-1.00 | 0.99-1.04 | 1.02-1.04 | 9 tie, 3 win | 12 tie |
| int8 quantize | 1.04-1.05 | 0.99-1.04 | 0.99-1.04 | 3 tie, 9 win | 1 loss, 2 tie, 9 win |
| add | 0.99-1.01 | 1.00-1.04 | 1.00-1.01 | 11 tie, 1 win | 1 loss, 9 tie, 2 win |
| mul | 0.99-1.01 | 1.00-1.05 | 1.00-1.01 | 11 tie, 1 win | 1 loss, 8 tie, 3 win |
| exp | 0.98-1.02 | 1.04-1.06 | 1.01-1.02 | 1 loss, 8 tie, 3 win | 1 loss, 9 tie, 2 win |
| SiLU-mul (gate/up halves in place) | 1.04-1.06 | 1.00-1.03 | 1.10-1.12 | 2 tie, 10 win | 3 tie, 9 win |
| RoPE (`hk.ops.rotary`) | 1.03-1.06 | 0.99-1.05 | 0.98-1.01 | 4 tie, 8 win | 2 loss, 3 tie, 7 win |

131 of 132 cases tie or win in this run (50 win, 81 tie). A second rotated
run gave 128, and the same tree without rotation 126, with all 66 cases past
32 rows a tie or a win (chapter 41: 120; its tree re-measured: 117;
chapter 37: 99). At 1-32 rows every arm sits at the card's floor, an empty
kernel's 3.80 us plus 0.05-0.1, and which arm gets the 0.1 follows its
output buffer. Examples: the fused add + RMSNorm at 2048x8192 takes 19.17 us
against tuned Triton's 19.38; at 8192x8192 rms_norm 37.2 us against tuned
Triton's 38.1 and softmax 38.1 against FlyDSL's 39.7; SiLU-mul at
8192x28672 226.1 against tuned Triton's 251.3. What still loses is in
[B4](#b4-losses-and-no-headroom-findings).

### B3. Kernels inside the engine

The vLLM nightly image, against what vLLM runs on the MI355X at its shapes
(`tools/hk-bench/engine_kernels_gfx950.py`, `paged_cdna.py`), each timed
burst queued behind a sleep longer than its enqueue: aiter's GEMM costs about
60 us of Python per call at M = 16, against a 12 us kernel.

**Linear layers (Qwen3-8B), against aiter's tuned GEMMs**
([chapter 42](history/42-mi355x-in-the-engine.md)), on chapter 41's GEMMs:
- one request: gate_up 1.21x and the LM head 1.04x, but qkv, o and down
  0.79-0.85x;
- 16 requests: the LM head 1.17x, the rest 0.86-1.02x;
- 64 requests: qkv, o and down 1.05-1.09x;
- prefill: qkv and gate_up 1.04-1.05x at 16384 rows, down 1.13x at 2048.

Before chapter 41, hk lost at every decode shape but two (0.69-0.93x). From
these kernel times, sending only the winning shapes to hk would save about 7%
of a one-request decode step and 1-2% elsewhere; end to end the 7% did not
show ([B1](#b1-end-to-end-in-vllm-aiter-on)).

**Prefill attention** at 8 x 2048 (chapter 42): Triton's prefill kernel,
what ROCM_ATTN runs, takes 0.906 ms; hk's `attn_cdna` 0.364 ms (2.49x; 1.87x
with the layout copies), HipKittens' GQA 0.318 ms and aiter's FA 0.319 ms.

**Paged attention** ([chapter 39](history/39-cdna-paged.md)). On gfx950
`hk.ops.paged` routes to `hk.ops.paged_cdna` (the split-KV decode and its
merge) and `hk.ops.paged_prefill` to the paged form of `hk.ops.attn_cdna`.
Qwen3-8B's attention (32/8 heads, D 128), bf16, us a layer. Each arm at its
own cache layout holding the same K/V, checked against fp32 first, its calls
rotating through caches past the 256 MB cache, captured in a CUDA graph and
replayed, rounds interleaved, median of 7. aiter's gluon unified attention
runs on hk's cache; ROCM_ATTN (vLLM's default on gfx950) and TRITON_ATTN on
their own pages of 16.

| decode, requests x context | hk | aiter gluon | ROCM_ATTN | TRITON_ATTN | hk / best | hk / ROCM_ATTN |
|---|---|---|---|---|---|---|
| 1 x 1024 | 9.1 | 15.5 | 9.4 | 9.1 | 1.00x | 1.03x |
| 1 x 8192 | 16.1 | 12.2 | 12.8 | 31.0 | 0.76x | 0.80x |
| 4 x 8192 | 28.2 | 27.3 | 30.7 | 42.5 | 0.97x | 1.09x |
| 16 x 1024 | 17.3 | 17.4 | 18.3 | 22.7 | 1.01x | 1.05x |
| 16 x 8192 | 88.7 | 89.9 | 99.0 | 131.8 | 1.01x | 1.12x |
| 32 x 8192 | 171.4 | 166.1 | 194.7 | 301.5 | 0.97x | 1.14x |
| 128 x 1024 | 84.0 | 85.0 | 104.4 | 118.4 | 1.01x | 1.24x |
| 128 x 8192 | 659.2 | 644.6 | 725.8 | 898.1 | 0.98x | 1.10x |

Over 40 decode shapes (1-128 requests, 512-8192 tokens): 0.90-1.05x the best
alternative at 8-128 requests and 1.04-1.27x ROCM_ATTN there; at one or two
requests 0.71-0.88x past 2048 tokens (the merge, and a lone wave's walk) and
0.81-1.02x at 512-1024. SDPA over gathered pages is 5-13x slower than any.

| prefill, cached prefix + new tokens | hk | aiter gluon | ROCM_ATTN | TRITON_ATTN | hk / best | hk / ROCM_ATTN |
|---|---|---|---|---|---|---|
| 0 + 2048 | 59.3 | 58.3 | 153.3 | 224.5 | 0.98x | 2.59x |
| 4096 + 512 | 106.2 | 92.0 | 283.7 | 228.3 | 0.87x | 2.67x |
| 4 x (2048 + 1024) | 171.8 | 192.6 | 600.6 | 920.8 | 1.12x | 3.50x |
| 8 x (0 + 2048) | 297.9 | 327.0 | 889.6 | 1511.2 | 1.10x | 2.99x |
| 0+300, 1000+77, 4000+1000, 64+2048 | 142.7 | 176.9 | 1908.3 | 683.9 | 1.24x | 13.4x |

**Norms and elementwise, against aiter** (chapter 42): aiter's fused add +
RMSNorm is 1.05-1.12x hk's; hk's RMSNorm is 0.74-1.05x aiter's, and its
SiLU-mul 0.96-1.16x Inductor's.

**MoE (Qwen3-30B-A3B), against aiter's fused_moe (CK)** (chapter 42): 37.7
against hk's 40.2 us at one token, 148 against 222 at 16, 209 against 318 at
64. At 2048 tokens hk's block_m 64 schedule fails to verify on gfx950 (the op
needs 16x32 fragments where it is given 16x16).

**4-bit GEMM**: `hk.ops.w4a16` runs on gfx950 on gfx1100's schedules, 2.3-4.0x
vLLM's Triton AWQ kernel at 1-16 tokens and 1.5-2.3x at 32-64 (chapter 39).
What the engine actually calls under torch.compile is a dequantize and a
hipBLASLt GEMM, which hk beats only at 1-16 tokens
([B1](#b1-end-to-end-in-vllm-aiter-on)).

### B4. Losses and no-headroom findings

Against aiter-on vLLM, hk's linear, norm and MoE kernels have little or
nothing to give, and some kernels lose:

- **The bf16 linear layers: about 1%.** hk's GEMMs are 0.79-1.21x aiter's
  tuned GEMMs at Qwen3-8B's shapes. The shipped table changes nothing end to
  end; a table of the points where hk wins adds about 1% over the attention
  backend, all of it in prefill (1.15x to 1.20x). The one-request decode step
  the kernel times promised 7% for was a tie (0.989x against the backend's
  1.010x). What it would take: a decode GEMM that beats aiter at every shape,
  and the one-request gap explained.
- **Norms and elementwise: nothing in the engine.** hk's RMSNorm is
  0.74-1.05x aiter's and aiter's fused add + RMSNorm 1.05-1.12x hk's;
  together the family is about 1 ms of a step. Against the tuned field (B2)
  what still loses moves from run to run: one- and 32-row cases at
  0.978-0.979x, at the 3.80 us floor (exp 1x5120 at 3.88 us against torch's
  3.80 in the first rotated run; the fused add at 32x4096 and mul and exp at
  1x8192 in the second), RoPE at 8192x40 once (0.977x, 24.70 against 24.13
  us; 0.983x and 0.985x in the other runs), and without buffer rotation six
  one- and 32-row cases at 0.972-0.980x ([chapter 43](history/43-cdna-last-rows.md)).
- **MoE: nothing.** hk's MoE GEMM is 0.66-0.94x CK's at 1-64 tokens, its
  block_m 64 schedule does not verify on gfx950, and vLLM's MoE patch
  rebinds the Triton experts, so with aiter's CK path on it routes nothing.
- **Paged decode at one or two long requests**: 0.71-0.88x the best past
  2048 tokens, and 0.81-1.02x at 512-1024; the prefill of one prompt after a
  4096-token prefix is 0.87x aiter's.
- **4-bit past 16 tokens**: hk's kernel loses to the dequantize-and-GEMM path
  stock runs, so it routes only 1-16 tokens; at 64 requests the request is
  1.020x, from a 0.922x loss before the routing changed.
- **The GEMM at the squares** is 0.97-0.98x the best (hipBLASLt), and the
  tightest routed point 0.91x; C's store and the copy path are the limits.
- **Attention forward**: hk's step is 2% slower than FlyDSL's; at most
  shapes it is 0.98-1.00x the best.
- **Not measured**: SGLang on gfx950, fp16 in hk's attention backend (so it
  cannot run with AWQ models), and gfx942 (no MI300X was available).

## Appendix: superseded results

Results that newer measurements replaced, kept because the chapters cite
them. Each section says what replaced it. Do not quote these as current.

### vLLM 0.16 end to end

Superseded by vLLM 0.31 ([A1](#vllm-031-on-gfx1100)). Qwen3-8B, bf16, one
W7900D, stock vLLM 0.16.1.dev0 choosing its own backends; arms alternate in
separate processes, each its minimum over rounds; every arm produced stock's
greedy tokens. vLLM 0.16 ran its skinny GEMMs on gfx9 only, so its stock
decode step was 41.3 ms at 1024 x 1 against 0.31's 24.5.

**Under vLLM's defaults, the decode captured whole**
([chapter 17](history/17-the-decode-step-captured-whole.md)). CUDA graphs,
prefix caching and chunked prefill on; 72 tokens, best of two rounds; `hk`
the attention backend, `hk+gemm` with the linear layers:

| ctx x batch | stock ms | hk | hk+gemm | step, stock ms | hk | hk+gemm |
|---|---|---|---|---|---|---|
| 1024 x 1 | 3206 | 1.007x | 1.85x | 41.32 | 0.999x | 1.94x |
| 1024 x 4 | 4033 | 1.028x | 1.69x | 42.31 | 0.996x | 1.91x |
| 1024 x 16 | 7628 | 1.065x | 1.48x | 46.32 | 0.989x | 1.74x |
| 1024 x 32 | 13059 | 1.159x | 1.49x | 61.64 | 1.209x | 1.98x |
| 4096 x 1 | 4448 | 1.116x | 1.81x | 42.21 | 0.999x | 1.91x |
| 4096 x 4 | 9195 | 1.264x | 1.74x | 45.29 | 1.010x | 1.85x |
| 4096 x 16 | 27923 | 1.379x | 1.67x | 56.20 | 0.997x | 1.52x |
| 4096 x 32 | 55372 | 1.479x | 1.74x | 109.38 | 1.550x | 2.17x |

**The linear-layer patch alone**
([a reduction across workgroups](history/11-cross-workgroup-reduction.md)).
Prefix caching and chunked prefill off; graph mode, 72 output tokens (128 at
32 and 64 requests). Total 1.84x / 1.62x / 1.35x / 1.27x / 1.18x at 1024 x
1 / 4 / 16 / 32 / 64, and 1.52x / 1.25x / 1.11x at 4096 x 1 / 4 / 16;
prefill 1.06-1.17x; decode step 1.23-1.95x. Nothing lost at any shape, in any
phase, in graph or eager mode.

**The attention backend before the decode GEMV**
([the linear layers](history/08-linear-layers.md)): it lost wherever the
request was mostly decode (decode step 0.834-0.849x) and won where prefill
dominated (1.293x total at 4096 x 16, 1.436x prefill), with both patches
1.364x at 4096 x 16. Eager, the same measurement was 1.29x on the
prefill-heavy shape and 0.87x at 1024 x 1
([against the engine](history/07-against-the-engine.md)). After the decode
GEMV ([the decode kernel](history/09-decode-kernel.md)), `gemm` alone was
1.104-1.717x per request and the best patch where decode dominated;
`hk+gemm` 1.297-1.476x, best on long-context prefill and wide decode.

**Routing by the measured table** ([the routing table](history/14-routing-table.md)).
In GEMM milliseconds per step for 36 layers plus the LM head, the measured
table beat the old hand-written rule by 1.07-1.34x at M = 96-784 and lost to
it by 0.5% and 2.1% at M = 1216 and 1728 (its 3% bar); out of sample it never
lost to the engine. End to end at 512 x 96 / 128, graph mode: 1.12x / 1.15x
stock per request with the table, against 1.07x with the rule.

**Paged decode on vLLM 0.16's cache.** [Phase 7](history/06-phase-7-paged-decode.md):
the backend's decode lost to vLLM's Triton attention everywhere but 1024 x
32, 1.14-1.19x Triton's time per step after the KV split (1.92x before), a
flat 7-9 ms that was the wrapper's torch ops. Captured whole, layer by layer
([chapter 17](history/17-the-decode-step-captured-whole.md)), hk's kernel
lost up to 8 requests (0.66-0.97x), won at 16 (1.02-1.04x) and was 2.6x at
32, where Triton's backend leaves its 16-segment kernel. On vLLM 0.31 with
triton-rocm 3.7.1 it was 0.83-0.95x at 16-64 requests
([chapter 20](history/20-vllm-031.md)), before chapter 38's packed transpose.

**Gate C of the MLIR move**: the GEMM patch in vLLM 0.16.1 serving Qwen3-8B
(eager, context 1024, 72 tokens) was 1.820x / 1.338x stock at batch 1 / 16
before the move and 1.824x / 1.340x after
([the MLIR dialect](history/15-mlir-dialect.md)).

**The vision tower** ([Phase 6](history/05-phase-6-torch-ops.md)). In vLLM
0.16.1.dev0 serving `OpenGVLab/InternVL2-2B`, the SDPA patch reached the
kernel (120 calls, 0 fallbacks over five requests); an end-to-end time cannot
resolve it, since the tower is about 4 ms of a ±25%, 250 ms request.

### torch 2.10's SDPA

Superseded by torch 2.13, whose SDPA is 3.3x faster causal on the W7900D
([A2](#a2-kernels-against-the-tuned-field)). These tables were also measured
before Phase 6 fixed a biased bf16 conversion, which costs attention 2.1%
at head_dim 128 and 3.4% at 64.

#### Against the handwritten C++

[Phase 4](history/03-phase-4-attention.md). Same process, interleaved, both
kernels loaded at once.

| shape | dsl ms | dsl TF | c++ ms | c++ TF | dsl/c++ |
|---|---|---|---|---|---|
| N=4096 | 7.880 | 61.0 | 7.895 | 60.9 | 1.002x |
| N=8192 | 30.528 | 63.0 | 30.527 | 63.0 | 1.000x |
| N=16384 | 120.334 | 64.0 | 120.300 | 64.0 | 1.000x |
| N=32768 | 475.309 | 64.8 | 475.132 | 64.8 | 1.000x |
| 480p/5s | 1100.228 | 64.9 | 1099.149 | 65.0 | 0.999x |
| causal 8192 | 16.628 | 57.9 | 16.754 | 57.4 | 1.008x |

A tie with HipKittens' C++, ahead under causal, at occupancy 6 against the
handwritten kernel's 5; 2.9-3.2x torch 2.10's SDPA, 4.0x causal.

#### Through each entry point

[Phase 6](history/05-phase-6-torch-ops.md), against aotriton, the backend
torch 2.10's SDPA picks on this chip; bit-identity checked before timing.
pybind, `torch.ops` and the SDPA drop-in run within 3% of each other;
aotriton is 0.35x at 1x56x4096x128, 0.33x at 1x56x16384x128 and 0.24x at
1x32x8192x128 causal. The drop-in costs 9.4-9.6 µs of host time a call
against pybind's 3.3-3.8.

#### Vision towers, on the call vLLM's ViT path makes

[Phase 6](history/05-phase-6-torch-ops.md). `(b, s, h, d)` permuted to
`(b, h, s, d)`, never contiguous, against aotriton:

| tower | B,H,N,D | hk ms | torch ms | speedup |
|---|---|---|---|---|
| CLIP ViT-L/14-336 (LLaVA) | 1,16,577,64 | 0.0978 | 0.1657 | 1.69x |
| InternViT-300M | 1,16,1025,64 | 0.1508 | 0.3759 | 2.49x |
| SigLIP so400m | 1,16,729,72 | 0.1945 (falls back) | 0.1850 | 0.95x |
| Qwen2.5-VL ViT | 1,16,1024,80 | 0.3406 (falls back) | 0.3295 | 0.97x |

head_dims 72 and 80 have no kernel and fall back to torch; those rows are
losses.

### Pre-strip norms

Superseded by the register-strip kernels of [chapter 27](history/27-strips.md)
on gfx1100 ([A2](#a2-kernels-against-the-tuned-field)).

#### Against `torch.compile`

[Phases 1 and 2](history/01-phases-1-2.md). 13 of 15 cases faster and the
other two (quantize 16384x4096 and layernorm 16384x5120) within 1% behind, a
tie. ms:

| case | hk | compile | eager |
|---|---|---|---|
| rmsnorm 4096x4096 | 0.060 | 0.093 | 0.418 |
| rmsnorm 16384x5120 | 0.486 | 0.775 | 2.823 |
| layernorm 4096x4096 | 0.061 | 0.081 | 0.063 |
| softmax 8192x16384 | 0.707 | 0.874 | 0.738 |
| silu_mul 8192x11008 | 0.761 | 0.763 | 1.218 |
| rope 128x1024x128 | 0.076 | 0.107 | 0.583 |
| quantize 16384x5120 | 0.367 | 0.369 | 6.118 |

#### Against what vLLM runs under torch.compile

[Strided globals](history/12-strided-globals.md). In the mode vLLM ships,
both ops come from Inductor, fused with their neighbours. Kernel time on
Qwen3-8B shapes, one process:

| op | shape | Inductor | hk |
|---|---|---|---|
| SiLU-mul | (4096, 2*12288) | 409.5 µs | 429.7 µs (contiguous halves) |
| RMSNorm | (4096, 4096) | 38.8 µs | 55.8 µs (hk unweighted) |
| RMSNorm, q/k heads | (163840, 128) | 42.9 µs | 70.4 µs (hk unweighted) |

A loss: the RMSNorm of the time lost by 44-64% before it had a weight to
multiply, so neither patch was built.

#### Against torch, at full size, after the cross-warp fix

[Chapter 16](history/16-validation-and-the-cross-warp-race.md):
`examples/validate.py` with PyTorch 2.10 (ROCm 7.2), hk against torch's eager
call, interleaved in one process. The norms trailed torch's own fused kernels:
rmsnorm 0.96x, layernorm 0.96x and softmax 0.88x at 8192x4096. On the strips,
the scoreboard's torch column has hk ahead at every norm and softmax shape.
The rows of that run that are still current (add, exp, SiLU-mul, quantize,
RoPE) are in [A2](#a2-kernels-against-the-tuned-field); its GEMM (1.12x),
decode GEMV (3.90-4.38x) and attention (2.76x, 3.94x causal) rows were
against torch 2.10's eager calls.

#### The tile kernels' diagnosis, and the LayerNorm backward on tiles

[Chapter 26](history/26-the-scoreboard-retuned.md) found three causes for the
tile norms' losses. At 1-32 rows it was the instruction stream (hk's folded
RMSNorm 846 instructions a warp against Triton's 221). At 2048 rows a 16-bit
row tile held each row twice, in WMMA's lane layouts; moving it once took
rms_norm from 27.17 to 24.72 us and softmax from 29.99 to 23.22. At 8192 rows
the unfolded plan was DRAM-bound and 3-9% behind. The tile kernels still run
rows wider than 32768. [Rows by index](history/19-rows-by-index.md)'s
LayerNorm backward on 16-row tiles was 1.12x / 0.92x / 0.69x / 0.66x the
Triton tutorial at 1024 / 2048 / 4096 / 8192 wide (0.040, 0.065, 0.153 and
0.534 ms), because past 2048 a tile read each row twice; the strips replaced
it.

### Chapter 18's scoreboard

Superseded by [chapter 26](history/26-the-scoreboard-retuned.md), which tuned
every baseline. [Chapter 18](history/18-the-scoreboard.md) measured hk's GEMM
1.03-1.08x ahead (1.05x, 1.08x, 1.08x, 1.05x, 1.03x at the five shapes above)
against FlyDSL's bare 128x128x32 default (1.743 ms at 4096³, where its
dispatcher runs 1.547) and Triton's seven tutorial configs; FlyDSL's
autotuner had never searched, and Triton's ranked cold calls. Its norms were
0.37-0.97x the best other arm, and its attention 3.65x and 2.75x torch 2.10.

### The handwritten GEMM and rocBLAS

[Phase 3](history/02-phase-3-gemm.md); overtaken by hk's later schedules. ROCm
7.2.4; same process, interleaved, bit-identity checked. `hk.ops.matmul` and
the handwritten `config<128,128,64,16,8,4,8,4>` compile to the same 209
VGPRs. At 4096³ both ran 75.0 TFLOP/s and rocBLAS 69.6 (0.93x); at
8192x8192x4096 77.9 and 78.0, rocBLAS 71.8 (0.92x).

### After the MLIR move

[The IR as an MLIR dialect](history/15-mlir-dialect.md), Gate B: old and new
kernels in one process, interleaved for 7 rounds, the old as control. A check
that the move changed nothing, not a comparison.

| kernel | shape | new / old |
|---|---|---|
| GEMM | 4096³ | 1.002x (71.6 TF) |
| attention d128 | N=4096, non-causal / causal | 1.001x / 0.993x |
| decode GEMV | M=1 N=4096, M=1 N=12288, M=4 N=4096 | 1.010x / 0.999x / 1.000x |
| rmsnorm | 8192x4096, 8192x8192, 32768x1024 | 0.998x / 1.002x / 1.002x |

Every gap is smaller than that row's own round-to-round spread.

### The first CDNA pass

Superseded by hk's own CDNA kernels and wave64 strips
([B2](#b2-kernels-against-the-tuned-field)). [Chapter 29](history/29-cdna.md);
one MI355X, `rocm/pytorch` (ROCm 7.2.3, PyTorch 2.10), best of 5,
interleaved in one process.

**Tiles planned for RDNA3**, the first run: vector add 0.61x `torch.add`,
RMSNorm 0.58x `F.rms_norm` and softmax 0.51x `torch.softmax` (`hk.ops`'
versions 0.72x and 0.76x); a GEMM through LDS correct at 4096³, 540 TFLOP/s,
0.35x hipBLASLt (1530).

**CDNA tiles for the elementwise and row kernels** (16x32 fragments, wider
blocks), against torch 2.10's eager call:

| op | 64x4096 | 2048x4096 | 8192x4096 | 16384x8192 |
|---|---|---|---|---|
| add | 0.59x | 0.95x | 0.90x | 0.99x |
| exp | 0.72x | 0.78x | 0.87x | 0.76x |
| rmsnorm (no weight) | 0.94x | 0.71x | 0.93x | 1.05x |
| rms_norm (weight) | 0.73x | 0.61x | 0.95x | 0.98x |
| layernorm | 0.99x | 0.89x | 0.98x | 0.98x |
| softmax | 0.68x | 0.87x | 1.04x | 0.91x |
| fused add + RMSNorm (in place) | 1.27x | 1.06x | 1.44x | 1.25x |
| silu_mul | 1.48x | 1.82x | 1.76x | 1.28x |

Before these tiles, at 8192x4096 the norms ran at 0.70-0.80x torch and the
fused add + RMSNorm at 1.19x.

**HipKittens' CDNA4 kernels as hk ops**
([chapter 29, part 5](history/29-cdna.md#part-5-hipkittens-cdna4-kernels-behind-hks-engine-layer);
[design/cdna-hipkittens-ops.md](design/cdna-hipkittens-ops.md)), best of 7
rounds. HipKittens' GEMM (M and N multiples of 256) ran 0.53x / 0.63x /
0.96x / 0.95x hipBLASLt at 1024³ / 2048³ / 4096³ / 8192³, 0.18-0.93x at
Llama-3-8B and Qwen3-8B's M = 256-1024 and 0.91-0.96x at M ≥ 4096, and won
4-27% only where hipBLASLt's heuristic picked a slow kernel; the gfx950 table
routed it at 37 of 206 points. Its GQA forward ran 1.28-2.62x torch 2.10's
SDPA and 0.92-1.00x FlyDSL's flash attention. hk's own GEMM and attention
(B2) replaced both.

**hk's own GEMM, earlier**: `examples/04_gemm.py` (64x64 blocks, no
pipelining) ran 631 TFLOP/s at 4096³, 0.41x hipBLASLt; chapter 33's routed
table had 108 of 122 linear points at 0.9x the best or better; the rows tied
or won 99 of 132 cases in chapter 37 and 120 in chapter 41 (the last column
in [B2](#norms-and-elementwise)).
