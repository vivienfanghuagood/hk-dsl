# How the numbers are taken

How a number was taken matters as much as the number. Each rule below exists
because breaking it once produced a plausible result that was wrong. The
numbers the project quotes are end-to-end serving results on
[the scorecard](scorecard.md); the same rules govern the kernel-level
measurements that guide the work (`hk.check`, the route tuner, the
benchmarks), which are not claims.

## Kernel against kernel

- **One process, interleaved.** Kernels being compared run in one process,
  round by round, with the order rotated. A kernel A/B across processes is
  invalid on these GPUs: the same kernel has measured 13-16% apart in two
  processes.
- **Checked first.** Every arm is checked against an fp32 torch reference
  before it is timed, and where a case has an exact answer (V all ones for
  attention, small integers for an add) against that too.
- **Past the cache.** A GPU with a large last-level cache (the W7900D's
  96 MB Infinity Cache, the MI355X's 256 MB) serves a weight read again from
  that cache. A decode-sized kernel is timed on copies of its weights rotated
  through more than the cache, so each round reads them from DRAM as a
  serving engine does.
- **Rounds queued behind a spin.** A launch that reaches the GPU shortly after
  it went idle pays extra; benchmarks queue each round's calls behind a GPU
  spin so every arm starts from a busy GPU.
- **The median of rounds, impossible rounds rejected.** A round that lost
  kernel records, or ran faster than the card can move its bytes, is
  rejected; the result is the median of the rest. A table that reports the
  best round says so.
- **Verdict bands.** A tie is within the larger of 2% and the two arms'
  measured noise.

## End to end

- **Each arm is its own engine.** An engine cannot be interleaved inside one
  process, so arms alternate across starts in separate processes, several
  starts each, and every start is reported. `python3 -m hk.integration ab`
  pairs the ratio round by round, takes the median, calls a result a win or a
  loss past a band of max(2%, noise), a tie inside a band of at most 5%, and
  noise for a single round or a band past 15%.
- **The same session.** An arm is compared with a stock arm run in the same
  session, on the same GPUs; the scorecard does not combine numbers from
  different sessions or tool versions.
- **Outputs compared.** Greedy tokens against stock's, and an accuracy set
  (GSM8K) with a paired test, so a speedup that changed the answers is not
  counted as one.
- **The engine as it ships**, unless a result says otherwise: vLLM's defaults
  (CUDA graphs and torch.compile, prefix caching, chunked prefill). A baseline
  is the engine at its best configuration -- its tuned paths on, its fastest
  backend for the GPU -- not its default when the default is slower.
- **The stack is part of the number.** The torch, Triton, ROCm and engine
  versions move results by more than many kernel wins; each result names the
  image or stack it ran on, and `python3 -m hk.integration.stack` says whether
  a vLLM venv is the measured one.

[Numbers not to trust](../agents.md#numbers-not-to-trust) lists the failure
each rule guards against.

## The scorecard scenarios

How each [scorecard](scorecard.md) scenario is measured. The page carries one small table per scenario; the
arms, ranges, internal checks and costs are here.

**Best stock.** For each scenario the stock arm is the best configuration found without changing code: the
engine's backends and switches, environment variables, and offline tuning (TunableOp). A stock configuration
counts only if it is lossless against plain stock on the scenario's quality check. On both scenarios below it is
not vLLM's default on this GPU: the default is much slower (MiniCPM-V-4.7-35B: 32k first token 14.2 s instead
of 4.40 s).

**How the tables are measured.** Since 2026-11-14 every row is the median that vLLM's own `vllm bench serve` prints
(`--dataset-name random`, `--save-result` keeps its JSON), against one server start per arm, with the commands in
[scripts/scorecard/README.md](../../scripts/scorecard/README.md). Before each measured run, the same command runs
once on another seed: that pays one-time compiles without priming the prefix cache for the measured prompts.
Earlier runs with our own client are kept below as cross-checks. The ratios agree; absolute time per output
token at 8 requests is higher with `vllm bench serve`, because its requests start staggered and new prompts'
prefills land between other requests' decode steps.

### MiniCPM-V-4.7-35B-A3B, vLLM, 2x Radeon PRO W7900D

- **Hardware:** two Radeon PRO W7900D (gfx1100), tensor parallel 2, all-reduce over PCIe (no P2P).
- **Engine:** vLLM 0.1.dev1+g10cc2f6ae (a ROCm 7 build of the production serving image, not public), torch 2.13;
  compile and CUDA graphs on, GPU memory 0.925, prefix caching on. The table: hk-dsl 0.9.23 from PyPI,
  `vllm bench serve`, 2026-11-14 (cards locked for the run). The earlier runs below: hk-dsl 0.9.20.
- **Arms:**
  - plain stock: the production arguments, `TRITON_ATTN`, no TunableOp;
  - best stock: plus `--attention-backend ROCM_SEGMENTED_ATTN` and TunableOp with 465 GEMMs tuned offline under
    CUDA graphs (one file per rank). Lossless against plain stock: GSM8K-200 91.0% vs 90.0%, synthetic-image
    VQA-100 100% vs 100%, McNemar p 0.63 on 300 paired items;
  - hk: hk-dsl's default routes on `TRITON_ATTN`, plus the same TunableOp file.
- **Earlier runs (own client, hk-dsl 0.9.20):** best stock and hk alternate, two server starts each, one driver.
  Ratios 1.53x / 1.20x first token at 128k / 32k, 1.31x / 1.12x time per output token at 8 / 1 requests,
  against 1.59x / 1.20x and 1.27x / 1.11x in the table.
- **Rows, earlier runs:**
  - first token: one uncached 32k-token and one 128k-token prompt; ranges over the starts: best stock 4.39-4.42 s
    and 35.7-35.8 s, hk 3.67-3.68 s and 23.4 s;
  - time per output token: 1 and 8 concurrent requests, prompts drawn from the request-length mix of real
    traffic (the table now uses random 2k-token prompts with 256 output tokens). Best stock 12.2 ms at 1 request and 31.4-33.3 ms
    at 8; hk 10.85 ms and 24.7-24.8 ms.
- **Quality:** GSM8K-200, greedy: 91.0% for stock on both starts, 89.5% and 91.5% on hk's; McNemar p = 1.0.
- **Also checked, not reproducible:** on a 25-minute replay of a real production peak (internal traffic), the
  first token's p90 was 0.91 s for best stock against 0.47-0.67 s with hk.
- **Where the time goes:** kernel traces of the best-stock arm: full attention is 32% of a 32k prefill and 66% of
  a 128k one; the MoE is 45-48% of a decode step at 8-32 requests.
- **hk's costs:** a cold start about 60 s longer; a one-time hk compile (~38 s) on the first long prompt after an
  empty hk cache. Warm restarts are level.

### Qwen3.8-27B GPTQ-Int4, vLLM, 1x Radeon PRO W7900D

- **Hardware:** one Radeon PRO W7900D (gfx1100).
- **Engine:** vLLM 0.33.0, torch 2.13.0 (HIP 7.2); `--dtype float16 --max-model-len 32768 --max-num-seqs 32
  --gpu-memory-utilization 0.90 --enable-prefix-caching`, CUDA graphs. Model `palmfuture/Qwen3.8-27B-GPTQ-Int4`.
- **hk:** hk-dsl 0.9.23's code (`vllm_gptq` on gfx1100 holds one copy of the weights and serves every batch, #399),
  measured as the 0.9.22 wheel plus #399's Python before 0.9.23 was tagged. In 0.9.21 the same route kept a second copy of the weights
  (KV cache 288k -> 94k tokens); 0.9.22 turned it off on this GPU.
- **Arms:** best stock = `--attention-backend ROCM_SEGMENTED_ATTN` + offline TunableOp (tuned on this card) + CUDA
  graphs; hk = hk-dsl's default routes on `TRITON_ATTN` + the same TunableOp file.
- **Starts:** the table is one `vllm bench serve` run per arm (hk-dsl 0.9.23 from PyPI, 2026-11-14). Before it, with
  our own client, the earlier script's run gave the same ratios (1.16x / 1.16x first token, 1.13x / 1.12x time per
  output token; KV cache 281,941 vs 283,306), and so did #399's acceptance: two campaigns of 2+2 alternating starts on one card, medians, 0 errored requests --
  time per output token 32.7 -> 28.5 ms (1.15x) / 44.3 -> 38.2 ms (1.16x) / 67.5 -> 59.8 ms (1.13x) at 1 / 4 / 8
  requests, first token 8.32 -> 7.05 s (1.18x) and 35.2 -> 30.4 s (1.16x) at 8k / 32k, KV cache 288,085 -> 288,768.
- **Rows not in the table:** time per output token at 4 requests 44.3 ms vs 38.5 ms (1.15x) in the earlier script's run.
- **Quality:** GSM8K-200, greedy: stock 91.0% and 91.5%, hk 91.5% and 91.0% on two runs each; McNemar p = 1.00.
- **Isolating the int4 route** (both arms hk on `TRITON_ATTN`, `HK_VLLM_GPTQ=0` against `=1`): time per output
  token 1.14x / 1.15x / 1.13x at 1 / 4 / 8 requests, first token 1.15x / 1.12x at 8k / 32k; so the decode gain is
  all this route's.
- **Also checked, not reproducible:** on a replay of 30 multi-turn sessions sampled from real traffic (internal), at
  4 concurrent sessions, output throughput was 68.8 -> 83.4 tokens/s (1.21x; ~1.13x of it the int4 route) and
  end-to-end p50 / p90 20.4 / 79.8 s -> 16.1 / 64.5 s.
- **hk's costs:** the first hk start after an upgrade compiles the new kernels on its first long prompt (8k first
  token 8.30 s on that start, 7.05 s warm).
