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
ranges, the arms and what was discarded are here.

**Best stock.** For each scenario the stock arm is the best configuration found without changing code: the
engine's backends and switches, environment variables, and offline tuning (TunableOp). A stock configuration
counts only if it is lossless against plain stock on the scenario's quality check.

### MiniCPM-V-4.7-35B-A3B, vLLM, 2x Radeon PRO W7900D

- **Hardware:** two Radeon PRO W7900D (gfx1100), tensor parallel 2, all-reduce over PCIe (no P2P).
- **Engine:** the production serving image's vLLM (a ROCm 7.14 build); compile and CUDA graphs on, GPU memory
  0.925, prefix caching on. The image is not public; [the script](../../scripts/scorecard/minicpmv-35b.sh)
  takes the image as an argument.
- **Arms:**
  - plain stock: the production arguments, `TRITON_ATTN`, no TunableOp;
  - best stock: plus `--attention-backend ROCM_SEGMENTED_ATTN` and TunableOp with 465 GEMMs tuned offline
    under CUDA graphs (one file per rank). Lossless against plain stock: GSM8K-200 91.0% vs 90.0%,
    synthetic-image VQA-100 100% vs 100%, McNemar p 0.63 on 300 paired items;
  - hk: hk-dsl 0.9.20 with its default routes, plus the same TunableOp file.
- **Starts:** best stock and hk alternate, two server starts each, the same driver, the cards locked.
- **Loads and what each row is:**
  - first token: one uncached 32k-token and one 128k-token prompt; ranges over the starts: best stock
    4.39-4.42 s and 35.7-35.8 s, hk 3.67-3.68 s and 23.4 s;
  - time per output token: a fixed number of concurrent requests on synthetic prompts; best stock 12.2 ms at
    1 request and 31.4-33.3 ms at 8, hk 10.85 ms and 24.7-24.8 ms;
  - quality: GSM8K-200, greedy; hk 91.5% and 89.5% on its two starts (stock 91.0% on both), McNemar p 1.0.
  - The scenario was also checked on a replay of real traffic (arrival times, prefix chains and cancels); that
    traffic is not public, so the page quotes no number from it.
- **Where the time goes:** kernel traces of the best-stock arm: full attention is 32% of a 32k prefill and 66% of
  a 128k one; the MoE is 45-48% of a decode step at 8-32 requests. hk's gains come from those two.
- **Discarded:** a start whose server failed or whose cards were touched by another process (the lock is held for
  the whole run); none were.
