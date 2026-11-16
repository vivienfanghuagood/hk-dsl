# The README's demo GIF

**The README's GIF** (2026-11-16) is the 35B scorecard scenario with CUDA graphs on both sides, recorded with
`record_serve.py` (below). The rest of this page describes the earlier, in-process recorder.

## The 35B recording (`record_serve.py`)

MiniCPM-V-4.7-35B-A3B on 2x W7900D does not fit twice in one process, so each arm is its own `vllm serve`,
started with [scripts/scorecard/minicpmv-35b.sh](../../../../scripts/scorecard/minicpmv-35b.sh)'s arguments
(stock: `HK_DISABLE=1`, `ROCM_SEGMENTED_ATTN`; hk: `TRITON_ATTN`; CUDA graphs and the same TunableOp file on
both). `record_serve.py ARM MODEL OUT/` sends two warm-up requests, then 3 rounds of the copy-the-log prompt below
at exactly 32,768 tokens, each streaming 256 tokens (greedy, `ignore_eos`), and records when each token arrived
and the token ids. `record_serve.py merge OUT/` pairs the arms into `demo.json`; `render.py` and agg then work
as below:

```bash
python3 render.py out/demo.json 0 demo.cast --titles "stock vLLM, CUDA graphs" "vLLM + hk-dsl, CUDA graphs" \
    --hold 4 --caption "2x Radeon PRO W7900D · MiniCPM-V-4.7-35B-A3B, TP2 · vLLM (ROCm 7) · hk-dsl 0.9.23 · real time"
agg --font-size 14 --fps-cap 10 --idle-time-limit 20 --last-frame-duration 4 demo.cast demo.gif
```

The recording is `demo-w7900d-minicpm-v-4.7-35b-0.9.23.json` (prompt ids dropped to keep it small). All 3
rounds had identical outputs. First token 4.35 s vs 3.62 s (1.20x), decode 77.6-78.0 vs 87.0-87.1 tokens/s
(1.12x). The GIF plays round 0.

## The earlier, in-process recorder (`record.py`)

The GIF at the top of the repository's README shows one request served by stock vLLM and by vLLM with hk-dsl,
on the same GPU, played side by side at real speed. Both sides get the same prompt and produce the same output.
This directory makes it again.

**The scenario** (`minicpm-v-4.7-1b.json`): MiniCPM-V-4.7-1B with the engine arguments of
[the scorecard](../../../../docs/performance/scorecard.md)'s row, which are its production ones (eager mode).
Prefix caching is off. hk wins both phases here:
- **prefill:** hk's prefill attention;
- **decode:** the `vllm_norm` route. An eager GemmaRMSNorm is ~8 PyTorch kernels, and hk runs it as one.

1. **Record** (`record.py`), on the GPU, in the vLLM environment the scorecard row used.
   - **Engines:** it builds both engines in one process, as the scorecard does. Stock has every hk vLLM route
     reverted; hk has the wheel's default-on routes. Both get the same TunableOp-tuned GEMMs when
     `PYTORCH_TUNABLEOP_ENABLED=1` and `PYTORCH_TUNABLEOP_FILENAME` are set.
   - **Rounds:** it warms both engines, then runs 5 rounds with the arm order alternating.
   - **The prompt:** each round builds ONE prompt of exactly 8,192 tokens (`DEMO_TOKENS`): a generated shipping log
     with a fresh first line, and a request to copy it out. Both arms get those same token ids.
   - **The output:** each arm decodes exactly 256 tokens (`DEMO_OUT`), greedy, `ignore_eos`.
   - **What it writes:** when each step's tokens arrived, the output token ids, and whether the two arms' ids are
     identical.

   ```bash
   python3 record.py minicpm-v-4.7-1b.json out/       # writes out/demo.json
   ```

2. **Render** (`render.py`) a round of `demo.json` as an asciicast. It refuses a round whose two prompts or
   outputs differ.
   - **Timing:** both panes start at t = 0, when that arm's request was added.
   - **Each pane shows:** the input and output token counts, the time to the first token, the decode rate so far
     (tokens after the first / seconds since the first), the elapsed time and the text.
   - **The last frame:** hk's two ratios, time to first token and decode rate.

   Then [agg](https://github.com/asciinema/agg) makes the GIF:

   ```bash
   python3 render.py out/demo.json 0 demo.cast --caption "W7900D · MiniCPM-V-4.7-1B, eager · hk-dsl X.Y.Z"
   agg --font-size 14 --fps-cap 10 --idle-time-limit 20 --last-frame-duration 4 demo.cast demo.gif
   ```

**Why the outputs are identical:** hk's kernels round differently from stock's (the norm within 1 bf16 ulp). On
an open-ended answer, greedy decoding can then pick a different token after a few dozen tokens. Copying a log is
a task where the top token wins by a wide margin, so both arms decode the same 256 tokens: in every round of
`demo-w7900d-minicpm-v-4.7-1b-0.9.19.json`.

**The README's recording** is `demo-w7900d-minicpm-v-4.7-1b-0.9.19.json`: a W7900D with hk-dsl 0.9.19, 5 rounds.

| | stock | hk | ratio |
|---|---|---|---|
| TTFT, 8,192 tokens | 0.60-0.61 s | 0.24 s | 2.52x median (2.51-2.54) |
| decode | 47.7-48.3 tok/s | 61.0-61.3 tok/s | 1.27x median (1.27-1.28) |

The GIF renders round 0, the one closest to both medians (2.51x, 1.27x). The two arms are recorded one after the
other in the same process (two engines at once would share the GPU), and played back together. Every time on
screen is the recorded one.
