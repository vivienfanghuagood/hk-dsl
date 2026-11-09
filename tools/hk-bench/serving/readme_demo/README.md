# The README's demo GIF

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
