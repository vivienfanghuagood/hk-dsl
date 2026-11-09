# The README's demo GIF

The GIF at the top of the repository's README shows one long-prompt request served by stock vLLM and by vLLM
with hk-dsl, on the same GPU, played side by side at real speed. This directory makes it again.

1. **Record** (`record.py`), on the GPU, in the vLLM environment the scorecard row used. It builds both engines in
   one process, as [the scorecard](../../../../docs/performance/scorecard.md) does (stock with hk's routes
   reverted, hk with the wheel's default-on routes, both with the same TunableOp-tuned GEMMs when
   `PYTORCH_TUNABLEOP_ENABLED=1` and `PYTORCH_TUNABLEOP_FILENAME` are set), warms both, then streams one fresh
   32k-token request per arm for 3 rounds, arm order alternating, and records when each step's tokens arrived:

   ```bash
   python3 record.py minicpm5.json out/       # writes out/demo.json
   ```

2. **Render** (`render.py`) a round of `demo.json` as an asciicast: both panes start at t = 0, when that arm's
   request was added, and show the tokens as they arrived. Then [agg](https://github.com/asciinema/agg) makes the GIF:

   ```bash
   python3 render.py out/demo.json 1 demo.cast --caption "W7900D · MiniCPM5-2B · vLLM · hk-dsl X.Y.Z"
   agg --font-family "DejaVu Sans Mono" --font-size 14 --fps-cap 10 --idle-time-limit 20 \
       --last-frame-duration 1.5 demo.cast demo.gif
   ```

**Why the two prompts differ by a few tokens** (32,752 and 32,783 in the README's round): every request gets a
document of its own -- a unique first line and its own random shipping entries -- so that no request can hit
the prefix cache of an earlier one. `record.py` sizes each document to 32,768 tokens within 64 (`DEMO_TOKENS`),
not exactly, so each request's prompt length is its own; `demo.json` records it per request. 31 tokens is 0.1%
of the prompt, against first tokens 7 s apart.

The two arms are recorded one after the other in the same process (two engines at once would share the GPU),
and played back together; every time on screen is the recorded one. The rendered round is named in the
README's caption. The numbers the project quotes are the scorecard's, measured over rounds; the GIF shows
one request.
