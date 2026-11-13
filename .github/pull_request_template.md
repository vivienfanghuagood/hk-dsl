<!--
hk-dsl PR template (owner decision D-20261113-12). Model: vllm-project/vllm#59132. Delete what does not apply,
but say so ("no performance claim", "no new switch") rather than leaving a section out silently.
Small and scoped: one change per PR, no stacked PRs. No number that this PR does not measure itself.
-->

## Summary

What this does, in two or three sentences.

- **Support matrix:** GPUs / archs (gfx1100, gfx1201, gfx950), engines and versions, dtypes, shapes or model
  features covered -- and what is not covered.
- **Default and switches:** on or off by default, on which GPUs; the variable or flag that turns it on/off
  (e.g. `HK_VLLM_PREFILL=0`); what happens when it does not apply (falls back to the engine, says so in the log).
- **Dependencies:** none new / which.

## Relationship to existing work

What this overlaps with (engine backends, other PRs, earlier hk routes) and how it differs. Say what the results
below are compared against and what they are *not* a comparison with.

## Validation

**Setup, once for all tables:** hardware (GPU model and count, TP), model (exact repo or snapshot), engine and
version (image digest or commit), ROCm / torch / hk-dsl versions, load tool and version, workload and its
parameters (input / output tokens, turns, concurrency), seed, and how many requests completed and errored
(must be zero errors, or say why). End to end (serving, including scheduling), not kernel-only.

**Baseline:** the engine at its best configuration without this change (fastest backend, tuned GEMMs, graphs),
named exactly -- not its default when the default is slower. Quality checked first (e.g. GSM8K-N, greedy or the
sampling used, paired test), and its scope said plainly.

### <config 1: GPU, model, TP>

| metric | baseline → candidate | change |
|---|---:|---:|
| | | |

One sentence on anything that got worse, and why if known.

Baseline command:

```bash
vllm serve ...
```

Candidate command (what differs):

```bash
...
```

### Reproducibility and scope

Exact source commit / image the numbers came from; what was single-run, what was not isolated, what was not
rerun on the final head.

## Tests

The commands run and their results (`N passed`), on which GPU tier / container. New tests added.

## AI assistance

Whether AI assistance was used, and that a human reviews every changed line.
