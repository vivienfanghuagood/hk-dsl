# Template: a performance page or a scorecard scenario

Every page or section in this project that quotes a performance number follows this shape (owner decision
D-20261113-12; model: vllm-project/vllm#59132). One scenario = one section.

## <Model>, <engine>, <N x GPU>

**Result, one sentence:** what hk changes for this scenario, against what.

**Setup:**
- Hardware: GPU model and count, tensor parallel, interconnect.
- Model: repository and snapshot.
- Engine: name, version, image digest or commit; ROCm, torch, hk-dsl versions.
- Load: tool and version, workload, input / output tokens, concurrency, number of requests, seed;
  completed / errored counts (zero errors, or why not).
- What a number is: mean or median, over how many starts or requests; end to end (serving), not kernel time.

**Baseline:** the engine's best configuration without hk, named exactly (backend, tuning, graphs), and why it is
the best one found. Quality checked against plain stock first.

| metric | best stock | hk | change |
|---|---:|---:|---:|
| 3 to 5 rows | | | |

Quality: the check, its size, and the result (equal / not, with the test used).

Anything that got worse, said in one sentence.

**Commands:** the exact baseline and hk commands, or one script that runs both
(`scripts/scorecard/<scenario>.sh`) with its hardware and time needs.

**Scope:** single runs or not, what was not isolated, what was not rerun, where the raw data is (or that it is not
public, and why).

Ranges, discard rules and the per-start numbers go to [the method](method.md), not into the table.
