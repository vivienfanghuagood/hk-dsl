# Serving rows: how the scorecard's end-to-end numbers are measured

The tools that produce the serving rows of [the scorecard](../../../docs/performance/scorecard.md):
a model served by vLLM, with and without hk, against a stock baseline that is given every lever it has
on its own (TunableOp, the engine's attention backends, a tuned MoE config). Each row on the scorecard
names the row and arms below, the image, the weights' revision and the hk-dsl version it ran; with
those, these commands reproduce it.

| file | what it does |
|---|---|
| `serve_bench.py` | one server start of one arm: `vllm serve` with the row's serving configuration and the arm's flags, warm-up, the benchmark specs through `vllm bench serve`, the quality probe, `result.json` |
| `run_row.sh` | a row: every arm, several starts each, alternated (the order reversed every other pass) |
| `table.py` | the row's table from its starts: medians, min-max over starts, ratios called only when the ranges do not overlap |
| `quality.py` | the judge-free quality probe: GSM8K (and ChartQA for a vision model), greedy, exact match, paired by item |
| `to_tune.py` | TunableOp's offline tuning of the GEMM shapes a recorded run used |
| `moe_tune.py` | vLLM's own Triton MoE tuner, with the search space cut to RDNA3's knobs |
| `rows.json` | each row's model and its revision, a serving configuration for it (the flags that matter for performance), benchmark specs and quality set |
| `arms.json` | each row's arms: what each changes against the row's stock |

## The arms

- **stock** (`ship`): the image's vLLM with the row's serving configuration, CUDA graphs, hk off (`HK_DISABLE=1`).
- **one lever each** (`ship_to`, `ship_triattn`, `ship_moe`, ...): so each lever's effect is visible alone.
- **stock-fair** (`fair`): every lever that helped, together. This is the baseline hk is compared with.
- **hk on stock-fair** (`hk_fair`): hk's default routes on top of stock-fair. Arms that isolate one route
  (`hk_fair_moe` / `hk_fair_nomoe`, `hk_fair_noprefill`) set that route's switch explicitly.

## Reproducing a row

In the row's image (the public vLLM ROCm image the row names), with hk-dsl installed in its Python
(`pip install hk-dsl==<the row's version>`) for the hk arms:

```bash
cd tools/hk-bench/serving
W=/data/rsc/mcpm5                                # the row's work directory
export SB_GPUS=0 SB_MODEL=/models/MiniCPM5-2B SB_WORK=$W
mkdir -p $W/data $W/to
# GSM8K's test split as JSON lines (the dataset's own fields are read as they are)
python3 -c "import datasets; datasets.load_dataset('openai/gsm8k', 'main', split='test').to_json('$W/data/gsm8k_test.jsonl')"

# 1. TunableOp: record the GEMM shapes the measured runs use (CUDA graphs, so padded batch sizes too),
#    then tune them offline with a rotating buffer larger than the GPU's last-level cache
python3 serve_bench.py --row mcpm5 --arm fair --mode to_record --gpus $SB_GPUS --model-path $SB_MODEL \
    --work $W --out $W/to_record
PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=256 python3 to_tune.py $W/to/mcpm5.csv $W/to_record/tunableop_untuned*.csv
cp $W/to/mcpm5.csv $W/to/mcpm50.csv              # one results file per rank (mcpm5%d.csv)
# the offline tuner skips M = 1 GEMMs: tune what is left online, in one start that runs the specs
python3 serve_bench.py --row mcpm5 --arm fair --mode to_online --gpus $SB_GPUS --model-path $SB_MODEL \
    --work $W --out $W/to_online
grep -vc Validator $W/to/mcpm50.csv               # the row states how many shapes were tuned

# 2. the row: 3 starts per arm, alternated
./run_row.sh $W/run mcpm5 3 "ship fair hk_fair"

# 3. the table
python3 table.py --ref fair --arms ship,fair,hk_fair \
    --cols "TPOT c8, ms:512/256/8/32:median_tpot_ms" --cols "output tok/s c32:512/256/32/96:output_throughput" \
    --cols "TTFT 16k, ms:16384/16/1/4:median_ttft_ms" $W/run
```

A vision-MoE row (`mcpmv`) also tunes its MoE config first, on one GPU of the run, and puts the files
where the arms look for them (`{work}/moecfg`):

```bash
HK_BENCHMARK_MOE=<vLLM source>/benchmarks/kernels/benchmark_moe.py \
python3 moe_tune.py --model $SB_MODEL --model-prefix text_config --tp-size 2 --tune --save-dir $W/moecfg \
    --batch-size 1 2 4 8 16 24 32 48 64 128 256 512 1024 2048 4096
```

## What makes a number count

- **Starts, not requests.** A row is at least 3 server starts per arm, alternated. A difference is called
  only when the arms' ranges over starts do not overlap (and it is at least 1%); otherwise the table says
  "=". Same-process A/Bs (switching a route between benches in one server) are used where the switch
  allows it, and the row says so.
- **The baseline is fair.** TunableOp is tuned on the shapes the served model actually runs, recorded under
  CUDA graphs, with a rotating buffer larger than the last-level cache: TunableOp's default buffer lets a
  96 MB Infinity Cache keep the weights hot, and then it picks kernels that are slower in serving.
- **Nothing else on the GPUs.** Every bench records which processes held its GPUs (KFD's sysfs). A bench
  with another process holding more than 64 MiB on one of them is discarded, as is one with failed requests.
- **Quality on every start.** GSM8K (and ChartQA for a vision model) after the benches, against the same
  server. A start below the row's floor stops that arm: an arm that is fast and wrong is not a result.
  `table.py` gives McNemar's exact p against the reference arm's starts.
- **The load is part of the result.** A row's numbers hold for its specs (`in/out/concurrency/requests`):
  short random prompts at 32 concurrent requests and one long prompt at a time measure different things,
  and a lever (the attention backend, say) can help one and hurt the other.
