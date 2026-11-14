# Reproducing the scorecard

Every number on the [scorecard](../../docs/performance/scorecard.md) comes from the summary that vLLM's own
benchmark tool, `vllm bench serve`, prints. Each scenario has two arms on the same install and the same GPUs:

- **Best stock vLLM:** hk-dsl is turned off with `HK_DISABLE=1`, and vLLM runs with the best configuration we found for
  it (the flags below, plus a TunableOp file tuned offline).
- **hk-dsl:** the same command with hk-dsl on.

Each arm is one `vllm serve` followed by the `vllm bench serve` runs. Run every bench command once with `--num-prompts 1 --seed 1`
first. That warm-up uses a different prompt from the measured run, so it pays the one-time compiles without
warming the prefix cache for the measured prompts. Each script below is a thin loop over these same commands
(under 30 lines). It runs both arms, saves vLLM's JSON results with `--save-result`, and prints the median TTFT and
TPOT from each JSON file.

```bash
pip install hk-dsl          # into the environment that has vLLM for ROCm
./minicpmv-35b.sh           # or ./qwen3.8-27b-gptq.sh; both need ./tunableop (step 0)
```

## Step 0, once per GPU model: TunableOp file for best stock

Best stock uses PyTorch TunableOp with GEMMs tuned offline, and both arms load the same file. First record the GEMM
shapes that the stock arm runs: serve stock (step 1 below) with these variables added, run the step 2 bench
commands against it, then stop the server.

```bash
mkdir -p tunableop
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
  PYTORCH_TUNABLEOP_FILENAME=$PWD/tunableop/rec%d.csv PYTORCH_TUNABLEOP_UNTUNED_FILENAME=$PWD/tunableop/untuned%d.csv
```

Then tune the recorded shapes on one GPU. This takes 15 min for the 27B model and 45 min for the 35B model.

```bash
PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=256 python3 - <<'PY'
import glob, torch
t = torch.cuda.tunable
t.enable(True); t.tuning_enable(True); t.set_filename("tunableop/k0.csv", insert_device_ordinal=False)
open("tunableop/all.csv", "w").writelines(sorted({l for f in glob.glob("tunableop/untuned*.csv") for l in open(f) if l.strip()}))
t.tune_gemm_in_file("tunableop/all.csv"); t.write_file()
PY
cp tunableop/k0.csv tunableop/k1.csv     # tensor parallel 2: one file per rank
```

## MiniCPM-V-4.7-35B-A3B, 2x Radeon PRO W7900 48 GB

```bash
export VLLM_ROCM_USE_AITER=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$PWD/tunableop/k%d.csv
# 1. best stock vLLM
HK_DISABLE=1 vllm serve openbmb/MiniCPM-V-4.7-35B-A3B --attention-backend ROCM_SEGMENTED_ATTN \
  --tensor-parallel-size 2 --dtype bfloat16 --moe-backend triton --disable-custom-all-reduce --max-model-len 262144 \
  --max-num-seqs 32 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.925 --trust-remote-code \
  --limit-mm-per-prompt '{"image":4,"video":1}' --mm-encoder-attn-backend TORCH_SDPA --mm-encoder-tp-mode data
# 2. in another shell, one command per row
vllm bench serve --model openbmb/MiniCPM-V-4.7-35B-A3B --trust-remote-code --dataset-name random --ignore-eos \
  --random-input-len 131072 --random-output-len 1 --max-concurrency 1 --num-prompts 3     # first token, 128k
#   --random-input-len 32768  --random-output-len 1   --max-concurrency 1 --num-prompts 3  # first token, 32k
#   --random-input-len 2048   --random-output-len 256 --max-concurrency 8 --num-prompts 32 # time per output token, 8 requests
#   --random-input-len 2048   --random-output-len 256 --max-concurrency 1 --num-prompts 4  # time per output token, 1 request
# 3. hk-dsl: the same server command without HK_DISABLE=1 and with --attention-backend TRITON_ATTN, then step 2 again
```

## Qwen3.8-27B GPTQ-Int4, 1x Radeon PRO W7900 48 GB

```bash
export VLLM_ROCM_USE_AITER=0 \
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$PWD/tunableop/k%d.csv
# 1. best stock vLLM
HK_DISABLE=1 vllm serve palmfuture/Qwen3.8-27B-GPTQ-Int4 --attention-backend ROCM_SEGMENTED_ATTN \
  --dtype float16 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90
# 2. in another shell, one command per row
vllm bench serve --model palmfuture/Qwen3.8-27B-GPTQ-Int4 --dataset-name random --ignore-eos \
  --random-input-len 32000 --random-output-len 1 --max-concurrency 1 --num-prompts 3      # first token, 32k
#   --random-input-len 8192 --random-output-len 1   --max-concurrency 1 --num-prompts 3    # first token, 8k
#   --random-input-len 2048 --random-output-len 256 --max-concurrency 1 --num-prompts 4    # time per output token, 1 request
#   --random-input-len 2048 --random-output-len 256 --max-concurrency 8 --num-prompts 32   # time per output token, 8 requests
# 3. hk-dsl: the same server command without HK_DISABLE=1 and with --attention-backend TRITON_ATTN, then step 2 again
```

The KV-cache row is the `GPU KV cache size: N tokens` line that each `vllm serve` prints at startup.

## Reading the result

The table quotes `Median TTFT` for the first-token rows and `Median TPOT` for the time-per-output-token rows. Both
come straight from the `vllm bench serve` summary, also saved as `median_ttft_ms` and `median_tpot_ms` in the JSON
file. hk-dsl's first start after install compiles its kernels into `~/.cache/hk`, which takes about 40 s. The
warm-up run pays that cost, and later starts reuse the cache.
