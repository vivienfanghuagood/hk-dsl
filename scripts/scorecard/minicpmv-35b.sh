#!/usr/bin/env bash
# MiniCPM-V-4.7-35B-A3B, vLLM, 2x Radeon PRO W7900 48 GB: best stock vLLM vs vLLM + hk-dsl, timed by vLLM's own
# `vllm bench serve`. Needs vLLM for ROCm, `pip install hk-dsl`, and ./tunableop/k0.csv + k1.csv (README.md, step 0).
# Per arm: `vllm serve`, then each `vllm bench serve` once on another seed (warm-up) and once measured.
set -euo pipefail
MODEL=${MODEL:-openbmb/MiniCPM-V-4.7-35B-A3B}; OUT=${OUT:-results-minicpmv-35b}; PORT=${PORT:-8000}; mkdir -p "$OUT"
export VLLM_ROCM_USE_AITER=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$PWD/tunableop/k%d.csv
SERVE=(--tensor-parallel-size 2 --dtype bfloat16 --moe-backend triton --disable-custom-all-reduce --max-model-len 262144
  --max-num-seqs 32 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.925 --trust-remote-code
  --limit-mm-per-prompt '{"image":4,"video":1}' --mm-encoder-attn-backend TORCH_SDPA --mm-encoder-tp-mode data)
RUNS=("32768 1 1 3" "131072 1 1 3" "2048 256 1 4" "2048 256 8 32")   # input tokens, output tokens, concurrency, prompts
for arm in stock hk; do
  if [ $arm = stock ]; then export HK_DISABLE=1; ATTN=ROCM_SEGMENTED_ATTN; else unset HK_DISABLE; ATTN=TRITON_ATTN; fi
  vllm serve "$MODEL" "${SERVE[@]}" --attention-backend $ATTN --port $PORT > "$OUT/$arm-serve.log" 2>&1 & SERVER=$!
  until curl -sf localhost:$PORT/health > /dev/null; do kill -0 $SERVER; sleep 5; done
  for run in "${RUNS[@]}"; do set -- $run
    BENCH=(vllm bench serve --model "$MODEL" --trust-remote-code --port $PORT --dataset-name random --ignore-eos
      --random-input-len $1 --random-output-len $2 --max-concurrency $3)
    "${BENCH[@]}" --num-prompts 1 --seed 1 > /dev/null
    "${BENCH[@]}" --num-prompts $4 --save-result --result-dir "$OUT" --result-filename $arm-$1in-$2out-c$3.json \
      | tee -a "$OUT/$arm.txt"
  done
  kill -INT $SERVER; wait $SERVER || true
done
for f in "$OUT"/*.json; do python3 -c "import json,sys; r=json.load(open(sys.argv[1]))
print(sys.argv[1].split('/')[-1], 'median TTFT %.0f ms, median TPOT %.2f ms' % (r['median_ttft_ms'], r['median_tpot_ms']))" "$f"; done
