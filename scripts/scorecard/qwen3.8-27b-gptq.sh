#!/usr/bin/env bash
# Qwen3.8-27B GPTQ-Int4, vLLM, 1x Radeon PRO W7900 48 GB: best stock vLLM vs vLLM + hk-dsl, timed by vLLM's own
# `vllm bench serve`. Needs vLLM for ROCm, `pip install hk-dsl`, and ./tunableop/k0.csv (README.md, step 0).
# Per arm: `vllm serve`, then each `vllm bench serve` once on another seed (warm-up) and once measured.
set -euo pipefail
MODEL=${MODEL:-palmfuture/Qwen3.8-27B-GPTQ-Int4}; OUT=${OUT:-results-qwen3.8-27b-gptq}; PORT=${PORT:-8000}; mkdir -p "$OUT"
export VLLM_ROCM_USE_AITER=0 \
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$PWD/tunableop/k%d.csv
SERVE=(--dtype float16 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90)
RUNS=("32000 1 1 3" "8192 1 1 3" "2048 256 1 4" "2048 256 8 32")   # input tokens, output tokens, concurrency, prompts
for arm in stock hk; do
  if [ $arm = stock ]; then export HK_DISABLE=1; ATTN=ROCM_SEGMENTED_ATTN; else unset HK_DISABLE; ATTN=TRITON_ATTN; fi
  vllm serve "$MODEL" "${SERVE[@]}" --attention-backend $ATTN --port $PORT > "$OUT/$arm-serve.log" 2>&1 & SERVER=$!
  until curl -sf localhost:$PORT/health > /dev/null; do kill -0 $SERVER; sleep 5; done
  for run in "${RUNS[@]}"; do set -- $run
    BENCH=(vllm bench serve --model "$MODEL" --port $PORT --dataset-name random --ignore-eos
      --random-input-len $1 --random-output-len $2 --max-concurrency $3)
    "${BENCH[@]}" --num-prompts 1 --seed 1 > /dev/null
    "${BENCH[@]}" --num-prompts $4 --save-result --result-dir "$OUT" --result-filename $arm-$1in-$2out-c$3.json \
      | tee -a "$OUT/$arm.txt"
  done
  kill -INT $SERVER; wait $SERVER || true
done
for f in "$OUT"/*.json; do python3 -c "import json,sys; r=json.load(open(sys.argv[1]))
print(sys.argv[1].split('/')[-1], 'median TTFT %.0f ms, median TPOT %.2f ms' % (r['median_ttft_ms'], r['median_tpot_ms']))" "$f"; done
grep -h "GPU KV cache size" "$OUT"/stock-serve.log "$OUT"/hk-serve.log
