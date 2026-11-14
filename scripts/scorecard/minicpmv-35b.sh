#!/usr/bin/env bash
# MiniCPM-V-4.7-35B-A3B on two Radeon PRO W7900 (gfx1100), vLLM, tensor parallel 2:
# best stock vLLM vs vLLM + hk-dsl, same process image, alternating server starts.
#
#   stock = --attention-backend ROCM_SEGMENTED_ATTN + an offline-tuned TunableOp file + CUDA graphs (vLLM default)
#   hk    = hk-dsl's vLLM plugin (--attention-backend TRITON_ATTN, which hk's prefill route runs on) + the same TunableOp file
#
# Measures: TTFT of one uncached 32k and 128k prompt, TPOT at 1 and 8 concurrent requests (2k prompt, 256 output
# tokens), optionally a replay file (REPLAY=path, JSONL of {"t": arrival_s, "prompt": n, "out": n}), and prints a table.
#
# Needs: 2x W7900 48 GB (or 7900 XTX 24 GB won't fit: 70 GB of bf16 weights), a ROCm vLLM install whose `vllm serve` runs
# this model (the measured one: vLLM 0.1.dev1+g10cc2f6ae, torch 2.13, ROCm 7.x), ~90 GB free disk for the weights,
# python3 with `requests`. Time: ~45 min TunableOp tuning (once; cached in $WORK/tunableop), then ~12 min per server
# start; the default plan (4 starts) is ~1.5 h.
#
#   ./minicpmv-35b.sh                                         # everything (hk-dsl $HK_VERSION from PyPI)
#   HK_VERSION=0.9.20 ./minicpmv-35b.sh                       # another hk-dsl release
#   PLAN="stock hk" ./minicpmv-35b.sh                         # one start each
#   MODEL=/path/to/local/snapshot ./minicpmv-35b.sh           # skip the download
#   SHORT=1 ./minicpmv-35b.sh                                 # smoke test: one start per arm, 32k TTFT only (~25 min with a tuned file)
set -euo pipefail
MODEL=${MODEL:-openbmb/MiniCPM-V-4.7-35B-A3B}
WORK=${WORK:-$PWD/minicpmv-35b-work}; mkdir -p "$WORK"
SHORT=${SHORT:-0}
PLAN=${PLAN:-$([ "$SHORT" = 1 ] && echo "stock hk" || echo "stock hk hk stock")}
PORT=${PORT:-8900}
HK_VERSION=${HK_VERSION:-0.9.21}   # hk-dsl release to test (default: the latest on PyPI when this script was written)
GPUS=${GPUS:-0,1}
ulimit -c 0; export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1

ARGS=(--served-model-name m --dtype bfloat16 --tensor-parallel-size 2 --distributed-executor-backend mp
  --moe-backend triton --disable-custom-all-reduce --max-model-len 262144 --max-num-seqs 32
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.925 --enable-prefix-caching
  --trust-remote-code --limit-mm-per-prompt '{"image":4,"video":1}' --mm-encoder-attn-backend TORCH_SDPA --mm-encoder-tp-mode data)
COMMON_ENV=(VLLM_ROCM_USE_AITER=0 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
  HIP_VISIBLE_DEVICES=$GPUS VLLM_NO_USAGE_STATS=1)
TO=(PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=$WORK/tunableop/k%d.csv)

SPID=
stop() { [ -n "$SPID" ] || return 0; kill -INT -- -"$SPID" 2>/dev/null || true
  for _ in $(seq 120); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done; kill -KILL -- -"$SPID" 2>/dev/null || true; SPID=; sleep 5; }
trap stop EXIT
serve() {  # LOG ENV... -- ARGS...
  local log=$1; shift; local envs=(); while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  setsid env "${COMMON_ENV[@]}" "${envs[@]}" vllm serve "$MODEL" "${ARGS[@]}" "$@" --port "$PORT" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1800); do curl -sf -m 2 "localhost:$PORT/health" >/dev/null && return 0
    kill -0 "$SPID" 2>/dev/null || { tail -20 "$log"; return 1; }; sleep 1; done; return 1
}

cat > "$WORK/client.py" <<'PY'
import json, random, statistics, sys, threading, time, requests
port, mode = int(sys.argv[1]), sys.argv[2]
import os
SHORT = os.environ.get("SHORT", "0") == "1"
URL = f"http://127.0.0.1:{port}/v1/completions"
rng = random.Random(int(time.time()))
def one(prompt, out):
    body = {"model": "m", "prompt": [rng.randrange(1000, 150000) for _ in range(prompt)], "max_tokens": out,
            "min_tokens": out, "ignore_eos": True, "temperature": 0, "stream": True}
    t0 = time.perf_counter(); first = None; n = 0
    with requests.post(URL, json=body, stream=True, timeout=3600) as r:
        for line in r.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                first = first or time.perf_counter(); n += 1
    t1 = time.perf_counter()
    return first - t0, (t1 - first) / max(1, out - 1)
def conc(c, prompt, out, rounds=3):
    tp = []
    for _ in range(rounds):
        res = [None] * c
        th = [threading.Thread(target=lambda i=i: res.__setitem__(i, one(prompt, out))) for i in range(c)]
        [t.start() for t in th]; [t.join() for t in th]
        tp += [r[1] for r in res]
    return statistics.median(tp)
if mode == "warm":
    one(64, 8); print("{}"); sys.exit()
r = {}
for L in ((32768,) if SHORT else (32768, 131072)):
    one(L, 1)                       # first one pays one-time kernel compilation; the second is measured
    r[f"ttft_{L // 1024}k_s"] = one(L, 1)[0]
if not SHORT:
    r["tpot_c1_ms"] = conc(1, 2048, 256) * 1e3
    r["tpot_c8_ms"] = conc(8, 2048, 256) * 1e3
print(json.dumps(r))
PY

# 1. TunableOp: record the GEMM shapes this config runs (CUDA graphs on), then tune them offline (once)
if [ ! -s "$WORK/tunableop/k0.csv" ]; then
  mkdir -p "$WORK/tunableop"
  echo "== TunableOp: recording shapes"
  serve "$WORK/record.log" PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 \
    PYTORCH_TUNABLEOP_FILENAME="$WORK/tunableop/rec%d.csv" PYTORCH_TUNABLEOP_UNTUNED_FILENAME="$WORK/tunableop/untuned%d.csv" \
    -- --attention-backend ROCM_SEGMENTED_ATTN
  python3 "$WORK/client.py" "$PORT" measure > /dev/null; stop
  echo "== TunableOp: tuning (256 MB rotating buffer)"
  HIP_VISIBLE_DEVICES=${GPUS%%,*} PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=256 python3 - "$WORK/tunableop" <<'PY'
import glob, os, sys, torch
d = sys.argv[1]; t = torch.cuda.tunable
t.enable(True); t.tuning_enable(True); t.set_filename(os.path.join(d, "k0.csv"), insert_device_ordinal=False)
lines = sorted({l for f in glob.glob(os.path.join(d, "untuned*.csv")) for l in open(f) if l.strip()})
open(os.path.join(d, "merged.csv"), "w").writelines(lines)
t.tune_gemm_in_file(os.path.join(d, "merged.csv")); t.write_file()
print(len(lines), "GEMMs tuned")
PY
  cp "$WORK/tunableop/k0.csv" "$WORK/tunableop/k1.csv"
fi

# 2. hk-dsl in its own target dir (the stock arm never sees it)
HK_DIR=$WORK/hk-$HK_VERSION
[ -d "$HK_DIR" ] || python3 -m pip install -q --no-deps --target "$HK_DIR" "hk-dsl==$HK_VERSION"
echo "hk-dsl $HK_VERSION" | tee "$WORK/versions.txt"; python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null | tee -a "$WORK/versions.txt"

# 3. alternating starts
i=0
for arm in $PLAN; do
  i=$((i + 1)); log="$WORK/$i-$arm.log"
  echo "== start $i: $arm"
  if [ "$arm" = stock ]; then serve "$log" "${TO[@]}" -- --attention-backend ROCM_SEGMENTED_ATTN
  else serve "$log" "${TO[@]}" PYTHONPATH="$HK_DIR" VLLM_PLUGINS=hk_attention HK_CACHE_DIR="$WORK/hk-cache-$HK_VERSION" \
         -- --attention-backend TRITON_ATTN; fi
  python3 "$WORK/client.py" "$PORT" warm > /dev/null
  SHORT=$SHORT python3 "$WORK/client.py" "$PORT" measure | sed "s/^/$arm /" | tee -a "$WORK/results.txt"
  stop
done

# 4. table (median over the starts of each arm)
python3 - "$WORK/results.txt" <<'PY'
import collections, json, statistics, sys
R = collections.defaultdict(list)
for l in open(sys.argv[1]):
    arm, js = l.split(" ", 1); R[arm].append(json.loads(js))
rows = [("TTFT, one uncached 128k prompt (s)", "ttft_128k_s"), ("TTFT, one uncached 32k prompt (s)", "ttft_32k_s"),
        ("TPOT, 8 concurrent requests (ms)", "tpot_c8_ms"), ("TPOT, 1 request (ms)", "tpot_c1_ms")]
med = lambda a, k: statistics.median(x[k] for x in R[a])
print(open(sys.argv[1].replace("results.txt", "versions.txt")).read())
print("\n| metric | best stock vLLM | vLLM + hk-dsl | hk speedup |\n|---|---|---|---|")
for name, k in rows:
    if k not in R["stock"][0]: continue
    s, h = med("stock", k), med("hk", k)
    print(f"| {name} | {s:.2f} | {h:.2f} | {s / h:.2f}x |")
PY
