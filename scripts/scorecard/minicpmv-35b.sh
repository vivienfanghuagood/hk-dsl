#!/bin/bash
# scripts/scorecard/minicpmv-35b.sh -- the scorecard's MiniCPM-V-4.7-35B-A3B scenario, end to end, in one command.
#
#   IMAGE=<a vLLM ROCm image> GPUS=0,1 bash scripts/scorecard/minicpmv-35b.sh [OUT_DIR]
#   RUNNER=local GPUS=0,1 bash scripts/scorecard/minicpmv-35b.sh [OUT_DIR]   # inside a container of the image,
#       with hk-dsl installed: the stock arm then runs with HK_DISABLE=1 (hk loads nothing, the engine is stock)
#
# What it does: on two GPUs (tensor parallel 2), it starts vLLM in two arms, alternating, START_PER_ARM starts
# each (default 2):
#   stock  -- the best stock configuration: --attention-backend ROCM_SEGMENTED_ATTN, and TunableOp reading the
#             tuned GEMM file TUNABLEOP_DIR/tunableop_results%d.csv if it exists (tune it first with TUNE=1);
#   hk     -- the same, plus hk-dsl from PyPI (HK_VERSION, default: the scorecard's) with its default routes.
# Each start measures: first token of one uncached 32k-token and one 128k-token synthetic prompt; time per
# output token at 1 and 8 concurrent requests (vllm bench serve, random prompts); GSM8K-200 greedy (first start
# of each arm). It prints the scorecard's table (medians over starts) and writes every raw number to OUT_DIR.
#
# Needs: two gfx1100 GPUs with 48 GB each (Radeon PRO W7900 / W7900D), docker, the model's weights
# (MODEL_PATH, default: downloaded by vLLM from Hugging Face), about 3.5 hours (two cold starts of ~6 min per arm,
# the prompts, the sweeps and GSM8K), 64 GB of host memory. Core dumps are off.
#
# The scorecard's numbers came from the production serving image (not public); with another vLLM build the
# numbers move with it -- quote the image with any number this prints.
set -euo pipefail
RUNNER=${RUNNER:-docker}
[ "$RUNNER" = local ] || IMAGE=${IMAGE:?set IMAGE to a vLLM ROCm image}
IMAGE=${IMAGE:-local}
SEED=${SEED:-20261113}
GPUS=${GPUS:-0,1}
MODEL=${MODEL:-openbmb/MiniCPM-V-4.7-35B-A3B}
MODEL_PATH=${MODEL_PATH:-}
HK_VERSION=${HK_VERSION:-0.9.20}
STARTS=${START_PER_ARM:-2}
TUNABLEOP_DIR=${TUNABLEOP_DIR:-$PWD/tunableop}
OUT=${1:-$PWD/scorecard-minicpmv-35b-$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-18031}
mkdir -p "$OUT" "$TUNABLEOP_DIR"
ulimit -c 0
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }

ARGS=(--served-model-name MiniCPM-V-4.7-35B-A3B --dtype bfloat16 --tensor-parallel-size 2
      --distributed-executor-backend mp --moe-backend triton --disable-custom-all-reduce
      --max-model-len 262144 --max-num-seqs 32 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.925
      --enable-prefix-caching --trust-remote-code --attention-backend ROCM_SEGMENTED_ATTN --port "$PORT")
MOUNTS=(-v "$OUT:/out" -v "$TUNABLEOP_DIR:/tunableop" -v "$HOME/.cache/huggingface:/hf")
[ -n "$MODEL_PATH" ] && MOUNTS+=(-v "$MODEL_PATH:/model:ro") && MODEL=/model

start() {   # start <arm> <n>: one server, in the background; waits until it serves
    local arm=$1 n=$2 name="sc35b-$1-$2"
    local env=(-e HF_HOME=/hf -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 -e VLLM_ROCM_USE_AITER=0 -e NCCL_P2P_DISABLE=1
               -e HIP_VISIBLE_DEVICES="$GPUS" -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING="${TUNE:-0}"
               -e PYTORCH_TUNABLEOP_FILENAME=/tunableop/tunableop_results%d.csv)
    local pre="true"
    [ "$arm" = hk ] && pre="pip install -q hk-dsl==$HK_VERSION" || env+=(-e HK_DISABLE=1)
    if [ "$RUNNER" = local ]; then
        local e=(); for x in "${env[@]}"; do [ "$x" != -e ] && e+=("${x/\/tunableop\//$TUNABLEOP_DIR/}"); done
        [ "$arm" = stock ] || e=("${e[@]/HK_DISABLE=1/}")
        ( for kv in "${e[@]}"; do [ -n "$kv" ] && export "${kv/HF_HOME=\/hf/HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}}"; done
          exec setsid vllm serve "$MODEL" "${ARGS[@]}" ) > "$OUT/server-$arm-$n.log" 2>&1 &
        echo $! > "$OUT/server-$arm-$n.pid"
    else
    docker run -d --rm --name "$name" --network host --ipc host --ulimit core=0 --device /dev/kfd --device /dev/dri \
        --group-add video --security-opt seccomp=unconfined "${env[@]}" "${MOUNTS[@]}" --entrypoint bash "$IMAGE" \
        -c "$pre && exec vllm serve $MODEL ${ARGS[*]}" > /dev/null
    fi
    for _ in $(seq 1 180); do
        sleep 10
        curl -sf "localhost:$PORT/v1/models" > /dev/null && { log "$arm start $n: serving"; return 0; }
        if [ "$RUNNER" = local ]; then kill -0 "$(cat "$OUT/server-$arm-$n.pid")" 2>/dev/null \
            || { log "$arm start $n: the server exited"; return 1; }
        else docker inspect "$name" > /dev/null 2>&1 || { log "$arm start $n: the server exited"; return 1; }; fi
    done
    log "$arm start $n: not serving after 30 min"; return 1
}
stop() {
    if [ "$RUNNER" = local ]; then
        local p; p=$(cat "$OUT/server-$1-$2.pid" 2>/dev/null) || return 0
        kill -INT -- -"$p" 2>/dev/null; for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
        kill -KILL -- -"$p" 2>/dev/null || true
    else docker stop -t 60 "sc35b-$1-$2" > /dev/null 2>&1 || true; fi
    sleep 10
}
in_image() { if [ "$RUNNER" = local ]; then bash -c "$1"; else docker run --rm --network host -v "$OUT:/out" --entrypoint bash "$IMAGE" -c "$1"; fi; }

ttft() {    # ttft <tokens> <seed>: seconds to the first token of one uncached synthetic prompt of about <tokens> tokens
    python3 - "$1" "$PORT" "$2" <<'PY'
import json, sys, time, urllib.request, random
n, port, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
random.seed(seed)                                 # a fresh server, and a seed per prompt: no shared prefix
words = " ".join(f"w{random.randrange(10**6)}" for _ in range(n // 2))
body = json.dumps({"model": "MiniCPM-V-4.7-35B-A3B", "prompt": words, "max_tokens": 1, "stream": True}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        if line.startswith(b"data:") and b"choices" in line:
            print(f"{time.time() - t0:.3f}"); break
PY
}

tpot() {    # tpot <concurrency> <tag>: mean time per output token (ms) of vllm bench serve on seeded random
            # prompts (4096 in, 256 out, 16 requests per stream); its full output is kept in OUT/bench-<tag>.txt
    in_image "vllm bench serve --backend vllm --model $MODEL --served-model-name MiniCPM-V-4.7-35B-A3B --port $PORT \
         --dataset-name random --random-input-len 4096 --random-output-len 256 --num-prompts $(( $1 * 16 )) \
         --max-concurrency $1 --seed $SEED --percentile-metrics tpot,ttft 2>&1" > "$OUT/bench-$2.txt"
    awk '/Mean TPOT/ {t=$NF} /Successful requests/ {o=$NF} /Failed requests/ {f=$NF} END {print t" ok="o" failed="(f==""?0:f)}' "$OUT/bench-$2.txt"
}

gsm8k() {   # GSM8K-200, greedy, through the server; prints the accuracy in %
    in_image "python3 - $PORT" <<'PY'
import json, re, sys, urllib.request
from datasets import load_dataset
port = sys.argv[1]
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(200))
ok = 0
for ex in ds:
    body = json.dumps({"model": "MiniCPM-V-4.7-35B-A3B", "temperature": 0, "max_tokens": 512,
                       "messages": [{"role": "user", "content": ex["question"] + "\nGive the final answer after ####."}]}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    want = ex["answer"].split("####")[-1].strip().replace(",", "")
    got = re.findall(r"-?\d[\d,]*\.?\d*", out.split("####")[-1])
    ok += bool(got) and got[-1].replace(",", "").rstrip(".") == want
print(f"{100 * ok / len(ds):.1f}")
PY
}

log "image $IMAGE, GPUs $GPUS, hk-dsl $HK_VERSION, $STARTS starts per arm -> $OUT"
for n in $(seq 1 "$STARTS"); do
    for arm in stock hk; do                          # alternating: stock, hk, stock, hk
        start "$arm" "$n" || { stop "$arm" "$n"; continue; }
        r="$OUT/$arm-$n"
        echo "ttft32k=$(ttft 32768 $((SEED + 10 * n + 1)))" > "$r"
        echo "ttft128k=$(ttft 131072 $((SEED + 10 * n + 2)))" >> "$r"
        read -r t c <<< "$(tpot 1 "$arm-$n-c1")"; echo "tpot_c1=$t" >> "$r"; echo "count_c1=$c" >> "$r"
        read -r t c <<< "$(tpot 8 "$arm-$n-c8")"; echo "tpot_c8=$t" >> "$r"; echo "count_c8=$c" >> "$r"
        [ "$n" = 1 ] && echo "gsm8k=$(gsm8k)" >> "$r"
        log "$arm start $n: $(tr '\n' ' ' < "$r")"
        stop "$arm" "$n"
    done
done

python3 - "$OUT" <<'PY'
import glob, os, statistics, sys
out = sys.argv[1]
def vals(arm, key):
    v = []
    for f in sorted(glob.glob(os.path.join(out, f"{arm}-*"))):
        for line in open(f):
            k, _, x = line.strip().partition("=")
            if k == key and x:
                v.append(float(x))
    return v
rows = [("first token, one 32k-token prompt (s)", "ttft32k", 1), ("first token, one 128k-token prompt (s)", "ttft128k", 1),
        ("time per output token, 1 request (ms)", "tpot_c1", 1), ("time per output token, 8 requests (ms)", "tpot_c8", 1)]
print("| metric | best stock | hk | hk vs stock |\n|---|---|---|---|")
for name, key, _ in rows:
    s, h = vals("stock", key), vals("hk", key)
    if s and h:
        ms, mh = statistics.median(s), statistics.median(h)
        print(f"| {name} | {ms:.3g} | {mh:.3g} | {ms / mh:.2f}x |")
s, h = vals("stock", "gsm8k"), vals("hk", "gsm8k")
if s and h:
    print(f"| GSM8K-200 (greedy) | {s[0]:.1f}% | {h[0]:.1f}% | |")
PY
