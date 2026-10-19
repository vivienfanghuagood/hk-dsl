#!/bin/bash
# Run a benchmark with its GPU's use and VRAM logged every 30 s beside it, so
# a run another job shared can be told from one it did not: on a shared node
# a second process on the same GPU slows every arm alike, which no rule on
# the rounds can see (docs/history/37-cdna-rows.md).
#
#   tools/hk-bench/watched.sh GPU LOG -- python3 tools/hk-bench/scoreboard.py ...
#
# GPU is the rocm-smi index; LOG gets the samples, then the largest VRAM%
# seen, which should be this process's own.
set -uo pipefail
gpu=$1 log=$2
shift 2
[ "${1:-}" = "--" ] && shift
( while true; do
    echo "$(date +%T) $(rocm-smi -d "$gpu" --showuse --showmemuse 2>/dev/null |
        grep -oE 'GPU use \(%\): [0-9]+|VRAM%\): [0-9]+' | tr '\n' ' ')"
    sleep 30
  done ) > "$log" 2>&1 &
watcher=$!
"$@"
status=$?
kill "$watcher"
echo "max VRAM% seen: $(grep -oE 'VRAM%\): [0-9]+' "$log" | grep -oE '[0-9]+$' | sort -n | tail -1)" >> "$log"
exit $status
