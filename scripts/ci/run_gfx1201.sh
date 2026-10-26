#!/bin/bash
# GPU CI, gfx1201, as a self-hosted Actions job (.github/workflows/gpu.yml): the checked-out commit
# on one idle Radeon AI PRO R9700 of the gfx1201 dev box (amd-G7466-M6), in rocm/pytorch, through
# scripts/ci/gfx1201/hkr9 -- our lock of the card, a check that no other process holds it (the box
# is shared, and its other users do not take our locks), core dumps off. The container sees that one
# GPU (its render node), the checkout, the run's directory, CI's own directory and the MLIR install,
# read-only; it runs container_run.sh, as gfx950's does.
#
# Host state is under the runner's own directory, $HK_CI_STATE (the runner's .env sets it to
# /home/amd/hfang-hk/gha/runner-gfx1201/state):
#   runs/gfx1201/RUN_ID-ATTEMPT/  the run: build.log, gpu/, gpusnap.log, result.json, report.txt
#   ci/                           /ci in the container: hk-build-cp312, pylib, hk-cache, home
# $HK_CI_MLIR (default /home/amd/hfang-hk/mlir/install-cp310) is MLIR 23.1.3 as the ci-mlir-23.1.3
# release has it. $HK_CI_GPUS: the cards it may take, in order of preference (default "3 2 1").
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${HK_CI_STATE:-/home/amd/hfang-hk/gha/runner-gfx1201/state}
MLIR=${HK_CI_MLIR:-/home/amd/hfang-hk/mlir/install-cp310}
CIDIR=${HK_CI_DIR:-$STATE/ci}
TIER=${HK_CI_TIER:-gfx1201}
GPUS=${HK_CI_GPUS:-3 2 1}
SHA=$GITHUB_SHA
KEY=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
RUN=$STATE/runs/$TIER/$KEY
OUT=$GITHUB_WORKSPACE/gha-report
[ -d "$MLIR/lib/cmake/mlir" ] || { echo "::error::no MLIR install at $MLIR (HK_CI_MLIR)"; exit 1; }
mkdir -p "$RUN" "$OUT" "$CIDIR"
NAME=hk-gha-$TIER-$KEY
SNAP=$HERE/gfx1201/gpusnap
trap 'docker rm -f "$NAME" >/dev/null 2>&1; exit 130' INT TERM

# The card is the first of $GPUS that is unlocked and idle; hkr9 waits up to an hour for one and
# writes the index it took to $RUN/gpu before the container starts.
HKR9_NAME=$NAME HKR9_GPU_FILE=$RUN/gpu HK_GPUS_PREF="$GPUS" \
HKR9_MOUNTS="-v $CIDIR:/ci -v $GITHUB_WORKSPACE:/ci/src -v $RUN:/ci/run -v $MLIR:/ci/mlir:ro -w /ci/src" \
    "$HERE/gfx1201/hkr9" auto -w 3600 -- \
    "HK_CI_ARCH=gfx1201 HK_CI_TIER=$TIER HK_CI_DEVICE=\"Radeon AI PRO R9700 (gfx1201), GPU \$(cat /ci/run/gpu)\" \
     /ci/src/scripts/ci/container_run.sh $SHA \$(cat /ci/run/gpu)" &
JOB=$!

# What else ran on the card while the suite did: a snapshot when the container starts and every five
# minutes after (gpusnap.log, in the artifact). The idle check only holds at the start.
while kill -0 $JOB 2>/dev/null && [ ! -s "$RUN/gpu" ]; do sleep 5; done
if [ -s "$RUN/gpu" ]; then
    touch "$RUN/started"    # minutes count from here, not from the wait
    GPU=$(cat "$RUN/gpu"); echo "GPU $GPU"
    "$SNAP" "$GPU" start >> "$RUN/gpusnap.log" 2>&1
    while kill -0 $JOB 2>/dev/null; do
        for _ in $(seq 60); do kill -0 $JOB 2>/dev/null || break; sleep 5; done
        kill -0 $JOB 2>/dev/null && "$SNAP" "$GPU" during >> "$RUN/gpusnap.log" 2>&1
    done
    "$SNAP" "$GPU" end >> "$RUN/gpusnap.log" 2>&1
fi
wait $JOB
rc=$?

cp -r "$RUN"/. "$OUT"/ 2>/dev/null
[ -f "$RUN/result.json" ] || { echo "::error::the container exited without a result (exit $rc)"; exit 1; }
cat "$RUN/gpusnap.log" 2>/dev/null
python3 "$HERE/report.py" "$TIER" "$RUN/result.json"
