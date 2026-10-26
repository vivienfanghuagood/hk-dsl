#!/bin/bash
# GPU CI, gfx1201, as a self-hosted Actions job (.github/workflows/gpu.yml): the checked-out commit
# on one idle Radeon AI PRO R9700 of the gfx1201 dev box (amd-G7466-M6), in rocm/pytorch, through
# scripts/ci/gfx1201/hkr9 -- a container that sees that one GPU (its render node), the checkout, the
# run's directory, CI's own directory and the MLIR install, read-only, with core dumps off; it runs
# container_run.sh, as gfx950's does.
#
# Host state is under the runner's own directory, $HK_CI_STATE (the runner's .env sets it to
# /home/amd/hfang-hk/gha/runner-gfx1201/state):
#   runs/gfx1201/RUN_ID-ATTEMPT/  the run: build.log, gpu/, gpusnap.log, result.json, report.txt
#   ci/                           /ci in the container: hk-build-cp312, pylib, hk-cache, home
# $HK_CI_MLIR (default /home/amd/hfang-hk/mlir/install-cp310) is MLIR 23.1.3 as the ci-mlir-23.1.3
# release has it. $HK_CI_GPUS: the cards it may take, in order of preference (default "7 6 5 4 3 2 1").
#
# The box is shared and its other users do not take our locks: a card is taken only when its lock
# (gpulock.sh) is free and no process of anyone's holds it (KFD sysfs), nor an hk-gha container
# has its render node. The lock is held until the container is gone.
set -uo pipefail
# The runner's listener is started from a non-interactive shell and may ignore SIGINT, which every
# job then inherits and bash cannot trap; the runner's cancel is a SIGINT. Start again, the same
# process, with the signals at their defaults (as run_gfx950.sh does).
[ -n "${HK_CI_SIGNALS:-}" ] || HK_CI_SIGNALS=default exec env --default-signal=INT,QUIT,HUP,TERM "${BASH_SOURCE[0]}" "$@"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${HK_CI_STATE:-/home/amd/hfang-hk/gha/runner-gfx1201/state}
MLIR=${HK_CI_MLIR:-/home/amd/hfang-hk/mlir/install-cp310}
CIDIR=${HK_CI_DIR:-$STATE/ci}
TIER=${HK_CI_TIER:-gfx1201}
GPUS=${HK_CI_GPUS:-7 6 5 4 3 2 1}
SHA=$GITHUB_SHA
KEY=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
RUN=$STATE/runs/$TIER/$KEY
OUT=$GITHUB_WORKSPACE/gha-report
[ -d "$MLIR/lib/cmake/mlir" ] || { echo "::error::no MLIR install at $MLIR (HK_CI_MLIR)"; exit 1; }
mkdir -p "$RUN" "$OUT" "$CIDIR"
NAME=hk-gha-$TIER-$KEY
RUNNER=${RUNNER_NAME:-manual}
SNAP=$HERE/gfx1201/gpusnap
. "$HERE/gfx1201/gpulock.sh"

# Cancellation (docs/testing.md, "Cancelled jobs and orphan containers"): the trap kills the
# container by name and waits until it is gone; for a SIGKILL, the reaper (cancel.sh --on-eof)
# does. Either holds the card's lock until then.
DPID= REAP=
stop() {
    trap - INT TERM HUP EXIT
    if [ -n "$DPID" ]; then     # cancelled while the container runs
        echo "cancelled: killing container $NAME"
        disown "$DPID" 2>/dev/null
        kill -KILL "$DPID" 2>/dev/null
    fi
    "$HERE/cancel.sh" container "$NAME" && [ -n "$REAP" ] && echo done >&"$REAP"
}
trap 'stop; exit 130' INT
trap 'stop; exit 143' TERM
trap 'stop; exit 129' HUP
trap 'stop' EXIT

# Orphans: this runner's hk-gha containers left by an earlier job. The runner is the box's only
# one and runs one job at a time, so any it finds now is an orphan; nothing needs asking.
for c in $(docker ps -aq --filter name=hk-gha- --filter "label=hk-gha.runner=$RUNNER"); do
    n=$(docker inspect --format '{{.Name}}' "$c" 2>/dev/null); n=${n#/}
    echo "orphan $n (this runner's, from an earlier job): killing it"
    "$HERE/cancel.sh" container "$n"
done

# The first of $GPUS whose lock is free at once, that no process holds, and whose render node no
# hk-gha container has, with its lock taken (fd LOCKFD, held by this script and the reaper until
# the container is gone). Waits up to an hour for one.
mapped() {
    docker ps -q --filter name=hk-gha- | xargs -r docker inspect --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null
}
LOCKFD= GPU=
pick() {
    local g m lk
    m=" $(mapped | tr '\n' ' ') "
    for g in $GPUS; do
        lk=$(gpulock_path "$g") || continue
        exec {LOCKFD}>>"$lk" || continue
        if flock -n "$LOCKFD"; then
            if [[ $m == *" /dev/dri/$(gpulock_render "$g") "* ]]; then
                echo "GPU $g: an hk-gha container has its render node"
            elif [ -z "$(gpu_foreign "$g")" ]; then
                printf 'host=%s pid=%s since=%s gpu_index=%s cmd=%s\n' "$(hostname)" "$$" \
                    "$(date -u +%FT%TZ)" "$g" "gpu.yml $TIER run $KEY" > "${lk%.lock}.owner"
                GPU=$g; return 0
            fi
        fi
        exec {LOCKFD}>&-
    done
    return 1
}
for i in $(seq 60); do
    pick && break
    echo "no idle, unlocked R9700 among $GPUS; waiting ($i)"
    sleep 60 >/dev/null 2>&1 & wait $!     # a trap runs at once, not after the sleep
done
[ -n "$GPU" ] || { echo "::error::no idle R9700 among $GPUS for an hour"; exit 1; }
echo "GPU $GPU ($(gpulock_render "$GPU")), holding $(gpulock_path "$GPU")"
echo "$GPU" > "$RUN/gpu"

# The reaper holds the lock too; its stdin's write end is REAP, which only this script holds.
exec {REAP}> >(exec env RUNNER_TRACKING_ID=0 setsid "$HERE/cancel.sh" --on-eof container "$NAME" \
               >>"$RUN/reaper.log" 2>&1)
touch "$RUN/started"    # minutes count from here, not from the wait
"$SNAP" "$GPU" start >> "$RUN/gpusnap.log" 2>&1
HKR9_LOCK_HELD=1 HKR9_NAME=$NAME \
HKR9_DOCKER_ARGS="--label hk-gha.runner=$RUNNER --label hk-gha.run=$GITHUB_RUN_ID --label hk-gha.attempt=$GITHUB_RUN_ATTEMPT --label hk-gha.tier=$TIER --label hk-gha.started=$(date +%s)" \
HKR9_MOUNTS="-v $CIDIR:/ci -v $GITHUB_WORKSPACE:/ci/src -v $RUN:/ci/run -v $MLIR:/ci/mlir:ro -w /ci/src" \
    "$HERE/gfx1201/hkr9" "$GPU" -- \
    "HK_CI_ARCH=gfx1201 HK_CI_TIER=$TIER HK_CI_DEVICE=\"Radeon AI PRO R9700 (gfx1201), GPU $GPU\" \
     /ci/src/scripts/ci/container_run.sh $SHA $GPU" \
    {REAP}>&- {LOCKFD}>&- < /dev/null &
DPID=$!

# What else ran on the card while the suite did: every five minutes (gpusnap.log, in the
# artifact). The idle check only holds at the start.
while kill -0 $DPID 2>/dev/null; do
    for _ in $(seq 60); do kill -0 $DPID 2>/dev/null || break; sleep 5 >/dev/null 2>&1 & wait $!; done
    kill -0 $DPID 2>/dev/null && "$SNAP" "$GPU" during >> "$RUN/gpusnap.log" 2>&1
done
wait $DPID
DPID=
"$SNAP" "$GPU" end >> "$RUN/gpusnap.log" 2>&1
# --rm: the container is gone once its client has returned; the EXIT trap checks, and tells the
# reaper so.

cp -r "$RUN"/. "$OUT"/ 2>/dev/null
[ -f "$RUN/result.json" ] || { echo "::error::the container exited without a result"; exit 1; }
cat "$RUN/gpusnap.log" 2>/dev/null
python3 "$HERE/report.py" "$TIER" "$RUN/result.json"
