#!/bin/bash
# GPU CI, gfx950, as a self-hosted Actions job (.github/workflows/gpu.yml): the
# checked-out commit on the lowest-numbered idle MI355X of the workstation, in
# rocm/pytorch. The container sees that one GPU (its render node), the
# checkout, the run's directory, CI's build directory, pip packages and
# compile cache, and the MLIR install, read-only; nothing else of the user's home.
#
# Host state is kept out of the checkout, under $HK_CI_STATE (default ~/gha), one per runner
# instance when several run gfx950 jobs at once:
#   runs/TIER/RUN_ID-ATTEMPT/  the run: build.log, gpu/, result.json, report.txt
#   x950/ci/                   /ci in the container: hk-build-cp312, pylib, hk-cache
#   x950v/ci/                  the same for the vLLM image (HK_CI_DIRNAME=x950v, gpu.yml's gfx950-vllm)
# GPUs: HK_CI_GPUS (default all eight) are the cards this instance may use. It takes the first
# that is idle and whose lock, $HK_CI_LOCKS/gfx950-gpu<N>.lock, it can take at once, and holds
# that lock until the container is gone; whoever else measures on these cards takes the same lock.
# The container is labelled with the runner and the run (hk-gha.runner, hk-gha.run,
# hk-gha.attempt), so that a later job of the same runner can tell an orphan from a live run.
# $HK_CI_MLIR (default ~/hk-ci/mlir/install-cp310) is MLIR 23.1.3 as
# scripts/build_mlir.sh installs it for CPython 3.10, the ci-mlir-23.1.3
# release's asset unpacked; container_run.sh builds the dialect against it.
set -uo pipefail
# A runner's listener started by `cmd &` from a non-interactive shell ignores SIGINT and SIGQUIT
# (SigIgn in /proc/PID/status; the workstation's did, 2026-10-26), every job inherits that, and
# bash silently cannot trap a signal ignored on entry -- the runner's cancel is a SIGINT. So start
# again, the same process, with the signals at their defaults.
[ -n "${HK_CI_SIGNALS:-}" ] || HK_CI_SIGNALS=default exec env --default-signal=INT,QUIT,HUP,TERM "${BASH_SOURCE[0]}" "$@"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Defaults are under the user's home from the passwd entry, not $HOME: a runner runs its jobs with
# HOME=~/gha/runner-<name>/home (supervise.sh).
USERHOME=$(getent passwd "$(id -un)" | cut -d: -f6)
STATE=${HK_CI_STATE:-$USERHOME/gha}
MLIR=${HK_CI_MLIR:-$USERHOME/hk-ci/mlir/install-cp310}
IMAGE=${HK_CI_IMAGE:-rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0}
# Another image (the vLLM deployment one, gpu.yml's gfx950-vllm) keeps its own build of the
# dialect and its own kernel cache: its CPython and glibc are not rocm/pytorch's. Two jobs never
# share one: an instance's jobs run one at a time, and each instance has its own HK_CI_STATE.
CIDIR=${HK_CI_DIR:-$STATE/${HK_CI_DIRNAME:-x950}/ci}
LOCKS=${HK_CI_LOCKS:-$USERHOME/.cache/hk-gpu-locks}
TIER=${HK_CI_TIER:-gfx950}
SHA=$GITHUB_SHA
KEY=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
RUN=$STATE/runs/$TIER/$KEY
OUT=$GITHUB_WORKSPACE/gha-report
declare -A RENDER=([0]=renderD136 [1]=renderD152 [2]=renderD144 [3]=renderD128
                   [4]=renderD168 [5]=renderD184 [6]=renderD176 [7]=renderD160)
# The job log is public: it names no host path (the runner's .env has them).
[ -d "$MLIR/lib/cmake/mlir" ] || { echo "::error::no MLIR install at HK_CI_MLIR"; exit 1; }
mkdir -p "$RUN" "$OUT" "$CIDIR" "$LOCKS"
NAME=hk-gha-$TIER-$KEY
RUNNER=${RUNNER_NAME:-manual}

# Cancellation (scripts/ci/cancel.sh). The runner sends SIGINT,
# then SIGTERM 7.5 s later, to the step's shell alone -- gpu.yml execs this script, so that is this
# script -- SIGKILLs it 2.5 s after that, and then every process the job left. The container belongs to dockerd and
# survives all of that unless killed by name: the trap does it, and, for a SIGKILL, the reaper
# (cancel.sh --on-eof) does. Either holds the GPU's lock until the container is gone.
DPID= REAP=
stop() {
    trap - INT TERM HUP EXIT
    if [ -n "$DPID" ]; then     # cancelled while the container runs
        echo "cancelled: killing container $NAME"
        disown "$DPID" 2>/dev/null     # no "Killed docker run ..." in the log
        kill -KILL "$DPID" 2>/dev/null     # its client can no longer start it
    fi
    "$HERE/cancel.sh" container "$NAME" && [ -n "$REAP" ] && echo done >&"$REAP"
}
trap 'stop; exit 130' INT
trap 'stop; exit 143' TERM
trap 'stop; exit 129' HUP
trap 'stop' EXIT

# Orphans: hk-gha containers of this runner left by an earlier job (any is one: a runner runs one
# job at a time) that the trap and the reaper missed. Killed only once GitHub says their run
# attempt has completed; never while it is queued or in progress. When GitHub cannot be asked,
# once older than the longest job's timeout (120 min).
sweep() {
    local c name run att started st age
    for c in $(docker ps -q --filter name=hk-gha- --filter "label=hk-gha.runner=$RUNNER"); do
        read -r name run att started < <(docker inspect --format '{{.Name}} {{index .Config.Labels "hk-gha.run"}} {{index .Config.Labels "hk-gha.attempt"}} {{index .Config.Labels "hk-gha.started"}}' "$c" 2>/dev/null)
        name=${name#/}
        [[ $name == hk-gha-* && $run =~ ^[0-9]+$ && $att =~ ^[0-9]+$ ]] || continue
        st=$(GH_PROMPT_DISABLED=1 timeout 30 gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run/attempts/$att" --jq .status 2>/dev/null)
        age=$(( $(date +%s) - ${started:-0} ))
        if [ "$st" = completed ]; then
            echo "orphan $name: run $run attempt $att has completed; killing it"
        elif [ -z "$st" ] && [[ $started =~ ^[0-9]+$ ]] && [ "$age" -gt 7200 ]; then
            echo "orphan $name: GitHub unreachable, started $((age / 60)) min ago; killing it"
        else
            echo "keeping $name: run $run attempt $att is ${st:-unknown (GitHub unreachable)}"
            continue
        fi
        "$HERE/cancel.sh" container "$name"
    done
}
sweep
unset GH_TOKEN      # only the sweep needs it

# The render nodes mapped into running hk-gha containers, whichever runner's or run's.
mapped() {
    docker ps -q --filter name=hk-gha- | xargs -r docker inspect --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null
}

# The first of HK_CI_GPUS whose lock is free and that is idle (under 10% use and 3% VRAM, and no
# hk-gha container has its render node), with its lock taken (fd LOCKFD, held by this script and
# the reaper until the container is gone). Never blocks on a lock: a held card is someone's, busy
# or not.
LOCKFD=
pick() {
    local g out use mem m
    m=" $(mapped | tr '\n' ' ') "
    for g in ${HK_CI_GPUS:-0 1 2 3 4 5 6 7}; do
        exec {LOCKFD}>>"$LOCKS/gfx950-gpu$g.lock" || continue
        if flock -n "$LOCKFD"; then
            out=$(rocm-smi -d $g --showuse --showmemuse 2>/dev/null)
            use=$(grep -o "GPU use (%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
            mem=$(grep -o "VRAM%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
            if [[ $m == *" /dev/dri/${RENDER[$g]} "* ]]; then
                echo "GPU $g: an hk-gha container has /dev/dri/${RENDER[$g]}"
            elif [ "${use:-100}" -le 10 ] && [ "${mem:-100}" -le 3 ]; then
                GPU=$g; return 0
            fi
        fi
        exec {LOCKFD}>&-
    done
    return 1
}
GPU=
# HK_CI_PICK_WAIT_MIN (default 60): minutes to wait for a GPU, e.g. while other users' processes,
# which take no lock, hold the runner's cards. The job's timeout-minutes (gpu.yml) covers the wait too.
for i in $(seq "${HK_CI_PICK_WAIT_MIN:-60}"); do
    pick && break
    echo "no idle, unlocked MI355X among ${HK_CI_GPUS:-0-7}; waiting ($i)"
    sleep 60 >/dev/null 2>&1 & wait $!     # a trap runs at once, not after the sleep
done
[ -n "$GPU" ] || { echo "::error::no idle MI355X for ${HK_CI_PICK_WAIT_MIN:-60} min"; exit 1; }
echo "holding gfx950-gpu$GPU.lock (HK_CI_LOCKS)"
echo "GPU $GPU (${RENDER[$GPU]})"

# The reaper holds the lock too; its stdin's write end is REAP, which only this script holds.
exec {REAP}> >(exec env -u GH_TOKEN RUNNER_TRACKING_ID=0 setsid "$HERE/cancel.sh" --on-eof container "$NAME" \
               >>"$RUN/reaper.log" 2>&1)
touch "$RUN/started"    # minutes count from here, not from the wait
docker run --rm --ulimit core=0 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 --name "$NAME" --user "$(id -u):$(id -g)" \
    --label hk-gha.runner="$RUNNER" --label hk-gha.run="$GITHUB_RUN_ID" --label hk-gha.attempt="$GITHUB_RUN_ATTEMPT" \
    --label hk-gha.tier="$TIER" --label hk-gha.started="$(date +%s)" \
    --device /dev/kfd --device "/dev/dri/${RENDER[$GPU]}" \
    --group-add 44 --group-add 109 --ipc=host \
    -e HOME=/ci/home -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e HK_CI_TIER="$TIER" -e HK_CI_TESTS="${HK_CI_TESTS:-}" \
    -v "$CIDIR:/ci" -v "$GITHUB_WORKSPACE:/ci/src" -v "$RUN:/ci/run" \
    -v "$MLIR:/ci/mlir:ro" \
    --entrypoint bash -w /ci/src "$IMAGE" /ci/src/scripts/ci/container_run.sh "$SHA" "$GPU" \
    {REAP}>&- {LOCKFD}>&- &
DPID=$!
wait $DPID
DPID=
# --rm: the container is gone once its client has returned; the EXIT trap checks, and tells the
# reaper so.

cp -r "$RUN"/. "$OUT"/ 2>/dev/null
[ -f "$RUN/result.json" ] || { echo "::error::the container exited without a result"; exit 1; }
python3 "$HERE/report.py" "$TIER" "$RUN/result.json"
