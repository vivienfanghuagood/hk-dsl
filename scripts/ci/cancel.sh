#!/bin/bash
# Stop a GPU CI job's work when the job is cancelled or times out. Idempotent; it touches only the job's own container or pod run.
#   cancel.sh container NAME           docker kill container NAME and wait until it is gone
#   cancel.sh pod JUMP POD RUNDIR      kill the pod's run in RUNDIR (pod_run.sh's whole session,
#                                      its timeout'd pytest included) unless it has finished
#   cancel.sh --on-eof ARGS...         a reaper: read stdin until "done" or end of file; at end
#                                      of file, the job's script is gone without saying done --
#                                      cancelled, or killed -- so do ARGS
# The job's script starts the reaper on a pipe it holds the only write end of, outside the
# runner's process tracking (RUNNER_TRACKING_ID=0): the runner SIGKILLs a cancelled job's leftover
# processes ("Cleaning up orphan processes"), which no trap sees, and the pipe's end of file is
# what is left. run_gfx950.sh also kills its container from its trap (about a second, within the
# runner's 7.5 s); run_gfx1100.sh leaves the pod to the reaper (a round trip takes 3-4 s).
set -u
if [ "${1:-}" = --on-eof ]; then
    shift
    while IFS= read -r line; do [ "$line" = done ] && exit 0; done
    echo "$(date -u +%FT%TZ) the job's script is gone without finishing: cancel.sh $1"   # not $*: the log is public
    EOF_=1
fi
case ${1:-} in
container)
    NAME=$2
    [ -n "${EOF_:-}" ] && sleep 2     # the docker client, killed with the script, can no longer start it
    for i in $(seq 60); do
        st=$(docker container inspect --format '{{.State.Status}}' "$NAME" 2>/dev/null) || exit 0
        case $st in
            running|restarting|paused) docker kill "$NAME" >/dev/null 2>&1 ;;
            *) docker rm -f "$NAME" >/dev/null 2>&1 ;;   # created, exited, or --rm removing it
        esac
        sleep 0.5
    done
    echo "cancel.sh: container $NAME still there after 30 s" >&2
    exit 1
    ;;
pod)
    JUMP=$2 POD=$3 R=$4
    [[ $R =~ ^[A-Za-z0-9_./-]+$ ]] || exit 2
    echo "cd $R 2>/dev/null && [ -f pid ] && [ ! -f result.json ] && pkill -KILL -s \$(cat pid) && echo killed the session of pod_run.sh \$(cat pid); true" |
        timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "$JUMP" "kubectl exec -i $POD -- bash -s"
    ;;
*)
    echo "usage: cancel.sh [--on-eof] container NAME | pod JUMP POD RUNDIR" >&2
    exit 2
    ;;
esac
