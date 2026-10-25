#!/bin/bash
# GPU CI, gfx950, as a self-hosted Actions job (.github/workflows/gpu.yml): the
# checked-out commit on the lowest-numbered idle MI355X of the workstation, in
# rocm/pytorch. The container sees that one GPU (its render node), the
# checkout, the run's directory, CI's build directory, pip packages and
# compile cache, and the MLIR install, read-only; nothing else of $HOME.
#
# Host state is kept out of the checkout, under $HK_CI_STATE (default ~/gha):
#   runs/gfx950/RUN_ID-ATTEMPT/  the run: build.log, gpu/, result.json, report.txt
#   x950/ci/                     /ci in the container: hk-build-cp312, pylib, hk-cache
# $HK_CI_MLIR (default ~/hk-ci/mlir/install-cp310) is MLIR 23.1.3 as
# scripts/build_mlir.sh installs it for CPython 3.10, the ci-mlir-23.1.3
# release's asset unpacked; container_run.sh builds the dialect against it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${HK_CI_STATE:-$HOME/gha}
MLIR=${HK_CI_MLIR:-$HOME/hk-ci/mlir/install-cp310}
IMAGE=${HK_CI_IMAGE:-rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0}
# Another image (the vLLM deployment one, gpu.yml's gfx950-vllm) keeps its own build of the
# dialect and its own kernel cache: its CPython and glibc are not rocm/pytorch's.
CIDIR=${HK_CI_DIR:-$STATE/x950/ci}
TIER=${HK_CI_TIER:-gfx950}
SHA=$GITHUB_SHA
KEY=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
RUN=$STATE/runs/$TIER/$KEY
OUT=$GITHUB_WORKSPACE/gha-report
declare -A RENDER=([0]=renderD136 [1]=renderD152 [2]=renderD144 [3]=renderD128
                   [4]=renderD168 [5]=renderD184 [6]=renderD176 [7]=renderD160)
[ -d "$MLIR/lib/cmake/mlir" ] || { echo "::error::no MLIR install at $MLIR (HK_CI_MLIR)"; exit 1; }
mkdir -p "$RUN" "$OUT" "$CIDIR"
NAME=hk-gha-$TIER-$KEY
trap 'docker rm -f "$NAME" >/dev/null 2>&1; exit 130' INT TERM

# Under 10% use and 3% VRAM.
pick() {
    local g out use mem
    for g in ${HK_CI_GPUS:-0 1 2 3 4 5 6 7}; do     # HK_CI_GPUS: only these cards
        out=$(rocm-smi -d $g --showuse --showmemuse 2>/dev/null)
        use=$(grep -o "GPU use (%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
        mem=$(grep -o "VRAM%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
        [ "${use:-100}" -le 10 ] && [ "${mem:-100}" -le 3 ] && { echo $g; return 0; }
    done
    return 1
}
GPU=
for i in $(seq 60); do GPU=$(pick) && break; echo "no idle MI355X; waiting ($i)"; sleep 60; done
[ -n "$GPU" ] || { echo "::error::no idle MI355X for an hour"; exit 1; }
echo "GPU $GPU (${RENDER[$GPU]})"

touch "$RUN/started"    # minutes count from here, not from the wait
docker run --rm --ulimit core=0 -e HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 --name "$NAME" --user "$(id -u):$(id -g)" \
    --device /dev/kfd --device "/dev/dri/${RENDER[$GPU]}" \
    --group-add 44 --group-add 109 --ipc=host \
    -e HOME=/ci/home -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -e HK_CI_TIER="$TIER" -e HK_CI_TESTS="${HK_CI_TESTS:-}" \
    -v "$CIDIR:/ci" -v "$GITHUB_WORKSPACE:/ci/src" -v "$RUN:/ci/run" \
    -v "$MLIR:/ci/mlir:ro" \
    --entrypoint bash -w /ci/src "$IMAGE" /ci/src/scripts/ci/container_run.sh "$SHA" "$GPU" &
wait $!

cp -r "$RUN"/. "$OUT"/ 2>/dev/null
[ -f "$RUN/result.json" ] || { echo "::error::the container exited without a result"; exit 1; }
python3 "$HERE/report.py" "$TIER" "$RUN/result.json"
