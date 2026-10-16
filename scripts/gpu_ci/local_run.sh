#!/bin/bash
# GPU CI, gfx950: one commit on one MI355X of this workstation, in rocm/pytorch.
#
#   local_run.sh SHA GPU      (started, detached, by orchestrate.py on a GPU it found idle)
#
# The container sees one GPU (its render node), the CI checkout, CI's own
# directories and the MLIR install, read-only; nothing of $HOME.
set -uo pipefail
SHA=${1:-}
GPU=${2:-}
[[ $SHA =~ ^[0-9a-f]{40}$ && $GPU =~ ^[0-7]$ ]] || { echo "usage: local_run.sh SHA GPU" >&2; exit 2; }

CI=${HK_CI_HOME:-$HOME/hk-ci}
RUN=$CI/runs/gfx950/$SHA
SRC=$CI/src-gfx950
IMAGE=rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0
declare -A RENDER=([0]=renderD136 [1]=renderD152 [2]=renderD144 [3]=renderD128
                   [4]=renderD168 [5]=renderD184 [6]=renderD176 [7]=renderD160)
mkdir -p "$RUN" "$CI/x950"
exec >"$RUN/driver.log" 2>&1
echo $$ >"$RUN/pid"
touch "$RUN/started"
export GIT_TERMINAL_PROMPT=0

fail() {
    echo "error: $*"
    python3 "$CI/summarize.py" --error "$*" --sha "$SHA" --tier gfx950 \
        --device "MI355X (gfx950), GPU $GPU" --out "$RUN"
    exit 1
}

if [ ! -d "$SRC/.git" ]; then
    git clone -q https://github.com/vivienfanghuagood/hk-dsl.git "$SRC" || fail "git clone"
fi
cd "$SRC" || fail "no checkout"
git fetch -q origin "$SHA" || fail "git fetch $SHA"
git checkout -q --force --detach "$SHA" || fail "git checkout $SHA"
git submodule sync -q && git submodule update -q --init --recursive || fail "the HipKittens submodule"
cp "$CI/container_run.sh" "$CI/summarize.py" "$CI/x950/"

NAME=hk-ci-gfx950-${SHA:0:12}
docker rm -f "$NAME" >/dev/null 2>&1
docker run --rm --name "$NAME" --user "$(id -u):$(id -g)" \
    --device /dev/kfd --device "/dev/dri/${RENDER[$GPU]}" \
    --group-add 44 --group-add 109 --ipc=host \
    -e HOME=/ci/home -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
    -v "$CI/x950:/ci" -v "$SRC:/ci/src" -v "$RUN:/ci/run" \
    -v "$CI/mlir/install-cp310:/ci/mlir:ro" \
    -w /ci/src "$IMAGE" bash /ci/container_run.sh "$SHA" "$GPU" \
    || [ -f "$RUN/result.json" ] || fail "the container exited without a result"
