#!/bin/bash
# GPU CI, gfx1100, the pod's side: one commit on one W7900D of the dev pod,
# installed and started, detached, by run_gfx1100.sh.
#   pod_run.sh SHA PROXY_IPV4 KEY
#
# Everything lives in $HK_CI_POD_DIR (default /workspace/gha): the clone,
# fetched through the gh-test proxy at the address the caller resolved, the
# dialect's build, the compile cache and runs/KEY. The GPU: an idle one of
# $HK_CI_POD_GPUS (default 5 6 7), each under $HK_CI_POD_DIR/gpuN.lock;
# failing that, GPU 0 under /workspace/ci/gpu0.lock, the lock everything that
# times GPU 0 holds. Nothing outside those and this process's environment is
# changed.
set -uo pipefail
SHA=$1; IP=$2; KEY=$3
[[ $SHA =~ ^[0-9a-f]{40}$ && $IP =~ ^[0-9.]+$ && $KEY =~ ^[0-9-]+$ ]] || exit 2
CI=${HK_CI_POD_DIR:-/workspace/gha}
GPUS=${HK_CI_POD_GPUS:-5 6 7}
RUN=$CI/runs/$KEY
SRC=$CI/hk-dsl
mkdir -p "$RUN" /workspace/ci
exec >"$RUN/driver.log" 2>&1
echo $$ >"$RUN/pid"
export PATH=/opt/venv/bin:$PATH GIT_TERMINAL_PROMPT=0 GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://gh-test.anruicloud.com/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.curloptResolve GIT_CONFIG_VALUE_1=gh-test.anruicloud.com:443:$IP
DEV="W7900D (gfx1100)"
fail() {
    echo "error: $*"
    python3 "$CI/summarize.py" --error "$*" --sha "$SHA" --tier gfx1100 --device "$DEV" --out "$RUN"
    exit 1
}

echo "$(date -u +%T) fetching $SHA"
if [ ! -d "$SRC/.git" ]; then
    git clone -q https://github.com/vivienfanghuagood/hk-dsl.git "$SRC" || fail "git clone through the proxy"
fi
cd "$SRC" || fail "no checkout"
git fetch -q origin "$SHA" || fail "git fetch $SHA through the proxy"
git checkout -q --force --detach "$SHA" || fail "git checkout $SHA"
git submodule sync -q && git submodule update -q --init --recursive || fail "the HipKittens submodule"

echo "$(date -u +%T) building the dialect"
HK_LLVM_ROOT=/workspace/llvm HK_MLIR_BUILD=$CI/hk-build-cp312 PYTHON=/opt/venv/bin/python3 JOBS=32 \
    scripts/build_hk_mlir.sh >"$RUN/build.log" 2>&1 || fail "the dialect did not build (build.log)"

idle() {
    local out use mem
    out=$(rocm-smi -d "$1" --showuse --showmemuse 2>/dev/null)
    use=$(grep -o "GPU use (%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
    mem=$(grep -o "VRAM%): [0-9]*" <<<"$out" | grep -o "[0-9]*$")
    [ "${use:-100}" -le 10 ] && [ "${mem:-100}" -le 3 ]
}
GPU=
for i in $(seq 120); do
    for g in $GPUS; do
        exec 9>"$CI/gpu$g.lock"
        if flock -n 9 && idle "$g"; then GPU=$g; break 2; fi
        exec 9>&-
    done
    exec 9>/workspace/ci/gpu0.lock
    if flock -n 9 && idle 0; then GPU=0; break; fi
    exec 9>&-
    echo "$(date -u +%T) no idle GPU among $GPUS or 0; waiting"
    sleep 60
done
[ -n "$GPU" ] || fail "no idle GPU for two hours"
touch "$RUN/started"    # minutes count from here, not from the wait
DEV="W7900D (gfx1100), GPU $GPU"
echo "$(date -u +%T) GPU $GPU; running the GPU tier"
export HK_CACHE_DIR=$CI/hk-cache FLYDSL_ROOT=/workspace/FlyDSL
HK_GPU=$GPU scripts/gpu_suite.sh "$RUN/gpu"
python3 "$CI/summarize.py" "$RUN/gpu" --sha "$SHA" --tier gfx1100 --device "$DEV" --out "$RUN"
echo "$(date -u +%T) done"
