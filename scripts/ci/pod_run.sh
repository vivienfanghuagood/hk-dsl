#!/bin/bash
# GPU CI, gfx1100, the pod's side: one commit on one W7900D of the dev pod,
# installed and started, detached, by run_gfx1100.sh.
#   pod_run.sh SHA PROXY_IPV4 KEY
#
# Everything lives in $HK_CI_POD_DIR (default /workspace/gha): the clone,
# fetched through the gh-test proxy at the address the caller resolved, the
# dialect's build, the compile cache and runs/KEY. The GPU: an idle one of
# $HK_CI_POD_GPUS (default 6), failing that GPU 0, each under the node-level
# lock of its physical card: /workspace/locks/gpu/<PCI address>.lock, one
# directory every pod on the node mounts, so a run here and a benchmark in
# another pod that sees the same cards cannot both take one (a lock inside
# this pod alone is invisible to them). Without that directory the run
# refuses to start rather than run unlocked. GPUs 5 and 7 left the default:
# the node allocates them to other users' serving pods. Nothing outside the
# locks and this process's environment is changed.
set -uo pipefail
SHA=$1; IP=$2; KEY=$3
[[ $SHA =~ ^[0-9a-f]{40}$ && $IP =~ ^[0-9.]+$ && $KEY =~ ^[0-9-]+$ ]] || exit 2
CI=${HK_CI_POD_DIR:-/workspace/gha}
GPUS=${HK_CI_POD_GPUS:-6}
LOCKS=/workspace/locks
RUN=$CI/runs/$KEY
SRC=$CI/hk-dsl
mkdir -p "$RUN"
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
[ -f "$LOCKS/NODE_LOCKS" ] || fail "no node-level GPU lock directory at $LOCKS"
lock_of() {     # the node-level lock file of this pod's GPU $1
    local pci
    pci=$(rocm-smi -d "$1" --showbus 2>/dev/null | grep -o "PCI Bus: [0-9A-Fa-f:.]*" | head -1 | awk '{print tolower($3)}')
    [ -n "$pci" ] && echo "$LOCKS/gpu/$pci.lock"
}
GPU=
for i in $(seq 120); do
    for g in $GPUS 0; do
        lk=$(lock_of "$g") || continue
        exec 9>>"$lk"
        if flock -n 9 && idle "$g"; then
            GPU=$g
            echo "pod=$(hostname) pid=$$ since=$(date -u +%FT%TZ) gpu_index=$g cmd=CI run $KEY ($SHA)" > "${lk%.lock}.owner"
            break 2
        fi
        exec 9>&-
    done
    echo "$(date -u +%T) no idle GPU among $GPUS 0; waiting"
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
