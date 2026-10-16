#!/bin/bash
# GPU CI, gfx1100: one commit on GPU 0 of the W7900D dev pod.
#
#   pod_run.sh SHA PROXY_IPV4      (installed and started, detached, by orchestrate.py)
#
# The cluster cannot reach github.com, so the commit is fetched through the
# gh-test.anruicloud.com proxy; the pod's DNS cannot resolve that name, so git
# is pointed at the address the orchestrator resolved. Nothing outside
# /workspace/ci and this process's environment is changed.
#
# A run holds /workspace/ci/gpu0.lock for as long as it uses the GPU. Anything
# else that times GPU 0 should hold it too:
#   flock /workspace/ci/gpu0.lock <command>
set -uo pipefail
SHA=${1:-}
IP=${2:-}
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo "usage: pod_run.sh SHA PROXY_IPV4" >&2; exit 2; }
[[ $IP =~ ^[0-9.]+$ ]] || { echo "usage: pod_run.sh SHA PROXY_IPV4" >&2; exit 2; }

CI=/workspace/ci
RUN=$CI/runs/$SHA
SRC=$CI/hk-dsl
mkdir -p "$RUN"
exec >"$RUN/driver.log" 2>&1
echo $$ >"$RUN/pid"
touch "$RUN/started"
export PATH=/opt/venv/bin:$PATH
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=url.https://gh-test.anruicloud.com/.insteadOf GIT_CONFIG_VALUE_0=https://github.com/
export GIT_CONFIG_KEY_1=http.curloptResolve GIT_CONFIG_VALUE_1=gh-test.anruicloud.com:443:$IP

fail() {
    echo "error: $*"
    python3 "$CI/summarize.py" --error "$*" --sha "$SHA" --tier gfx1100 \
        --device "W7900D (gfx1100), GPU 0" --out "$RUN"
    exit 1
}

# The GPU: the lock first, then GPU 0 idle, for at most two hours.
exec 9>"$CI/gpu0.lock"
flock 9
for i in $(seq 120); do
    use=$(rocm-smi -d 0 --showuse 2>/dev/null | grep -o "GPU use (%): [0-9]*" | grep -o "[0-9]*$")
    mem=$(rocm-smi -d 0 --showmemuse 2>/dev/null | grep -o "VRAM%): [0-9]*" | grep -o "[0-9]*$")
    [ "${use:-100}" -le 10 ] && [ "${mem:-100}" -le 3 ] && break
    [ "$i" = 120 ] && fail "GPU 0 stayed busy for two hours"
    echo "$(date -u +%T) GPU 0 busy (use ${use:-?}%, vram ${mem:-?}%); waiting"
    sleep 60
done
touch "$RUN/started"    # the run's minutes count from here, not from the wait

# The commit.
if [ ! -d "$SRC/.git" ]; then
    git clone -q https://github.com/vivienfanghuagood/hk-dsl.git "$SRC" || fail "git clone through the proxy"
fi
cd "$SRC" || fail "no checkout"
git fetch -q origin "$SHA" || fail "git fetch $SHA through the proxy"
git checkout -q --force --detach "$SHA" || fail "git checkout $SHA"
git submodule sync -q && git submodule update -q --init --recursive || fail "the HipKittens submodule"

# The dialect, against the pod's MLIR, in a build directory of CI's own.
HK_LLVM_ROOT=/workspace/llvm HK_MLIR_BUILD=$CI/hk-build-cp312 PYTHON=/opt/venv/bin/python3 JOBS=32 \
    scripts/build_hk_mlir.sh >"$RUN/build.log" 2>&1 || fail "the dialect did not build (build.log)"

# The GPU tier, with a compile cache of CI's own.
export HK_CACHE_DIR=$CI/hk-cache FLYDSL_ROOT=/workspace/FlyDSL
HK_GPU=0 scripts/gpu_suite.sh "$RUN/gpu"
python3 "$CI/summarize.py" "$RUN/gpu" --sha "$SHA" --tier gfx1100 \
    --device "W7900D (gfx1100), GPU 0" --out "$RUN"
