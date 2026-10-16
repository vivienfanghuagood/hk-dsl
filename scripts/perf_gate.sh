#!/bin/bash
# The performance gate, on one Radeon: the scoreboard with the baseline's
# arguments, then every ratio t_arm / t_hk checked against
# tests/perf/baselines/<arch>.json (tools/hk-bench/check_regression.py).
#
#   HK_GPU=3 FLYDSL_ROOT=/workspace/FlyDSL scripts/perf_gate.sh [OUTDIR]
#   HK_GPU=3 FLYDSL_ROOT=/workspace/FlyDSL scripts/perf_gate.sh --refresh [N]
#
# Exits 0 if every gated pair is within tolerance, 1 on a regression, 2 if
# the run could not be checked (the scoreboard failed, or a gated arm is
# missing -- FlyDSL's needs FLYDSL_ROOT). --refresh runs the scoreboard N
# times (default 3), each its own process, and rewrites the baseline from
# them; review its diff before committing it. Run it alone on the GPU: the
# gate is a timing.
set -u
cd "$(dirname "$0")/.."
gpu=${HK_GPU:-0}
arch=${HK_ARCH:-gfx1100}
py=${PYTHON:-python3}
base=tests/perf/baselines/$arch.json
export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"

scoreboard() {  # scoreboard <json> <log>
    # shellcheck disable=SC2046 -- the baseline's arguments, word-split on purpose
    HIP_VISIBLE_DEVICES=$gpu timeout --signal=KILL 3600 "$py" tools/hk-bench/scoreboard.py \
        $("$py" tools/hk-bench/check_regression.py --print-args --baseline "$base") \
        --json "$1" > "$2" 2>&1
}

if [ "${1:-}" = "--refresh" ]; then
    n=${2:-3}
    out=$(mktemp -d /tmp/hk-perf-refresh.XXXX)
    runs=()
    for i in $(seq 1 "$n"); do
        scoreboard "$out/run$i.json" "$out/run$i.log" || { echo "run $i failed: $out/run$i.log"; exit 2; }
        runs+=("$out/run$i.json")
    done
    "$py" tools/hk-bench/check_regression.py --refresh --baseline "$base" "${runs[@]}"
    exit $?
fi

out=${1:-/tmp/hk-perf-gate}
mkdir -p "$out"
scoreboard "$out/scoreboard.json" "$out/scoreboard.log" || {
    echo "scoreboard failed: $out/scoreboard.log"; exit 2; }
"$py" tools/hk-bench/check_regression.py "$out/scoreboard.json" --baseline "$base" \
    | tee "$out/gate.txt"
exit "${PIPESTATUS[0]}"
