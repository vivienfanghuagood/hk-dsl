#!/bin/bash
# The GPU tier (tests/hk/gpu) on one Radeon, in groups that run side by side.
#
#   HK_GPU=3 scripts/gpu_suite.sh [OUTDIR]
#
# Each group's pytest log lands in OUTDIR (default /tmp/hk-gpu-run), its exit
# status in OUTDIR/status, and `done` last; the script exits non-zero if any
# group failed. Groups run three at a time because they compile different
# kernels: hipcc keeps the CPUs busy while the GPU runs another group's tests,
# and three keeps a small pod's memory clear of the compile.
#
# HK_PERF_GATE=1 then runs the performance gate (scripts/perf_gate.sh) alone,
# after every group, as group `perf`; it needs FLYDSL_ROOT for FlyDSL's arm.
set -u
cd "$(dirname "$0")/.."
out=${1:-/tmp/hk-gpu-run}
gpu=${HK_GPU:-0}
export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$out"; mkdir -p "$out"
G=tests/hk/gpu

run() {  # run <label> <test files...>
    local label=$1; shift
    HIP_VISIBLE_DEVICES=$gpu timeout --signal=KILL 5400 \
        python3 -m pytest -q -p no:cacheprovider "$@" > "$out/$label.log" 2>&1
    echo "$label exit $?" >> "$out/status"
}

run gemm   $G/test_gemm.py $G/test_autotune.py &
run attn   $G/test_attn.py $G/test_sdpa.py &
run decode $G/test_decode_gemm.py $G/test_paged.py $G/test_load_scalar.py &
wait
run phase2 $G/test_phase2.py $G/test_target_rmsnorm.py &
run elem   $G/test_elementwise.py $G/test_bounded.py $G/test_strided.py $G/test_while.py $G/test_indexed.py &
run torch  $G/test_torch_op.py $G/test_engine.py $G/test_vllm_gemm.py $G/test_flydsl_gemm.py &
wait
rest=$(ls $G/test_*.py | grep -v -E "test_(gemm|autotune|attn|sdpa|decode_gemm|paged|load_scalar|phase2|target_rmsnorm|elementwise|bounded|strided|while|indexed|torch_op|engine|vllm_gemm|flydsl_gemm)\.py")
# Two processes: every kernel a test builds is a pybind module of its own,
# each takes one of a process's 1024 pthread TSS keys, and one process ran
# out ("could not initialize the TSS key") once gfx950's strips had
# preloaded twins.
set -- $rest
half=$(( ($# + 1) / 2 ))
run rest "${@:1:$half}" &
run rest2 "${@:$((half + 1))}" &
wait
if [ "${HK_PERF_GATE:-0}" = 1 ]; then
    HK_GPU=$gpu scripts/perf_gate.sh "$out/perf" > "$out/perf.log" 2>&1
    echo "perf exit $?" >> "$out/status"
fi
echo done >> "$out/status"
! grep -q -v -E " exit 0$|^done$" "$out/status"
