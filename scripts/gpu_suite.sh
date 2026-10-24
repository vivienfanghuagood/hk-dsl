#!/bin/bash
# The GPU tier (tests/hk/gpu) in groups that run side by side, on one GPU or sharded over several.
#
#   HK_GPU=3 scripts/gpu_suite.sh [OUTDIR]           one GPU
#   HK_GPUS="0 1 2 3" scripts/gpu_suite.sh [OUTDIR]  sharded: each group bound to one of them
#
# Each group's pytest log lands in OUTDIR (default /tmp/hk-gpu-run), its exit status in
# OUTDIR/status, and `done` last; the script exits non-zero if any group failed. Groups are
# handed out longest-first to HK_LANES (default 3) slots per GPU: a slot is one pytest process
# bound to its GPU by HIP_VISIBLE_DEVICES. Three a GPU because the groups compile different
# kernels -- hipcc keeps the CPUs busy while the GPU runs another group's tests -- and three
# keeps a small pod's memory clear of the compile. A group's GPU is the first line of its log.
#
# HK_PERF_GATE=1 then runs the performance gate (scripts/perf_gate.sh) alone, on the first GPU,
# after every group, as group `perf`; it needs FLYDSL_ROOT for FlyDSL's arm.
set -u
cd "$(dirname "$0")/.."
out=${1:-/tmp/hk-gpu-run}
read -ra gpus <<< "$(tr ',' ' ' <<< "${HK_GPUS:-${HK_GPU:-0}}")"
lanes=${HK_LANES:-3}
export PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$out"; mkdir -p "$out"
G=tests/hk/gpu

# The groups, longest first (measured on gfx1100 with a cold compile cache): the dispatcher
# below starts them in this order as slots free up.
labels=(); files=()
group() { labels+=("$1"); shift; files+=("$*"); }
group gemm   $G/test_gemm.py $G/test_autotune.py
group attn   $G/test_attn.py $G/test_sdpa.py
group torch  $G/test_torch_op.py $G/test_engine.py $G/test_vllm_gemm.py $G/test_flydsl_gemm.py
group decode $G/test_decode_gemm.py $G/test_paged.py $G/test_load_scalar.py
group phase2 $G/test_phase2.py $G/test_target_rmsnorm.py
group elem   $G/test_elementwise.py $G/test_bounded.py $G/test_strided.py $G/test_while.py $G/test_indexed.py
rest=$(ls $G/test_*.py | grep -v -E "test_(gemm|autotune|attn|sdpa|decode_gemm|paged|load_scalar|phase2|target_rmsnorm|elementwise|bounded|strided|while|indexed|torch_op|engine|vllm_gemm|flydsl_gemm|templates_[a-z]+)\.py")
# Every kernel a test builds is a pybind module of its own, each takes one of a process's 1024
# pthread TSS keys, and one process ran out ("could not initialize the TSS key") once gfx950's
# strips had preloaded twins: the rest in one process per GPU-slot-sized share, at least two.
set -- $rest
nrest=$(( ${#gpus[@]} > 2 ? ${#gpus[@]} : 2 ))
for ((i = 0; i < nrest; i++)); do
    share=()
    for ((j = i + 1; j <= $#; j += nrest)); do share+=("${!j}"); done
    [ ${#share[@]} -gt 0 ] && group "rest$([ $i -gt 0 ] && echo $((i + 1)))" "${share[@]}"
done
group dx tests/dx            # the DX suite (bench/dx): every task and solution on small shapes
# The templates build a kernel per plan they test: a process of their own each (TSS keys).
for t in $G/test_templates_*.py; do
    group "tmpl_$(basename "$t" .py | sed 's/^test_templates_//')" "$t"
done

# Slot s runs on gpus[s % #gpus]; pid[s] is its process, empty when free.
nslots=$(( ${#gpus[@]} * lanes ))
pid=()
run() {  # run <slot> <label> <test files...>
    local s=$1 label=$2; shift 2
    local g=${gpus[$(( s % ${#gpus[@]} ))]}
    { echo "gpu=$g"
      HIP_VISIBLE_DEVICES=$g timeout --signal=KILL 5400 python3 -m pytest -q -p no:cacheprovider "$@"
    } > "$out/$label.log" 2>&1
    echo "$label exit $?" >> "$out/status"
}
for ((i = 0; i < ${#labels[@]}; i++)); do
    while true; do      # the first free slot, waiting for a group to finish if none is
        slot=-1
        for ((s = 0; s < nslots; s++)); do
            if [ -z "${pid[$s]:-}" ] || ! kill -0 "${pid[$s]}" 2>/dev/null; then slot=$s; break; fi
        done
        [ $slot -ge 0 ] && break
        wait -n 2>/dev/null || sleep 1
    done
    # shellcheck disable=SC2086
    run "$slot" "${labels[$i]}" ${files[$i]} &
    pid[$slot]=$!
done
wait
if [ "${HK_PERF_GATE:-0}" = 1 ]; then
    HK_GPU=${gpus[0]} scripts/perf_gate.sh "$out/perf" > "$out/perf.log" 2>&1
    echo "perf exit $?" >> "$out/status"
fi
echo done >> "$out/status"
! grep -q -v -E " exit 0$|^done$" "$out/status"
