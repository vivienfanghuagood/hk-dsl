#!/bin/bash
# run_row.sh OUT ROW STARTS "ARM ARM ..." [MODE] -- the server starts of one scorecard row: for s in 1..STARTS,
# every arm once, the order reversed on even s, one serve_bench.py start each, 60 s apart. The arguments
# serve_bench.py needs besides --row/--arm/--out/--mode come from the environment:
#   SB_GPUS (HIP indices), SB_MODEL (the weights), SB_WORK (the row's work directory), and optionally
#   SB_HK_SRC (a checkout for the hk arms) and SB_LOCK_DIR (an flock per GPU).
# A start that hangs at startup (serve_bench exit 5) is retried once. In measure mode, a start whose GSM8K is
# below 80% (rows.json quality.gsm_min for a model that scores lower) or whose quality probe had errors stops
# that arm's remaining starts (OUT/abort.ARM): a fast arm that answers wrong is not a result.
# Then: python3 table.py --ref ARM --arms ... --cols ... OUT
set -u
OUT=$1; ROW=$2; STARTS=$3; ARMS=$4; MODE=${5:-measure}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SB_GPUS:?set SB_GPUS}" "${SB_MODEL:?set SB_MODEL}" "${SB_WORK:?set SB_WORK}"
mkdir -p "$OUT"
ulimit -c 0; export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
read -ra AS <<< "$ARMS"
log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/row.log"; }
log "row $ROW starts=$STARTS arms=$ARMS mode=$MODE gpus=$SB_GPUS"
start() {   # ARM DIR
  python3 "$HERE/serve_bench.py" --row "$ROW" --arm "$1" --mode "$MODE" --out "$2" --gpus "$SB_GPUS" \
      --model-path "$SB_MODEL" --work "$SB_WORK" ${SB_HK_SRC:+--hk-src "$SB_HK_SRC"} \
      ${SB_LOCK_DIR:+--lock-dir "$SB_LOCK_DIR"} > "$2.log" 2>&1
}
quality_ok() {  # DIR -> prints "gsm X errors N"; returns 1 when the floor trips
  python3 - "$1/quality.json" "$HERE/rows.json" "$ROW" <<'PY'
import json, sys
floor = json.load(open(sys.argv[2]))[sys.argv[3]].get("quality", {}).get("gsm_min", 0.80)
try:
    s = json.load(open(sys.argv[1]))["summary"]
except Exception as e:  # noqa: BLE001
    print(f"no quality result ({type(e).__name__})"); sys.exit(1)
acc, err = s.get("gsm8k_acc"), s.get("errors", 0)
print(f"gsm {acc} errors {err} (floor {floor})")
sys.exit(0 if (acc is not None and acc >= floor and not err) else 1)
PY
}
for s in $(seq 1 "$STARTS"); do
  ORDER=("${AS[@]}")
  if [ $((s % 2)) = 0 ]; then ORDER=(); for ((i=${#AS[@]}-1; i>=0; i--)); do ORDER+=("${AS[$i]}"); done; fi
  for arm in "${ORDER[@]}"; do
    if [ -e "$OUT/abort.$arm" ]; then log "skip $arm-s$s (arm aborted: $(cat "$OUT/abort.$arm"))"; continue; fi
    log "start $arm-s$s"
    start "$arm" "$OUT/$arm-s$s"; rc=$?
    log "end $arm-s$s rc=$rc $(tail -1 "$OUT/$arm-s$s.log")"
    if [ $rc = 5 ]; then
      mv "$OUT/$arm-s$s" "$OUT/$arm-s$s.hang"; mv "$OUT/$arm-s$s.log" "$OUT/$arm-s$s.hang.log"; sleep 60
      log "retry $arm-s$s"
      start "$arm" "$OUT/$arm-s$s"; rc=$?
      log "end $arm-s$s rc=$rc $(tail -1 "$OUT/$arm-s$s.log")"
    fi
    if [ "$MODE" = measure ]; then
      q=$(quality_ok "$OUT/$arm-s$s"); qrc=$?
      log "quality $arm-s$s: $q"
      if [ $rc != 0 ] || [ $qrc != 0 ]; then echo "rc $rc; $q" > "$OUT/abort.$arm"; log "ABORT arm $arm (rc $rc; $q)"; fi
    fi
    sleep 60
  done
done
log "ROW-DONE"
