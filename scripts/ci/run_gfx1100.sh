#!/bin/bash
# GPU CI, gfx1100, as a self-hosted Actions job (.github/workflows/gpu.yml):
# the commit on a W7900D of the dev pod, through `ssh $HK_CI_JUMP` and
# `kubectl exec`. The pod fetches the commit itself, through the gh-test
# proxy: its DNS resolves no GitHub name and github.com is blocked by name, so
# git is pointed at the address resolved here. This side installs pod_run.sh
# and summarize.py from the checkout, starts the pod's run detached, follows
# its log and brings back its result. No code from the repository runs on the
# jump host.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POD=${HK_CI_POD:-hfang-hk-rdna3-b}
JUMP=${HK_CI_JUMP:-0004}
PROXY=${HK_CI_PROXY:-gh-test.anruicloud.com}
DIR=${HK_CI_POD_DIR:-/workspace/gha}       # the pod's checkout, build, cache and runs
GPUS=${HK_CI_POD_GPUS:-6}                  # tried first; then GPU 0 (pod_run.sh)
SHA=$GITHUB_SHA
KEY=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT
R=$DIR/runs/$KEY
OUT=$GITHUB_WORKSPACE/gha-report
mkdir -p "$OUT"
# Run the script on stdin in the pod.
pod() { ssh -o BatchMode=yes -o ServerAliveInterval=30 "$JUMP" "kubectl exec -i $POD -- bash -s" ; }
cancel() { echo "kill -- -\$(cat $R/pid) 2>/dev/null; true" | pod; exit 130; }
trap cancel INT TERM

IP=$(ssh -o BatchMode=yes "$JUMP" "getent ahostsv4 $PROXY | head -1 | cut -d' ' -f1")
[[ $IP =~ ^[0-9.]+$ ]] || { echo "::error::cannot resolve $PROXY on $JUMP"; exit 1; }
{ echo "mkdir -p $DIR"
  echo "base64 -d > $DIR/pod_run.sh <<'B64'"; base64 -w0 "$HERE/pod_run.sh"; echo; echo "B64"
  echo "base64 -d > $DIR/summarize.py <<'B64'"; base64 -w0 "$HERE/summarize.py"; echo; echo "B64"
  echo "HK_CI_POD_DIR=$DIR HK_CI_POD_GPUS='$GPUS' nohup setsid bash $DIR/pod_run.sh $SHA $IP $KEY >/dev/null 2>&1 < /dev/null &"
} | pod || { echo "::error::cannot start the run in $POD"; exit 1; }
echo "started on $POD (run $KEY)"

seen=0
for i in $(seq 360); do     # 3 hours
    sleep 30
    st=$(echo "cd $R 2>/dev/null || { echo missing 0; exit; }; [ -f result.json ] && s=done || s=running; echo \$s \$(wc -l < driver.log)" | pod 2>/dev/null)
    n=$(awk '{print $2}' <<<"$st")
    if [[ ${n:-0} =~ ^[0-9]+$ ]] && [ "$n" -gt "$seen" ]; then
        echo "sed -n '$((seen + 1)),${n}p' $R/driver.log" | pod 2>/dev/null; seen=$n
    fi
    [[ $st == done* ]] && break
done
echo "cd $R && tar -cf - result.json report.txt build.log driver.log 2>/dev/null" | pod | tar -xf - -C "$OUT"
[ -f "$OUT/result.json" ] || { echo "::error::no result from the pod"; exit 1; }
python3 "$HERE/report.py" gfx1100 "$OUT/result.json"
