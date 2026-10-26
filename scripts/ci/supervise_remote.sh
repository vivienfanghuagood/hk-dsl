#!/bin/bash
# supervise_remote.sh NAME HOST RDIR [LABELS]
#
# One ephemeral self-hosted runner for hk-dsl on another machine, kept registered from this one:
# the gfx1201 dev box (docs/testing.md, "The gfx1201 runner"). The runner itself -- actions-runner,
# its work directory and its HOME -- lives on HOST in RDIR; what stays here is the one thing the dev
# box must not hold, a credential that can mint registration tokens (gh's login on this machine).
# Each round: a registration token from `gh api`, sent over ssh's stdin to config.sh on HOST (never on
# a command line here, never in a file or a log), then run.sh started there detached, for one job.
# When the listener exits, the next round registers again. Started and stopped like supervise.sh:
#   nohup setsid supervise_remote.sh gfx1201 amd@10.170.38.101 /home/amd/hfang-hk/gha/runner-gfx1201 \
#       gfx1201 >> ~/gha/logs/gfx1201.log 2>&1 < /dev/null &
#   touch ~/gha/STOP-gfx1201        # within 15 s: SIGTERM to the listener there, unregister, exit
# NAME is the runner's GitHub name prefix (amd-g7466-NAME), its pid file and its stop file here.
# The listener runs with HOME=RDIR/home, never the box's real HOME: actions/checkout writes global
# git config (the job's token as an extraheader) under $HOME.
set -u
NAME=$1
HOST=$2
RDIR=$3
LABELS=${4:-$NAME}
REPO=vivienfanghuagood/hk-dsl
GHA=$HOME/gha
mkdir -p "$GHA"
echo $$ > "$GHA/supervise-$NAME.pid"
log() { echo "$(date -u +%FT%TZ) $*"; }
stopped() { [ -f "$GHA/STOP" ] || [ -f "$GHA/STOP-$NAME" ]; }
r() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST" "$@"; }
# 0: the listener is running there; 1: it is not; anything else: could not tell (ssh failed).
alive() { r "pgrep -u \$(id -u) -f '$RDIR/bin/Runner.Listener' >/dev/null"; }

while ! stopped; do
    alive; a=$?
    if [ $a = 0 ]; then sleep 30; continue; fi           # a round from before a restart, still running
    if [ $a != 1 ]; then log "$HOST unreachable"; sleep 60; continue; fi
    TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token) \
        || { log "token failed"; sleep 60; continue; }
    printf '%s\n' "$TOKEN" | r "cd '$RDIR' && rm -f .runner .credentials .credentials_rsaparams &&
        read -r T && HOME='$RDIR/home' ./config.sh --unattended --ephemeral --replace \
            --url https://github.com/$REPO --token \"\$T\" --name 'amd-g7466-$NAME' \
            --labels 'hk-gpu,$LABELS' --work _work > '$RDIR/config.log' 2>&1"
    rc=$?
    unset TOKEN
    [ $rc = 0 ] || { log "config failed ($rc; $RDIR/config.log there)"; sleep 60; continue; }
    r "cd '$RDIR' && HOME='$RDIR/home' setsid nohup ./run.sh >> '$RDIR/run.log' 2>&1 < /dev/null &" \
        || { log "run.sh did not start"; sleep 60; continue; }
    log "registered amd-g7466-$NAME ($LABELS) on $HOST, waiting for a job"
    sleep 15
    while :; do
        if stopped; then
            log "stop requested"
            r "pkill -TERM -u \$(id -u) -f '$RDIR/bin/Runner.Listener'"
            for _ in $(seq 30); do alive; [ $? = 1 ] && break; sleep 2; done
            break
        fi
        alive; a=$?
        [ $a = 1 ] && { log "listener exited"; break; }
        sleep 15
    done
    sleep 5
done
TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token 2>/dev/null) &&
    printf '%s\n' "$TOKEN" | r "cd '$RDIR' && [ -f .runner ] && read -r T &&
        HOME='$RDIR/home' ./config.sh remove --token \"\$T\" > /dev/null 2>&1"
unset TOKEN
log "stopped"
