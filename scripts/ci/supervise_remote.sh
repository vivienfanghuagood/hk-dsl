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
#   nohup setsid supervise_remote.sh gfx1201 <user>@<rdna4-host> <runner-root> \
#       gfx1201 >> ~/gha/logs/gfx1201.log 2>&1 < /dev/null &
# (the real host and paths live in this machine's ~/gha, not in the repository)
#   touch ~/gha/STOP-gfx1201        # within 15 s: SIGTERM to the listener there, unregister, exit
# The runner is named $HK_RUNNER_PREFIX-NAME on GitHub (default hk-gha-NAME); NAME also names its pid
# file and its stop file here.
# The listener runs with HOME=RDIR/home, never the box's real HOME: actions/checkout writes global
# git config (the job's token as an extraheader) under $HOME.
set -u
NAME=$1
HOST=$2
RDIR=$3
LABELS=${4:-$NAME}
REPO=vivienfanghuagood/hk-dsl
RNAME=${HK_RUNNER_PREFIX:-hk-gha}-$NAME
GHA=$HOME/gha
mkdir -p "$GHA"
echo $$ > "$GHA/supervise-$NAME.pid"
log() { echo "$(date -u +%FT%TZ) $*"; }
stopped() { [ -f "$GHA/STOP" ] || [ -f "$GHA/STOP-$NAME" ]; }
r() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$HOST" "$@"; }
# 0: the listener is running there; 1: it is not; anything else: could not tell (ssh failed).
# Anchored: the remote shell running pgrep has the path on its own command line.
LISTENER="^$RDIR/bin/Runner\\.Listener"
alive() { r "pgrep -u \$(id -u) -f '$LISTENER' >/dev/null"; }

up=0          # whether this loop has seen the listener of its last registration running
while ! stopped; do
    alive; a=$?
    if [ $a = 0 ]; then up=1; sleep 15; continue; fi    # running (or waiting for a job)
    if [ $a != 1 ]; then log "$HOST unreachable"; sleep 60; continue; fi
    [ $up = 1 ] && log "listener exited"
    up=0
    TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token) \
        || { log "token failed"; sleep 60; continue; }
    printf '%s\n' "$TOKEN" | r "cd '$RDIR' && rm -f .runner .credentials .credentials_rsaparams &&
        read -r T && HOME='$RDIR/home' ./config.sh --unattended --ephemeral --replace \
            --url https://github.com/$REPO --token \"\$T\" --name '$RNAME' \
            --labels 'hk-gpu,$LABELS' --work _work > '$RDIR/config.log' 2>&1"
    rc=$?
    unset TOKEN
    [ $rc = 0 ] || { log "config failed ($rc; $RDIR/config.log there)"; sleep 60; continue; }
    # setsid -f: forked and detached, so ssh returns -- a trailing & would leave a subshell
    # holding ssh's stdout, and ssh would wait for the job to end.
    r "cd '$RDIR' && HOME='$RDIR/home' setsid -f nohup ./run.sh >> '$RDIR/run.log' 2>&1 < /dev/null" \
        || { log "run.sh did not start"; sleep 60; continue; }
    # run-helper.sh takes a few seconds to start the listener: wait for it, so the next pass
    # does not take "not yet" for "exited".
    for _ in $(seq 30); do alive && { up=1; break; }; sleep 2; done
    [ $up = 1 ] && log "registered $RNAME ($LABELS) on $HOST, waiting for a job" \
                || log "registered $RNAME, but its listener did not start ($RDIR/run.log there)"
done
log "stop requested"
alive && {
    r "pkill -TERM -u \$(id -u) -f '$LISTENER'"
    for _ in $(seq 30); do alive; [ $? = 1 ] && break; sleep 2; done
}
TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token 2>/dev/null) &&
    printf '%s\n' "$TOKEN" | r "cd '$RDIR' && [ -f .runner ] && read -r T &&
        HOME='$RDIR/home' ./config.sh remove --token \"\$T\" > /dev/null 2>&1"
unset TOKEN
log "stopped"
