#!/bin/bash
# One self-hosted runner for hk-dsl's GPU tier, ephemeral: registered for one
# job, then registered again. Installed as ~/gha/supervise.sh (docs/testing.md,
# "Starting and stopping the runners"):
#   install -m 755 scripts/ci/supervise.sh ~/gha/.supervise.new && mv ~/gha/.supervise.new ~/gha/supervise.sh
#   supervise.sh NAME [LABELS]
#   supervise.sh gfx950|gfx1100|gfx1100-x4             (LABELS default: NAME)
#   supervise.sh gfx950-b gfx950,gfx950-b               (a second gfx950 instance)
# NAME is the runner's directory (runner-NAME), its GitHub name (smci355-NAME),
# its pid file and its stop file; LABELS (comma-separated) are its labels
# besides hk-gpu. Several instances can carry the same label (gfx950): each has
# its own directory, and its .env its own GPUs (HK_CI_GPUS) and state
# (HK_CI_STATE) -- see docs/testing.md, "Several gfx950 runners".
# Stop: touch ~/gha/STOP (all) or ~/gha/STOP-<name>. Within 10 s the listener
# gets SIGTERM, which cancels a job in progress, and this loop exits.
# The listener and its jobs run with HOME=runner-NAME/home (also set in the
# runner's .env), never the real one: actions/checkout writes global git config
# (the job's token as an extraheader, for submodules) under $HOME, so CI never
# touches ~/.gitconfig. This loop's own gh calls keep the real HOME (gh's auth).
# Replace this file with mv, never edit it in place: running instances read it
# as they go.
set -u
NAME=$1
LABELS=${2:-$NAME}
REPO=vivienfanghuagood/hk-dsl
GHA=$HOME/gha
DIR=$GHA/runner-$NAME
cd "$DIR" || exit 1
mkdir -p home && chmod 700 home
echo $$ > "$GHA/supervise-$NAME.pid"
stopped() { [ -f "$GHA/STOP" ] || [ -f "$GHA/STOP-$NAME" ]; }
while ! stopped; do
    rm -f .runner .credentials .credentials_rsaparams
    TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" --jq .token) \
        || { echo "$(date -u +%FT%TZ) token failed"; sleep 60; continue; }
    ./config.sh --unattended --ephemeral --replace --url "https://github.com/$REPO" \
        --token "$TOKEN" --name "smci355-$NAME" --labels "hk-gpu,$LABELS" --work _work \
        || { echo "$(date -u +%FT%TZ) config failed"; sleep 60; continue; }
    echo "$(date -u +%FT%TZ) registered smci355-$NAME ($LABELS), waiting for a job"
    HOME=$DIR/home ./run.sh &
    RUN=$!
    while kill -0 $RUN 2>/dev/null; do
        if stopped; then
            echo "$(date -u +%FT%TZ) stop requested"
            pkill -TERM -f "$DIR/bin/Runner.Listener"
            wait $RUN
            break
        fi
        sleep 10
    done
    wait $RUN 2>/dev/null
    echo "$(date -u +%FT%TZ) run.sh exited"
    sleep 5
done
TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/remove-token" --jq .token 2>/dev/null) \
    && [ -f .runner ] && ./config.sh remove --token "$TOKEN" >/dev/null 2>&1
echo "$(date -u +%FT%TZ) stopped"
