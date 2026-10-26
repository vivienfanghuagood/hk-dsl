# gpulock.sh -- GPU locks on the gfx1201 dev box (8x Radeon AI PRO R9700), named by physical PCI
# address.
#
# The box is shared: other users run on these cards with no lock of any kind. These locks only keep *our* sessions off each other's card; the idle check
# (gpu_foreign) is what keeps us off everyone else's. Both are needed before a GPU run.
# Lock: $GPULOCK_DIR/gpu/<pci>.lock (flock); holder: $GPULOCK_DIR/gpu/<pci>.owner.
#
#   GPULOCK_DIR=<dir> . scripts/ci/gfx1201/gpulock.sh
#   gpulock_take IDX [WAIT_S]   # IDX: rocm-smi index on the host; takes fd 9
#   gpulock_pci IDX             # prints the card's PCI address, e.g. 0000:63:00.0
#   gpulock_render IDX          # prints the card's render node, e.g. renderD128
#   gpu_foreign IDX             # prints the KFD processes holding VRAM/queues on it (empty: idle)
# or:  scripts/ci/gfx1201/gpulock IDX [-w WAIT_S] -- CMD...
# GPULOCK_DIR (required): the lock directory every session on the box shares, with a NODE_LOCKS
# marker in it; set in the runner's .env, and in a person's shell.

GPULOCK_DIR=${GPULOCK_DIR:-}

gpulock_pci() {
    local pci
    pci=$(rocm-smi -d "$1" --showbus 2>/dev/null | grep -o "PCI Bus: [0-9A-Fa-f:.]*" | head -1 | awk '{print tolower($3)}')
    [ -n "$pci" ] || { echo "gpulock: no PCI address for GPU index $1" >&2; return 4; }
    echo "$pci"
}

gpulock_render() {
    local pci
    pci=$(gpulock_pci "$1") || return $?
    basename "$(ls -d /sys/bus/pci/devices/$pci/drm/renderD* 2>/dev/null | head -1)"
}

gpulock_path() {
    local pci
    [ -n "$GPULOCK_DIR" ] || { echo "gpulock: set GPULOCK_DIR to the box's lock directory" >&2; return 2; }
    pci=$(gpulock_pci "$1") || return $?
    echo "$GPULOCK_DIR/gpu/$pci.lock"
}

# KFD processes on GPU IDX (queues or VRAM there), one "pid user vram_MiB cmd" a line.
gpu_foreign() {
    local r minor gid p pid on v
    r=$(gpulock_render "$1") || return $?
    minor=${r#renderD}
    for n in /sys/class/kfd/kfd/topology/nodes/*; do
        grep -qx "drm_render_minor $minor" "$n/properties" 2>/dev/null && gid=$(cat "$n/gpu_id")
    done
    [ -n "${gid:-}" ] || { echo "gpulock: no KFD node for $r" >&2; return 4; }
    for p in /sys/class/kfd/kfd/proc/[0-9]*; do
        pid=${p##*/}; on=0
        grep -qx "$gid" "$p"/queues/*/gpuid 2>/dev/null && on=1
        v=$(cat "$p/vram_$gid" 2>/dev/null); [ "${v:-0}" != 0 ] && on=1
        [ $on = 1 ] && echo "$pid $(ps -o user= -p "$pid" 2>/dev/null) $(( ${v:-0} >> 20 ))MiB $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-80)"
    done
    return 0
}

gpulock_take() {
    local idx=$1 wait=${2:-3600} lk
    [ -f "$GPULOCK_DIR/NODE_LOCKS" ] || { echo "gpulock: $GPULOCK_DIR has no NODE_LOCKS marker" >&2; return 7; }
    lk=$(gpulock_path "$idx") || return $?
    exec 9>>"$lk"
    if ! flock -w "$wait" 9; then
        echo "gpulock: $lk busy for ${wait}s; holder: $(cat "${lk%.lock}.owner" 2>/dev/null)" >&2
        exec 9>&-
        return 3
    fi
    printf 'host=%s pid=%s since=%s gpu_index=%s cmd=%s\n' "$(hostname)" "$$" "$(date -u +%FT%TZ)" "$idx" \
        "${GPULOCK_WHAT:-$(tr '\0' ' ' < /proc/$$/cmdline | cut -c1-200)}" > "${lk%.lock}.owner"
}
