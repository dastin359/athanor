#!/bin/bash
# Watchdog + event feed for tools/heartbeat.sh.
#
# **Why this exists.** The heartbeat used to run AS the harness Monitor's
# command, so the Monitor's timeout was also the heartbeat's lifetime: on
# 2026-08-08 the Monitor expired and took heartbeat.sh (pid 3196) down with it,
# silently. Nothing reported the loss -- heartbeat.sh's own `daemon_check` lists
# supervisor.sh, preserve_evidence.sh and context_watch.py, but not the heartbeat
# itself, so the one daemon that could not report its own death was the one that
# died. Same defect this project keeps hitting: the check passes by not running.
#
# So the heartbeat is now detached (setsid, own log) and this script only WATCHES
# it. A Monitor timeout now costs an event feed, not the daemon; and when the
# daemon does die, the relaunch below both fixes it and says so. Verified both
# ways before arming: against a dead heartbeat it relaunched and emitted one
# line, and the relaunched heartbeat outlived the watchdog's own death; against a
# live one it stayed silent and spawned no duplicate.
#
# Liveness is argv-element-exact, never a substring -- the trap that has bitten
# this project three times. This file's own path ends in `heartbeat_watch.sh`,
# which `-x .*/heartbeat\.sh` cannot match, so the watchdog can never mistake
# itself for the thing it watches.
SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
LOG="$SP/heartbeat.log"
HB="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/heartbeat.sh"

hb_alive() {
    local d argv
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        argv=$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null)
        grep -qx '.*/heartbeat\.sh' <<< "$argv" && return 0
    done
    return 1
}

touch "$LOG"
POS=$(wc -l < "$LOG")

while true; do
    if ! hb_alive; then
        echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') heartbeat.sh DOWN — relaunching detached"
        setsid nohup bash "$HB" >> "$LOG" 2>&1 < /dev/null &
        sleep 3
    fi
    # Drain only what is new. POS starts at the log's current length so a re-arm
    # does not replay history as fresh events.
    TOT=$(wc -l < "$LOG")
    if [ "$TOT" -gt "$POS" ]; then
        sed -n "$((POS + 1)),${TOT}p" "$LOG" \
            | grep -E 'FINISHED|DAEMON DOWN|NO RUNNER|LAST FAILURE|SWEEP COMPLETE|Traceback'
        POS=$TOT
    fi
    sleep 60
done
