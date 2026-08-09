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
# So the heartbeat is now launched detached (own session, own log, reparented to
# init) and this script only WATCHES it.
#
# **`setsid` alone was not enough, and the first version of this comment claimed
# it was.** It said the relaunched heartbeat "outlived the watchdog's own death",
# which was a true observation of the wrong thing: that instance had been started
# from a shell that exited, so it was already an orphan with ppid 1. A heartbeat
# started BY this watchdog is a live child of it, and `setsid` changes the
# session, not the parent -- whatever reaps a finished Monitor walks descendants,
# so it took the heartbeat with it every time. Measured: relaunch lines at 16:26,
# 16:58 and 17:28 PDT, one per 30-minute Monitor cycle, while supervisor.sh and
# preserve_evidence.sh (ppid 1) sailed through all three.
#
# Hence the double fork below: the subshell backgrounds the heartbeat and exits
# immediately, orphaning it to init, which is what the surviving daemons look
# like. The watchdog remains the safety net -- if the heartbeat dies for any
# other reason it is relaunched within one poll and the loss is announced --
# but it is no longer the thing keeping it alive.
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
# **`2>/dev/null` on `tr` does not silence the redirection.** `< "$d/cmdline"` is
# opened by the SHELL before `tr` exists, so when the process exits between the
# glob and the read -- which it does, `/proc` is a live directory -- the failure is
# reported on the shell's stderr and the `2>/dev/null` never sees it. Measured
# 2026-08-09: `bash: line 43: /proc/19946/cmdline: No such file or directory`
# reached the heartbeat's output and the Monitor forwarded it as an alert. An
# alarm channel that emits noise is one people stop reading, which is the same
# reason the daemon report is not always-on. Braces put the redirection inside the
# group, so the group's stderr covers it.
        argv=$({ tr '\0' '\n' < "$d/cmdline"; } 2>/dev/null)
        grep -qx '.*/heartbeat\.sh' <<< "$argv" && return 0
    done
    return 1
}

touch "$LOG"
POS=$(wc -l < "$LOG")

while true; do
    if ! hb_alive; then
        echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') heartbeat.sh DOWN — relaunching detached"
        # Double fork: the subshell exits at once, so the heartbeat reparents
        # to init instead of hanging off this watchdog. `setsid` gives it its
        # own session; only the orphaning survives a descendant sweep.
        ( setsid bash "$HB" >> "$LOG" 2>&1 < /dev/null & )
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
