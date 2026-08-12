#!/usr/bin/env bash
# Relaunch a ccarc3 shell daemon that has died. Runs detached, at ppid 1.
#
# **Why this is a daemon and not a Monitor.** The watch was a `Monitor` three
# times over, and it timed out three times in ninety minutes: the runtime clamps
# a monitor's lifetime to thirty minutes whatever the caller asks for, so the
# thing guarding the daemons kept dying before they did. The same clamp is
# recorded against `persistent: true`. A watch that has to be re-armed by hand
# every half hour is a watch that lapses the first time nobody is looking, which
# is exactly when it is needed.
#
# `setsid` + ppid 1 is measured protection, not a guess: during the bp35
# validation run a session-worker restart killed the `claude` solver outright and
# left `supervisor.sh`, `heartbeat.sh` and `preserve_evidence.sh` running,
# because those three are detached and reparented to init. This joins them.
#
# **Liveness is argv-exact, never `pgrep -f`.** A `pgrep -f supervisor.sh` matches
# any process whose command line merely mentions the name -- including the check
# itself, and including this script, which names all three. That defect has been
# hit repeatedly here; the most recent instance was a probe that excluded
# `*grep*` to drop self-matches and thereby dropped the real watchdog, whose
# script contains `pgrep`. So: read `/proc/*/cmdline`, split on NUL, and require
# an argv element to equal the path.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${CCARC3_WATCHDOG_LOG:-/tmp/ccarc3_watchdog.log}"
INTERVAL="${CCARC3_WATCHDOG_INTERVAL:-120}"
DAEMONS="${CCARC3_WATCHDOG_DAEMONS:-supervisor.sh heartbeat.sh preserve_evidence.sh}"

stamp() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z'; }

# Every pid whose argv contains this script as an *element*, under either the
# absolute or the relative spelling. One implementation, used by both callers --
# it started as two, and the copy that only knew the absolute path could not see
# a watchdog started the way watchdogs are actually started.
# Our own process group. `setsid` makes this equal to our pid, and every
# subshell we fork inherits it -- which is the point: a `$(...)` command
# substitution is a *forked copy of this script*, argv and all, so a scan that
# excludes only `$$` finds that copy and calls it "another watchdog". It did,
# and the guard refused every launch including the first, leaving the box with
# no watchdog at all while `--check` cheerfully reported one running. Excluding
# by process group covers the subshell at any nesting depth, while a genuinely
# separate watchdog -- launched under `setsid`, as they are -- has its own group
# and is still seen.
MY_PGID="$({ read -r _ rest < /proc/self/stat; echo "${rest##*) }"; } 2>/dev/null | cut -d' ' -f3)"
[ -n "${MY_PGID:-}" ] || MY_PGID="$$"

# Field 5 of /proc/PID/stat is the process group. Fields must be taken *after*
# the last ")": `comm` is unquoted and may itself contain spaces and parens.
pgid_of() {
    local raw
    raw="$(cat "/proc/$1/stat" 2>/dev/null)" || return 1
    [ -n "$raw" ] || return 1
    set -- ${raw##*) }
    echo "${3:-}"
}

pids_of() {
    local script="$1" want="$REPO/tools/$1" pid cmd pg
    for dir in /proc/[0-9]*; do
        pid="${dir#/proc/}"
        [ "$pid" = "$$" ] && continue
        # **A pid with no readable pgid is gone, not a rival.** `pgid_of` reads
        # `/proc/$pid/stat`, and the pid most likely to vanish mid-scan is *our
        # own command-substitution subshell* -- which carries this script's argv,
        # so it is a candidate, and which exits the moment `pids_of` returns.
        # Empty then compared unequal to `MY_PGID`, the dead subshell counted as
        # another watchdog, and the guard refused to start with no watchdog
        # running at all. Intermittent, because it depends on whether the
        # subshell has been reaped yet: a launch would refuse three times and
        # succeed on the fourth, which is why this read as contention for hours
        # before it read as a bug.
        pg="$(pgid_of "$pid")"
        [ -z "$pg" ] && continue
        [ "$pg" = "$MY_PGID" ] && continue
        [ -r "$dir/cmdline" ] || continue
        # **A `--check` invocation is not a running daemon.** It carries this
        # script's argv for the half-second it lives, so a watchdog started in
        # the same breath as a health check sees the check as a rival and
        # refuses. That is how the launch failed for two hours while `--check`
        # reported no watchdog running: the two invocations were looking at each
        # other. Reporting on the daemons is not being one.
        case " $({ tr '\0' ' ' < "$dir/cmdline"; } 2>/dev/null) " in
            *" --check "*) continue;;
        esac
        while IFS= read -r -d '' cmd; do
            if [ "$cmd" = "$want" ] || [ "$cmd" = "tools/$script" ]; then
                echo "$pid"
                break
            fi
        done < "$dir/cmdline"
    done
}

# Is a daemon running? Argv-exact, never `pgrep -f`: this script names all three
# daemons, so a substring check finds *itself* and reports a dead daemon alive.
running() { [ -n "$(pids_of "$1")" ]; }

# One watchdog per *daemon set*. A second watching the same set would double
# every relaunch; a second watching a different set is not a duplicate at all,
# and refusing it would mean a test could never start one beside the live
# instance -- which is how the first version of this guard was written, and it
# made the behaviour untestable while the production watchdog ran.
#
# The set is not in argv, so it is read from the candidate's own environment.
# `/proc/PID/environ` is NUL-separated; a missing or unreadable one is treated as
# the default set, which is what an instance launched without the variable is
# actually watching.
already_watching() {
    local pid theirs
    for pid in $(pids_of "daemon_watchdog.sh"); do
        # **The redirection is what fails here, not the command.** A process can
        # exit between `pids_of` listing it and this line opening its environ,
        # and `2>/dev/null` on `tr` does not silence a failed *input
        # redirection* -- the shell reports that itself, before `tr` runs. This
        # project has the trap on record and it was reintroduced here anyway;
        # the fix is to redirect the whole group's stderr.
        # Same reasoning as the pgid read above: an unreadable environ means the
        # process is gone. Defaulting to the standard daemon set here would make
        # a corpse look like a rival watching exactly what we watch.
        [ -r "/proc/$pid/environ" ] || continue
        theirs="$({ tr '\0' '\n' < "/proc/$pid/environ" \
                    | sed -n 's/^CCARC3_WATCHDOG_DAEMONS=//p' | head -1; } 2>/dev/null)"
        [ -z "$theirs" ] && theirs="supervisor.sh heartbeat.sh preserve_evidence.sh"
        [ "$theirs" = "$DAEMONS" ] && return 0
    done
    return 1
}

if [ "${1:-}" = "--check" ]; then
    for d in $DAEMONS; do
        running "$d" && echo "$d alive" || echo "$d GONE"
    done
    already_watching && echo "watchdog already running" || echo "no other watchdog"
    exit 0
fi

if already_watching; then
    echo "$(stamp) another watchdog is already running; exiting" >> "$LOG"
    exit 0
fi

echo "$(stamp) watchdog up (pid $$, ppid $PPID, every ${INTERVAL}s)" >> "$LOG"

while true; do
    for d in $DAEMONS; do
        if ! running "$d"; then
            echo "$(stamp) DAEMON-LOSS $d" >> "$LOG"
            setsid bash "$REPO/tools/$d" >> "/tmp/${d%.sh}.relaunch.log" 2>&1 < /dev/null &
            echo "$(stamp) RELAUNCHED $d" >> "$LOG"
        fi
    done
    sleep "$INTERVAL"
done
