#!/bin/bash
# Keep the baseline-free arm running whenever quota allows, and only then.
#
# Replaces quota_guard.sh, which was one-directional: it stopped the runner at
# the ceiling and exited, so resuming after the weekly reset needed a human or a
# wake-up to notice. The operator's framing — *"only run when we have enough
# quota. Weekly limit will reset within 18 hours"* — is a duty cycle, not a
# single stop, so this loops both ways.
#
# **Stops politely.** At the ceiling it waits for every started game to have a
# result.json before killing the runner. Killing mid-game is the worst option:
# the solver is a child that keeps burning quota while the parent that writes
# result.json is gone, so the work is lost and the spend buys nothing. Learned
# when stopping the batch after lp85.
#
# **Starts only what is sanctioned.** batch6.py is retired — it hands
# baseline_actions to the solver. The runner is now an argument (default: the
# rollout driver), but every runner it will start installs the baseline strip
# and asserts it, per "all runs after the 13th scored game must be baseline
# free" and "no control arm from now on".
#
# Process matching is argv-element-exact, never a substring: this script's own
# command line contains the runner's name, so a `pgrep -f` would match itself and
# conclude the arm was always running. That bug has bitten this project three
# times.
SP=/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad
# **The runner is an argument, and it comes from the repo.** It was hardcoded to
# the scratchpad copy of ablate_baselines.py -- volatile, so a container
# replacement could leave the supervisor driving whatever the image snapshot
# held, and wrong besides once the arm finished and the queue moved on to the
# rollout driver. That driver has no quota guard of its own, so running it
# outside this loop means running it unguarded.
#
#   bash tools/supervisor.sh                     # default: the rollout driver
#   bash tools/supervisor.sh tools/rerun_losses.py
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUNNER="${1:-$REPO/tools/clean_rollouts.py}"
case "$RUNNER" in /*) ;; *) RUNNER="$REPO/$RUNNER" ;; esac
[ -f "$RUNNER" ] || { echo "supervisor: no such runner: $RUNNER" >&2; exit 2; }
RUNNER_RE=".*/$(basename "$RUNNER" | sed 's/\./\\./g')"
# Where that runner puts its game workspaces. The politeness check and the
# orphan sweep both walk it, and both were hardcoded to the arm's directory --
# so pointed at any other runner the supervisor would have seen nothing
# pending, killed mid-game, and left the solver burning quota with no parent
# left to write result.json. That is the exact failure the politeness exists
# to prevent.
# **`CCARC3_SWEEP_DIR` moves the driver's directory and must move this one too.**
# `clean_rollouts.py` reads that variable to give a submission sweep its own
# directory; nothing else did. Exported here and left unset elsewhere, the
# supervisor would walk `clean_rollouts` while the driver worked in
# `clean_rollouts_submission` -- seeing nothing pending, killing mid-game, and
# leaving a solver burning quota with no parent to write `result.json`. Which is
# the failure the comment above already describes, arriving through the door that
# was opened to fix something else.
case "$(basename "$RUNNER")" in
    ablate_baselines.py) WORK="ablate_nobaseline";;
    rerun_losses.py)     WORK="rerun_losses";;
    *)                   WORK="${CCARC3_SWEEP_DIR:-clean_rollouts}";;
esac
LOG="$SP/$(basename "$RUNNER" .py).log"
# Repo first: the scratchpad copy is whatever the image snapshot held.
QUOTA="$REPO/tools/quota.sh"; [ -f "$QUOTA" ] || QUOTA="$SP/quota.sh"
LIMIT=0.98          # operator's ceiling, raised from 0.95 on 2026-08-03
RESUME=0.90         # after a *ceiling* stop, wait for util to fall below this
                    # before restarting. Without the hysteresis the loop would
                    # kill the runner at 0.98 and restart it one second later,
                    # churning a game per cycle at the ceiling.
                    #
                    # It applies ONLY to a ceiling stop. A runner that exits
                    # because it finished its queue is restarted whenever util is
                    # under LIMIT -- otherwise extending the queue from 13 to 25
                    # games would strand the extra 12 until the weekly reset,
                    # which is exactly the gap this comment exists to close.

pids_of() {
    local d
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        tr '\0' '\n' < "$d/cmdline" 2>/dev/null | grep -qx "$1" && echo "${d#/proc/}"
    done
}

# **A VOID window reads as 0, not as its last utilization.**
#
# quota.sh derives its readings from `rate_limit_event` lines in solver
# stream.jsonl files, so they only refresh while a solver is running. That is
# fine in flight and deadlocks the moment the arm is parked:
#
#   no solver -> no fresh reading -> util stays 0.98 -> supervisor never starts
#   -> no solver
#
# and the reset does not break it. When `resetsAt` passes, quota.sh marks the
# window VOID in its note but still *prints* the stale `utilization` field, so
# the number stays 0.98 forever while the real window sits at zero. The arm
# would have waited out its whole queue.
#
# VOID means resetsAt is in the past, which means the window rolled over and
# utilization is 0. Reading it that way is what lets the arm resume unattended.
util_now() {
    bash "$QUOTA" 2>/dev/null | awk '/seven_day/ {
        if (index($0, "VOID")) { print "0.00"; exit }
        for (i = 1; i <= NF; i++) if ($i ~ /^util=/) { sub("util=", "", $i); print $i }
    }'
}

ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }

stop_politely() {
    echo "$(date -u +%H:%M) util=$1 >= $LIMIT — stopping after the in-flight game"
    # **Ask whether a solver is running, not whether every directory on disk was
    # tidied.** This counted any workspace with a `trace.jsonl` and no
    # `result.json` as "in flight" — over every attempt that has ever existed,
    # not the games actually running. One abandoned attempt pins that to 1
    # permanently: the loop then burns its full ~2h on every quota stop and kills
    # mid-game regardless, which is the exact outcome the politeness exists to
    # prevent. Zero such directories exist today, so it works and would break the
    # first time a container is replaced mid-write — on this box, hourly.
    #
    # A live solver has its cwd inside the work tree. That is the thing itself.
    for _ in $(seq 1 240); do          # up to ~2h
        pending=0
        for pid in $(pids_of '.*/claude'); do
            c=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
            case "$c" in "$SP"/"$WORK"/*) pending=1;; esac
        done
        [ "$pending" -eq 0 ] && break
        sleep 30
    done
    for p in $(pids_of "$RUNNER_RE"); do
        kill "$p" 2>/dev/null && echo "  stopped runner $p"
    done
    sleep 2
    for pid in $(pgrep -f claude); do
        c=$(readlink "/proc/$pid/cwd" 2>/dev/null)
        case "$c" in
            *"$WORK"/*)
                [ -f "$c/result.json" ] || {
                    kill "$pid" 2>/dev/null && echo "  killed orphaned solver $pid"
                };;
        esac
    done
}

# **Refresh the outbound proxy before every launch, and refuse to launch without
# one that answers.** Outbound HTTPS leaves this box through an agent proxy on a
# loopback port, and that port CHANGES when the session worker restarts. A daemon
# keeps whatever it inherited: on 2026-08-07 this supervisor, started hours
# earlier, still exported 127.0.0.1:37827 while the live proxy had moved to
# :41751 -- so every driver it spawned inherited a dead proxy and died on its
# first `list_games()` with [Errno 111] Connection refused. Retrospectively that
# is also the unexplained incident earlier the same day, where seven games churned
# on Connection refused and the cause was never found.
#
# The session is the only thing that can see the current value, so it writes
# `$SP/proxy_env` and this reads it back. The validation matters as much as the
# refresh: without it a stale proxy produces a driver that starts, fails, exits,
# and is restarted ten minutes later forever, with each cycle writing a fresh
# traceback nobody reads. A launch that cannot possibly work should be a loud
# skip, not a quiet retry.
refresh_proxy() {
    [ -f "$SP/proxy_env" ] && . "$SP/proxy_env"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
           "${HTTPS_PROXY:-http://127.0.0.1:1}/__agentproxy/status" 2>/dev/null)
    if [ "$code" != "200" ]; then
        echo "$(date -u +%H:%M) SKIPPING LAUNCH — proxy ${HTTPS_PROXY:-unset} did not"\
             "answer (got '${code:-no response}'). Every outbound call would fail."\
             "Refresh $SP/proxy_env from a live session."
        return 1
    fi
    return 0
}

start() {
    refresh_proxy || return 1
    echo "$(date -u +%H:%M) util=$1 < $RESUME — starting $(basename "$RUNNER")"
    set -a
    . "$SP/arc3/.env"
    set +a
    cd /home/user/athanor || return
    setsid nohup .venv/bin/python "$RUNNER" >> "$LOG" 2>&1 < /dev/null &
}

ceiling_stopped=0

while true; do
    u=$(util_now)
    running=$(pids_of "$RUNNER_RE")

    # A non-numeric reading ("<threshold") is not a reason to act either way.
    if [ -n "$u" ] && [ "${u#*[0-9]}" != "$u" ]; then
        if [ -n "$running" ] && ge "$u" "$LIMIT"; then
            stop_politely "$u"
            ceiling_stopped=1
        elif [ -z "$running" ]; then
            if [ "$ceiling_stopped" = "1" ]; then
                # held back by hysteresis until the window recovers
                ! ge "$u" "$RESUME" && { start "$u"; ceiling_stopped=0; }
            elif ! ge "$u" "$LIMIT"; then
                # exited on its own -- finished the queue, was recycled, or died.
                # Restart while there is any quota left at all.
                start "$u"
            fi
        fi
    fi
    sleep 600
done
