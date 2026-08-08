#!/bin/bash
# Heartbeat over the active runner: snapshot results, emit progress, exit when it
# ends.
#
# **The header said "over batch6" long after batch6 was retired**, and the body
# note twelve lines down says exactly that -- the file contradicted itself top to
# bottom. batch6 survives here only as the name of the argv-matching bug below,
# which is worth keeping because it is the bug, not the batch.
#
# **Liveness is checked per-argv-element, not by substring.** `pgrep -f batch6.py`
# and `case "$cmdline" in *bin/python*batch6.py*)` both match this very script,
# whose source contains both strings -- so the watcher sees itself, concludes the
# batch is alive, and loops forever after batch6 has exited. That exact bug has
# now bitten this project three times (once as a two-process deadlock). Splitting
# /proc/<pid>/cmdline on NUL and requiring a *whole argv element* to be a path
# ending in batch6.py fixes it: this script's own command line is one big `-c`
# string, which can never equal such a path. Verified: the substring form matched
# 3 pids (including a transient shell of the monitor's own), this form matched 1.
SP=/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad

# **Watch whatever the supervisor actually launches, not a name from last week.**
# This matched `ablate_baselines.py` long after `clean_rollouts.py` became the
# runner, so `runner_alive` was permanently false -- and the loop below reads
# false as "the arm ended", prints four lines of a log last written days ago, and
# breaks on its FIRST poll. A heartbeat that exits immediately is not a heartbeat
# that reports nothing wrong; it is a heartbeat that cannot report at all, and it
# looked identical to a healthy one from the outside.
#
# The name comes from the supervisor rather than being written twice: the two
# drifted apart once and the drift was invisible precisely because this file kept
# answering.
#
# batch6 survives below only as the name of the argv-matching bug, which is worth
# keeping because it is the bug, not the batch.
RUNNER_NAME=$(grep -oE 'RUNNER="\$\{1:-\$REPO/tools/[a-z_]+\.py\}"' \
                   /home/user/athanor/tools/supervisor.sh 2>/dev/null \
              | grep -oE '[a-z_]+\.py' | head -1)
RUNNER_NAME="${RUNNER_NAME:-clean_rollouts.py}"

runner_alive() {
    local d
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        if tr '\0' '\n' < "$d/cmdline" 2>/dev/null \
           | grep -qx ".*/${RUNNER_NAME//./\\.}"; then
            return 0
        fi
    done
    return 1
}

# **"No runner" is the steady state between games, not an ending.** The old loop
# broke the moment `runner_alive` was false, which after the arm finished meant
# on the first poll, every time. The supervisor starts a runner within ten
# minutes whenever there is work and quota; absence for one poll says nothing.
# Only absence sustained past that window, with work still pending, is a report.
IDLE_POLLS=0
IDLE_LIMIT=3            # 3 x 420s = 21 min, twice the supervisor's 10-min cycle
ANNOUNCED_COMPLETE=0

# **The daemons that have to outlive this script.** Detached work here is mortal
# -- a container restart killed the supervisor once, 22 minutes before a quota
# reset -- and nothing else notices. Checked argv-element-exact for the reason
# given above: a substring match finds this script, which names all three.
daemon_check() {
    local d name argv
    local -A seen=()
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        argv=$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null)
        for name in supervisor.sh preserve_evidence.sh context_watch.py; do
            grep -qx ".*/${name//./\\.}" <<< "$argv" && seen[$name]=1
        done
    done
    for name in supervisor.sh preserve_evidence.sh context_watch.py; do
        [ -n "${seen[$name]}" ] || \
            echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') DAEMON DOWN: $name"
    done
}

# **Ask the driver where it works; do not rebuild the path.** Reconstructing it
# as $REPO/$CCARC3_SWEEP_DIR looked right and pointed at a directory that does
# not exist -- so `banked` read 0 against 25 games and "work pending" would have
# been true forever, on a sweep that finished days ago. The driver resolves OUT
# from the scratchpad, and it is the only thing that knows. Same defect as the
# one this file is being fixed for: a value that agrees with the truth in the
# environment you happen to test it in.
work_pending() {
    /home/user/athanor/.venv/bin/python - <<'EOF' 2>/dev/null
import sys
sys.path[:0] = ["/home/user/athanor/tools", "/home/user/athanor/src"]
try:
    import clean_rollouts as cr
except Exception:
    sys.exit(0)                      # cannot tell -> assume work, stay watching
banked = len(list(cr.OUT.glob("*/clean_result.json")))
sys.exit(0 if banked < len(cr.GAMES) else 1)
EOF
}

while true; do
    ARC_API_KEY=dummy /home/user/athanor/.venv/bin/python "$SP/snapshot_results.py" \
        >/dev/null 2>&1 || true

    if ! runner_alive; then
        IDLE_POLLS=$((IDLE_POLLS + 1))
        if ! work_pending; then
            # **Say it once, then hold.** Breaking here was correct as a report
            # and wrong as a heartbeat: this process is also what keeps the
            # session and its VM alive, so exiting three seconds after arming
            # left nothing running and produced a re-arm on every turn. An alarm
            # that fires immediately and ends is one you learn to ignore.
            #
            # So the idle state is quiet, not absent: the daemons are still
            # checked every poll, and anything that changes still wakes someone.
            [ "$ANNOUNCED_COMPLETE" = "1" ] || {
                echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') SWEEP COMPLETE — every game banked; holding, will report daemon loss or new work"
                ANNOUNCED_COMPLETE=1
            }
            daemon_check
            sleep 420
            continue
        fi
        ANNOUNCED_COMPLETE=0
        if [ "$IDLE_POLLS" -ge "$IDLE_LIMIT" ]; then
            echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') NO RUNNER for $((IDLE_POLLS * 7)) min with work pending — supervisor is not starting one"
            IDLE_POLLS=0
        fi
        sleep 420
        continue
    fi
    IDLE_POLLS=0

    # Emit progress *and* the signals worth waking for. A filter that only ever
    # reports forward movement is silent through a stall, which reads identically
    # to healthy running.
    #
    # **Every field here read the retired arm.** `arm`, `fail`, `d`, `n`, `age`,
    # `done_n` and `total` all came off `$SP/ablate.log` and
    # `$SP/ablate_nobaseline/`, last written 2026-08-06/07. The commit that
    # repointed `runner_alive` at the live runner left this block behind, so the
    # emitted line described a sweep that had already ended -- pinned at
    # `[25/25] wa30: already finished, skipping | 0 acts (?s ago) | 25/25 done`
    # whatever the live runner was doing. `fail` was the worst of them: the one
    # alarm this block exists for, grepping a file nothing writes to.
    #
    # The log is derived the way supervisor.sh derives it (`$SP/<runner>.log`),
    # and the progress counts come from the driver itself rather than from a log
    # format. Both are the thing rather than a proxy for it.
    LOG="$SP/$(basename "$RUNNER_NAME" .py).log"

    # **Anchored, because the obvious pattern is a permanent false alarm.**
    # Bare `429` matches ephemeral proxy ports and any scorecard UUID containing
    # those digits: on the live log that is 32 hits, of which ZERO are HTTP 429s,
    # and the newest is the healthy startup line
    # `arc_proxy: http://127.0.0.1:44297 (key withheld...)`. Since `tail -1`
    # takes the newest match, an unanchored pattern would print that every poll
    # and bury the 36 real tracebacks underneath it. Bare `quota` is the same
    # mistake: it appears in ordinary quota accounting. A status line that cries
    # failure continuously is no better than one that never does.
    fail=$(grep -E 'Traceback \(most recent call last\)|FAILED|PROOFREAD DID NOT RUN|aborting:|rate.?limit|out_of_credits|HTTP Error 429|\b429[: ]+(Too Many|Client Error)|status[ =:]+429' \
                "$LOG" 2>/dev/null | tail -1)

    # The formats clean_rollouts.py actually writes (:509 and :787).
    arm=$(grep -E '^(=== |--- pass )' "$LOG" 2>/dev/null | tail -1)

    read -r done_n total < <(/home/user/athanor/.venv/bin/python - <<'EOF' 2>/dev/null
import sys
sys.path[:0] = ["/home/user/athanor/tools", "/home/user/athanor/src"]
try:
    import clean_rollouts as cr
    print(len(list(cr.OUT.glob("*/clean_result.json"))), len(cr.GAMES))
except Exception:
    print("?", "?")
EOF
)
    done_n=${done_n:-?}; total=${total:-?}

    d=$(/home/user/athanor/.venv/bin/python -c '
import sys; sys.path[:0]=["/home/user/athanor/tools","/home/user/athanor/src"]
try:
    import clean_rollouts as cr
    ws=sorted(cr.OUT.glob("*/attempt_*/*/trace.jsonl"), key=lambda p: p.stat().st_mtime)
    print(ws[-1].parent if ws else "")
except Exception:
    print("")' 2>/dev/null)
    n=0
    [ -n "$d" ] && [ -f "$d/trace.jsonl" ] && n=$(wc -l < "$d/trace.jsonl")
    # Age of the newest trace write. A long age is not proof of a stall: a solver
    # reasoning over the trace writes the stream, not the ledger. Check the
    # process before concluding anything from it.
    age="?"
    [ -n "$d" ] && [ -f "$d/trace.jsonl" ] && age=$(( $(date +%s) - $(stat -c %Y "$d/trace.jsonl") ))
    q=$(bash "$SP/quota.sh" 2>/dev/null | grep -E 'five_hour|seven_day' | tr -s ' ' | tr '\n' ';')

    echo "$(TZ=America/Los_Angeles date '+%H:%M %Z') ${arm:-starting} | ${n} acts (${age}s ago) | ${done_n}/${total} done | ${q}${fail:+ | LAST FAILURE: $fail}"
    sleep 420
done
