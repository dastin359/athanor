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

# **Baseline-free arm only.** batch6.py was retired on operator instruction --
# "all runs after the 13th scored game must be baseline free", "no control arm
# from now on" -- so nothing here should watch it. Reporting a control line after
# it stopped was actively misleading: it kept printing the last batch6.log entry
# alongside a stale trace from a different game.
runner_alive() {
    local d
    for d in /proc/[0-9]*; do
        [ -r "$d/cmdline" ] || continue
        if tr '\0' '\n' < "$d/cmdline" 2>/dev/null | grep -qx '.*/ablate_baselines\.py'; then
            return 0
        fi
    done
    return 1
}

while true; do
    ARC_API_KEY=dummy /home/user/athanor/.venv/bin/python "$SP/snapshot_results.py" \
        >/dev/null 2>&1 || true

    if ! runner_alive; then
        echo "BASELINE-FREE ARM ENDED >>> $(tail -n 4 "$SP/ablate.log" 2>/dev/null | tr '\n' ' ')"
        break
    fi

    # Emit progress *and* the signals worth waking for. A filter that only ever
    # reports forward movement is silent through a stall, which reads identically
    # to healthy running.
    arm=$(grep -E '^\[[0-9]+/' "$SP/ablate.log" 2>/dev/null | tail -1)
    fail=$(grep -E 'FAILED|Traceback|quota|rate.?limit|429' "$SP/ablate.log" 2>/dev/null | tail -1)
    d=$(ls -dt "$SP"/ablate_nobaseline/*/ 2>/dev/null | head -1)
    n=0
    [ -n "$d" ] && [ -f "$d/trace.jsonl" ] && n=$(wc -l < "$d/trace.jsonl")
    # Age of the newest trace write. A long age is not proof of a stall: a solver
    # reasoning over the trace writes the stream, not the ledger. Check the
    # process before concluding anything from it.
    age="?"
    [ -n "$d" ] && [ -f "$d/trace.jsonl" ] && age=$(( $(date +%s) - $(stat -c %Y "$d/trace.jsonl") ))
    done_n=$(ls "$SP"/ablate_nobaseline/*/result.json 2>/dev/null | wc -l)
    # Total comes from the runner's own log line, not a constant: the queue grew
    # from 13 to 25 and a hardcoded denominator silently understated progress.
    total=$(echo "$arm" | grep -oE '^\[[0-9]+/[0-9]+\]' | grep -oE '[0-9]+\]' | tr -d ']')
    total=${total:-?}
    q=$(bash "$SP/quota.sh" 2>/dev/null | grep seven_day | tr -s ' ')

    echo "baseline-free: ${arm:-starting} | ${n} acts (${age}s ago) | ${done_n}/${total} done |${q}${fail:+ | LAST FAILURE: $fail}"
    sleep 420
done
