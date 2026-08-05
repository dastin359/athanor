#!/bin/bash
# Append one line identifying the container we are on, then push it.
#
# **Why.** This session lost work five times to what looked like filesystem
# rollbacks: the checkout kept reappearing at commit bf90a03 with a scratchpad
# holding exactly 13 run directories. Every occurrence was caught by accident --
# some command failed -- and no timestamps were kept, so "how often does this
# happen" and "can it be predicted" were both unanswerable.
#
# Two hypotheses worth separating, which this distinguishes:
#
#   replacement   the container is destroyed and rebuilt, cloning the repo fresh
#                 at whatever origin's default branch points to. Predicts a NEW
#                 boot_id and a small uptime after each event.
#   alternation   the session is being routed between two long-lived boxes, one
#                 carrying the work and one frozen. Predicts boot_ids that
#                 REPEAT, alternating between a stable set.
#
# One observation already favours replacement: a box investigated mid-session had
# been up 1 day 8 hours, and a later one 39 minutes. But a single pair is not a
# series, and the distinction matters -- replacement can only be survived by
# pushing, whereas alternation could be detected and pinned.
#
# Run from any autopilot turn. Cheap, append-only, and lives in the repo so it
# survives the very event it measures.
set -u

REPO="/home/user/athanor"
LOG="$REPO/evidence/box_fingerprint.tsv"
BRANCH="claude/athanor-cc-harness-variant-jpqw7t"
SP="/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"

mkdir -p "$(dirname "$LOG")"
[ -f "$LOG" ] || printf 'utc\tboot_id\tuptime_s\thead\tscratch_dirs\trerun_present\tnote\n' > "$LOG"

now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
up=$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null)
head=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
dirs=$(ls -1 "$SP/ablate_nobaseline" 2>/dev/null | wc -l)
rerun=$([ -d "$SP/rerun_losses" ] && echo yes || echo no)

# A boot_id we have logged before means the same kernel boot -- so the session
# came back to a box it had already been on, which is alternation, not
# replacement. Worth flagging in the row itself rather than left to be noticed.
note="-"
if [ -f "$LOG" ] && grep -q "	$boot	" "$LOG" 2>/dev/null; then
    prev=$(grep -c "	$boot	" "$LOG")
    note="seen_boot_id_before(x$prev)"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$now" "$boot" "$up" "$head" "$dirs" "$rerun" "$note" >> "$LOG"

cd "$REPO" || exit 1
if [ -n "$(git status --porcelain "$LOG")" ]; then
    git add "$LOG"
    git commit -q -m "evidence: box fingerprint $now (uptime ${up}s, head $head)"
    for i in 1 2 3 4; do
        git push -q origin "$BRANCH" 2>/dev/null && break
        sleep $((2**i))
    done
fi

tail -5 "$LOG"
