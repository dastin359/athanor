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

# **Derived, not written.** The hard-coded `/home/user/athanor` is this
# container's clone location; a fresh container, or the same repo checked out
# by a different account, gets a different one. `readlink -f` first, so the
# derivation still holds if this script is reached through a symlink.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
LOG="$REPO/evidence/box_fingerprint.tsv"
# Overridable via `CCARC3_BRANCH`; a session on another account works on
# another branch. Not derived from HEAD -- see preserve_evidence.sh.
BRANCH="${CCARC3_BRANCH:-claude/athanor-cc-harness-variant-jpqw7t}"
# Overridable via `CCARC3_SCRATCH`: the path encodes a session UUID and the
# container is recycled every 10-50 minutes, so a hard-coded copy points at a
# directory that stops existing. See tools/clean_rollouts.py for the full note.
SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"

mkdir -p "$(dirname "$LOG")"
[ -f "$LOG" ] || printf 'utc\tboot_id\tuptime_s\thead\tscratch_dirs\trerun_present\tnote\n' > "$LOG"

# The `utc` COLUMN stays UTC: it is machine-readable, ISO-8601, explicitly
# Z-suffixed, and every existing row is in it. CLAUDE.md's Pacific rule is about
# reports, status lines and commit messages -- text a person reads and mistakes
# for local time -- so the commit message below gets a Pacific rendering instead.
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
now_local=$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z')
boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)
up=$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null)
head=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
# **`scratch_dirs` counted the RETIRED arm.** It was `ls "$SP/ablate_nobaseline"`,
# so through a submission sweep the column would sit frozen at 25 and say nothing
# about the work actually at risk -- in a row whose entire purpose is noticing
# that a container replacement took the disk with it. Same stale-arm shape as the
# snapshot batch list and the live panel's cwd filter.
#
# It now counts game directories across every work root: the retired arm, the
# live sweep (whatever CCARC3_SWEEP_DIR names) and any rerun_*. The column name
# always read as "work on disk"; this is what it now measures.
#
# **The basis changed on 2026-08-09**, so rows before that date count only
# ablate_nobaseline and are not comparable with later ones. Recorded here because
# a series that silently changes meaning is worse than one that jumps.
dirs=0
for _root in "$SP/ablate_nobaseline" "$SP/${CCARC3_SWEEP_DIR:-clean_rollouts}" "$SP"/rerun_*; do
    [ -d "$_root" ] || continue
    dirs=$(( dirs + $(ls -1 "$_root" 2>/dev/null | wc -l) ))
done
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

# **The exit status has to mean "the row reached origin".** This script's last
# command was `tail -5 "$LOG"`, so its status was `tail`'s — the log file is
# readable — and `rehydrate_box.sh` read that as "fingerprint logged" while a
# failed commit or four failed pushes went unmentioned. A fingerprint row that
# exists only on a box that is about to be replaced is the one case it is for.
pushed=1                     # nothing to push is already-pushed
if [ -n "$(git status --porcelain "$LOG")" ]; then
    pushed=0
    git add "$LOG"
    if git commit -q -m "evidence: box fingerprint $now_local (uptime ${up}s, head $head)"; then
        for i in 1 2 3 4; do
            git push -q origin "$BRANCH" 2>/dev/null && { pushed=1; break; }
            sleep $((2**i))
        done
    else
        echo "fingerprint: COMMIT FAILED — row not recorded" >&2
    fi
fi

tail -5 "$LOG"
[ "$pushed" = 1 ] || {
    echo "fingerprint: PUSH FAILED — the row is local only and will block the" >&2
    echo "  next ff-only merge on this box." >&2
    exit 1
}
