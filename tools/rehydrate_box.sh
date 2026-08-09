#!/usr/bin/env bash
# Put a freshly-replaced container back into a working state.
#
# Containers here are *replaced*, not restarted: a new boot_id, a working tree
# rewound to whatever the image snapshot held, and a scratchpad missing whatever
# was written since. On 2026-08-06 this happened three times in 45 minutes, each
# needing the same four steps by hand. This is those four steps.
#
# Every one of them is idempotent, so running it on a box that is already healthy
# does nothing and costs nothing. Run it first thing on any autopilot wake-up.
#
#   1. Fast-forward the repo from origin. The tree rewinds to the image commit
#      (bf90a03 as of this writing) and origin is always ahead -- everything
#      durable is pushed, so origin is the source of truth, never the local tree.
#   2. Re-point the scratchpad symlink the hourly refresh instruction invokes.
#   3. Log the box fingerprint, so replacement cadence stays measurable.
#   4. Restore the banked arm results, without which the arm runner would treat
#      13 finished games as unstarted. See tools/restore_banked_results.py.
#
# Then report what is running and whether anything finished, which is what the
# wake-up needs to decide next.
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
# Overridable via `CCARC3_BRANCH`; a session on another account works on
# another branch. Not derived from HEAD -- see preserve_evidence.sh.
BRANCH="${CCARC3_BRANCH:-claude/athanor-cc-harness-variant-jpqw7t}"
SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
cd "$REPO" || exit 1

echo "box $(cat /proc/sys/kernel/random/boot_id) up $(cut -d' ' -f1 /proc/uptime)s"

# 1. Repo. Retry the fetch: a replaced box often races container networking.
#
# **"already current" used to cover three different outcomes.** HEAD is
# unchanged when the tree really is up to date, when all five fetch retries
# failed (the networking race the retry loop exists for), and when the ff-only
# merge was REFUSED because HEAD had diverged — unpushed local commits, which is
# precisely the state a rollback recovery has to notice. Both commands sent their
# errors to /dev/null and neither status was tested, so the one line printed was
# "already current" in all three, and the recovery step that this script's own
# header calls the load-bearing one reported success for having done nothing.
fetched=0
for delay in 0 2 4 8 16; do
  [ "$delay" = 0 ] || sleep "$delay"
  git fetch origin "$BRANCH" -q 2>/dev/null && { fetched=1; break; }
done
[ "$fetched" = 1 ] || echo "repo   FETCH FAILED after 5 tries — origin unreachable; the tree is NOT known current"

before="$(git rev-parse --short HEAD)"
if git merge --ff-only "origin/$BRANCH" -q 2>/dev/null; then
  after="$(git rev-parse --short HEAD)"
  if [ "$before" = "$after" ]; then
    [ "$fetched" = 1 ] && echo "repo   $after (already current)"
  else
    echo "repo   $before -> $after (recovered from rollback)"
  fi
else
  echo "repo   FF-ONLY REFUSED — HEAD $before has diverged from origin/$BRANCH"
  echo "       (unpushed local commits). The tree is NOT current and this script"
  echo "       will not resolve it: rebase or push by hand."
fi

# 2. Standing instructions call these by their old scratchpad paths.
#
# **Symlinks, not copies, and this is why.** `heartbeat.sh` was a copy here and
# drifted five days behind the repo: the scratchpad version still watched
# `ablate_baselines.py` after `clean_rollouts.py` became the runner, so
# `runner_alive` was permanently false, the loop read that as "the arm ended",
# and it broke on its first poll while looking exactly like a healthy heartbeat.
# The scratchpad reverts to an image snapshot when the container is replaced --
# CLAUDE.md records that as the reason a rule kept only there is a rule that
# expires -- and the same is true of a script. A symlink cannot drift.
mkdir -p "$SP"
# `snapshot_results.py` joined this list on 2026-08-09. It had lived ONLY in
# the scratchpad -- the one store that reverts -- which is the worst possible
# home for the script whose whole purpose is surviving data loss.
for tool in refresh_audit.sh heartbeat.sh snapshot_results.py; do
    ln -sfn "$REPO/tools/$tool" "$SP/$tool"
done

# **The standing agenda and the memory index live in the repo now.** Both were
# scratchpad-only until 2026-08-08, and `rehydrate_bootstrap.sh` had already
# written down why that is unsafe: there is no on-disk location that reliably
# survives a replacement, and origin is the only store that has never lost
# anything. AUTOPILOT.md came through one replacement by predating the snapshot
# -- luck, not durability -- and it is the file every wake-up is told to read, so
# losing it or letting it go stale is what makes an unattended session act on
# work that finished days ago.
ln -sfn "$REPO/docs/ccarc3_autopilot.md" "$SP/AUTOPILOT.md"
ln -sfn "$REPO/docs/ccarc3_memory.md"    "$SP/MEMORY.md"
ln -sfn "$REPO/tools/quota.sh"           "$SP/quota.sh"

# 3+4. Fingerprint and banked results.
#
# **Failures are reported, not swallowed.** These two restores are what stop a
# replacement from presenting finished games as unstarted, so a driver relaunched
# after one silently re-runs them -- the exact failure the last block of this
# script refuses to tolerate from `refresh_audit.sh`. `2>/dev/null` here read as
# "nothing to restore" whether that was true or the script had crashed.
# **`&& echo "fingerprint logged"` tested `tail`, not the fingerprint.**
# box_fingerprint.sh runs without `set -e` or `pipefail` and ends with
# `tail -5 "$LOG"`, so its exit status was "the log file is readable" — a failed
# commit and four failed pushes both left it 0. It exits on its own success now.
if bash "$REPO/tools/box_fingerprint.sh" >/dev/null 2>&1; then
  echo "fingerprint logged"
else
  echo "fingerprint NOT recorded — the row is local only, or was never written"
fi
for restore in restore_banked_results restore_clean_rollouts; do
  # The rollout driver reads its banked markers off the scratchpad too, and
  # restores only attempts that finished without an error.
  if out="$(python3 "$REPO/tools/$restore.py" 2>&1)"; then
    echo "$out" | head -1
  else
    echo "restore: $restore.py FAILED -- finished games may look unstarted"
    echo "$out" | tail -3
  fi
done

# What is running, argv-element-exact -- a substring match finds this script.
# `clean_rollouts.py` is the driver actually in use and was missing from this
# list, so a wake-up after a replacement could not see whether it had survived.
# `supervisor.sh` stays: it is the quota duty-cycle loop and the first thing the
# autopilot check asks about. (It lives in `tools/` now -- this parenthesis used
# to say it did not, which was true when written and stopped being true without
# the line changing. A comment that contradicts the tree beside it is the same
# defect as a check that reads a proxy: both keep answering after the thing they
# describe has moved.)
for name in clean_rollouts.py ablate_baselines.py rerun_losses.py preserve_evidence.sh supervisor.sh; do
  n=0
  for d in /proc/[0-9]*; do
    [ -r "$d/cmdline" ] || continue
    tr '\0' '\n' < "$d/cmdline" 2>/dev/null | grep -qx ".*/${name}\|.*${name}" && n=$((n+1))
  done
  [ "$n" -gt 0 ] && echo "running: $name ($n)"
done

bash "$REPO/tools/refresh_audit.sh"
case $? in
  0) echo "refresh: nothing new" ;;
  10) echo "refresh: NEW RESULTS above — rebuild and republish" ;;
  *) echo "refresh: FAILED — investigate, do not read as a no-op" ;;
esac
