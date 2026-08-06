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
BRANCH=claude/athanor-cc-harness-variant-jpqw7t
SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
cd "$REPO" || exit 1

echo "box $(cat /proc/sys/kernel/random/boot_id) up $(cut -d' ' -f1 /proc/uptime)s"

# 1. Repo. Retry the fetch: a replaced box often races container networking.
for delay in 0 2 4 8 16; do
  [ "$delay" = 0 ] || sleep "$delay"
  git fetch origin "$BRANCH" -q 2>/dev/null && break
done
before="$(git rev-parse --short HEAD)"
git merge --ff-only "origin/$BRANCH" -q 2>/dev/null
after="$(git rev-parse --short HEAD)"
if [ "$before" = "$after" ]; then
  echo "repo   $after (already current)"
else
  echo "repo   $before -> $after (recovered from rollback)"
fi

# 2. The hourly instruction calls refresh_audit.sh by its old scratchpad path.
mkdir -p "$SP"
ln -sf "$REPO/tools/refresh_audit.sh" "$SP/refresh_audit.sh"

# 3+4. Fingerprint and banked results.
bash "$REPO/tools/box_fingerprint.sh" >/dev/null 2>&1 && echo "fingerprint logged"
python3 "$REPO/tools/restore_banked_results.py" 2>/dev/null | head -1
# Same problem, different experiment: the rollout driver reads its banked
# markers off the scratchpad, so a replacement makes finished games look
# unstarted. Restores only attempts that finished without an error.
python3 "$REPO/tools/restore_clean_rollouts.py" 2>/dev/null | head -1

# What is running, argv-element-exact -- a substring match finds this script.
for name in supervisor.sh ablate_baselines.py rerun_losses.py preserve_evidence.sh; do
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
