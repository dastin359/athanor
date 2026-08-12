#!/usr/bin/env bash
# Bootstrap for tools/rehydrate_box.sh. Seed a copy into the scratchpad; keep
# this one in the repo as the copy of record.
#
# A container replacement rewinds the working tree to an image snapshot, deleting
# tools/rehydrate_box.sh along with everything else added since, so the recovery
# script cannot restore itself. These are the two git commands that fetch it back.
#
# **The scratchpad is not a safe haven, and an earlier version of this comment
# said it was.** It claimed "the scratchpad has survived every replacement so far;
# the repo tree has survived none." The next replacement falsified it: AUTOPILOT.md
# and quota.sh were still there, and this file — written forty minutes earlier —
# was gone. The scratchpad reverts to a snapshot exactly like the repo tree does;
# older files survive because they predate it, not because the directory is
# durable. `rerun_losses/` disappeared the same way.
#
# So there is no on-disk location that reliably survives a replacement, and the
# honest recovery procedure is: run these two git commands by hand, because
# **origin is the only store that has never lost anything.** Seeding a copy here
# still helps within one box's lifetime; it just cannot be relied on across one.
#
# Deliberately minimal: fetch, fast-forward, hand off. Everything else belongs
# in tools/rehydrate_box.sh where it is version-controlled.
set -uo pipefail
# **Not derived from `${BASH_SOURCE[0]}` here, unlike every other tool.**
# This file exists to be COPIED into the scratchpad, so the derivation would
# resolve to wherever the copy sits. An explicit default with an env override
# is the honest form for a script whose whole purpose is to run from outside
# the tree it is repairing.
REPO="${CCARC3_REPO:-/home/user/athanor}"
# Overridable via `CCARC3_BRANCH`; a session on another account works on
# another branch. Not derived from HEAD -- see preserve_evidence.sh.
BRANCH="${CCARC3_BRANCH:-codexarc3}"
cd "$REPO" || exit 1
for delay in 0 2 4 8 16; do
  [ "$delay" = 0 ] || sleep "$delay"
  git fetch origin "$BRANCH" -q 2>/dev/null && break
done
git merge --ff-only "origin/$BRANCH" -q 2>/dev/null

# A rolled-back HEAD is fixed by the merge above. A file merely missing from the
# working tree is not -- the merge is a no-op when HEAD already contains it, so
# the checkout is what covers that case. Scoped to the one path on purpose: a
# blanket `git checkout -- .` would silently discard uncommitted work.
[ -f "$REPO/tools/rehydrate_box.sh" ] || git checkout HEAD -- tools/rehydrate_box.sh 2>/dev/null

if [ ! -f "$REPO/tools/rehydrate_box.sh" ]; then
  echo "rehydrate: tools/rehydrate_box.sh missing and not recoverable from HEAD" >&2
  exit 1
fi
exec bash "$REPO/tools/rehydrate_box.sh" "$@"
