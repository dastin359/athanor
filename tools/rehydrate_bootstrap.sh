#!/usr/bin/env bash
# Bootstrap for tools/rehydrate_box.sh, kept OUTSIDE the repo on purpose.
#
# A container replacement rewinds the working tree to the image commit, which
# deletes tools/rehydrate_box.sh along with everything else added since. The
# recovery script cannot restore itself, so the two git commands that fetch it
# back have to live somewhere the rollback does not reach. The scratchpad has
# survived every replacement so far; the repo tree has survived none.
#
# Deliberately minimal: fetch, fast-forward, hand off. Everything else belongs
# in the repo copy where it is version-controlled.
set -uo pipefail
REPO=/home/user/athanor
BRANCH=claude/athanor-cc-harness-variant-jpqw7t
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
