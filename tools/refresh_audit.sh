#!/usr/bin/env bash
# Has any game finished since the trace-audit page was last built?
#
# Prints nothing and exits 0 when there is nothing new, so an hourly caller can
# treat silence as "no-op". Prints the new game ids and exits 10 when there are.
# Any other exit code is a real failure and should NOT be read as a no-op.
#
# `--record` marks everything currently on disk as refreshed. Run it after a
# successful rebuild+republish. It exists because the alternative was pasting a
# 27-entry `echo` by hand, which is how a malformed marker got written once.
#
# Repo-backed on purpose. The previous version of this script lived only in the
# scratchpad, along with the page generator and the generated page, and all three
# were gone by 2026-08-06 -- the hourly net had been failing silently into a
# missing file. Anything the loop depends on has to survive a container.
set -uo pipefail

SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
# Resolve through symlinks. The standing hourly instruction invokes this by its
# old scratchpad path, which is now a symlink here; without readlink -f, `dirname`
# gives the *symlink's* directory, REPO points into the scratchpad, the marker is
# read from a path that does not exist, and every game reports as new.
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
MARKER="$REPO/evidence/ccarc3/trace_audit/.last_refresh"

# Directories that can hold finished games. rerun_losses is the live one; the
# arm batches are listed because a resumed run can still write into them.
ROOTS=("rerun_losses" "ablate_nobaseline")

# One entry per line, matched with grep -Fxq. The first version stored them
# space-separated on one line and matched with a `case` glob requiring a space on
# each side; an entry that landed at a line boundary had no leading space, never
# matched, and would have been reported as new on every run forever.
stamps() {
  local root dir gid result
  for root in "${ROOTS[@]}"; do
    [ -d "$SP/$root" ] || continue
    for result in "$SP/$root"/*/result.json; do
      [ -e "$result" ] || continue
      dir="$(dirname "$result")"
      gid="$(basename "$dir")"
      # A crashed or clock-interrupted run writes result.json with an "error".
      # That is not a result to score, and the driver will retry it.
      python3 -c "
import json,sys
try: d=json.load(open('$result'))
except Exception: sys.exit(1)
sys.exit(1 if d.get('error') else 0)
" 2>/dev/null || continue
      # Keyed by root as well as game, because the same game_id exists in both
      # the arm batch and the re-run driver and they are *different runs*.
      #
      # Keyed by content, not mtime. Restoring the 13 banked arm results from the
      # trace-audit store rewrote those files with identical content and fresh
      # mtimes, and an mtime key called all 13 new -- a refresh that would have
      # re-ingested and republished 13 unchanged games.
      echo "$root/$gid:$(md5sum < "$result" | cut -c1-12)"
    done
  done
}

if [ "${1:-}" = "--record" ]; then
  mkdir -p "$(dirname "$MARKER")"
  stamps | sort > "$MARKER"
  echo "recorded $(wc -l < "$MARKER") results as refreshed"
  exit 0
fi

touch "$MARKER"
new="$(stamps | sort | grep -Fxv -f "$MARKER" || true)"
[ -z "$new" ] && exit 0

echo "new results since last refresh:"
echo "$new" | sed 's/:.*//; s/^/  /'
echo
echo "rebuild with:"
args=""
while IFS= read -r stamp; do
  key="${stamp%%:*}"          # <root>/<game_id>
  gid="${key##*/}"
  # A re-run is stored under its own id so it sits beside the arm run it
  # supersedes rather than overwriting it in the page.
  if [ "${key%%/*}" = "rerun_losses" ]; then
    args="$args --ingest $SP/${key%%/*}/$gid --as $gid@rerun"
  else
    args="$args --ingest $SP/${key%%/*}/$gid"
  fi
done <<< "$new"
echo "  python3 $REPO/tools/build_trace_audit.py$args"
echo "then republish to artifact 7447856a-b587-4d52-9c5c-a839de3eb6ee"
echo "and record with:  bash $REPO/tools/refresh_audit.sh --record"
exit 10
