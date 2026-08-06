#!/usr/bin/env bash
# Has any game finished since the trace-audit page was last built?
#
# Prints nothing and exits 0 when there is nothing new, so an hourly caller can
# treat silence as "no-op". Prints the new game ids and exits 10 when there are.
# Any other exit code is a real failure and should NOT be read as a no-op.
#
# Repo-backed on purpose. The previous version of this script lived only in the
# scratchpad, along with the page generator and the generated page, and all three
# were gone by 2026-08-06 -- the hourly net had been failing silently into a
# missing file. Anything the loop depends on has to survive a container.
set -uo pipefail

SP="${CCARC3_SCRATCH:-/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="$REPO/evidence/ccarc3/trace_audit/.last_refresh"

# Directories that can hold finished games. rerun_losses is the live one; the
# arm batches are listed because a resumed run can still write into them.
ROOTS=("$SP/rerun_losses" "$SP/ablate_nobaseline")

seen=""
[ -f "$MARKER" ] && seen="$(cat "$MARKER")"

new=""
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  for result in "$root"/*/result.json; do
    [ -e "$result" ] || continue
    dir="$(dirname "$result")"
    gid="$(basename "$dir")"
    # A crashed run writes result.json with an "error" -- not a result to score.
    if python3 -c "
import json,sys
try: d=json.load(open('$result'))
except Exception: sys.exit(1)
sys.exit(1 if d.get('error') else 0)
" 2>/dev/null; then
      # Keyed by root as well as game, because the same game_id exists in both
      # the arm batch and the re-run driver and they are *different runs*.
      # Collapsing them would hide a re-run behind the arm result it replaced.
      stamp="$(basename "$root")/$gid:$(stat -c %Y "$result")"
      case " $seen " in
        *" $stamp "*) ;;
        *) new="$new $stamp" ;;
      esac
    fi
  done
done

new="$(echo "$new" | xargs -n1 2>/dev/null | sort | xargs 2>/dev/null)"
[ -z "$new" ] && exit 0

echo "new results since last refresh:"
for stamp in $new; do echo "  ${stamp%%:*}"; done
echo
echo "rebuild with:"
args=""
for stamp in $new; do
  key="${stamp%%:*}"          # <root>/<game_id>
  root="$SP/${key%%/*}"
  gid="${key##*/}"
  # A re-run is stored under its own id so it sits beside the arm run it
  # supersedes rather than overwriting it in the page.
  if [ "${key%%/*}" = "rerun_losses" ]; then
    args="$args --ingest $root/$gid --as $gid@rerun"
  else
    args="$args --ingest $root/$gid"
  fi
done
echo "  python3 $REPO/tools/build_trace_audit.py$args"
echo "then republish to artifact 7447856a-b587-4d52-9c5c-a839de3eb6ee"
echo "and record: echo '$(echo "$seen $new" | xargs)' > $MARKER"
exit 10
