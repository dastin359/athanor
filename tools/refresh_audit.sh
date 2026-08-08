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

# Directories that can hold finished games, in the flat
# <root>/<game_id>/result.json layout. The arm batch is listed because a resumed
# run can still write into it.
#
# **`rerun_losses` used to be hard-coded here and that is a bug pattern, not a
# list.** Ad-hoc re-run directories get created per investigation -- the bp35
# resume of 2026-08-08 went into `rerun_bp35_fixed` -- and a hard-coded name
# means the net goes blind to the only run in flight, reporting "nothing new" at
# exactly the moment it exists to fire. Caught on 2026-08-08 with that run live.
# Any `rerun_*` directory is now globbed instead, in BOTH layouts, below.
ROOTS=("ablate_nobaseline")
# clean_rollouts is handled separately: its finished runs are marked by a banked
# clean_result.json at the game directory, while the run itself lives a further
# two levels down at <game>/attempt_N/<game>/. A plain <root>/*/result.json glob
# matches neither, so three finished clean rollouts were invisible here and the
# hourly check reported "nothing new" while the artifact had never seen them.
CLEAN_ROOT="clean_rollouts"

# One entry per line, matched with grep -Fxq. The first version stored them
# space-separated on one line and matched with a `case` glob requiring a space on
# each side; an entry that landed at a line boundary had no leading space, never
# matched, and would have been reported as new on every run forever.
# Hash the result's *content*, not its bytes.
#
# A plain md5 of the file makes the check sensitive to JSON formatting, and that
# is not hypothetical: after a container replacement the banked results were
# restored from evidence, which rewrote them with different indentation than the
# driver had used. Three already-published games re-flagged as new, and a refresh
# that re-ingests and republishes unchanged games is wasted work that also looks
# like a real event in the log. Canonicalising with sorted keys makes the key
# depend on what the run actually did.
canon_hash() {
  python3 -c "
import hashlib, json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(hashlib.md5(open(sys.argv[1],'rb').read()).hexdigest()[:12]); raise SystemExit
blob = json.dumps(d, sort_keys=True, separators=(',', ':')).encode()
print(hashlib.md5(blob).hexdigest()[:12])
" "$1" 2>/dev/null || md5sum < "$1" | cut -c1-12
}

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
      echo "$root/$gid:$(canon_hash "$result")"
    done
  done
  # Ad-hoc re-run directories, whatever they are called this week.
  #
  # Two layouts, because both occur: `rerun_losses` wrote
  # <root>/<gid>/result.json, while a `run_game(out_dir=X)` resume writes the
  # nested <root>/<gid>/attempt_N/<gid>/result.json. Matching only the flat one
  # is how the bp35 resume would have finished unnoticed.
  local rroot rname
  for rroot in "$SP"/rerun_*/; do
    [ -d "$rroot" ] || continue
    rname="$(basename "$rroot")"
    for result in "$rroot"*/result.json "$rroot"*/attempt_*/*/result.json; do
      [ -e "$result" ] || continue
      python3 -c "
import json,sys
try: d=json.load(open('$result'))
except Exception: sys.exit(1)
sys.exit(1 if d.get('error') else 0)
" 2>/dev/null || continue
      # The game id is the <gid> directory directly under the root, so the two
      # layouts produce the SAME key and the existing marker stays valid --
      # rewriting the key format would re-flag all 50 recorded results as new.
      gid="$(echo "${result#$rroot}" | cut -d/ -f1)"
      echo "$rname/$gid:$(canon_hash "$result")"
    done
  done

  # Banked clean rollouts.
  for result in "$SP/$CLEAN_ROOT"/*/clean_result.json; do
    [ -e "$result" ] || continue
    gid="$(basename "$(dirname "$result")")"
    echo "$CLEAN_ROOT/$gid:$(canon_hash "$result")"
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
  case "${key%%/*}" in
    rerun_*)
      # Flat layout first, then the nested one a `run_game` resume produces.
      rroot="$SP/${key%%/*}"
      if [ -f "$rroot/$gid/result.json" ]; then
        args="$args --ingest $rroot/$gid --as $gid@rerun"
      else
        for a in "$rroot/$gid"/attempt_*/"$gid"; do
          [ -f "$a/result.json" ] || continue
          args="$args --ingest $a --as $gid@rerun"; break
        done
      fi ;;
    clean_rollouts)
      # Point at the attempt whose workspace holds the clean result.
      for a in "$SP/$CLEAN_ROOT/$gid"/attempt_*/"$gid"; do
        [ -f "$a/result.json" ] || continue
        python3 -c "
import json,sys
sys.exit(1 if json.load(open('$a/result.json')).get('error') else 0)" 2>/dev/null \
          && { args="$args --ingest $a --as $gid@clean"; break; }
      done ;;
    *)
      args="$args --ingest $SP/${key%%/*}/$gid" ;;
  esac
done <<< "$new"
echo "  python3 $REPO/tools/build_trace_audit.py$args"
echo "then republish to artifact 7447856a-b587-4d52-9c5c-a839de3eb6ee"
echo "and record with:  bash $REPO/tools/refresh_audit.sh --record"
exit 10
