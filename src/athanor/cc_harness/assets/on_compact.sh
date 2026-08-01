#!/usr/bin/env bash
# SessionStart(compact) hook — Athanor CC harness.
#
# Claude Code compacts context on its own schedule. Athanor's ICAE mechanism
# handles the same moment by distilling research state into a checkpoint and
# resuming from it. This variant keeps that state on disk continuously, so
# recovery is just a matter of replaying it into the fresh context window.
set -uo pipefail

cd "__WORKSPACE__" 2>/dev/null || exit 0

echo "--- Athanor CC harness: context was compacted. Distilled research state follows. ---"
echo
"__PYTHON__" gate.py status 2>&1 || true
echo
echo "Re-read CLAUDE.md and the tail of NOTES.md before doing anything else."
echo "Exploratory Python state did not survive the compaction; anything you need"
echo "must be re-derived by running a script under explore/."
