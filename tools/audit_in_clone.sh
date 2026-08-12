#!/usr/bin/env bash
# Run the mutation battery against a throwaway git worktree, never the live tree.
#
# **Why this exists.** `mutation_check` edits the source file in place, so a
# battery that runs for an hour leaves the working tree dirty for an hour. That
# collides with this repo's standing rule that changes are committed and pushed:
# the dirt looks exactly like unfinished work, and the one resolution that must
# never be taken -- committing it -- publishes deliberately broken source. It
# came up for real on 2026-08-10, with `grids.py` carrying a live `palette_low`
# mutant at the moment a commit was requested.
#
# The harness's own restore-on-signal is sound and is not the issue. The issue is
# that "is the tree clean?" stops being answerable while an audit runs, and that
# question is load-bearing for everything else.
#
# A worktree costs a few hundred milliseconds and removes the conflict outright:
# mutants land on a copy that is deleted afterwards, and the live tree is never
# touched, so the audit can run as long as it likes.
#
#   bash tools/audit_in_clone.sh              # every module
#   bash tools/audit_in_clone.sh proxy grids  # named modules
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
WT="$(mktemp -d "${TMPDIR:-/tmp}/ccarc3_audit_XXXXXX")"

cleanup() {
    cd "$REPO" || exit
    git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
}
trap cleanup EXIT INT TERM

# Detached at HEAD: the audit scores what is committed, which is also what a
# reviewer would read. Uncommitted work is deliberately NOT audited -- scoring a
# tree nobody else can see is how a green result stops meaning anything.
if ! git -C "$REPO" worktree add --detach "$WT" HEAD >/dev/null 2>&1; then
    echo "audit_in_clone: could not create a worktree at $WT" >&2
    exit 3
fi

# `mutation_check` resolves PYTEST as `<repo>/.venv/bin/pytest`, and a worktree
# has no `.venv`. Symlink rather than copy: the interpreter is read-only here and
# a real copy would cost more than the audit.
ln -sfn "$REPO/.venv" "$WT/.venv"

if [ -z "$(git -C "$REPO" status --porcelain)" ]; then
    echo "audit_in_clone: live tree clean; auditing HEAD ($(git -C "$REPO" rev-parse --short HEAD))"
else
    # Loud, because the difference matters: a mutant that the uncommitted work
    # would have killed still reads as a survivor here.
    echo "audit_in_clone: NOTE — the live tree has uncommitted changes, and this" >&2
    echo "  audits HEAD without them. Commit first if they touch the audited code." >&2
fi

cd "$WT" || exit 3
.venv/bin/python -u tools/mutation_battery_ccarc3.py "$@"
