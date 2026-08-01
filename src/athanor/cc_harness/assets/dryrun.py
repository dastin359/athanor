#!/usr/bin/env python3
"""Free dry-run: score solution/solve.py against the training pairs.

    python dryrun.py

Costs nothing, consumes no iteration, records nothing. Use it as often as you
like. Every failure it catches is a failure you did not spend a budgeted
`gate.py submit` on — the gate is a checkpoint, not a debugger.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc import check  # noqa: E402


def main() -> int:
    solution = ROOT / "solution" / "solve.py"
    if not solution.is_file():
        print("solution/solve.py does not exist yet.")
        return 1

    spec = importlib.util.spec_from_file_location("candidate_solution", solution)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:  # noqa: BLE001 - report, do not traceback-spam
        print(f"solution/solve.py failed to load: {type(exc).__name__}: {exc}")
        return 1

    solve = getattr(module, "solve", None)
    if not callable(solve):
        print("solution/solve.py does not define a callable solve(grid).")
        return 1

    summary = check(solve)
    _durability_nudge()
    return 0 if summary["all_train_correct"] else 1


def _durability_nudge() -> None:
    """Warn when nothing has been committed to the invariant ledger.

    Running experiments and *recording* what they established are different
    acts, and the second is easy to skip during a long search — exactly when it
    matters most, because a context compaction discards everything the ledger
    does not hold.

    Deliberately not a count target: recording trivia to satisfy a number is the
    same failure as recording a tautology. The point is durability.
    """
    from arc import invariants

    try:
        if invariants():
            return
    except Exception:  # noqa: BLE001 - a nudge must never break the dry run
        return
    print(
        "\n== nothing recorded in the invariant ledger yet.\n"
        "   If your context is compacted, everything you have worked out so far goes with it —\n"
        "   `gate.py status` replays only what arc.verify() recorded. Commit the facts your\n"
        "   current reading actually rests on."
    )


if __name__ == "__main__":
    raise SystemExit(main())
