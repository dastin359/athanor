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
    return 0 if summary["all_train_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
