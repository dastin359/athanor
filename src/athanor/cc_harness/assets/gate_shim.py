#!/usr/bin/env python3
"""Verification gate for this puzzle workspace.

    python gate.py status    distilled research state (run this after a compaction)
    python gate.py submit    formal iteration: solution/hypothesis.md + solution/solve.py
    python gate.py accept    finalize, using the last submission and solution/audit.md

Thin shim over ``athanor.cc_harness.gate``; the source path below is baked in
when the workspace is created.
"""

from __future__ import annotations

import sys

ATHANOR_SRC = r"__ATHANOR_SRC__"

if ATHANOR_SRC not in sys.path:
    sys.path.insert(0, ATHANOR_SRC)

from athanor.cc_harness.gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
