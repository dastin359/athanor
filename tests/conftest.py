"""Shared fixtures for the CC-harness test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from athanor.cc_harness.config import CCRunConfig  # noqa: E402
from athanor.cc_harness.workspace import build_workspace  # noqa: E402

#: A synthetic task whose rule is "mirror each row horizontally".
#: Small enough to reason about, asymmetric enough that identity fails.
MIRROR_TASK = {
    "train": [
        {"input": [[1, 2, 0], [0, 3, 0]], "output": [[0, 2, 1], [0, 3, 0]]},
        {"input": [[4, 0, 0], [0, 0, 5]], "output": [[0, 0, 4], [5, 0, 0]]},
        {"input": [[7, 8, 9], [1, 1, 2]], "output": [[9, 8, 7], [2, 1, 1]]},
    ],
    "test": [
        {"input": [[6, 0, 3], [0, 4, 0]], "output": [[3, 0, 6], [0, 4, 0]]},
    ],
}

MIRROR_SOLVE = "def solve(grid):\n    return [row[::-1] for row in grid]\n"
IDENTITY_SOLVE = "def solve(grid):\n    return [row[:] for row in grid]\n"

LONG_HYPOTHESIS = (
    "## Rule\n\nEvery output grid is the input grid with each row reversed left to right; "
    "the grid keeps its dimensions and its colour palette, and no cell is recoloured. "
    "Step 1: read the H by W input grid. Step 2: for each row, emit the same cells in "
    "reverse column order. Step 3: return the resulting H by W grid. Edge cases: a grid "
    "of width one is returned unchanged, since reversing a single element is a no-op; "
    "there is no background colour to special-case and no object extraction involved. "
    "Generalization: the rule is a pure geometric reflection about the vertical axis, so "
    "it is independent of grid size, palette, and object content. No ambiguity remains, "
    "so a single candidate is emitted for the test input.\n"
)


@pytest.fixture
def task_json(tmp_path: Path) -> Path:
    """The synthetic task written out the way the dataset stores tasks."""
    path = tmp_path / "dataset" / "mirror01.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MIRROR_TASK), encoding="utf-8")
    return path


@pytest.fixture
def config() -> CCRunConfig:
    return CCRunConfig(max_iterations=4, best_effort_iterations=1, visual=False)


@pytest.fixture
def workspace(tmp_path: Path, config: CCRunConfig):
    return build_workspace(
        task_id="mirror01",
        puzzle_data=MIRROR_TASK,
        root=tmp_path / "run" / "workspace",
        config=config,
    )


def write_solution(workspace, *, hypothesis: str = LONG_HYPOTHESIS, code: str = MIRROR_SOLVE) -> None:
    (workspace.root / "solution" / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    (workspace.root / "solution" / "solve.py").write_text(code, encoding="utf-8")
