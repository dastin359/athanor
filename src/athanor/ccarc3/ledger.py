"""The transition ledger: an append-only record of ``(before, action) -> after``.

The SDK already records enough to reconstruct transitions -- ``FrameData``
carries ``action_input``, so the action that produced a frame travels with it --
but it records *states*, and hypothesis testing is about *transitions*. This
module writes the transition view and reads it back, so that no solver has to
re-derive a loader and every run's trace has the same shape.

The design rule this serves (``docs/ccarc3_design.md`` §3): **frames leave
context, they never leave disk.** A cleared level's frames stop being worth
attending to, but they remain evidence, and :mod:`athanor.ccarc3.rules` replays
against all of them.

Nothing here imports ``arc_agi_3``. Records are plain dicts of the shape
``FrameData.model_dump()`` produces, so this is testable without the SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .grids import as_grid

__all__ = [
    "ACTION_NAMES",
    "Transition",
    "TraceWriter",
    "load",
    "action_name",
    "infer_levels",
]

ACTION_NAMES = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


def action_name(raw: Any) -> str:
    """Normalise an action id to its name.

    ``GameAction`` sets ``_value_`` to the numeric id, so a JSON dump yields an
    int; callers holding the enum itself pass a name. Accept either.
    """
    if isinstance(raw, str):
        return raw.upper()
    if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly
        raise ValueError(f"not an action id: {raw!r}")
    if isinstance(raw, int):
        try:
            return ACTION_NAMES[raw]
        except KeyError:
            raise ValueError(f"no GameAction with id {raw}") from None
    name = getattr(raw, "name", None)
    if isinstance(name, str):
        return name.upper()
    raise ValueError(f"cannot read an action name from {raw!r}")


@dataclass(frozen=True)
class Transition:
    """One action and the state change it produced."""

    index: int
    """Position in the game, counting every action including RESET."""

    level: int
    action: str
    params: dict[str, Any]
    before: np.ndarray | None
    """``None`` for the first transition, which has no predecessor."""

    after: np.ndarray
    intermediate: tuple[np.ndarray, ...]
    """Every grid this action rendered, ``after`` included.

    One action can render several frames; the intermediate ones are where a
    mechanic is often visible, so they are kept rather than collapsed.
    """

    score_before: int
    score_after: int
    state: str
    full_reset: bool
    available_actions: tuple[str, ...]
    hypothesis: Any = None
    """Whatever was stamped into ``ActionInput.reasoning`` -- the server stores
    and echoes it verbatim, which makes the trace self-describing for free."""

    @property
    def score_delta(self) -> int:
        return self.score_after - self.score_before

    @property
    def changed(self) -> bool:
        """Did the board change at all? A no-op action is itself a finding."""
        if self.before is None:
            return True
        if self.before.shape != self.after.shape:
            return True
        return not np.array_equal(self.before, self.after)

    @property
    def is_game_over(self) -> bool:
        return self.state == "GAME_OVER"

    @property
    def is_win(self) -> bool:
        return self.state == "WIN"


def _grids(frame_field: Sequence[Any]) -> list[np.ndarray]:
    return [as_grid(g) for g in frame_field]


class TraceWriter:
    """Appends one record per action to a JSONL ledger.

    Level tracking is the caller's business. :func:`infer_levels` offers a
    score-based heuristic, but whether levels really are delimited by a score
    increment is an open question (``docs/ccarc3_design.md`` §7.2), so nothing
    here assumes it silently.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def append(self, frame: dict[str, Any], *, level: int = 0) -> dict[str, Any]:
        """Record one frame dict as a transition. Returns the record written."""
        action_input = frame.get("action_input") or {}
        record = {
            "i": self._index,
            "level": level,
            "action": action_name(action_input.get("id", 0)),
            "params": {
                k: v for k, v in (action_input.get("data") or {}).items()
                if k in ("x", "y")
            },
            "hypothesis": action_input.get("reasoning"),
            "frames": [
                np.asarray(g, dtype=np.int16).tolist() for g in frame.get("frame", [])
            ],
            "score": int(frame.get("score", 0)),
            "state": str(frame.get("state", "NOT_PLAYED")),
            "full_reset": bool(frame.get("full_reset", False)),
            "available_actions": [
                action_name(a) for a in frame.get("available_actions", [])
            ],
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._index += 1
        return record


def _records(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(path: str | Path) -> list[Transition]:
    """Read a ledger back as transitions, chaining ``before`` from ``after``."""
    out: list[Transition] = []
    previous: np.ndarray | None = None
    previous_score = 0
    for rec in _records(path):
        grids = _grids(rec.get("frames", []))
        if not grids:
            # An action that returned no frame at all: the SDK drops it rather
            # than raising, and a ledger that hid it would misalign the index.
            continue
        out.append(
            Transition(
                index=int(rec["i"]),
                level=int(rec.get("level", 0)),
                action=rec["action"],
                params=dict(rec.get("params") or {}),
                before=previous,
                after=grids[-1],
                intermediate=tuple(grids),
                score_before=previous_score,
                score_after=int(rec.get("score", 0)),
                state=str(rec.get("state", "NOT_PLAYED")),
                full_reset=bool(rec.get("full_reset", False)),
                available_actions=tuple(rec.get("available_actions") or ()),
                hypothesis=rec.get("hypothesis"),
            )
        )
        previous = grids[-1]
        previous_score = int(rec.get("score", 0))
    return out


def infer_levels(scores: Sequence[int]) -> list[int]:
    """Heuristic: a score increment marks a level boundary.

    HEURISTIC, NOT VERIFIED. It holds for locally authored games, where
    ``next_level`` is what raises the score, but ARC-AGI-3 scores are
    server-side and a game may award points within a level. Treat the output as
    a labelling convenience, and prefer an explicit level from the caller when
    one is available. See ``docs/ccarc3_design.md`` §7.2.
    """
    level = 0
    out: list[int] = []
    previous = scores[0] if scores else 0
    for s in scores:
        if s > previous:
            level += 1
        out.append(level)
        previous = s
    return out
