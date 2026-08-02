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

    crosses_level: bool = False
    """This action ended a level, so ``before`` and ``after`` are different boards.

    Such a transition is not a transition *within* a game state: the whole board
    is replaced. A movement rule checked across one sees the avatar "teleport"
    and reports a violation that never happened -- and by §5.4 an
    applicable-and-violated result is the highest-signal event there is, so a
    false one is the most damaging kind of error this ledger can produce.

    Found exactly that way: "ACTION1 moves the cursor up" held 7/7 on level 0 and
    showed 13/1 on level 1, and the single violation was the level boundary, with
    1467 cells changed at once.

    Rules about movement, adjacency or anything else spatial should exclude these
    in their ``applies``.
    """

    wasted: bool = False
    """The action returned no frame and changed nothing, but was still counted.

    ``perform_action`` short-circuits any non-RESET action while the state is
    GAME_OVER or WIN, returning ``frame=[]`` without stepping the game. The
    action still costs budget. These are kept rather than dropped precisely
    because "actions burned after a death, before the solver noticed it had to
    RESET" is a number worth being able to read off a trace.
    """

    @property
    def board_replaced(self) -> bool:
        """``before`` and ``after`` are different boards, for any reason.

        Two things cause it: completing a level (:attr:`crosses_level`) and a
        full reset, which rewinds the whole game to level 0. Both make a spatial
        comparison meaningless, and ``crosses_level`` alone only catches the
        first -- a full reset moves the level *down*, so it slips past a test
        that looks for an increase.

        **This is the check a spatial rule wants**, not ``crosses_level``.
        """
        return self.crosses_level or self.full_reset

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
            # arcengine 0.9.3 renamed `score` to `levels_completed`; arc_agi_3
            # 0.0.1 still sends `score`. Reading only one of them silently
            # records zero for every frame the other produced.
            "score": int(
                frame.get("score", frame.get("levels_completed", 0)) or 0
            ),
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
    """Yield ledger records, tolerating a half-written final line.

    **A trace is appended to while it is read.** ``athanor ccarc3 report`` is
    most useful *during* a batch, which is exactly when the last line may be
    mid-write -- and this raised ``JSONDecodeError: Unterminated string`` on a
    live run rather than reporting the 20 complete actions before it.

    Only the **final** line is forgiven. A malformed line anywhere earlier is
    real corruption, and dropping it silently would quietly shorten a level's
    action count and change a score; that still raises, and says where.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                return          # a writer is mid-append; everything before is good
            raise ValueError(
                f"{p}: line {number} of {len(lines)} is not valid JSON. A bad line "
                f"in the middle of a trace is corruption, not a partial write."
            ) from None


def load(path: str | Path) -> list[Transition]:
    """Read a ledger back as transitions, chaining ``before`` from ``after``."""
    out: list[Transition] = []
    previous: np.ndarray | None = None
    previous_score = 0
    previous_level = 0
    for rec in _records(path):
        grids = _grids(rec.get("frames", []))
        wasted = not grids
        if wasted:
            # A non-RESET action issued while GAME_OVER or WIN: the engine
            # short-circuits and returns frame=[] without stepping. It still
            # cost budget, so it is recorded, not dropped. With no predecessor
            # to carry forward there is nothing to represent, so skip only then.
            if previous is None:
                continue
            grids = [previous]
        out.append(
            Transition(
                index=int(rec["i"]),
                level=int(rec.get("level", 0)),
                action=rec["action"],
                params=dict(rec.get("params") or {}),
                before=previous,
                after=grids[-1],
                intermediate=tuple(grids),
                wasted=wasted,
                # The first transition has no predecessor, so it cannot have crossed
                # anything -- comparing it against a level-0 default would flag
                # any trace that starts mid-game.
                crosses_level=(
                    previous is not None and int(rec.get("level", 0)) > previous_level
                ),
                score_before=previous_score,
                score_after=int(rec.get("score", 0)),
                state=str(rec.get("state", "NOT_PLAYED")),
                # A level that goes *down* is a full reset whatever the server
                # said. It recorded ``full_reset: False`` on a transition that
                # took the game from level 6 to level 0 — which meant
                # ``board_replaced``, documented as the check a spatial rule
                # wants, returned False on the largest board replacement in the
                # whole trace. ``crosses_level`` only catches an increase.
                full_reset=bool(rec.get("full_reset", False)) or (
                    previous is not None and int(rec.get("level", 0)) < previous_level
                ),
                available_actions=tuple(rec.get("available_actions") or ()),
                hypothesis=rec.get("hypothesis"),
            )
        )
        previous = grids[-1]
        previous_score = int(rec.get("score", 0))
        previous_level = int(rec.get("level", 0))
    return out


def infer_levels(scores: Sequence[int]) -> list[int]:
    """A score increment marks a level boundary.

    Verified against ``arcengine`` 0.9.3: ``next_level()`` is the *only* thing
    that touches the score, and it adds exactly 1. The score is a count of
    completed levels, which is why the engine's own ``FrameData`` renamed the
    field to ``levels_completed``.

    Two caveats before trusting it on the live API. The installed
    ``arc_agi_3`` 0.0.1 still calls the field ``score`` and types it
    ``0..254``, so a server-side game could in principle award points within a
    level; and a level *reset* does not decrement, so replays of a failed level
    are correctly attributed but a full reset (score back to 0) is not handled
    here. Prefer an explicit level from the caller when one is available, and
    watch ``full_reset``. See ``docs/ccarc3_design.md`` §7.
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
