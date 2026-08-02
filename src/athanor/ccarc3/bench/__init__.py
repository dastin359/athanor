"""Local bench: drive an ``arcengine`` game and record a real trace.

Everything in :mod:`athanor.ccarc3` can be unit-tested against hand-built
dicts, but hand-built dicts are written by the same person who wrote the code
and agree with it by construction. This module closes that loop by driving an
actual game engine and recording what it actually emits, which is how the
``levels_completed`` / ``score`` skew and the wasted-action short-circuit were
found rather than assumed.

``arcengine`` is deliberately **not** an athanor dependency. Imports are local
and callers should skip when it is missing::

    pytest.importorskip("arcengine")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from ..ledger import TraceWriter, load
from ..grids import as_grid

__all__ = ["play", "PlayResult", "scripted"]


class PlayResult:
    """What a bench run produced."""

    def __init__(self, path: Path, states: list[str], scores: list[int]) -> None:
        self.path = path
        self.states = states
        self.scores = scores

    @property
    def actions_used(self) -> int:
        return len(self.states)

    @property
    def won(self) -> bool:
        return "WIN" in self.states

    @property
    def deaths(self) -> int:
        """GAME_OVER transitions, not GAME_OVER frames.

        Counting frames would multiply-count every wasted action taken while
        already dead, which is exactly the overhead ``Transition.wasted``
        exists to keep separate.
        """
        return sum(
            1
            for prev, cur in zip(["NOT_PLAYED", *self.states], self.states)
            if cur == "GAME_OVER" and prev != "GAME_OVER"
        )

    def transitions(self):
        return load(self.path)

    def __repr__(self) -> str:
        return (
            f"<PlayResult actions={self.actions_used} score={self.scores[-1] if self.scores else 0} "
            f"deaths={self.deaths} won={self.won}>"
        )


def play(
    game: Any,
    policy: Callable[[list[dict[str, Any]]], int | None],
    path: str | Path,
    *,
    max_actions: int = 1000,
    append: bool = False,
) -> PlayResult:
    """Drive ``game`` with ``policy``, recording every action to a ledger.

    ``policy`` receives the frame dicts so far and returns an action id; 0 is
    RESET. The default ``max_actions`` matches the design note's §2.1
    recommendation -- high enough that the counter is not what ends a run, so
    the action count becomes something measured rather than something capped.

    Levels are labelled from the score, which ``arcengine`` guarantees is a
    count of completed levels (§2.5).
    """
    from arcengine import ActionInput, GameAction

    path = Path(path)
    if path.exists() and not append:
        # The ledger is append-only by design, so re-running a bench against the
        # same path silently welds two runs together -- and `load()` then chains
        # one run's `before` from the other's `after`, inventing transitions that
        # never happened. Three bench runs here produced 2038 transitions from 19
        # actions before this was noticed. Truncate by default; opt in to append.
        path.unlink()

    writer = TraceWriter(path)
    history: list[dict[str, Any]] = []
    states: list[str] = []
    scores: list[int] = []
    level = 0

    for _ in range(max_actions):
        action_id = policy(history)
        if action_id is None:
            break
        frame = game.perform_action(ActionInput(id=GameAction.from_id(action_id)))
        record = frame.model_dump()
        history.append(record)

        score = int(record.get("levels_completed", record.get("score", 0)) or 0)
        # Label with the level the frame reports, *before* writing -- the same
        # convention ArcClient uses. The action that clears a level therefore
        # carries the new level number, which is what lets load() infer
        # crosses_level from an increase. Labelling it with the old level (the
        # level it was "played on", which reads more naturally) silently breaks
        # that inference: the clearing action is exactly the one whose before
        # and after straddle two different boards, and it would never be
        # flagged. Two conventions and one inference is a bug waiting to
        # happen, and did.
        level = score
        writer.append(record, level=level)

        state = str(record.get("state", ""))
        state = getattr(state, "value", state)
        states.append(state.split(".")[-1])
        scores.append(score)

        if states[-1] == "WIN":
            break

    return PlayResult(Path(path), states, scores)


def scripted(actions: Sequence[int]) -> Callable[[list[dict[str, Any]]], int | None]:
    """A policy that replays a fixed action list, then stops.

    It returns ``None`` rather than idling on RESET, and that is not a detail.
    ``_action_count`` is zeroed when a level advances, so a RESET issued as the
    very next action takes ``handle_reset``'s ``_action_count == 0`` branch and
    performs a **full game reset**, wiping the score to zero. An earlier version
    of this helper idled on RESET and silently destroyed the progress of every
    bench run that finished a level. See §2.3 of the design note.
    """

    def policy(history: list[dict[str, Any]]) -> int | None:
        i = len(history)
        return actions[i] if i < len(actions) else None

    return policy
