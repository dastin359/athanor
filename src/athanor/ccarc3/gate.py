"""The level-boundary gate.

ARC-AGI-2's harness refuses a submission that does not reproduce every training
pair. That refusal is the reason the solver's belief and its evidence stay in
contact: you cannot *assert* your way past it.

ARC-AGI-3 has no submission to gate, but it has a moment that serves the same
purpose, and it is free. A level boundary is the one point where the solver is
guaranteed to have just learned something and to be about to need it. So the
gate is: **the first action of a new level is refused until the rule book has
been updated.**

Why structural and not advisory. The knowledge that transfers between levels is
the only thing standing between a solver and re-deriving the game from scratch
on every level. An instruction to "remember what you learned" competes with the
immediate pull of the new level. A refusal does not.

What the gate deliberately does *not* do is judge the content. It checks that
something was written for this level and moves on. A gate that graded the
hypothesis would be a reviewer, and this harness does not have one.
"""
# **This docstring is solver-visible, and it used to leak.** `__init__.py` does
# `from .gate import GateRefusal, LevelGate`, which binds the submodule, so
# `arc.gate.__doc__` and `pydoc athanor.ccarc3.gate` both print it -- and the
# paragraph above once justified the gate by quoting a real per-game baseline
# total, in the operator's own voice, reachable by introspection from an object
# the workspace `session.py` hands the solver by name. The argument survives
# without the number; the number does not survive being printed -- and neither
# does a note like this one that repeats it while explaining the removal, which
# is how the figure was still here months later.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rules import RuleBook

__all__ = ["GateRefusal", "LevelGate"]


class GateRefusal(RuntimeError):
    """The gate declined an action. Nothing was sent and the game did not step."""


@dataclass
class LevelGate:
    """Requires a rule-book entry at every level boundary.

    The gate is armed by a level *advance*, not by entering a level, so it never
    fires on the opening RESET -- there is nothing to have learned yet.
    """

    rulebook_path: Path | str = "rules.json"
    pending_level: int | None = None
    acknowledged: dict[int, str] = field(default_factory=dict)
    last_level: int = 0
    refusals: int = 0
    on_change: Any = None
    """Called with no arguments whenever the gate's held/open state changes.

    ``ArcClient`` sets this to its state saver. Without it an acknowledgement is
    lost the moment the process ends: the client persists after each *action*,
    so clearing the gate and then exiting leaves ``gate_pending`` still set on
    disk, and the next process restores a gate that was already satisfied.

    Found by reading what a real solver had to write around it -- a helper whose
    docstring read "Gate state is per-process; re-clear it after re-importing
    session." A harness the solver has to work around is a harness bug.
    """

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def observe(self, level: int) -> None:
        """Record the level reported by the latest frame."""
        if level > self.last_level:
            # A boundary already acknowledged has nothing new to record. This is
            # what makes a *replay* usable: after a full reset the solver
            # re-crosses every boundary it has already documented, and demanding
            # a fresh entry at each one would cost a turn apiece to restate what
            # the rule book already holds. The gate exists to catch knowledge
            # about to be lost, not to bill for knowledge already kept.
            self.pending_level = None if level in self.acknowledged else level
        elif level < self.last_level:
            # A full reset rewound the game. Whatever was pending is moot; the
            # rule book keeps what it already learned, which is the point of
            # keeping knowledge separate from progress.
            self.pending_level = None
        self.last_level = level

    def check(self) -> None:
        """Raise if a level advance is still unacknowledged."""
        if self.pending_level is None:
            return
        self.refusals += 1
        raise GateRefusal(
            f"level {self.pending_level} reached and the rule book has not been "
            f"updated. Before acting again, record what level "
            f"{self.pending_level - 1} established: which mechanics you believe "
            f"are game-scoped (they carry, as priors about what to test first), "
            f"which rules were level-scoped (they do not carry), and what you "
            f"ruled out. Call gate.acknowledge(...) to proceed."
        )

    def acknowledge(
        self,
        summary: str,
        *,
        mechanics: list[str] | None = None,
        refuted: list[str] | None = None,
        untested: list[str] | None = None,
        book: RuleBook | None = None,
    ) -> RuleBook:
        """Record what the finished level established, and clear the gate.

        Three lists, because there are three states and collapsing any two of
        them loses the distinction the whole design rests on:

        - ``mechanics`` — game-scoped beliefs worth carrying forward, as priors
          about what to test first on the next level.
        - ``refuted`` — tested and found false. Cheaper to carry than positive
          rules and transfers at least as well.
        - ``untested`` — never exercised, so nothing is known either way.

        The third exists because a real solver put *"actions that would move
        into a wall were never tested yet"* into ``refuted``, having nowhere
        else to put it. A claim never exercised is not a claim disproved, and
        filing it as a refutation makes the solver stop asking -- the opposite
        of what an untested question deserves.
        """
        if self.pending_level is None:
            raise GateRefusal("nothing pending; the gate is not holding anything")
        level = self.pending_level
        if not summary.strip():
            raise GateRefusal("summary is empty; the gate wants a claim, not a token")

        book = book or RuleBook.load(self.rulebook_path)
        for m in mechanics or []:
            book.verified.append({"rule": m, "scope": "game", "level": level - 1, "note": summary})
        for r in refuted or []:
            book.refuted.append({"rule": r, "scope": "game", "level": level - 1, "note": summary})
        for u in untested or []:
            book.open_questions.append(f"level {level - 1}: UNTESTED — {u}")
        if not (mechanics or refuted or untested):
            # A level can genuinely teach nothing portable, and saying so is a
            # real answer. It is recorded as an open question rather than
            # silently dropped, so a run that keeps producing them is visible.
            book.open_questions.append(f"level {level - 1}: {summary}")
        book.save(self.rulebook_path)

        self.acknowledged[level] = summary
        self.pending_level = None
        self._changed()
        return book

    @property
    def held(self) -> bool:
        return self.pending_level is not None

    def status(self) -> str:
        if self.held:
            return f"GATE HELD at level {self.pending_level} — acknowledge to continue"
        return f"gate open (level {self.last_level}, {len(self.acknowledged)} acknowledged)"
