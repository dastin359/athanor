"""Three-valued rule checking, with level-local refutation authority.

The single most important constraint in this module is that a predicate returns
**three** values, not two:

    HOLDS | VIOLATED | NOT_APPLICABLE

and only ``VIOLATED`` is evidence against a rule.

Most cross-level "failures" in ARC-AGI-3 are vacuous. A rule that references a
red block, checked on a level with no red block, has an unmet precondition -- it
is not refuted. A boolean predicate cannot tell the two apart: it returns
``False`` for both, and a true rule dies. This is the same failure
``arc.unreached()`` exists to catch in the ARC-AGI-2 toolkit, where a branch that
never executed passed itself off as a branch that passed. The doctrine already
carries the general form, "unanimity is not verification"; the ARC-AGI-3 sibling
is **"failure is not refutation unless the rule was applicable."**

The second constraint follows from levels sharing lineage but not setup: rules
are level-scoped and do not port, while *mechanics* are game-scoped and port as
priors about what to test first. So there are two entry points with deliberately
different powers:

    verify(rule, transitions)  -- current level only. CAN refute.
    survey(rule, transitions)  -- every level. CANNOT refute.

Conflating them is what destroys knowledge: replaying a level-3 rule against
level 1 and believing the result would false-refute a correct rule.

See ``docs/ccarc3_design.md`` §5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .ledger import Transition

__all__ = [
    "Outcome",
    "effective_actions",
    "level_pace",
    "predict",
    "PredictionReport",
    "Rule",
    "Counts",
    "VerifyResult",
    "Survey",
    "verify",
    "survey",
    "regressions",
    "RuleBook",
]


class Outcome(str, Enum):
    """The result of checking a rule against one transition."""

    HOLDS = "HOLDS"
    VIOLATED = "VIOLATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Rule:
    """A named, scoped predicate over transitions.

    The precondition and the claim are separate callables *by construction*, so
    a rule cannot accidentally collapse "did not apply" into "did not hold".

    ``scope`` records what the author believes, not what has been shown:

    - ``"level"`` -- concrete, tied to this level's setup. Does not port.
    - ``"game"`` -- a mechanic. Ports to later levels as a prior about what to
      test first, never as an established truth.
    """

    name: str
    applies: Callable[[Transition], bool]
    holds: Callable[[Transition], bool]
    scope: str = "level"
    note: str = ""

    def __post_init__(self) -> None:
        if self.scope not in ("level", "game"):
            raise ValueError(f"scope must be 'level' or 'game', got {self.scope!r}")

    def __call__(self, t: Transition) -> Outcome:
        if not self.applies(t):
            return Outcome.NOT_APPLICABLE
        return Outcome.HOLDS if self.holds(t) else Outcome.VIOLATED


@dataclass(frozen=True)
class Counts:
    """Outcome tally over some set of transitions."""

    holds: int = 0
    violated: int = 0
    not_applicable: int = 0

    @property
    def applicable(self) -> int:
        return self.holds + self.violated

    @property
    def total(self) -> int:
        return self.holds + self.violated + self.not_applicable

    def __str__(self) -> str:
        return f"{self.holds}/{self.violated}/{self.not_applicable}"


def _tally(rule: Rule, transitions: Iterable[Transition]) -> tuple[Counts, list[Transition]]:
    holds = violated = na = 0
    offenders: list[Transition] = []
    for t in transitions:
        outcome = rule(t)
        if outcome is Outcome.HOLDS:
            holds += 1
        elif outcome is Outcome.VIOLATED:
            violated += 1
            offenders.append(t)
        else:
            na += 1
    return Counts(holds, violated, na), offenders


@dataclass(frozen=True)
class VerifyResult:
    """The verdict on a rule within a single level."""

    rule: str
    level: int
    counts: Counts
    violations: list[Transition] = field(default_factory=list)

    @property
    def refuted(self) -> bool:
        """True iff the rule was applicable somewhere and failed there."""
        return self.counts.violated > 0

    @property
    def verified(self) -> bool:
        """Held at least once, never violated.

        Vacuous truth does not count: a rule that was never applicable has been
        tested by nothing and is not verified by anything.
        """
        return self.counts.violated == 0 and self.counts.holds > 0

    @property
    def vacuous(self) -> bool:
        return self.counts.applicable == 0

    def __str__(self) -> str:
        if self.vacuous:
            verdict = "VACUOUS (never applicable)"
        elif self.refuted:
            verdict = f"REFUTED on {self.counts.violated}"
        else:
            verdict = "VERIFIED"
        return f"{self.rule} @ L{self.level}: {self.counts} -> {verdict}"


def verify(
    rule: Rule,
    transitions: Sequence[Transition],
    *,
    level: int | None = None,
) -> VerifyResult:
    """Check a rule against **one level only**. This is the call that can refute.

    ``level`` defaults to the last level present in ``transitions`` -- the level
    currently being played. Restricting the evidence is the point: a rule
    checked against a level whose setup it was never about produces vacuous
    failures, and treating those as refutations throws away true rules.

    This is the ARC-AGI-3 analogue of ARC-AGI-2's ``check()``: a rule graduates
    from hypothesis to verified because it reproduces recorded history, not
    because the solver believes it. It is free to call, and should be called
    without restraint.
    """
    if level is None:
        # **The LAST level, not the highest.** This line read
        # `max(t.level for t in transitions)`, which is not what the docstring
        # three paragraphs up promises and not what the caller needs: the two
        # agree only while levels ascend, i.e. within a single playthrough.
        #
        # `ledger.load` concatenates every playthrough — it even forces
        # `full_reset=True` when the recorded level goes *down* — so after a full
        # reset the ledger holds an abandoned play's deep levels while play
        # resumes at 0. `verify` then scoped its evidence to the abandoned
        # play's deepest level and returned a confident verdict about a level
        # nobody is playing.
        #
        # That is this module's own named catastrophe, from its docstring
        # ("replaying a level-3 rule against level 1 and believing the result
        # would false-refute a correct rule"), with the levels swapped — and it
        # is not vacuous, so the three-valued defence never fires: the stale
        # level has plenty of applicable transitions, `violated > 0`, and the
        # solver deletes a correct rule about the level it is actually on.
        #
        # Measured across the 30 preserved traces: 23 have two or more
        # playthroughs, and each spends 70-1855 transitions in the window where
        # max != last. `ls20-9607627b` spends 67% of its run there. The
        # unconditional-replay doctrine makes multi-play traces the norm, so this
        # gets worse precisely as the strategy is adopted.
        #
        # Found by an adversarial audit that ran the mutation: swapping this line
        # to the documented behaviour broke NO test, because every fixture in the
        # suite is single-level or ascending — including
        # `test_verify_defaults_to_the_current_level`, which uses levels [0, 3]
        # where max and last coincide.
        level = transitions[-1].level if transitions else 0
    scoped = [t for t in transitions if t.level == level]
    counts, violations = _tally(rule, scoped)
    return VerifyResult(rule=rule.name, level=level, counts=counts, violations=violations)


@dataclass(frozen=True)
class Survey:
    """Per-level outcome map for a rule. Diagnostic only -- it cannot refute."""

    rule: str
    per_level: dict[int, Counts]

    @property
    def levels_applicable(self) -> list[int]:
        return [lv for lv, c in sorted(self.per_level.items()) if c.applicable]

    @property
    def levels_violated(self) -> list[int]:
        return [lv for lv, c in sorted(self.per_level.items()) if c.violated]

    @property
    def instantiations(self) -> int:
        """How many distinct levels the rule was applicable on.

        A mechanic independently instantiated on k levels earns confidence as k
        rises. That is the accumulation -- measured, not declared.
        """
        return len(self.levels_applicable)

    def __str__(self) -> str:
        body = " | ".join(
            f"L{lv}: {c}" for lv, c in sorted(self.per_level.items())
        )
        return f"{self.rule}  {body}   (holds/violated/n-a)"


def survey(rule: Rule, transitions: Sequence[Transition]) -> Survey:
    """Report where a rule holds across **every** level. Never refutes.

    Reading the output: ``L0: 0/0/47 | L1: 12/0/19 | L2: 31/0/0`` says
    "applicable from L1 onward, never violated where applicable" -- a strong
    rule. ``L0: 40/7/0`` is a genuine violation and deserves attention.

    What this buys is **prioritisation, not verification**. A mechanic seen
    instantiated on several levels is a high-prior hypothesis on the next one --
    worth two actions to confirm rather than fifteen to discover. It sets the
    search order, never the truth value.
    """
    levels = sorted({t.level for t in transitions})
    per_level: dict[int, Counts] = {}
    for lv in levels:
        counts, _ = _tally(rule, [t for t in transitions if t.level == lv])
        per_level[lv] = counts
    return Survey(rule=rule.name, per_level=per_level)


def regressions(
    rules: Iterable[Rule],
    transitions: Sequence[Transition],
    *,
    level: int | None = None,
) -> list[VerifyResult]:
    """Game-scoped rules that are **applicable and violated** on a level.

    Levels escalate by introducing roughly one new mechanic, so a mechanic that
    held for a hundred transitions and then breaks is the highest-signal event
    available: the game has just pointed at what it changed.

    Only ``scope="game"`` rules are checked, and only genuine violations are
    returned. A mechanic that is merely ``NOT_APPLICABLE`` on the new level is
    silent -- a boolean predicate would have fired on both and trained everyone
    to ignore the alarm.
    """
    out = []
    for rule in rules:
        if rule.scope != "game":
            continue
        result = verify(rule, transitions, level=level)
        if result.refuted:
            out.append(result)
    return out


def effective_actions(
    transitions: Sequence[Transition],
    *,
    level: int | None = None,
) -> dict[str, tuple[int, int]]:
    """Which actions actually *do* anything: ``{action: (changed, tried)}``.

    ``available_actions`` tells you what the game accepts. It does not tell you
    what has an effect, and the two are not the same. A run that failed outright
    spent **almost one action in three on no-ops** -- 46 of 157 clicks left the
    board untouched, 20% across all 336 of its actions -- while every winning run
    in the same batch wasted none. Nothing in the harness surfaced that while it
    was happening.

    (The percentage read 20% beside the 46-of-157 fraction until 2026-08-08, with
    the fraction offered as its evidence. 46/157 is 29%; 20% is 67 dead of 336,
    the whole-run rate, and the two denominators had been welded together. Both
    numbers are true and both are now shown with the denominator they belong to
    -- the doctrine's §7a always rendered this evidence correctly.)

    This is free: it reads the ledger rather than spending actions. Call it after
    a handful of moves, and stop paying for whatever reads ``0/n``.

    Board-replaced transitions are excluded -- a level swap changes everything
    and would make every action look effective.
    """
    out: dict[str, list[int]] = {}
    for t in transitions:
        if t.before is None or t.board_replaced or t.wasted:
            continue
        if level is not None and t.level != level:
            continue
        row = out.setdefault(t.action, [0, 0])
        row[1] += 1
        if t.changed:
            row[0] += 1
    return {a: (c, n) for a, (c, n) in out.items()}


def level_pace(
    transitions: Sequence[Transition],
    baselines: Sequence[int],
) -> dict[int, tuple[int, int, float]]:
    """Per level: ``{level: (spent, baseline, ratio)}``, final playthrough only.

    The absolute ratio is worth knowing and the *relative* one is worth more.
    Measured over 26 level-attempts across five games:

    - **24 of 25 cleared levels finished at or under 0.92x their baseline**,
      median 0.52x. Finishing a level you understand costs less than the
      published figure, because that figure includes a human's own hesitation.
    - The two attempts that crossed 1.0x are the two the batch flagged
      independently: one at 3.44x that was eventually cleared by a run that
      averaged 0.56x, and one at 6.13x that never cleared at all.
    - There is **nothing between 0.92x and 3.44x** in the sample, so any
      threshold in that range separates the data identically. Do not read a
      precise cutoff into it -- read the shape: normal is comfortably under 1.

    Against a run's own median the outlier is far louder than in absolute
    terms: that 3.44x level was **6.2x** the median of the levels the same run
    had already cleared. A game may simply run above or below baseline
    throughout; a level that runs above *its own run* is the signal.

    Transitions before the last full reset are dropped -- a replayed level
    otherwise counts twice and every ratio doubles.
    """
    cut = max((i for i, t in enumerate(transitions) if t.full_reset), default=0)
    spent: dict[int, int] = {}
    for t in transitions[cut:]:
        spent[t.level] = spent.get(t.level, 0) + 1
    out: dict[int, tuple[int, int, float]] = {}
    for level, n in sorted(spent.items()):
        if 0 <= level < len(baselines) and baselines[level]:
            base = baselines[level]
            out[level] = (n, base, n / base)
    return out


@dataclass(frozen=True)
class PredictionReport:
    """How a forward model fared against every recorded transition."""

    name: str
    correct: int = 0
    wrong: int = 0
    skipped: int = 0
    failures: list[int] = field(default_factory=list)
    """Indices of transitions the model got wrong -- go and look at these."""

    @property
    def tested(self) -> int:
        return self.correct + self.wrong

    @property
    def accuracy(self) -> float:
        return self.correct / self.tested if self.tested else 0.0

    @property
    def perfect(self) -> bool:
        """Every transition it was applicable to, predicted exactly."""
        return self.wrong == 0 and self.correct > 0

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.correct}/{self.tested} exact"
            f"{f' ({self.accuracy:.0%})' if self.tested else ''}"
            f", {self.skipped} skipped"
            f"{f'; first failures at {self.failures[:5]}' if self.failures else ''}"
        )


def predict(
    step: Callable[[Any, str, dict[str, Any]], Any],
    transitions: Sequence[Transition],
    *,
    name: str = "forward model",
    level: int | None = None,
) -> PredictionReport:
    """Check a forward model of the game against everything recorded.

    ``step(before, action, params)`` should return the predicted next board, or
    ``None`` to decline (the transition is then skipped, not counted wrong).

    A predicate says *a* property survived. A simulator that reproduces every
    recorded board exactly says the mechanics are understood, which is a far
    stronger claim and the one worth aiming at.

    **Measured caveat, recorded because it undercuts the case for this function.**
    Solvers write step functions unaided -- the first real run on `ls20` wrote
    one before anything here supported it -- but across seven runs and 539
    commands, **not one of them ever called this**. They kept their models in
    their own code. Nothing in `rules` has been used by a solver; see design
    note §9.8a. Offering an abstraction is not the same as it being adopted.

    Boards replaced wholesale (level boundaries, full resets) and wasted actions
    are skipped automatically: no forward model can or should predict those, and
    counting them as failures would hide the real ones.

    ``perfect`` is the thing to chase. A model at 95% is not 95% right about the
    mechanics; it is missing one, and the ``failures`` indices say where.
    """
    correct = wrong = skipped = 0
    failures: list[int] = []
    for t in transitions:
        if level is not None and t.level != level:
            continue
        if t.before is None or t.board_replaced or t.wasted:
            skipped += 1
            continue
        try:
            predicted = step(t.before, t.action, t.params)
        except Exception:  # noqa: BLE001 -- a model that crashes has not predicted
            predicted = None
        if predicted is None:
            skipped += 1
            continue
        import numpy as _np

        if _np.array_equal(_np.asarray(predicted), t.after):
            correct += 1
        else:
            wrong += 1
            failures.append(t.index)
    return PredictionReport(name, correct, wrong, skipped, failures)


@dataclass
class RuleBook:
    """The small, durable tier of memory: what is believed, and what is ruled out.

    Refutations are carried deliberately. "ACTION5 has no observable effect in
    any state seen" is worth roughly five actions on every later level if
    remembered, and costs five per level if forgotten. Negative knowledge is
    cheaper to hold than positive knowledge and transfers at least as well.
    """

    verified: list[dict[str, Any]] = field(default_factory=list)
    refuted: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def record(self, result: VerifyResult, *, scope: str = "level", note: str = "") -> None:
        entry = {
            "rule": result.rule,
            "scope": scope,
            "level": result.level,
            "counts": str(result.counts),
            "note": note,
        }
        if result.refuted:
            self.refuted.append(entry)
        elif result.verified:
            self.verified.append(entry)
        else:
            self.open_questions.append(
                f"{result.rule}: never applicable on L{result.level}; untested"
            )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RuleBook":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))
