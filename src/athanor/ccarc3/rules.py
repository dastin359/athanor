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
        level = max((t.level for t in transitions), default=0)
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
