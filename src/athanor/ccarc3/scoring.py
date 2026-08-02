"""Official ARC-AGI-3 scoring: Relative Human Action Efficiency (RHAE).

The benchmark does **not** score "did you finish the game". It scores how many
actions you took relative to a human who already knew the rules, squared, per
level, weighted toward later levels, and capped by how far you actually got.

    S_l  = min(1.15, (h_l / a_l) ** 2)      per completed level; 0 if not completed
    E^raw = sum(l * S_l) / sum(l)           weights are the 1-indexed level numbers
    C    = sum(1..k) / sum(1..n)            k = sequential levels completed
    E    = min(C, E^raw)                    an environment cannot exceed its cap
    T    = mean(E) over the evaluated set

Three consequences that are easy to get wrong:

- **The ratio is squared, then capped.** Twice the human's actions scores 0.25,
  not 0.5. Ten times scores 0.01. Efficiency dominates.
- **Later levels are worth more.** A five-level environment weights 1,2,3,4,5, so
  the last level alone is a third of the total.
- **The completion cap binds.** Being very fast early cannot compensate for not
  finishing. Clearing 4 of 5 levels caps the environment at 10/15 = 66.67%
  however efficient those four were.

Implements the rubric as specified, and is checked against its published
validation example in ``tests/test_ccarc3_scoring.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "LEVEL_SCORE_CAP",
    "LevelScore",
    "EnvironmentScore",
    "actions_per_level",
    "environment_score",
    "score_run",
    "total_score",
    "score_environment",
]

LEVEL_SCORE_CAP = 1.15
"""Maximum per-level score, applied **after** squaring. 115%."""


@dataclass(frozen=True)
class LevelScore:
    """One level's contribution, kept so a total can be explained rather than asserted."""

    level: int
    """1-indexed, which is also the weight."""

    human: int
    agent: int | None
    """``None`` when the level was not completed."""

    score: float

    @property
    def completed(self) -> bool:
        return self.agent is not None

    @property
    def capped(self) -> bool:
        """The raw ratio exceeded 1.15 and was clipped — i.e. beat the human."""
        return self.completed and (self.human / self.agent) ** 2 > LEVEL_SCORE_CAP


@dataclass(frozen=True)
class EnvironmentScore:
    """One environment's RHAE score, with the arithmetic that produced it."""

    levels: list[LevelScore]
    raw: float
    """Weighted mean of level scores, before the completion cap."""

    cap: float
    """Weighted fraction of levels completed."""

    score: float
    """``min(cap, raw)`` — the number that enters the benchmark total."""

    @property
    def completed(self) -> int:
        return sum(1 for level in self.levels if level.completed)

    @property
    def cap_binds(self) -> bool:
        """The completion cap, not efficiency, is what limited this environment."""
        return self.cap < self.raw


def environment_score(
    human_baselines: Sequence[int],
    agent_actions: Sequence[int | None],
) -> float:
    """One environment's score in ``[0.0, 1.0]``. The reference behaviour."""
    return score_environment(human_baselines, agent_actions).score


def score_environment(
    human_baselines: Sequence[int],
    agent_actions: Sequence[int | None],
) -> EnvironmentScore:
    """As :func:`environment_score`, but keeps the working.

    ``agent_actions[i]`` is the number of actions used to *complete* level
    ``i``, or ``None`` if it was not completed. Levels are sequential: once one
    is incomplete every later one must be too, and claiming otherwise is an
    error rather than something to silently score.
    """
    if len(human_baselines) != len(agent_actions):
        raise ValueError("baseline and agent-action arrays must have equal length")
    if not human_baselines:
        raise ValueError("an environment must contain at least one level")
    if any(h <= 0 for h in human_baselines):
        raise ValueError("human baselines must be positive")

    seen_incomplete = False
    completed = 0
    weighted_sum = 0.0
    total_weight = 0
    levels: list[LevelScore] = []

    for index, (human, agent) in enumerate(zip(human_baselines, agent_actions), start=1):
        weight = index
        total_weight += weight

        if agent is None:
            seen_incomplete = True
            score = 0.0
        else:
            if seen_incomplete:
                raise ValueError(
                    "levels are sequential: a later level cannot be completed "
                    "after an earlier one was left incomplete"
                )
            if agent <= 0:
                raise ValueError("completed-level action counts must be positive")
            completed += 1
            score = min(LEVEL_SCORE_CAP, (human / agent) ** 2)

        weighted_sum += weight * score
        levels.append(LevelScore(level=index, human=human, agent=agent, score=score))

    raw = weighted_sum / total_weight
    cap = sum(range(1, completed + 1)) / total_weight
    return EnvironmentScore(levels=levels, raw=raw, cap=cap, score=min(raw, cap))


def total_score(environment_scores: Sequence[float]) -> float:
    """Benchmark total as a percentage in ``[0.0, 100.0]``.

    Every environment carries equal weight. **Score each environment first, then
    take the unweighted mean** — averaging all levels globally is a different and
    wrong number, because it would let a long environment outvote a short one.

    Pass ``0.0`` for environments that were not attempted: on the published
    benchmark the denominator is the whole evaluated set, so an unplayed
    environment scores zero rather than being excluded.
    """
    if not environment_scores:
        raise ValueError("at least one environment score is required")
    if any(not 0.0 <= score <= 1.0 for score in environment_scores):
        raise ValueError("environment scores must lie in [0.0, 1.0]")
    return 100.0 * sum(environment_scores) / len(environment_scores)


def actions_per_level(
    transitions: Sequence["object"],
    n_levels: int,
    *,
    cumulative: bool = True,
) -> list[int | None]:
    """Actions used to complete each level, from a ledger. ``None`` if not completed.

    **Attribution is by the level the action was taken *from*, not the level it
    landed in**, and that distinction is worth a paragraph because getting it
    backwards silently changes the score.

    The trace records ``level`` as ``levels_completed`` *after* the action. So
    the action that finishes level 3 is recorded at level 4. Grouping by the
    recorded level therefore credits every level's final, decisive action to the
    next one, shortening each level by one and inventing a phantom level at the
    end.

    Verified against ground truth rather than reasoned about alone: on `ls20`
    this returns 18 actions for level 0, which is exactly what that solver
    reported for itself; grouping by the recorded level gives 17.

    **``cumulative=True`` (the default) counts a replayed level twice**, and
    that is the conservative reading of an ambiguity in the rubric. It says
    ``a_l`` is "the agent's action count for completing level l" without saying
    how multiple playthroughs combine. Two things decide it in favour of
    summing:

    - The SDK's own aggregation is asymmetric: ``Card.high_score = max(scores)``
      but ``Card.total_actions = sum(actions)``. Score is best-of, actions
      accumulate. That asymmetry exists precisely so that replaying a game to
      learn its route cannot pay — and if ``a_l`` were per-play, the summing
      would be pointless and the benchmark trivially gameable by grinding out
      the optimal route and then walking it.
    - Choosing otherwise means reporting the flattering reading of a rule you
      have not resolved.

    Pass ``cumulative=False`` for the per-play reading, which counts only the
    playthrough that finished. On `ls20` — the one run here with a full reset —
    the two give **0.799** and **1.000**. Nothing else in the set is affected.

    Actions spent on abandoned attempts and on failed level retries are included
    either way. They were spent, and "actions used to complete this level" is
    what the rubric divides by.
    """
    from .ledger import Transition  # noqa: F401 -- documents the expected element type

    kept = list(transitions)
    if not cumulative:
        cut = max((i for i, t in enumerate(kept) if t.full_reset), default=0)
        kept = kept[cut:]

    counts: dict[int, int] = {}
    previous: int | None = None
    for t in kept:
        source = previous if previous is not None else 0
        counts[source] = counts.get(source, 0) + 1
        previous = t.level

    completed = max((t.level for t in kept), default=0)
    return [counts.get(i, 0) if i < completed else None for i in range(n_levels)]


def score_run(
    transitions: Sequence["object"],
    baselines: Sequence[int],
    *,
    cumulative: bool = True,
) -> EnvironmentScore:
    """RHAE score for one run, straight from its ledger and the game's baselines.

    Defaults to the conservative reading of the replay ambiguity — see
    :func:`actions_per_level`.
    """
    return score_environment(
        baselines, actions_per_level(transitions, len(baselines), cumulative=cumulative)
    )
