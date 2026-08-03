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
    "server_actions_per_level",
    "disagreements_with_server",
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
    cumulative: bool = False,
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

    **Replayed levels are NOT summed, and that is measured, not assumed.**
    A live probe of the scorecard settles it. The API records, per game:

    ===================  ==================================================
    ``total_plays``      how many plays this scorecard holds
    ``actions``          actions **per play**, e.g. ``[3, 1]``
    ``actions_by_level`` **cumulative** actions at each level's completion,
                         per play -- see the warning below
    ``total_actions``    the sum across plays — a *budget* figure
    ===================  ==================================================

    A ``RESET`` while the action counter is non-zero is a *level* reset:
    ``total_plays`` stays put and ``resets`` increments. A ``RESET`` when the
    counter is zero — which is the state immediately after a level advance —
    **starts a new play**: a new guid, a new ``actions`` row and a new
    ``actions_by_level`` row.

    **``actions_by_level`` is cumulative, despite its name.** Measured on a live
    run of `tu93`: it returned ``[[1, 18], [2, 45], [3, 64]]`` while
    ``total_actions`` was **67**. Those entries sum to 127, which is the tell --
    they are ``[level, actions spent by the time that level fell]``, so the
    per-level ``a_l`` the rubric divides by is the **difference** between
    consecutive entries: 18, 27, 19. Reading them as per-level counts inflates
    every level after the first and would have scored this run roughly a third
    of what it earns.

    A one-level game cannot show this -- cumulative and per-level coincide -- so
    the first probe, on `lp85`, looked like agreement and was not evidence.

    So for a one-level game with a human baseline of 7, won in 10 and then
    replayed and won in 7: ``actions_by_level`` is ``[[10], [7]]`` and
    ``total_actions`` is 17. Since ``Card.high_score = max(scores)`` scores the
    *best* play, ``a_l`` is **7**. The 17 is budget spent, not the denominator.

    This module briefly defaulted to summing, on the argument that
    ``total_actions`` accumulates. That argument conflated a game-level total
    with a per-level denominator; the server keeps both, separately.

    ``cumulative=True`` still gives the summed reading for comparison. On
    `ls20` — the one run here that was replayed — the two give **1.000** and
    **0.799**. Nothing else in the set is affected.

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
    for i, t in enumerate(kept):
        source = previous if previous is not None else 0
        # **A RESET that starts a play is not an action, and the server agrees.**
        # Measured against a live scorecard: seven ACTION6 calls preceded by the
        # opening RESET were reported as `actions: [7]` and
        # `actions_by_level: [[[1, 7]]]`, not 8. A *level* reset -- a RESET while
        # the counter is non-zero, taken after a death -- is counted, and that
        # also matched: a padded play of 13 clicks, a level reset and 7 more came
        # back as 21, exactly what was spent after the opening RESET.
        #
        # So the exclusion is precisely the play-starting RESET: the first
        # transition of the trace, and any RESET that performs a full reset.
        if t.action == "RESET" and (i == 0 or t.full_reset):
            previous = t.level
            continue
        counts[source] = counts.get(source, 0) + 1
        previous = t.level

    completed = max((t.level for t in kept), default=0)
    return [counts.get(i, 0) if i < completed else None for i in range(n_levels)]


def score_run(
    transitions: Sequence["object"],
    baselines: Sequence[int],
    *,
    cumulative: bool = False,
) -> EnvironmentScore:
    """RHAE score for one run, straight from its ledger and the game's baselines.

    Counts the play that finished, which is what the server records and what
    best-of-plays scoring selects — see :func:`actions_per_level`.
    """
    return score_environment(
        baselines, actions_per_level(transitions, len(baselines), cumulative=cumulative)
    )


def server_actions_per_level(scorecard: dict, game_id: str, *, play: int = -1) -> list[int]:
    """Per-level actions as **the server** records them, from a saved scorecard.

    Exists so a run can check itself rather than trusting this module. Every
    figure this project reported for eleven games was derived from the trace and
    had never once been compared against ARC's own numbers, because the card is
    gone by the time anyone looks — see ``ArcClient._snapshot_scorecard_if_due``.

    **Differences the cumulative entries**, which is the whole point: the API
    field is named ``actions_by_level`` but holds
    ``[level, actions spent when that level fell]``. See :func:`actions_per_level`.

    ``play`` selects which playthrough; the default is the most recent. Levels in
    the response are 1-indexed and the returned list is 0-indexed, matching
    :func:`actions_per_level`.
    """
    card = (scorecard.get("cards") or {}).get(game_id)
    if not card:
        raise KeyError(f"no card for {game_id} in this scorecard")
    plays = card.get("actions_by_level") or []
    if not plays:
        return []
    out: list[int] = []
    previous = 0
    for level, cumulative in plays[play]:
        if level - 1 != len(out):
            raise ValueError(
                f"{game_id}: levels arrived out of order at {level}; "
                f"differencing assumes they are consecutive and ascending"
            )
        out.append(cumulative - previous)
        previous = cumulative
    return out


def disagreements_with_server(
    transitions: Sequence["object"],
    scorecard: dict,
    game_id: str,
    n_levels: int,
) -> list[tuple[int, int, int]]:
    """``(level, ours, theirs)`` wherever the trace and the server disagree.

    Empty means the two agree on every level the server has scored. Levels the
    server has not recorded yet — an in-flight run — are not compared, so this
    is safe to call on a live game.
    """
    theirs = server_actions_per_level(scorecard, game_id)
    ours = actions_per_level(transitions, n_levels)
    return [
        (i, ours[i], theirs[i])
        for i in range(min(len(theirs), len(ours)))
        if ours[i] != theirs[i]
    ]
