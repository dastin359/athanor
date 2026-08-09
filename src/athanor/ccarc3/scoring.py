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
        """The raw ratio exceeded 1.15 and was clipped — i.e. beat the human.

        **Strictly greater, and the boundary is unreachable rather than
        untested.** A mutation pass flags `>` -> `>=` as surviving; it is an
        equivalent mutant. No IEEE double squares to exactly 1.15 -- the two
        nearest candidates give 1.1499999999999997 and 1.1500000000000001 -- so
        no `(human / agent)` can land on the boundary and the two forms cannot
        be distinguished by any input. `>` is still the correct spelling: at
        exactly the cap nothing is clipped, so nothing was capped.
        """
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
                # **Deliberately refused, and `client.py` disagreed about it.**
                # `_send` records 0 for the extra levels of a multi-level
                # advance, and its comment claimed "Zero is read as 'cleared for
                # free' by `_play_score`". It is not: this raises, and that
                # comment has been corrected rather than this rule relaxed.
                #
                # Scoring 0 as "cleared for free" means the cap, 1.15 -- the
                # maximum a level can earn. So the two ways to be wrong are not
                # symmetric: if a zero ever arrives from a parsing fault rather
                # than a real double advance, accepting it awards a defect the
                # best possible score and the run reads as excellent. Refusing
                # costs a crash and a look. A double advance has never been
                # observed in 50 preserved runs; a parsing fault has, more than
                # once, in this project.
                raise ValueError(
                    f"completed-level action counts must be positive, got {agent}. "
                    f"If the server really credited two levels at once, that level "
                    f"cost nothing and needs a deliberate decision here — it is not "
                    f"scored automatically, because 0 would earn the cap."
                )
            completed += 1
            # **`agent == 0` is unreachable, and the branch that handled it said
            # the opposite of the rule above.** The guard four lines up raises on
            # `agent <= 0`, so the old ternary --
            # `LEVEL_SCORE_CAP if agent == 0 else ...` -- could never run. Dead
            # code is not why it mattered: it implemented, in the line
            # immediately after, exactly the policy the comment spends a
            # paragraph refusing ("0 would earn the cap"). Anyone relaxing the
            # guard to admit a real double advance would have silently inherited
            # the cap-award instead of making the deliberate decision the guard's
            # own message demands. Found by mutation: flipping that branch to 0.0
            # left all 1070 tests green, because nothing can reach it.
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

    **The ground-truth check that stood here has gone stale, and inverted.** It
    named a per-level figure for `ls20` level 0 and offered a second, lower one
    as what the *rejected* attribution would give. Re-measured on the trace it
    names, `evidence/ccarc3/ablate_nobaseline/ls20-9607627b`, this function now
    returns the lower of the two — so a reader re-deriving the check matches it
    against the rejected reading and concludes the code uses the rule this
    docstring argues against.

    Nothing regressed: the play-starting RESET was later excluded from the count
    (see the block further down), which moved level 0 down by exactly one action.
    The check was simply never re-derived afterwards.

    It is also no longer discriminating — both attributions return the same
    per-level list on that trace, so the example cannot separate them any more,
    and finding one where they still differ is work this note does not do.
    Stated rather than papered over, because a check that agrees with both
    answers is not evidence for either.

    (The figures are described rather than printed. Per-level action counts are
    published medians for some game -- the leak guard in
    `tests/test_ccarc3_no_medians_in_source.py` caught the first draft of this
    paragraph doing exactly that -- and this file is on the solver's path.)

    **Replayed levels are NOT summed, and that is measured, not assumed.**
    A live probe of the scorecard settles it. The API records, per game:

    ===================  ==================================================
    ``total_plays``      how many plays this scorecard holds
    ``actions``          actions **per play**, e.g. ``[3, 1]``
    ``actions_by_level`` **cumulative** actions at each level's completion,
                         per play -- see the warning below
    ``total_actions``    the sum across plays — never a scoring denominator
    ===================  ==================================================

    A ``RESET`` while the action counter is non-zero is a *level* reset:
    ``total_plays`` stays put and ``resets`` increments. A ``RESET`` when the
    counter is zero — which is the state immediately after a level advance —
    **starts a new play**: a new guid, a new ``actions`` row and a new
    ``actions_by_level`` row.

    **The field is a list of PLAYS, each a list of ``[level, cumulative]``
    pairs**, and both worked examples below used to omit that outer list -- one
    showed a single play's pairs, the other dropped the level index entirely.
    Either shape would raise or mis-parse in
    :func:`server_actions_per_level`, which is the function that reads it.
    Verified against a preserved two-play card::

        [[[1, 27], [2, 80], …, [8, 623]],     # play 1
         [[1, 14], [2, 46], …, [8, 488]]]     # play 2

    **And it is cumulative, despite its name.** Measured on a live run of
    `tu93`, whose single play returned ``[[1, 18], [2, 45], [3, 64]]`` while
    ``total_actions`` was **67**. Those entries sum to 127, which is the tell --
    they are ``[level, actions spent by the time that level fell]``, so the
    per-level ``a_l`` the rubric divides by is the **difference** between
    consecutive entries: 18, 27, 19. Reading them as per-level counts inflates
    every level after the first and would have scored this run roughly a third
    of what it earns.

    A one-level game cannot show this -- cumulative and per-level coincide -- so
    the first probe, on `lp85`, looked like agreement and was not evidence.

    So for a one-level game with a human baseline of 7, won in 10 and then
    replayed and won in 7: ``actions_by_level`` is ``[[[1, 10]], [[1, 7]]]`` --
    two plays, each one ``[level, cumulative]`` pair -- and ``total_actions``
    is 17. Since ``Card.high_score = max(scores)`` scores the
    *best* play, ``a_l`` is **7**. The 17 is the running total, not the denominator.

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


def plays(transitions: Sequence["object"]) -> list[list["object"]]:
    """Split a ledger into playthroughs, one list per play.

    A full reset starts a new play — a new guid, a new ``actions`` row and a new
    ``actions_by_level`` row server-side — so the boundaries are exactly the
    transitions flagged ``full_reset``, excluding the one that opens the trace.

    Note the flag is only partly the server's: :mod:`athanor.ccarc3.ledger` ORs
    ``full_reset`` with "the recorded level went down", because the API returned
    ``full_reset: False`` on a transition that took a game from level 6 to 0. A
    new play always returns to level 0, so the fallback catches what the flag
    misses in every case seen so far.
    """
    kept = list(transitions)
    starts = [0] + [i for i, t in enumerate(kept) if t.full_reset and i > 0]
    return [kept[a:b] for a, b in zip(starts, starts[1:] + [len(kept)])]


def score_run(
    transitions: Sequence["object"],
    baselines: Sequence[int],
    *,
    cumulative: bool = False,
    select: str = "best",
) -> EnvironmentScore:
    """RHAE score for one run, straight from its ledger and the game's baselines.

    **Scores the best play, because that is what ARC scores.** This function used
    to score the play that *finished*, on the stated grounds that doing so "is
    what best-of-plays scoring selects". Those are not the same thing, and a live
    card settles which one ARC uses. On a probe run — play 1 clearing level 1
    cleanly, play 2 deliberately fumbling the same level — the card came back
    with::

        runs[0].score = higher     (level_scores [ higher, ...])
        runs[1].score = lower      (level_scores [ lower,  ...])
        environments[0].score = the higher of the two

    The environment took the **maximum**, not the last.

    The two conventions agree whenever a run's last play is also its best, which
    was true of all seventeen games scored before this changed — so no recorded
    result moves. They came within one play of disagreeing on one seven-level
    game, whose middle play was *completed* and still scored below its
    predecessor. Had the run ended there, this
    function would have reported 1.1270 against ARC's 1.1500. The trajectory is
    not monotone, so "last" is not a safe proxy for "best".

    ``select="last"`` restores the old behaviour for comparison. ``cumulative``
    sums every play instead and implies ``select="last"``, since summing has
    already collapsed the plays — see :func:`actions_per_level` for why summing
    is wrong against a per-play denominator.
    """
    if cumulative or select == "last":
        return score_environment(
            baselines, actions_per_level(transitions, len(baselines), cumulative=cumulative)
        )
    if select != "best":
        raise ValueError(f"select must be 'best' or 'last', not {select!r}")

    candidates = [
        score_environment(baselines, actions_per_level(play, len(baselines)))
        for play in plays(transitions)
    ]
    if not candidates:
        # **Unreachable today, and kept deliberately rather than as decoration.**
        # `plays()` always yields at least one list -- its `starts` begins with a
        # literal 0 -- so an empty ledger arrives here as `[[]]`, one empty play,
        # and is scored 0.0 by the live path below. The comment that used to sit
        # on this line ("an empty ledger still deserves a score") described the
        # branch as the thing that handles that case, which it is not; a reader
        # relaxing `plays()` to return `[]` would have believed the empty ledger
        # was already covered. It is covered, by the line below, for a different
        # reason. Found by mutation: deleting this branch changes nothing.
        return score_environment(baselines, [None] * len(baselines))
    # Rank by score, then by raw so two capped plays are separated by efficiency,
    # then by fewest actions so the choice is deterministic rather than incidental.
    return max(
        candidates,
        key=lambda s: (
            s.score,
            s.raw,
            -sum(level.agent or 0 for level in s.levels),
        ),
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


def card_disagreement(scorecard: dict, game_id: str, result: dict) -> str:
    """Why ARC's own card contradicts a run's ``result.json``, or "" if it agrees.

    **A run nobody but us can confirm is not a clean run.** The card is the
    server's per-level count and the only source independent of our trace. When
    a proxy bug started 404ing the reads, the first rollout of ``sb26`` finished
    8/8 in 124 actions with a card frozen at level 3 -- and nothing else showed a
    symptom, because actions carry a ``guid`` and are not card-scoped. The game
    plays perfectly and the numbers you score from stop moving.

    **Only this attempt's plays are in scope, which is why ``playthroughs`` is
    read.** ``levels_completed`` holds one entry per play, and under a shared
    scorecard the server keeps appending to the same ``cards[game_id]`` entry
    across *attempts* of the same game. A plain ``max()`` over it therefore
    carries the high-water mark of every discarded earlier attempt: a game whose
    attempt_1 reached level 8 and was thrown away, and whose attempt_2 has a card
    that stopped updating, still reads ``best=8 >= reached=8`` and banks as
    corroborated -- the exact ``sb26`` failure, now invisible. The last
    ``playthroughs`` entries are the plays this attempt produced; nothing earlier
    belongs in the comparison.

    A card holding *fewer* plays than the trace recorded is itself the failure
    this is named for, so it is reported rather than passed over.

    Action counts are deliberately not compared: our ledger and ARC's have always
    differed by a few for reasons already documented. Levels are the check.
    """
    entry = (scorecard.get("cards") or {}).get(game_id) or {}
    done = list(entry.get("levels_completed") or [])
    plays = int(result.get("playthroughs") or 1)
    if plays < 1:
        # `or 1` coerces a missing or zero count, and lets a negative through --
        # where it becomes a *slice length*, silently choosing a different set of
        # the card's rows to compare. This function is the one check that does
        # not read our own trace, so a corrupted `result.json` is exactly what it
        # may be needed to catch; computing a comparison from the corruption is
        # the one thing it must not do.
        return f"result claims {plays} playthrough(s), which is not a count"

    # **Where this attempt's rows begin, when the attempt recorded it.** Taking
    # the last `plays` rows is right on a per-game card and wrong on a shared
    # one: the server appends every attempt of a game to the same entry, so
    # prior attempts' rows always pad the list and `len(done) < plays` can never
    # fire on a retried game. When the banked attempt contributed ZERO rows --
    # the frozen-card case this whole function is for -- the slice silently
    # returns a DISCARDED attempt's rows and corroborates the run with them.
    # Whether that passes depends only on how far the thrown-away run got.
    # `ArcClient.open` snapshots the boundary against the lent card.
    before = result.get("card_plays_at_open")
    before = int(before) if before is not None else -1
    if before >= 0:
        mine = done[before:]
        if len(mine) < plays:
            return (f"card holds {len(mine)} play(s) for this attempt "
                    f"(rows {before}..{len(done)}), result claims {plays}")
    else:
        if len(done) < plays:
            return f"card holds {len(done)} play(s), result claims {plays}"
        mine = done[-plays:]
    best = max(mine) if mine else 0
    reached = result.get("levels_reached") or 0
    if best < reached:
        return f"card {best} vs result {reached} levels"
    return ""


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

    **Both sides read the LAST play, which is deliberate and is not the play
    that gets scored.** ``server_actions_per_level`` defaults to ``play=-1``,
    and :func:`actions_per_level` restarts its accumulation at a full reset, so
    a whole-trace read already yields the final play -- checked on a real
    two-play card, where the two sides match element for element. The
    comparison is like-for-like.

    What it does not corroborate is the *scored* play: :func:`score_run` selects
    the **best**, and on a run whose best is not its last this agrees about a
    different playthrough than the one the score came from. Every result banked
    so far has best == last, so nothing recorded turns on it.
    """
    theirs = server_actions_per_level(scorecard, game_id)
    ours = actions_per_level(transitions, n_levels)
    return [
        (i, ours[i], theirs[i])
        for i in range(min(len(theirs), len(ours)))
        if ours[i] != theirs[i]
    ]
