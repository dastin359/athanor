"""A completed level costing zero actions is refused, never scored.

Found by mutating `scoring.py`: flipping the zero-action branch from
`LEVEL_SCORE_CAP` to `0.0` left all 1070 tests green. The branch was
unreachable -- `agent <= 0` raises four lines above it -- but it implemented, in
the line immediately after, exactly the policy the guard's own comment spends a
paragraph refusing:

    "Scoring 0 as 'cleared for free' means the cap, 1.15 -- the maximum a level
     can earn. [...] if a zero ever arrives from a parsing fault rather than a
     real double advance, accepting it awards a defect the best possible score
     and the run reads as excellent."

Dead code is not why it mattered. Anyone relaxing the guard to admit a genuine
double advance would have silently inherited the cap-award rather than making
the deliberate decision the guard demands. These pin the rule so the two cannot
drift apart again.
"""

from __future__ import annotations

import inspect

import pytest

from athanor.ccarc3 import scoring
from athanor.ccarc3.scoring import (
    LEVEL_SCORE_CAP,
    score_environment,
)


def test_a_zero_action_completed_level_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        score_environment([10], [0])


def test_a_zero_among_real_levels_raises_too() -> None:
    """The realistic shape: a double advance mid-run, not a one-level fixture."""
    with pytest.raises(ValueError, match="must be positive"):
        score_environment([10, 12, 14], [5, 0, 7])


def test_a_negative_count_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        score_environment([10], [-1])


def test_no_completed_level_can_score_above_the_cap() -> None:
    """The property the dead branch would have broken if it ever ran.

    One action against a 500-action baseline is the most extreme ratio the
    benchmark admits; it must still clip to the cap.
    """
    result = score_environment([500], [1])
    assert result.levels[0].score == LEVEL_SCORE_CAP
    assert all(level.score <= LEVEL_SCORE_CAP for level in result.levels)


def test_the_source_no_longer_awards_the_cap_for_zero() -> None:
    """Pinned by name: the ternary must not come back.

    A behavioural test cannot catch its return, because the guard makes it
    unreachable -- which is precisely how it survived unnoticed. Reading the
    source is the only check that fails when it reappears.
    """
    # **Comments stripped first.** The first version searched the raw source,
    # and the comment explaining the removal quotes the very ternary it forbids
    # -- so the test failed on the fixed code. It also made the mutation harness
    # report an unrelated mutant as "killed", because this failure appeared
    # under every mutant alike. A check that reads prose as code is the same
    # proxy error as the rest of this file, one layer up.
    src = "\n".join(ln for ln in inspect.getsource(scoring.score_environment).splitlines()
                    if not ln.lstrip().startswith("#"))
    assert "LEVEL_SCORE_CAP if agent == 0" not in src, (
        "the zero-action cap-award is back. It is unreachable today, but it "
        "states the opposite of the guard above it, and relaxing that guard "
        "would silently award 1.15 to a level that cost nothing."
    )


def test_the_guard_still_precedes_any_scoring() -> None:
    """If the raise ever moves below the score line, zero becomes scorable."""
    src = inspect.getsource(scoring.score_environment)
    guard = src.index("if agent <= 0:")
    scored = src.index("score = min(LEVEL_SCORE_CAP")
    assert guard < scored, (
        "the zero-action guard no longer runs before the level is scored"
    )
