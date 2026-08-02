"""Tests for the official ARC-AGI-3 RHAE scoring rubric.

The published validation example is the load-bearing test here. Everything else
in this file guards a specific way the rubric is easy to implement wrongly:
squaring after the cap instead of before, weighting levels equally, averaging
levels globally instead of per environment, or forgetting that the completion
cap is what stops early speed compensating for late failure.
"""

from __future__ import annotations

import pytest

from athanor.ccarc3.scoring import (
    LEVEL_SCORE_CAP,
    environment_score,
    score_environment,
    total_score,
)


def test_the_published_validation_example():
    """Five levels, baseline 10 each, agent [10, 20, 10, 5, incomplete].

    Level scores 1.00, 0.25, 1.00, 1.15 (capped from 4.00), 0.00.
    Raw = (1*1.00 + 2*0.25 + 3*1.00 + 4*1.15 + 5*0) / 15 = 9.1/15 ≈ 0.6067.
    Cap = (1+2+3+4)/15 = 10/15 ≈ 0.6667. Score = min = 0.6067.
    """
    result = score_environment([10, 10, 10, 10, 10], [10, 20, 10, 5, None])
    assert result.raw == pytest.approx(9.1 / 15)
    assert result.cap == pytest.approx(10 / 15)
    assert result.score == pytest.approx(0.6067, abs=1e-4)


def test_the_ratio_is_squared_so_twice_the_actions_scores_a_quarter():
    """Not a half. This is what makes efficiency dominate the benchmark."""
    assert environment_score([10], [20]) == pytest.approx(0.25)
    assert environment_score([10], [100]) == pytest.approx(0.01)


def test_the_cap_is_applied_after_squaring_not_before():
    """Capping the *ratio* at 1.15 then squaring gives 1.3225 — a different rubric.

    Baseline 10 against 5 actions: ratio 2.0, squared 4.0, capped to 1.15.
    """
    (level,) = score_environment([10], [5]).levels
    assert level.score == LEVEL_SCORE_CAP
    assert level.capped


def test_later_levels_are_worth_more_than_earlier_ones():
    """Weights are the 1-indexed level numbers, so level 3 counts triple level 1."""
    early = score_environment([10, 10, 10], [5, 20, 20]).raw
    late = score_environment([10, 10, 10], [20, 20, 5]).raw
    assert late > early, "being fast on the last level must be worth more"


def test_the_completion_cap_stops_early_speed_paying_for_late_failure():
    """Three perfect levels out of five cannot score above 6/15."""
    result = score_environment([10] * 5, [1, 1, 1, None, None])
    assert result.raw == pytest.approx((1 + 2 + 3) * LEVEL_SCORE_CAP / 15)
    assert result.cap == pytest.approx(6 / 15)
    assert result.score == pytest.approx(6 / 15)
    assert result.cap_binds


def test_an_environment_cannot_exceed_one_even_when_every_level_caps():
    """Every level at 1.15 gives raw 1.15; the completion cap holds it to 1.0.

    This is the case every winning run in this project lands in, which is why
    the benchmark score here is effectively a count of environments finished.
    """
    result = score_environment([100] * 4, [1, 1, 1, 1])
    assert result.raw == pytest.approx(LEVEL_SCORE_CAP)
    assert result.cap == 1.0
    assert result.score == 1.0


def test_finishing_nothing_scores_zero():
    assert environment_score([10, 10], [None, None]) == 0.0


def test_a_level_completed_after_an_incomplete_one_is_an_error():
    """Levels are sequential; accepting this would silently score an impossible run."""
    with pytest.raises(ValueError, match="sequential"):
        score_environment([10, 10], [None, 10])


def test_total_is_the_unweighted_mean_of_environment_scores():
    """Not a global average over levels — that would let a long game outvote a short one."""
    assert total_score([1.0, 0.0]) == pytest.approx(50.0)
    assert total_score([1.0, 1.0, 0.5]) == pytest.approx(83.333, abs=1e-3)


def test_unattempted_environments_are_zeros_in_the_denominator():
    """The published benchmark divides by the whole evaluated set.

    Seven environments won out of 25 is 28%, not 100% — scoring only what was
    attempted is the single easiest way to overstate a result here.
    """
    attempted = [1.0] * 7
    assert total_score(attempted) == pytest.approx(100.0)
    assert total_score(attempted + [0.0] * 18) == pytest.approx(28.0)


def test_malformed_input_is_refused_rather_than_scored():
    with pytest.raises(ValueError, match="equal length"):
        score_environment([10, 10], [10])
    with pytest.raises(ValueError, match="at least one level"):
        score_environment([], [])
    with pytest.raises(ValueError, match="baselines must be positive"):
        score_environment([0], [10])
    with pytest.raises(ValueError, match="must be positive"):
        score_environment([10], [0])
    with pytest.raises(ValueError, match="must lie in"):
        total_score([1.5])
