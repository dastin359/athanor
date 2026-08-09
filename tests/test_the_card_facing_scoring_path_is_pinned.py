"""The half of ``scoring.py`` that the shared-scorecard sweep runs through.

The RHAE core carries mutation evidence in its own docstrings. The functions
added later for card corroboration -- ``plays``, ``score_run``'s best-of-plays
selection, ``server_actions_per_level``, ``card_disagreement``,
``disagreements_with_server`` -- did not, and 11 of 23 mutants against them
survived the suite. None of them was a code defect: every pristine behaviour
checked out under a live probe. They were all holes, in the code the 25-game
sweep depends on for deciding whether a run may be banked.

The two worth naming:

- ``score_run`` could be made to score the **first** play instead of the best
  and nothing failed, even though scoring the *last* play was caught. Every
  fixture in the suite had its best play first or last, so "best" was only ever
  pinned from one side. The middle-is-best case is the one that separates all
  three readings, and it did not exist.
- ``card_disagreement`` could accept a card one level behind the result, or read
  the last of an attempt's plays instead of the best, and stay silent. That
  function exists because a frozen card once let a run finish 8/8 against a card
  stuck at level 3 with no other symptom -- so a hole in it is a hole in the
  only check that is independent of our own trace.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from athanor.ccarc3.scoring import (
    actions_per_level,
    card_disagreement,
    disagreements_with_server,
    plays,
    score_run,
    server_actions_per_level,
)


class Step(NamedTuple):
    """The only three attributes ``plays`` and ``actions_per_level`` read."""

    action: str
    level: int
    full_reset: bool = False


def play(per_level: list[int], *, opens_with_full_reset: bool = False) -> list[Step]:
    """A playthrough that spends ``per_level[i]`` actions clearing level ``i``.

    Levels are recorded as *levels completed after the action*, so the action
    that finishes level i is stamped i+1 -- the attribution ``actions_per_level``
    documents at length. The opening RESET is not an action and is excluded.
    """
    steps = [Step("RESET", 0, opens_with_full_reset)]
    for level, spend in enumerate(per_level):
        steps += [Step("ACTION1", level) for _ in range(spend - 1)]
        steps.append(Step("ACTION1", level + 1))
    return steps


def trace(*per_level_plays: list[int]) -> list[Step]:
    out: list[Step] = []
    for index, spend in enumerate(per_level_plays):
        out += play(spend, opens_with_full_reset=index > 0)
    return out


# --------------------------------------------------------------------------- #
# plays()
# --------------------------------------------------------------------------- #


def test_the_opening_reset_does_not_open_an_empty_play():
    """A trace whose very first transition is flagged ``full_reset``.

    Dropping the ``i > 0`` guard puts a zero-length play at the front, which
    ``score_run`` then scores as a playthrough that completed nothing.
    """
    steps = [Step("RESET", 0, True), Step("ACTION1", 0), Step("ACTION1", 1)]
    assert [len(p) for p in plays(steps)] == [3]


def test_every_play_holds_at_least_one_transition():
    steps = trace([2], [3], [4])
    assert [len(p) for p in plays(steps)] == [3, 4, 5]
    assert all(p for p in plays(steps))


def test_the_fixture_itself_reproduces_the_intended_action_counts():
    """A positive control: if `play()` miscounts, every test below is vacuous."""
    assert actions_per_level(play([60, 95]), 2) == [60, 95]
    assert actions_per_level(play([2]), 2) == [2, None]


# --------------------------------------------------------------------------- #
# score_run: best of plays, from both sides
# --------------------------------------------------------------------------- #


def test_the_best_play_is_scored_even_when_it_is_neither_first_nor_last():
    """Three plays, the middle one best -- the only shape that separates
    "best" from "first", "last" and "worst" at once."""
    baselines = [100, 200]
    ts = trace([150], [90, 150], [120])       # only play 2 clears level 1
    best = score_run(ts, baselines)
    assert [level.agent for level in best.levels] == [90, 150]
    assert best.completed == 2

    from athanor.ccarc3.scoring import score_environment
    each = [score_environment(baselines, actions_per_level(p, 2)) for p in plays(ts)]
    assert best.score > each[0].score and best.score > each[2].score
    assert best.score == max(s.score for s in each)


def test_select_last_still_scores_the_final_play():
    ts = trace([150], [90, 150], [120])
    last = score_run(ts, [100, 200], select="last")
    assert last.completed == 1


def test_two_plays_tied_on_score_are_separated_by_efficiency():
    """Both cap out, so ``score`` cannot choose; ``raw`` must.

    The pair is chosen so the two tie-breaks *disagree*: play A spends 155
    actions for a raw of 1.1500, play B spends 154 for 1.1439. Fewest-actions
    would take B. Efficiency is the intended rule and takes A -- which is what
    makes this fixture able to tell the two apart at all.
    """
    baselines = [100, 200]
    ts = trace([60, 95], [94, 60])
    from athanor.ccarc3.scoring import score_environment
    a, b = (score_environment(baselines, actions_per_level(p, 2)) for p in plays(ts))
    assert a.score == b.score, "the fixture must actually tie on score"
    assert a.raw > b.raw and sum([60, 95]) > sum([94, 60]), "and the rules must disagree"

    assert [level.agent for level in score_run(ts, baselines).levels] == [60, 95]


def test_two_plays_tied_on_score_and_efficiency_are_separated_by_actions():
    """Same capped level score, so ``raw`` ties too; fewest actions decides,
    which is what makes the choice deterministic rather than incidental."""
    baselines = [10, 20]
    ts = trace([3], [2])                       # both cap; 2 actions is the cheaper
    from athanor.ccarc3.scoring import score_environment
    a, b = (score_environment(baselines, actions_per_level(p, 2)) for p in plays(ts))
    assert (a.score, a.raw) == (b.score, b.raw), "the fixture must tie on both"

    assert [level.agent for level in score_run(ts, baselines).levels] == [2, None]


def test_the_choice_does_not_depend_on_the_order_the_plays_appear_in():
    baselines = [100, 200]
    forwards = score_run(trace([150], [90, 150]), baselines)
    backwards = score_run(trace([90, 150], [150]), baselines)
    assert forwards.score == backwards.score
    assert [l.agent for l in forwards.levels] == [l.agent for l in backwards.levels]


def test_an_empty_ledger_scores_zero_rather_than_raising():
    result = score_run([], [10, 20])
    assert result.score == 0.0 and result.completed == 0


# --------------------------------------------------------------------------- #
# server_actions_per_level
# --------------------------------------------------------------------------- #


def test_the_default_play_is_the_most_recent_not_the_first():
    """Under a shared scorecard the server keeps appending plays to one entry,
    so "which row" is the whole question. Defaulting to row 0 reads the oldest
    play on the card -- possibly a discarded attempt's."""
    card = {"cards": {"g": {"actions_by_level": [[[1, 10]], [[1, 33]]]}}}
    assert server_actions_per_level(card, "g") == [33]
    assert server_actions_per_level(card, "g", play=0) == [10]
    assert server_actions_per_level(card, "g", play=-1) == [33]


def test_a_game_with_no_card_is_refused_rather_than_read_as_empty():
    """Returning [] would make ``disagreements_with_server`` compare nothing and
    report agreement -- a missing card reading as a clean one."""
    with pytest.raises(KeyError, match="no card for zz99"):
        server_actions_per_level({"cards": {"other": {}}}, "zz99")


# --------------------------------------------------------------------------- #
# card_disagreement
# --------------------------------------------------------------------------- #


def _card(levels_completed: list[int]) -> dict:
    return {"cards": {"g": {"levels_completed": levels_completed}}}


def test_a_card_one_level_behind_the_result_is_reported():
    """Off by one is the whole failure: a frozen card lags, it does not vanish."""
    message = card_disagreement(_card([6]), "g", {"playthroughs": 1, "levels_reached": 7})
    assert message and "6" in message and "7" in message


def test_a_card_that_agrees_says_nothing():
    assert card_disagreement(_card([7]), "g", {"playthroughs": 1, "levels_reached": 7}) == ""


def test_a_card_ahead_of_the_result_is_not_a_disagreement():
    """ARC scores the best play; our bar is stricter, so the card may legitimately
    stand higher than what we chose to claim."""
    assert card_disagreement(_card([9]), "g", {"playthroughs": 1, "levels_reached": 7}) == ""


def test_the_cards_best_play_is_read_not_its_last():
    """The attempt's rows are [4, 7, 5]. ARC scores the max, so 7 corroborates a
    result of 7; reading the last row sees 5 and invents a disagreement."""
    result = {"playthroughs": 3, "levels_reached": 7}
    assert card_disagreement(_card([4, 7, 5]), "g", result) == ""


def test_an_attempt_that_contributed_no_rows_is_reported():
    """The frozen-card case this function is named for, under a shared card:
    the boundary says this attempt starts at row 2 and the card still holds 2
    rows, so the attempt put nothing there."""
    message = card_disagreement(
        _card([4, 6]), "g",
        {"playthroughs": 1, "levels_reached": 5, "card_plays_at_open": 2},
    )
    assert message and "0 play(s)" in message


def test_a_prior_attempts_rows_cannot_corroborate_this_one():
    """Without the boundary, the last-N slice would hand back the discarded
    attempt's level 8 and bank the run."""
    lent = _card([8, 8])
    assert card_disagreement(lent, "g", {"playthroughs": 1, "levels_reached": 8}) == ""
    scoped = card_disagreement(
        lent, "g",
        {"playthroughs": 1, "levels_reached": 8, "card_plays_at_open": 2},
    )
    assert scoped, "rows belonging to a previous attempt must not corroborate"


# --------------------------------------------------------------------------- #
# disagreements_with_server
# --------------------------------------------------------------------------- #


def test_a_disagreement_is_reported_whichever_side_is_higher():
    """Reporting only one direction hides half the failures, and which half it
    hides depends on whether our count drifted up or down."""
    card = {"cards": {"g": {"actions_by_level": [[[1, 60], [2, 155]]]}}}
    ours_lower = disagreements_with_server(play([59, 95]), card, "g", 2)
    ours_higher = disagreements_with_server(play([61, 95]), card, "g", 2)
    assert ours_lower and ours_lower[0][0] == 0
    assert ours_higher and ours_higher[0][0] == 0


def test_agreement_is_reported_as_nothing():
    card = {"cards": {"g": {"actions_by_level": [[[1, 60], [2, 155]]]}}}
    assert disagreements_with_server(play([60, 95]), card, "g", 2) == []


def test_a_nonsense_playthrough_count_is_refused_not_used_as_a_slice():
    """``plays`` becomes a slice length. A negative one quietly compares a
    different set of the card's rows, and this function exists to catch a
    corrupted result rather than compute from one."""
    for bad in (-1, -3):
        message = card_disagreement(_card([7]), "g",
                                    {"playthroughs": bad, "levels_reached": 7})
        assert "not a count" in message, f"playthroughs={bad} was used anyway"


def test_a_missing_or_zero_playthrough_count_still_means_one():
    """The existing coercion is deliberate and must survive the new guard."""
    assert card_disagreement(_card([7]), "g", {"levels_reached": 7}) == ""
    assert card_disagreement(_card([7]), "g",
                             {"playthroughs": 0, "levels_reached": 7}) == ""
