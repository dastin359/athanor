"""What ``ArcClient`` reports about a play, and what it refuses to compute.

Ten of twenty-six mutants against ``client.py`` survived the suite. Two of them
are the reason this file is named the way it is: ``done`` could be made to
return True on ``GAME_OVER`` and ``dead`` True on ``WIN``, and nothing failed.
Both are solver-facing -- the client's own docstring shows ``while not c.done``
as the play loop -- so a loss reading as a win ends the loop on a dead game and
banks it.

The rest are RHAE arithmetic and the per-level effect tally: a level credited
without an action scoring zero instead of the cap, the completion cap not
applied to the current score, a shorter-than-expected baseline list scoring
anyway rather than declining, and a stale tally carried across a board that no
longer exists.

None of these was a code defect. All of them were untested.
"""

from __future__ import annotations

import pytest

from athanor.ccarc3 import ArcClient, GameInfo
from athanor.ccarc3 import client as client_mod


def _frame(state="NOT_FINISHED", score=0, grids=None):
    return {
        "game_id": "g1",
        "frame": [[[0]]] if grids is None else grids,
        "state": state,
        "score": score,
        "full_reset": False,
        "available_actions": [0, 1],
        "action_input": {"id": 1},
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A client wired to a fake server, with published baselines."""
    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "card-1"}
        if url.endswith("/scorecard/close"):
            return {"closed": True}
        name = url.rsplit("/", 1)[-1]
        reply = _frame()
        reply["action_input"] = {"id": 0 if name == "RESET" else int(name.removeprefix("ACTION"))}
        return reply

    monkeypatch.setattr(client_mod, "_post", fake_post)
    monkeypatch.setenv("ARC_API_KEY", "k")
    info = GameInfo("g1", baseline_actions=(10, 20, 30))
    return ArcClient("g1", trace_path=tmp_path / "t.jsonl", info=info).open()


# --------------------------------------------------------------------------- #
# terminal states
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state, expect_done, expect_dead",
    [("WIN", True, False), ("GAME_OVER", False, True),
     ("NOT_FINISHED", False, False), ("NOT_PLAYED", False, False)],
)
def test_done_and_dead_name_exactly_one_state_each(client, state, expect_done, expect_dead):
    """The client's own play loop is ``while not c.done``. A loss that reads as
    done ends that loop on a dead game."""
    client.state = state
    assert client.done is expect_done
    assert client.dead is expect_dead


def test_a_play_is_never_both_won_and_lost(client):
    for state in ("WIN", "GAME_OVER", "NOT_FINISHED"):
        client.state = state
        assert not (client.done and client.dead)


# --------------------------------------------------------------------------- #
# withholding: hiding a number must not switch a limit off
# --------------------------------------------------------------------------- #


def test_the_level_budget_survives_a_withheld_baseline(client):
    """``baseline_here`` returns None under ``quiet_pace`` so the solver cannot
    read it. ``level_budget`` must still enforce it -- reading the *visible*
    value would silently disable the ceiling for exactly the runs that hide it.
    """
    client.level_budget_multiple = 4
    client.quiet_pace = False
    loud = client.level_budget
    assert loud == 40

    client.quiet_pace = True
    assert client.baseline_here is None, "the solver can still see the baseline"
    assert client.level_budget == loud, "hiding the number switched the ceiling off"


def test_a_withheld_score_is_none_rather_than_computed(client):
    client.quiet_pace = True
    assert client.score_now is None
    assert client.score_ceiling is None


# --------------------------------------------------------------------------- #
# RHAE arithmetic
# --------------------------------------------------------------------------- #


def test_a_level_credited_without_an_action_takes_the_cap(client):
    """A multi-level advance credits a level that cost nothing. It beat the human
    by definition, so it scores the cap -- and, more sharply, dividing by its
    zero cost would raise."""
    client.win_levels = 1
    client.level_costs = (0,)
    score = client.score_now
    assert score is not None and score > 0.0


def test_a_zero_cost_level_does_not_divide_by_zero(client):
    client.win_levels = 2
    client.level_costs = (0, 5)
    assert client.score_now is not None            # no ZeroDivisionError


def test_the_current_score_is_limited_by_levels_actually_cleared(client):
    """``score_now`` applies the completion cap; ``score_ceiling`` does not.

    Dropping the cap makes a play that cleared one level of three report what a
    play that cleared all three would -- which is the number a replay decision
    turns on.
    """
    # `win_levels` is the game's TOTAL level count, not the number won; the
    # levels actually finished are the entries in `level_costs`. One of three,
    # played far under baseline, so `raw` caps out and only the completion cap
    # can hold the current score down.
    client.win_levels = 3
    client.level_costs = (1,)
    now, ceiling = client.score_now, client.score_ceiling
    assert now is not None and ceiling is not None
    assert now <= client.completion_cap + 1e-12
    assert ceiling > now, "the ceiling must not be held down by the completion cap"


def test_a_score_needs_a_baseline_for_every_level_it_counts(client):
    """Fewer baselines than levels means the arithmetic cannot be done. None is
    the honest answer; indexing past the list is not."""
    client.info = GameInfo("g1", baseline_actions=(10,))
    client.win_levels = 3
    client.level_costs = (5, 5, 5)
    assert client.score_now is None


def test_later_levels_weigh_more_than_earlier_ones(client):
    """RHAE weights level i by i. Spending the same effort late is worth more,
    and an unweighted mean would report a different number entirely."""
    client.win_levels = 2
    client.info = GameInfo("g1", baseline_actions=(10, 10))
    client.level_costs = (5, 20)                   # good early, poor late
    poor_late = client.score_now
    client.level_costs = (20, 5)                   # poor early, good late
    good_late = client.score_now
    assert poor_late is not None and good_late is not None
    assert good_late > poor_late


# --------------------------------------------------------------------------- #
# the per-level effect tally
# --------------------------------------------------------------------------- #


def test_a_wasted_action_across_a_level_boundary_clears_the_tally(client):
    """A wasted action returns no frame, so it is no evidence about the action --
    but if the *level* changed anyway the tally describes a board that no longer
    exists, and carrying it forward reports the old level's waste as the new
    level's."""
    client.level_tried, client.level_dead, client.level_repeats = 7, 3, 2
    client.level_revisits = 4
    client._account_effect({"frame": []}, "ACTION1", {}, board_replaced=True)
    assert (client.level_tried, client.level_dead, client.level_repeats) == (0, 0, 0)
    assert client.level_revisits == 0


def test_a_wasted_action_within_a_level_leaves_the_tally_alone(client):
    client.level_tried, client.level_dead = 7, 3
    client._account_effect({"frame": []}, "ACTION1", {}, board_replaced=False)
    assert (client.level_tried, client.level_dead) == (7, 3)


def test_a_no_op_is_counted_dead_and_not_as_a_revisit(client):
    """A no-op leaves the board on a key already seen. Counting it as a revisit
    double-reports what ``level_dead`` already covers."""
    client._account_effect(_frame(grids=[[[1]]]), "ACTION1", {}, board_replaced=True)
    before = client.level_revisits
    client._account_effect(_frame(grids=[[[1]]]), "ACTION1", {}, board_replaced=False)
    assert client.level_dead == 1, "an unchanged board is a dead action"
    assert client.level_revisits == before, "a no-op is not a revisit"


def test_returning_to_an_earlier_board_is_a_revisit(client):
    """The positive control: the counter the test above holds at zero must be
    reachable, or it is asserting an absence nothing could produce."""
    client._account_effect(_frame(grids=[[[1]]]), "ACTION1", {}, board_replaced=True)
    client._account_effect(_frame(grids=[[[2]]]), "ACTION1", {}, board_replaced=False)
    client._account_effect(_frame(grids=[[[1]]]), "ACTION1", {}, board_replaced=False)
    assert client.level_revisits == 1


def test_two_clicks_at_different_places_are_not_the_same_dead_action(client):
    """``ACTION6`` carries coordinates. Keying repeats on the name alone makes a
    second click anywhere look like a repeat of the first."""
    client._account_effect(_frame(grids=[[[1]]]), "ACTION6", {"x": 1, "y": 1},
                           board_replaced=True)
    client._account_effect(_frame(grids=[[[1]]]), "ACTION6", {"x": 1, "y": 1},
                           board_replaced=False)
    client._account_effect(_frame(grids=[[[1]]]), "ACTION6", {"x": 9, "y": 9},
                           board_replaced=False)
    assert client.level_dead == 2
    assert client.level_repeats == 0, "different coordinates are different actions"
