"""Tests for the local bench game.

Skipped unless ``arcengine`` is importable — it is a bench dependency, not a
runtime one, so athanor does not carry it. Install it (``pip install
'athanor[bench]'``) to run these.

The bench exists to make §5's central claim falsifiable rather than merely
stated: rules are level-scoped and do not transfer, mechanics are game-scoped
and do. `lineage` is built so those two come apart — the goal *colour* changes
at level 1 while the goal *mechanic* does not — and these tests check the game
really has that structure. A bench that quietly failed to separate them would
let the claim pass unexamined, which is worse than having no bench.
"""

from __future__ import annotations

import pytest

pytest.importorskip("arcengine")

from athanor.ccarc3 import load  # noqa: E402
from athanor.ccarc3.bench import play, scripted  # noqa: E402
from athanor.ccarc3.bench.lineage import (  # noqa: E402
    GOAL_COLOUR,
    LETHAL_COLOUR,
    MOVES,
    WALL_COLOUR,
    build_game,
)

WALK_TO_GOAL = [0] + [4] * 9 + [2] * 9  # RESET, then right x9, down x9


def test_the_goal_colour_changes_between_levels():
    """The whole point of the bench: a level rule that must not transfer."""
    assert GOAL_COLOUR[0] != GOAL_COLOUR[1]
    assert len({GOAL_COLOUR[k] for k in GOAL_COLOUR}) > 1


def test_the_game_has_four_levels_and_a_win_condition():
    g = build_game()
    assert g.win_score == 4
    assert len(g._levels) == 4


def test_walking_to_the_goal_clears_level_zero(tmp_path):
    g = build_game()
    r = play(g, scripted(WALK_TO_GOAL), tmp_path / "t.jsonl")
    assert r.scores[-1] == 1, "reaching the goal must advance the level"
    assert r.deaths == 0
    assert r.actions_used == len(WALK_TO_GOAL)


def test_every_action_lands_in_the_ledger_with_its_frames(tmp_path):
    g = build_game()
    path = tmp_path / "t.jsonl"
    play(g, scripted(WALK_TO_GOAL), path)
    ts = load(path)
    assert len(ts) == len(WALK_TO_GOAL)
    assert ts[0].action == "RESET"
    assert all(t.after.shape == (64, 64) for t in ts)


def test_the_level_advance_is_flagged_as_replacing_the_board(tmp_path):
    """Spatial rules must exclude it, so the bench has to produce one."""
    g = build_game()
    path = tmp_path / "t.jsonl"
    play(g, scripted(WALK_TO_GOAL), path)
    ts = load(path)
    # The clearing action is the one whose before/after straddle two boards.
    assert ts[-1].crosses_level, "the action that cleared the level must be flagged"
    assert not any(t.crosses_level for t in ts[:-1])
    # The opening RESET is a genuine full reset, so it is board_replaced too --
    # correctly, though its `before` is None so spatial rules skip it anyway.
    assert ts[0].full_reset and ts[0].board_replaced
    assert ts[0].before is None


def test_a_second_run_does_not_inherit_the_first_trace(tmp_path):
    """An append-only ledger welds runs together and invents transitions."""
    path = tmp_path / "t.jsonl"
    play(build_game(), scripted(WALK_TO_GOAL), path)
    r = play(build_game(), scripted([0, 4, 4]), path)
    assert r.actions_used == 3
    assert len(load(path)) == 3


def test_appending_is_available_but_must_be_asked_for(tmp_path):
    path = tmp_path / "t.jsonl"
    play(build_game(), scripted([0, 4]), path)
    play(build_game(), scripted([0, 4]), path, append=True)
    assert len(load(path)) == 4


def test_a_scripted_policy_stops_rather_than_idling_on_reset(tmp_path):
    """Idling on RESET once destroyed a completed level: the server's action
    counter is zero right after an advance, so that RESET is a full reset."""
    g = build_game()
    r = play(g, scripted(WALK_TO_GOAL), tmp_path / "t.jsonl", max_actions=500)
    assert r.actions_used == len(WALK_TO_GOAL), "it must stop, not keep resetting"
    assert r.scores[-1] == 1, "the completed level must survive"


def test_the_lethal_and_wall_colours_are_distinct_from_the_goals():
    """Level 2 and 3 introduce mechanics that must be recognisable as new."""
    assert LETHAL_COLOUR not in GOAL_COLOUR.values()
    assert WALL_COLOUR not in GOAL_COLOUR.values()
    assert LETHAL_COLOUR != WALL_COLOUR


def test_moves_cover_the_four_directions():
    assert sorted(MOVES) == [1, 2, 3, 4]
    assert len(set(MOVES.values())) == 4


# --------------------------------------------------------------------------- #
# planning, end to end against a real engine
# --------------------------------------------------------------------------- #


def test_a_planned_route_actually_clears_a_level(tmp_path):
    """shortest_path had only synthetic tests. This drives a real engine with it.

    Parse the board out of the 64x64 render, plan with the search, execute the
    plan blind, and check the level cleared. If the search were wrong the level
    would not advance -- there is no partial credit here.
    """
    import numpy as np
    from arcengine import ActionInput, GameAction

    from athanor.ccarc3 import shortest_path

    AVATAR, SIZE = 2, 10

    def board(frame):
        # 10 cells across a 64px viewport: 6px each with a 2px letterbox.
        return np.asarray(frame)[2:62:6, 2:62:6]

    def locate(b, colour):
        ys, xs = np.nonzero(b == colour)
        return (int(ys[0]), int(xs[0])) if len(ys) else None

    game = build_game()
    frame = game.perform_action(ActionInput(id=GameAction.from_id(0))).model_dump()
    b = board(frame["frame"][-1])
    start, goal = locate(b, AVATAR), locate(b, GOAL_COLOUR[0])
    assert start is not None and goal is not None

    def step(pos, action):
        dx, dy = MOVES[action]
        nxt = (pos[0] + dy, pos[1] + dx)
        return pos if not (0 <= nxt[0] < SIZE and 0 <= nxt[1] < SIZE) else nxt

    route = shortest_path(step, start, goal, actions=(1, 2, 3, 4))
    assert route is not None
    assert len(route) == abs(goal[0] - start[0]) + abs(goal[1] - start[1]), "must be optimal"

    for action in route:
        frame = game.perform_action(ActionInput(id=GameAction.from_id(action))).model_dump()
    assert frame["levels_completed"] == 1, "executing the plan must clear the level"
