"""Tests for search over a forward model.

This module exists because a real solver hand-rolled BFS twice and the harness
offered nothing. The second rewrite happened when it discovered launchers --
cells that slide the avatar across the board rather than moving it one square --
which is why the search takes a step function and not a grid.
"""

from __future__ import annotations

import pytest

from athanor.ccarc3 import reachable, shortest_path

WALLS = {(1, 1), (1, 2), (1, 3)}
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def grid_step(pos, action):
    dy, dx = DIRS[action]
    nxt = (pos[0] + dy, pos[1] + dx)
    if not (0 <= nxt[0] < 5 and 0 <= nxt[1] < 5) or nxt in WALLS:
        return pos          # a blocked move is a self-loop, not an error
    return nxt


def test_it_finds_the_shortest_route():
    path = shortest_path(grid_step, (0, 0), (0, 4))
    assert path == [4, 4, 4, 4]


def test_it_routes_around_a_wall():
    path = shortest_path(grid_step, (0, 1), (2, 2))
    assert path is not None
    pos = (0, 1)
    for a in path:
        pos = grid_step(pos, a)
    assert pos == (2, 2)
    # Manhattan distance is 3, but the wall at (1,1) forces a detour via
    # column 0: left, down, down, right, right.
    assert len(path) == 5, "the direct route is blocked, so it must go around"


def test_an_unreachable_goal_returns_none():
    boxed = lambda p, a: p            # nothing ever moves
    assert shortest_path(boxed, (0, 0), (4, 4)) is None


def test_the_start_being_the_goal_is_an_empty_path():
    assert shortest_path(grid_step, (2, 2), (2, 2)) == []


def test_a_goal_predicate_works_as_well_as_a_state():
    path = shortest_path(grid_step, (0, 0), lambda p: p[0] == 4)
    assert path is not None and len(path) == 4


def test_it_handles_a_launcher_that_is_not_a_one_cell_move():
    """The mechanic that forced the solver's second rewrite."""
    def step(pos, action):
        nxt = grid_step(pos, action)
        return (4, 4) if nxt == (0, 2) else nxt

    path = shortest_path(step, (0, 0), (4, 4))
    assert path == [4, 4], "two moves onto the launcher, not a walk across"


def test_custom_action_sets_are_respected():
    right_only = shortest_path(grid_step, (0, 0), (0, 3), actions=(4,))
    assert right_only == [4, 4, 4]
    assert shortest_path(grid_step, (0, 0), (3, 0), actions=(4,)) is None


def test_reachable_reports_the_whole_component():
    seen = reachable(grid_step, (0, 0))
    assert (0, 0) in seen and (4, 4) in seen
    assert not (WALLS & seen), "walls are not states you can occupy"


def test_a_runaway_step_function_is_caught_not_hung():
    counter = {"n": 0}

    def never_repeats(pos, action):
        counter["n"] += 1
        return (pos[0], counter["n"])   # a brand new state every call

    with pytest.raises(RuntimeError, match="exceeded"):
        shortest_path(never_repeats, (0, 0), (99, 99), max_states=500)
