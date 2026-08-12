"""`reachable` returned a partial answer shaped exactly like a complete one.

Both searches in `planning.py` carry the same `max_states` bound against the
same hazard -- a step function that mints a fresh state every call, so the
frontier never closes. They disagreed about what to do when it fires:

    shortest_path:  if len(prev) > max_states: raise RuntimeError(...)
    reachable:      while queue and len(seen) <= max_states:   # ...then return seen

`shortest_path` tells you. `reachable` hands back the states it happened to
visit, as a `set`, with nothing marking it short.

**What makes this worse than an ordinary silent truncation** is `reachable`'s
own docstring, which tells the reader how to interpret a small result: *"An
unexpectedly small reachable set usually means the model thinks a wall is
somewhere it is not."* So the cap firing manufactures precisely the observation
the documentation teaches you to blame on your model. A solver would go
debugging a step function that was working, on the authority of the docstring,
and the harness would never say otherwise. Both modules are exported from
`athanor.ccarc3` for solvers to call.

These tests fail against the truncating version: the runaway case returns a set
instead of raising.
"""
from __future__ import annotations

import pytest

from athanor.ccarc3 import reachable, shortest_path


def _runaway(state, action):
    """A step function that never revisits a state, so the search cannot close."""
    return (state[0] + 1, state[1] + action)


def _box(width, height):
    """A well-behaved grid: bounded, blocked moves are self-loops."""
    def step(state, action):
        x, y = state
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            return state
        return (nx, ny)
    return step


def test_reachable_raises_rather_than_returning_a_short_set():
    with pytest.raises(RuntimeError) as excinfo:
        reachable(_runaway, (0, 0), max_states=500)
    assert "exceeded 500 states" in str(excinfo.value)


def test_both_searches_answer_a_runaway_the_same_way():
    """The bug was that they disagreed. Pin the agreement, not one side of it."""
    errors = []
    for search in (
        lambda: shortest_path(_runaway, (0, 0), (99, 99), max_states=500),
        lambda: reachable(_runaway, (0, 0), max_states=500),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            search()
        errors.append(str(excinfo.value))
    assert errors[0] == errors[1], (
        "the two searches report the same condition differently; that divergence "
        "is what let one of them start truncating"
    )


def test_a_bounded_board_is_returned_whole_and_does_not_raise():
    """The guard must not fire on a search that legitimately finishes."""
    seen = reachable(_box(6, 5), (0, 0), max_states=500)
    assert len(seen) == 30
    assert (5, 4) in seen


def test_a_board_exactly_at_the_cap_is_not_a_runaway():
    """`max_states` is a ceiling on the search, not a ceiling minus one.

    An off-by-one here turns the largest legitimate board into a crash, which is
    the same class of wrongness as truncating -- just in the other direction.
    """
    seen = reachable(_box(10, 10), (0, 0), max_states=100)
    assert len(seen) == 100


def test_a_short_set_now_means_the_board_and_not_the_search():
    """The reading the docstring recommends is only sound if the cap cannot fire quietly."""
    walled = _box(2, 2)
    seen = reachable(walled, (0, 0), max_states=200_000)
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}
