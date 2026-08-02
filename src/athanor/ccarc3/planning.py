"""Search over a forward model.

Added because a real solver needed it and the harness had none. Reading what it
wrote in its own scratch space, it had hand-rolled breadth-first search **twice**
-- once over a passability test, then again over a step function once it
understood that some cells launch the avatar across the board rather than moving
it one square.

That is the more interesting half of a finding about where the difficulty
actually sits. The design's most carefully-reasoned component is §5's
three-valued rule checking, and across a 370-action run the solver used
``Rule``, ``verify`` and ``survey`` exactly **zero** times, despite all three
being documented in the workspace it was given. What it wrote instead was
parsers, a simulator, and search.

The reason is structural. On ARC-AGI-2 you cannot test a hypothesis without
spending your one submission, so verification machinery is the whole game. On
ARC-AGI-3 testing *is* acting: one action tells you whether you were right, for
the price of one action. Verification gets cheap and **planning gets expensive**
-- the hard question stops being "is my rule correct" and becomes "given rules I
already believe, what is the shortest route".

So this module exists, and §5 stays because it is cheap and correct, but it is
no longer the centre of gravity.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Hashable, Iterable, Sequence

__all__ = ["shortest_path", "reachable"]


def shortest_path(
    step: Callable[[Any, int], Any],
    start: Hashable,
    goal: Hashable | Callable[[Any], bool],
    actions: Sequence[int] = (1, 2, 3, 4),
    *,
    max_states: int = 200_000,
) -> list[int] | None:
    """Fewest actions from ``start`` to ``goal``, or ``None`` if unreachable.

    ``step(state, action) -> state`` is your forward model. It should return the
    state unchanged when an action does nothing -- a blocked move is a self-loop,
    not an error -- and may return any hashable state, not just a coordinate.

    ``goal`` is a state to reach, or a predicate over states.

    Actions default to the four directions. Pass the game's real
    ``available_actions`` when they differ; searching over actions the game does
    not accept wastes nothing here but will mislead you about the route length.

    Because it searches over a *step function* rather than a grid, it handles
    mechanics that are not one-cell moves -- launchers that slide the avatar
    until it hits something, teleports, wrap-around edges -- as long as your
    model of them is in ``step``. That generality is the point: the solver that
    prompted this wrote a grid version first and had to rewrite it when it found
    launchers.

    The path is optimal *for your model*. If the model is wrong the path is
    wrong, which is what :func:`athanor.ccarc3.predict` is for -- check the model
    against the ledger before trusting a route from it.
    """
    reached = goal if callable(goal) else (lambda s: s == goal)
    if reached(start):
        return []

    prev: dict[Any, tuple[Any, int] | None] = {start: None}
    queue: deque[Any] = deque([start])
    while queue:
        if len(prev) > max_states:
            raise RuntimeError(
                f"search exceeded {max_states} states; the step function is "
                f"probably returning a new state every call (is something in it "
                f"changing that should not?)"
            )
        current = queue.popleft()
        for action in actions:
            nxt = step(current, action)
            if nxt in prev:
                continue
            prev[nxt] = (current, action)
            if reached(nxt):
                path = []
                node: Any = nxt
                while prev[node] is not None:
                    node, act = prev[node]
                    path.append(act)
                return path[::-1]
            queue.append(nxt)
    return None


def reachable(
    step: Callable[[Any, int], Any],
    start: Hashable,
    actions: Sequence[int] = (1, 2, 3, 4),
    *,
    max_states: int = 200_000,
) -> set[Any]:
    """Every state reachable from ``start`` under ``step``.

    Useful for the question that comes before routing: *is the thing I want even
    reachable, or have I misread the board?* An unexpectedly small reachable set
    usually means the model thinks a wall is somewhere it is not.
    """
    seen = {start}
    queue: deque[Any] = deque([start])
    while queue and len(seen) <= max_states:
        current = queue.popleft()
        for action in actions:
            nxt = step(current, action)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen
