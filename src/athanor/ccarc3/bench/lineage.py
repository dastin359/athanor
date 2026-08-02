"""``lineage`` — a local bench game built to falsify the CCARC3 design claims.

The design (``docs/ccarc3_design.md`` §5) rests on a claim that is easy to state
and easy to get wrong: **rules do not transfer across levels, mechanics do.**
A bench that cannot distinguish those two cases cannot test it, so this game is
constructed so they come apart on purpose.

Four levels, and what each is for:

===== ===================================================== =========================
level what it introduces                                    what it tests
===== ===================================================== =========================
0     avatar moves on ACTION1-4; goal colour 3 advances      baseline
1     **the goal colour changes to 4**                       a level rule that does
                                                             NOT transfer, while the
                                                             mechanic behind it does
2     **colour 9 becomes lethal**                            a game-scoped mechanic
                                                             going applicable-and-
                                                             VIOLATED, i.e. the
                                                             regressions() signal
3     **colour 8 walls block movement**                      a second late mechanic,
                                                             so level 2 is not a
                                                             one-off
===== ===================================================== =========================

The level-1 goal-colour switch is the important one. A solver that learned
"colour 3 is the goal" on level 0 and ported it as a *rule* is wrong on level 1.
A solver that learned "some colour advances the level on contact" and ported it
as a *mechanic* has a two-action confirmation rather than a fifteen-action
rediscovery. That difference is the entire payoff claimed in §5.1, and this
game is where it can be measured instead of asserted.

Level 2 is the other half. "Walking onto a non-background cell is safe" holds on
levels 0 and 1 and is *applicable and violated* on level 2 — not merely
inapplicable. That is precisely the distinction three-valued predicates exist to
draw, and the event ``regressions()`` is supposed to fire on.

Requires ``arcengine``; it is not an athanor dependency, so the import is local
to the constructor and the module is safe to import without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["GOAL_COLOUR", "LETHAL_COLOUR", "WALL_COLOUR", "MOVES", "build_game"]

BACKGROUND = 0
AVATAR_COLOUR = 2
WALL_COLOUR = 8
LETHAL_COLOUR = 9

GOAL_COLOUR = {0: 3, 1: 4, 2: 4, 3: 4}
"""Per level. The change at level 1 is the point -- see the module docstring."""

MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
"""ACTION1-4 -> (dx, dy). Up, down, left, right."""

_SIZE = 10


def _sprite(colour: int, name: str, x: int, y: int, **kw: Any):
    from arcengine import Sprite

    return Sprite(pixels=np.array([[colour]]), name=name, x=x, y=y, **kw)


def _levels() -> list[Any]:
    from arcengine import Level

    built = []

    # Level 0 -- the mechanics in their simplest form.
    built.append(
        Level(
            sprites=[
                _sprite(GOAL_COLOUR[0], "goal", _SIZE - 1, _SIZE - 1),
                _sprite(AVATAR_COLOUR, "avatar", 0, 0),
            ],
            grid_size=(_SIZE, _SIZE),
            name="plain",
        )
    )

    # Level 1 -- same mechanic, different colour. The rule breaks, not the idea.
    built.append(
        Level(
            sprites=[
                _sprite(GOAL_COLOUR[1], "goal", 0, _SIZE - 1),
                _sprite(AVATAR_COLOUR, "avatar", _SIZE - 1, 0),
            ],
            grid_size=(_SIZE, _SIZE),
            name="recoloured-goal",
        )
    )

    # Level 2 -- lethality appears. Placed away from the direct path, so a
    # solver only meets it by exploring rather than by walking to the goal.
    built.append(
        Level(
            sprites=[
                _sprite(GOAL_COLOUR[2], "goal", _SIZE - 1, _SIZE - 1),
                _sprite(LETHAL_COLOUR, "hazard", 4, 0),
                _sprite(LETHAL_COLOUR, "hazard2", 0, 4),
                _sprite(AVATAR_COLOUR, "avatar", 0, 0),
            ],
            grid_size=(_SIZE, _SIZE),
            name="lethal",
        )
    )

    # Level 3 -- walls, on top of lethality. Two mechanics live at once.
    walls = [
        _sprite(WALL_COLOUR, f"wall{i}", 5, i)
        for i in range(_SIZE - 2)
    ]
    built.append(
        Level(
            sprites=[
                _sprite(GOAL_COLOUR[3], "goal", _SIZE - 1, _SIZE - 1),
                _sprite(LETHAL_COLOUR, "hazard", 7, 0),
                *walls,
                _sprite(AVATAR_COLOUR, "avatar", 0, 0),
            ],
            grid_size=(_SIZE, _SIZE),
            name="walled",
        )
    )
    return built


def build_game():
    """Construct the ``lineage`` game. Importing ``arcengine`` is deferred."""
    from arcengine import ARCBaseGame

    class Lineage(ARCBaseGame):
        """Four levels sharing mechanics but not rules."""

        def __init__(self) -> None:
            super().__init__(
                game_id="lineage",
                levels=_levels(),
                available_actions=[1, 2, 3, 4],
                win_score=4,
            )

        def step(self) -> None:
            action_id = getattr(getattr(self.action, "id", None), "value", None)
            move = MOVES.get(action_id)
            if move is not None:
                hits = self.try_move("avatar", move[0], move[1])
                names = {getattr(h, "name", "") or "" for h in hits}
                if any(n.startswith("hazard") for n in names):
                    # Lethal contact. From level 2 only -- the mechanic does not
                    # exist earlier, which is what makes it a genuine mid-game
                    # introduction rather than something missed on level 0.
                    self.lose()
                elif "goal" in names:
                    self.next_level()
                # Walls are simply blocking: try_move reports the collision and
                # declines the move, so nothing more is needed here.
            self.complete_action()

    return Lineage()
