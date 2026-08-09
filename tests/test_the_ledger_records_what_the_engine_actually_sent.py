"""What survives the trip from a frame dict to a ``Transition``.

The ledger is the only durable record of a run: every rule check, every forward
model, every audit and every scoring pass reads it and nothing re-reads the
server. So a field this module drops is a field nobody can recover, and a field
it mangles is one everything downstream believes.

Nine mutations of it survived the suite. The two that matter most:

- ``TraceWriter.append`` could stop writing ``full_reset`` entirely and nothing
  failed. That flag is what ``board_replaced`` reads to know a spatial
  comparison is meaningless, and what ``level_pace`` cuts a replayed
  playthrough at.
- ``load`` chained ``before`` from ``grids[-1]``; taking ``grids[0]`` instead
  passed everything. The two are the same number until an action renders more
  than one frame -- which ``grids.py`` opens by warning is the normal case
  ("One grid per action is the wrong assumption") -- and then every subsequent
  ``before`` is an intermediate frame that was never a resting state.

Plus one real defect, in ``append``'s compatibility seam: see
``test_a_null_score_falls_through_to_the_other_name``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from athanor.ccarc3.ledger import TraceWriter, action_name, infer_levels, load


def _frame(action_id=1, grids=None, **kw):
    frame = {
        "game_id": "toy",
        "frame": [[[0]]] if grids is None else grids,
        "state": "NOT_FINISHED",
        "score": 0,
        "action_input": {"id": action_id, "data": {"x": 1, "y": 2}},
        "full_reset": False,
        "available_actions": [0, 1],
    }
    frame.update(kw)
    return frame


# --------------------------------------------------------------------------- #
# action_name
# --------------------------------------------------------------------------- #


def test_an_enum_like_name_is_normalised_like_a_string_one():
    """The string branch upper-cases and is tested; the enum branch was not."""

    class Lowercase:
        name = "action4"

    assert action_name(Lowercase()) == "ACTION4"


def test_an_unknown_action_id_is_refused_rather_than_named():
    with pytest.raises(ValueError, match="no GameAction with id 99"):
        action_name(99)


# --------------------------------------------------------------------------- #
# TraceWriter
# --------------------------------------------------------------------------- #


def test_a_full_reset_survives_the_round_trip(tmp_path):
    """Dropping this flag broke nothing, and it drives two other functions."""
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(action_id=0))
    writer.append(_frame(action_id=0, full_reset=True))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["full_reset"] for r in records] == [False, True]

    ts = load(path)
    assert [t.full_reset for t in ts] == [False, True]
    assert ts[1].board_replaced, "a full reset replaces the board"


def test_available_actions_are_recorded_as_names_not_ids(tmp_path):
    path = tmp_path / "t.jsonl"
    TraceWriter(path).append(_frame(available_actions=[0, 3, 6]))
    record = json.loads(path.read_text().splitlines()[0])
    assert record["available_actions"] == ["RESET", "ACTION3", "ACTION6"]
    assert load(path)[0].available_actions == ("RESET", "ACTION3", "ACTION6")


def test_only_the_click_coordinates_are_kept_from_action_data(tmp_path):
    """The filter is deliberate: whatever else a client hangs off ``data`` is
    not part of the game's state and does not belong in a durable trace."""
    path = tmp_path / "t.jsonl"
    TraceWriter(path).append(
        _frame(action_input={"id": 6, "data": {"x": 4, "y": 5, "note": "scratch",
                                               "token": "should-not-be-written"}})
    )
    record = json.loads(path.read_text().splitlines()[0])
    assert record["params"] == {"x": 4, "y": 5}
    assert "token" not in path.read_text()


def test_the_record_index_advances(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    for _ in range(3):
        writer.append(_frame())
    assert [json.loads(line)["i"] for line in path.read_text().splitlines()] == [0, 1, 2]
    assert [t.index for t in load(path)] == [0, 1, 2]


def test_a_null_score_falls_through_to_the_other_name(tmp_path):
    """The compatibility seam, through its third door.

    ``arcengine`` calls the field ``levels_completed``; ``arc_agi_3`` calls it
    ``score``. ``frame.get("score", fallback)`` reaches the fallback only when
    the key is *absent*, so a frame carrying ``score: None`` beside a real
    ``levels_completed`` recorded 0 -- the same silent zero the code comment
    was already written about, one case it did not cover.

    A zeroed score is not a small error. ``infer_levels`` reads score
    increments as level boundaries, so it collapses every level into level 0.
    """
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(score=None, levels_completed=5))
    writer.append(_frame(score=7))
    writer.append(_frame(score=None, levels_completed=None))
    scores = [json.loads(line)["score"] for line in path.read_text().splitlines()]
    assert scores == [5, 7, 0]


def test_an_explicit_zero_score_is_believed(tmp_path):
    """``score: 0`` is a real reading, not a missing one -- the fix must not
    turn a legitimate zero into a fallback lookup."""
    path = tmp_path / "t.jsonl"
    TraceWriter(path).append(_frame(score=0, levels_completed=6))
    assert json.loads(path.read_text().splitlines()[0])["score"] == 0


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_before_chains_from_the_resting_board_not_an_intermediate_frame(tmp_path):
    """One action can render several frames; only the last one is a state.

    Chaining from ``grids[0]`` passes every single-frame fixture, which is
    every fixture anyone writes without thinking about it.
    """
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(action_id=0, grids=[[[0]]]))
    writer.append(_frame(grids=[[[1]], [[2]], [[3]]]))     # three renders, one action
    writer.append(_frame(grids=[[[9]]]))

    ts = load(path)
    assert ts[1].after.tolist() == [[3]]
    assert [g.tolist() for g in ts[1].intermediate] == [[[1]], [[2]], [[3]]]
    assert ts[2].before is not None
    assert ts[2].before.tolist() == [[3]], "chained from an intermediate frame"


def test_every_rendered_frame_is_kept(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(grids=[[[1]], [[2]], [[3]]]))
    assert len(load(path)[0].intermediate) == 3


# --------------------------------------------------------------------------- #
# Transition.changed
# --------------------------------------------------------------------------- #


def test_a_shape_change_counts_as_a_change(tmp_path):
    path = tmp_path / "t.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(action_id=0, grids=[[[0, 0]]]))
    writer.append(_frame(grids=[[[0, 0], [0, 0]]]))
    ts = load(path)
    assert ts[1].before.shape != ts[1].after.shape
    assert ts[1].changed


def test_the_opening_transition_counts_as_a_change(tmp_path):
    """It has no predecessor, so "did anything change" cannot be answered by
    comparison. True is the safe answer: a RESET does put a board on screen."""
    path = tmp_path / "t.jsonl"
    TraceWriter(path).append(_frame(action_id=0))
    first = load(path)[0]
    assert first.before is None and first.changed


# --------------------------------------------------------------------------- #
# infer_levels
# --------------------------------------------------------------------------- #


def test_a_score_that_drops_is_not_a_level_boundary():
    """A full reset sends the score back to 0. Counting that as a boundary
    would promote a run that just lost everything."""
    assert infer_levels([0, 1, 2, 0, 1]) == [0, 1, 2, 2, 3]


def test_levels_are_counted_from_the_first_score_not_from_zero():
    """A trace that starts mid-game starts at level 0 of what it recorded.

    Seeding the comparison at 0 makes the very first frame of any resumed trace
    look like a level-up, shifting every level by one.
    """
    assert infer_levels([2, 2, 3]) == [0, 0, 1]
    assert infer_levels([0, 0, 1]) == [0, 0, 1]
    assert infer_levels([]) == []


def test_a_flat_score_never_advances_a_level():
    assert infer_levels([4, 4, 4, 4]) == [0, 0, 0, 0]
