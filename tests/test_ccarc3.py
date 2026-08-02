"""Tests for the ARC-AGI-3 harness primitives.

The tests that matter most here are the three-valued ones. A boolean predicate
would pass every "does the checker work" test and still destroy knowledge in the
field, because it reports ``False`` for "the rule was violated" and for "the rule
never applied" alike. Those two cases are asserted apart deliberately.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from athanor.ccarc3 import (
    Outcome,
    Rule,
    RuleBook,
    TraceWriter,
    action_name,
    as_grid,
    block_size,
    diff,
    infer_levels,
    load,
    logical,
    objects,
    regressions,
    render,
    survey,
    verify,
)


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #


def test_render_is_one_char_per_cell_over_sixteen_colours():
    grid = [[0, 1, 9], [10, 15, 5]]
    assert render(grid) == "019\naf5"


def test_render_is_about_three_times_cheaper_than_bracketed_rows():
    """Measured, not asserted from intuition: 12,416 chars -> 4,159 for 64x64.

    The saving is ~3x against a bare ``str(grid)``. Against the SDK's actual
    ``pretty_print_3d`` it is larger, since that adds per-grid headers and
    two-space indentation, and one action can return several grids.
    """
    grid = np.zeros((64, 64), dtype=int).tolist()
    assert len(render(grid)) * 2.5 < len(str(grid))


def test_as_grid_rejects_out_of_palette_values():
    with pytest.raises(ValueError, match=r"\[0, 16\)"):
        as_grid([[0, 16]])


def test_as_grid_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        as_grid([[[1]]])


def test_diff_reports_changed_cells_only():
    changes = diff([[1, 2], [3, 4]], [[1, 7], [3, 4]])
    assert len(changes) == 1
    assert (changes[0].y, changes[0].x) == (0, 1)
    assert (changes[0].before, changes[0].after) == (2, 7)


def test_diff_refuses_a_shape_change_rather_than_burying_it():
    with pytest.raises(ValueError, match="shape mismatch"):
        diff([[1, 2]], [[1, 2, 3]])


def test_block_size_recovers_the_render_factor():
    logical_board = np.array([[1, 2], [3, 4]])
    rendered = np.kron(logical_board, np.ones((16, 16), dtype=int))
    assert block_size(rendered) == 16
    assert np.array_equal(logical(rendered), logical_board)


def test_block_size_is_one_when_no_factor_holds():
    assert block_size([[1, 2], [3, 4]]) == 1


def test_logical_is_idempotent():
    board = np.array([[1, 2], [3, 4]])
    assert np.array_equal(logical(logical(np.kron(board, np.ones((4, 4), int)))), board)


def test_block_size_does_not_claim_a_factor_that_fails_anywhere():
    rendered = np.kron(np.array([[1, 2], [3, 4]]), np.ones((16, 16), dtype=int))
    rendered[0, 0] = 7  # one cell breaks the 16x16 uniformity
    assert block_size(rendered) != 16


def test_objects_finds_components_and_infers_background():
    grid = [
        [5, 5, 5, 5],
        [5, 2, 2, 5],
        [5, 5, 5, 5],
        [3, 5, 5, 5],
    ]
    found = objects(grid)
    assert [o.colour for o in found] == [2, 3]
    assert found[0].size == 2
    assert found[0].bbox == (1, 1, 1, 2)


def test_objects_respects_connectivity():
    grid = [[1, 0], [0, 1]]
    assert len(objects(grid, background=0, connectivity=4)) == 2
    assert len(objects(grid, background=0, connectivity=8)) == 1


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def _frame(action_id, grids, score=0, state="NOT_FINISHED", reasoning=None):
    return {
        "game_id": "toy",
        "frame": grids,
        "state": state,
        "score": score,
        "action_input": {"id": action_id, "data": {"x": 1, "y": 2}, "reasoning": reasoning},
        "guid": "g",
        "full_reset": False,
        "available_actions": [0, 1, 2],
    }


def test_action_name_accepts_ids_names_and_enum_likes():
    assert action_name(0) == "RESET"
    assert action_name(6) == "ACTION6"
    assert action_name("action3") == "ACTION3"

    class Fake:
        name = "ACTION7"

    assert action_name(Fake()) == "ACTION7"


def test_action_name_rejects_bools_and_unknown_ids():
    with pytest.raises(ValueError):
        action_name(True)
    with pytest.raises(ValueError, match="no GameAction with id 9"):
        action_name(9)


def test_writer_and_loader_round_trip_and_chain_before_from_after(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(0, [[[1, 1], [1, 1]]]), level=0)
    writer.append(_frame(1, [[[1, 2], [1, 1]]]), level=0)

    transitions = load(path)
    assert len(transitions) == 2
    assert transitions[0].before is None
    assert transitions[0].action == "RESET"
    assert np.array_equal(transitions[1].before, np.array([[1, 1], [1, 1]]))
    assert np.array_equal(transitions[1].after, np.array([[1, 2], [1, 1]]))
    assert transitions[1].changed


def test_loader_keeps_every_intermediate_grid_of_a_multi_frame_action(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(1, [[[1]], [[2]], [[3]]]))
    (t,) = load(path)
    assert len(t.intermediate) == 3
    assert np.array_equal(t.after, np.array([[3]]))


def test_writer_captures_the_reasoning_blob_as_the_hypothesis(tmp_path):
    path = tmp_path / "trace.jsonl"
    TraceWriter(path).append(_frame(1, [[[1]]], reasoning={"h": "walls block"}))
    (t,) = load(path)
    assert t.hypothesis == {"h": "walls block"}


def test_score_delta_and_terminal_states(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(1, [[[1]]], score=0))
    writer.append(_frame(1, [[[2]]], score=1, state="GAME_OVER"))
    a, b = load(path)
    assert a.score_delta == 0
    assert b.score_delta == 1
    assert b.is_game_over and not b.is_win


def test_records_are_one_json_object_per_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(0, [[[1]]]))
    writer.append(_frame(1, [[[2]]]))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert [json.loads(line)["i"] for line in lines] == [0, 1]


def test_infer_levels_tracks_score_increments():
    assert infer_levels([0, 0, 1, 1, 2]) == [0, 0, 1, 1, 2]
    assert infer_levels([]) == []


def test_writer_reads_levels_completed_as_well_as_score(tmp_path):
    """arcengine 0.9.3 renamed the field; reading one name silently zeroes the other."""
    path = tmp_path / "trace.jsonl"
    frame = _frame(1, [[[1]]])
    del frame["score"]
    frame["levels_completed"] = 3
    TraceWriter(path).append(frame)
    (t,) = load(path)
    assert t.score_after == 3


def test_actions_after_a_death_are_recorded_as_wasted_not_dropped(tmp_path):
    """perform_action returns frame=[] while GAME_OVER, but the action still costs."""
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(1, [[[1, 1]]], state="NOT_FINISHED"))
    writer.append(_frame(2, [], state="GAME_OVER"))
    writer.append(_frame(3, [], state="GAME_OVER"))

    transitions = load(path)
    assert len(transitions) == 3
    assert [t.wasted for t in transitions] == [False, True, True]
    # The board is carried forward, so downstream analysis sees no phantom change.
    assert not transitions[1].changed
    assert sum(t.wasted for t in transitions) == 2


def test_a_leading_empty_frame_has_nothing_to_carry_and_is_skipped(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    writer.append(_frame(1, []))
    writer.append(_frame(1, [[[1]]]))
    assert [t.index for t in load(path)] == [1]


# --------------------------------------------------------------------------- #
# rules — the three-valued core
# --------------------------------------------------------------------------- #


def _t(index, level, action, before, after, state="NOT_FINISHED"):
    from athanor.ccarc3.ledger import Transition

    return Transition(
        index=index,
        level=level,
        action=action,
        params={},
        before=None if before is None else as_grid(before),
        after=as_grid(after),
        intermediate=(as_grid(after),),
        score_before=0,
        score_after=0,
        state=state,
        full_reset=False,
        available_actions=("RESET", "ACTION1"),
    )


def _has_colour(colour):
    return lambda t: bool((t.after == colour).any())


RED_MOVES = Rule(
    name="red-block-moves-on-ACTION1",
    applies=lambda t: t.action == "ACTION1" and _has_colour(2)(t),
    holds=lambda t: t.changed,
    scope="game",
)


def test_outcome_is_three_valued_not_boolean():
    applicable_and_ok = _t(0, 0, "ACTION1", [[0]], [[2]])
    applicable_and_bad = _t(1, 0, "ACTION1", [[2]], [[2]])
    inapplicable = _t(2, 0, "ACTION1", [[0]], [[0]])

    assert RED_MOVES(applicable_and_ok) is Outcome.HOLDS
    assert RED_MOVES(applicable_and_bad) is Outcome.VIOLATED
    assert RED_MOVES(inapplicable) is Outcome.NOT_APPLICABLE


def test_an_inapplicable_rule_is_not_refuted():
    """The whole point. A boolean predicate would kill this rule."""
    transitions = [_t(i, 0, "ACTION1", [[0]], [[0]]) for i in range(20)]
    result = verify(RED_MOVES, transitions, level=0)
    assert not result.refuted
    assert result.vacuous
    assert result.counts.violated == 0


def test_a_vacuous_rule_is_not_verified_either():
    transitions = [_t(i, 0, "ACTION1", [[0]], [[0]]) for i in range(20)]
    assert not verify(RED_MOVES, transitions, level=0).verified


def test_verify_refutes_only_on_a_genuine_violation():
    transitions = [
        _t(0, 0, "ACTION1", [[0]], [[2]]),
        _t(1, 0, "ACTION1", [[2]], [[2]]),  # applicable, did not change
    ]
    result = verify(RED_MOVES, transitions, level=0)
    assert result.refuted
    assert [t.index for t in result.violations] == [1]


def test_verify_is_level_local_and_ignores_other_levels():
    transitions = [
        _t(0, 0, "ACTION1", [[2]], [[2]]),  # a violation, but on level 0
        _t(1, 1, "ACTION1", [[0]], [[2]]),  # level 1 is clean
    ]
    assert verify(RED_MOVES, transitions, level=1).verified
    assert verify(RED_MOVES, transitions, level=0).refuted


def test_verify_defaults_to_the_current_level():
    transitions = [
        _t(0, 0, "ACTION1", [[2]], [[2]]),
        _t(1, 3, "ACTION1", [[0]], [[2]]),
    ]
    result = verify(RED_MOVES, transitions)
    assert result.level == 3
    assert result.verified


def test_survey_reports_every_level_and_never_refutes():
    transitions = [
        _t(0, 0, "ACTION1", [[0]], [[0]]),  # n/a
        _t(1, 1, "ACTION1", [[0]], [[2]]),  # holds
        _t(2, 2, "ACTION1", [[2]], [[2]]),  # violated
    ]
    s = survey(RED_MOVES, transitions)
    assert str(s.per_level[0]) == "0/0/1"
    assert str(s.per_level[1]) == "1/0/0"
    assert str(s.per_level[2]) == "0/1/0"
    assert s.levels_applicable == [1, 2]
    assert s.levels_violated == [2]
    assert not hasattr(s, "refuted")


def test_survey_counts_instantiations_for_confidence():
    transitions = [
        _t(0, 0, "ACTION1", [[0]], [[2]]),
        _t(1, 1, "ACTION1", [[0]], [[2]]),
        _t(2, 2, "ACTION1", [[0]], [[0]]),  # not applicable on L2
    ]
    assert survey(RED_MOVES, transitions).instantiations == 2


def test_regressions_fire_only_for_game_scoped_applicable_violations():
    level_rule = Rule(
        name="level-only",
        applies=lambda t: True,
        holds=lambda t: False,
        scope="level",
    )
    transitions = [_t(0, 4, "ACTION1", [[2]], [[2]])]  # applicable + violated

    assert [r.rule for r in regressions([RED_MOVES], transitions)] == [RED_MOVES.name]
    assert regressions([level_rule], transitions) == []


def test_regressions_stay_silent_when_the_mechanic_merely_does_not_apply():
    transitions = [_t(0, 4, "ACTION1", [[0]], [[0]])]
    assert regressions([RED_MOVES], transitions) == []


def test_rule_scope_is_validated():
    with pytest.raises(ValueError, match="scope must be"):
        Rule(name="x", applies=lambda t: True, holds=lambda t: True, scope="galaxy")


# --------------------------------------------------------------------------- #
# rule book
# --------------------------------------------------------------------------- #


def test_rulebook_sorts_results_into_verified_refuted_and_untested(tmp_path):
    book = RuleBook()
    book.record(verify(RED_MOVES, [_t(0, 0, "ACTION1", [[0]], [[2]])], level=0))
    book.record(verify(RED_MOVES, [_t(1, 1, "ACTION1", [[2]], [[2]])], level=1))
    book.record(verify(RED_MOVES, [_t(2, 2, "ACTION1", [[0]], [[0]])], level=2))

    assert len(book.verified) == 1
    assert len(book.refuted) == 1
    assert len(book.open_questions) == 1

    path = tmp_path / "rules.json"
    book.save(path)
    assert RuleBook.load(path).refuted == book.refuted


def test_rulebook_load_of_a_missing_file_is_empty(tmp_path):
    assert RuleBook.load(tmp_path / "nope.json").verified == []
