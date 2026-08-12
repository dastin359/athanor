"""Tests for the ARC-AGI-3 harness primitives.

The tests that matter most here are the three-valued ones. A boolean predicate
would pass every "does the checker work" test and still destroy knowledge in the
field, because it reports ``False`` for "the rule was violated" and for "the rule
never applied" alike. Those two cases are asserted apart deliberately.
"""

from __future__ import annotations

import json
import pathlib

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
    cell_boundaries,
    collapse,
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


def test_collapse_handles_scaling_that_has_no_integer_factor():
    """A 10-cell board in a 64-px viewport has no common factor; logical() stalls."""
    board = np.array([[1, 0], [0, 2]])
    # 6 and 7 pixel cells, as the real renderer produces.
    rendered = np.repeat(np.repeat(board, 6, axis=0), 7, axis=1)
    assert block_size(rendered) == 1  # nothing exact to find
    assert np.array_equal(logical(rendered), rendered)  # so logical is a no-op
    assert np.array_equal(collapse(rendered), board)  # collapse still recovers it


def test_collapse_merges_identical_adjacent_bands_and_loses_position():
    """Documented, deliberate: structure survives, metric position does not."""
    sparse = np.zeros((9, 9), dtype=int)
    sparse[1, 1] = 2
    sparse[7, 7] = 3
    out = collapse(sparse)
    assert out.shape == (5, 5)  # not 9x9 -- the empty bands merged
    assert (out == 2).sum() == 1 and (out == 3).sum() == 1


def test_cell_boundaries_needs_many_frames_and_never_invents_splits():
    board_a = np.zeros((12, 12), dtype=int)
    board_a[0:6, 0:6] = 1
    board_b = np.zeros((12, 12), dtype=int)
    board_b[6:12, 6:12] = 1

    one = cell_boundaries([board_a])
    both = cell_boundaries([board_a, board_b])
    assert len(both[0]) >= len(one[0])  # pooling only ever adds boundaries
    assert all(0 <= r < 12 for r in both[0])
    assert both[0][0] == 0


def test_cell_boundaries_on_nothing_is_empty():
    assert cell_boundaries([]) == ([], [])


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


def test_as_grid_rejects_negative_values():
    """The lower palette bound, which no test read until a mutant removed it."""
    with pytest.raises(ValueError, match=r"\[0, 16\)"):
        as_grid([[0, -1]])


def test_diff_reports_changes_in_row_major_order():
    """The docstring promises an order, so something has to hold it to it.

    A solver taking ``diff(...)[0]`` as "the first cell that changed" is reading
    that promise, and reversing the list passed the whole suite.
    """
    before = [[0, 0, 0], [0, 0, 0]]
    after = [[0, 1, 0], [2, 0, 3]]
    positions = [(c.y, c.x) for c in diff(before, after)]
    assert positions == [(0, 1), (1, 0), (1, 2)]
    assert positions == sorted(positions)


def test_block_size_is_always_safe_to_divide_by():
    """``logical`` slices by whatever this returns, including on an empty grid."""
    assert block_size(np.zeros((0, 4), dtype=int)) == 1
    assert block_size(np.zeros((4, 0), dtype=int)) == 1
    assert logical(np.zeros((0, 4), dtype=int)).shape == (0, 4)


def test_counts_reports_every_colour_present_and_no_others():
    from athanor.ccarc3 import counts

    assert counts([[5, 5, 2], [5, 3, 3]]) == {5: 3, 2: 1, 3: 2}
    assert counts([[5]]) == {5: 1}
    assert counts(np.zeros((0, 0), dtype=int)) == {}


def test_everything_the_package_exports_is_exported_by_its_own_module():
    """A derived rule, because an enumerated list has gone stale here four times.

    Two ``__all__`` lists had already drifted when this was written: ``counts``
    was absent from ``grids.__all__`` and ``ledger_facts`` from
    ``session.__all__``. Both were harmless only by luck -- ``ccarc3/__init__``
    imports by name, which ignores ``__all__`` entirely -- so nothing failed, and
    a ``from .grids import *`` anywhere would have started losing symbols
    silently.

    The invariant is not "every public function must be exported" (plenty are
    deliberately internal to the package). It is the narrower, checkable one:
    anything the *package* publishes must be published by the module that
    defines it.
    """
    import importlib

    import athanor.ccarc3 as pkg

    drift = []
    for name in pkg.__all__:
        obj = getattr(pkg, name)
        home = getattr(obj, "__module__", None)
        if not home or not home.startswith("athanor.ccarc3."):
            continue                       # a constant, or re-exported from afar
        mod = importlib.import_module(home)
        if hasattr(mod, "__all__") and name not in mod.__all__:
            drift.append(f"{home}.__all__ is missing {name!r}")
    assert not drift, "; ".join(drift)


def test_no_module_promises_a_name_it_does_not_have():
    import importlib

    import athanor.ccarc3 as pkg

    ghosts = []
    for name in pkg.__all__:
        obj = getattr(pkg, name)
        home = getattr(obj, "__module__", None)
        if not home or not home.startswith("athanor.ccarc3."):
            continue
        mod = importlib.import_module(home)
        ghosts += [f"{home}.{n}" for n in getattr(mod, "__all__", ()) if not hasattr(mod, n)]
    assert not ghosts, "; ".join(sorted(set(ghosts)))


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


def test_png_writes_an_image_in_the_official_palette(tmp_path):
    """The agent can Read the file back and actually look at the board."""
    from athanor.ccarc3 import PALETTE, png
    from PIL import Image

    grid = np.arange(16).reshape(4, 4)
    out = png(grid, tmp_path / "g.png", scale=4)
    img = Image.open(out).convert("RGB")
    assert img.size == (16, 16)
    # cell (0,0) is colour 0 -> white, (3,3) is colour 15 -> purple
    assert img.getpixel((1, 1)) == (255, 255, 255)
    assert "#%02X%02X%02X" % img.getpixel((13, 13)) == PALETTE[15]


def test_png_creates_missing_parent_directories(tmp_path):
    from athanor.ccarc3 import png

    out = png([[1, 2]], tmp_path / "deep" / "nested" / "g.png")
    assert pathlib.Path(out).exists()


def test_a_level_boundary_transition_is_flagged(tmp_path):
    """`before` and `after` straddle two different boards there.

    A movement rule checked across one sees the avatar teleport and reports a
    violation that never happened -- and an applicable-and-violated result is
    supposed to be the highest-signal event in a run, so a false one is the
    most damaging error the ledger can produce. Found on a real ls20 trace:
    "ACTION1 moves the cursor up" held 7/7 on level 0 and showed 13/1 on level
    1, and the single violation was the boundary, 1467 cells changing at once.
    """
    path = tmp_path / "t.jsonl"
    w = TraceWriter(path)
    w.append(_frame(1, [[[1]]], score=0), level=0)
    w.append(_frame(1, [[[2]]], score=1), level=1)   # this action ended level 0
    w.append(_frame(1, [[[3]]], score=1), level=1)

    a, b, c = load(path)
    assert [t.crosses_level for t in (a, b, c)] == [False, True, False]


def test_excluding_boundaries_is_what_a_spatial_rule_must_do(tmp_path):
    from athanor.ccarc3 import Rule, survey

    path = tmp_path / "t.jsonl"
    w = TraceWriter(path)
    w.append(_frame(1, [[[1, 1]]], score=0), level=0)
    w.append(_frame(1, [[[9, 9]]], score=1), level=1)  # boundary: board replaced
    w.append(_frame(1, [[[9, 9]]], score=1), level=1)

    ts = load(path)
    naive = Rule(name="board never changes wholesale",
                 applies=lambda t: t.before is not None,
                 holds=lambda t: not t.changed, scope="game")
    careful = Rule(name="same, boundaries excluded",
                   applies=lambda t: t.before is not None and not t.crosses_level,
                   holds=lambda t: not t.changed, scope="game")

    assert survey(naive, ts).levels_violated == [1]
    assert survey(careful, ts).levels_violated == []


def test_a_full_reset_also_replaces_the_board(tmp_path):
    """crosses_level only catches level *increases*; a full reset moves it down.

    Both make a spatial comparison meaningless, so board_replaced is what a
    movement rule should actually exclude on.
    """
    path = tmp_path / "t.jsonl"
    w = TraceWriter(path)
    w.append(_frame(1, [[[0]]], score=2), level=2)
    w.append(_frame(1, [[[1]]], score=2), level=2)
    frame = _frame(0, [[[9]]], score=0)
    frame["full_reset"] = True
    w.append(frame, level=0)

    _first, a, b = load(path)
    assert not b.crosses_level, "the level went down, not up"
    assert b.full_reset and b.board_replaced
    assert not a.board_replaced


def test_a_full_reset_is_detected_even_when_the_server_denies_it(tmp_path):
    """The server flag is a hint. The level going down is the fact.

    Taken verbatim from the `ls20` trace: index 369 read level 6, index 370
    read level 0, and `full_reset` was **False** on it. So `board_replaced` --
    the check documented as the one a spatial rule wants -- returned False on
    the single largest board replacement in the whole 860-action trace, and
    every downstream count of full resets read zero for a run that replayed the
    entire game.
    """
    path = tmp_path / "t.jsonl"
    w = TraceWriter(path)
    w.append(_frame(2, [[[1]]], score=5), level=5)
    w.append(_frame(2, [[[2]]], score=6), level=6)     # cleared level 5
    lying = _frame(0, [[[9]]], score=0)
    lying["full_reset"] = False                        # what the server sent
    w.append(lying, level=0)

    _first, advance, wipe = load(path)
    assert advance.crosses_level and not advance.full_reset
    assert wipe.full_reset and wipe.board_replaced, "6 -> 0 is a full reset"


def test_the_first_transition_never_counts_as_crossing_a_level(tmp_path):
    """It has no predecessor. Comparing against a level-0 default would flag
    any trace that starts mid-game."""
    path = tmp_path / "t.jsonl"
    TraceWriter(path).append(_frame(0, [[[1]]], score=3), level=3)
    (t,) = load(path)
    assert t.before is None and not t.crosses_level and not t.board_replaced


# --------------------------------------------------------------------------- #
# forward models
# --------------------------------------------------------------------------- #


def _tr(index, level, action, before, after, **kw):
    from athanor.ccarc3.ledger import Transition

    return Transition(
        index=index, level=level, action=action, params={},
        before=None if before is None else as_grid(before),
        after=as_grid(after), intermediate=(as_grid(after),),
        score_before=0, score_after=0, state="NOT_FINISHED",
        full_reset=kw.get("full_reset", False), available_actions=(),
        crosses_level=kw.get("crosses_level", False), wasted=kw.get("wasted", False),
    )


def test_a_perfect_forward_model_is_reported_as_perfect():
    from athanor.ccarc3 import predict

    ts = [_tr(0, 0, "ACTION1", [[0, 0]], [[1, 0]]),
          _tr(1, 0, "ACTION1", [[1, 0]], [[2, 0]])]
    step = lambda b, a, p: np.array([[b[0][0] + 1, 0]])
    r = predict(step, ts, name="increment")
    assert r.perfect and r.correct == 2 and r.wrong == 0


def test_a_model_that_is_wrong_somewhere_says_where():
    from athanor.ccarc3 import predict

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]),
          _tr(7, 0, "ACTION1", [[1]], [[5]])]
    r = predict(lambda b, a, p: np.array([[b[0][0] + 1]]), ts)
    assert not r.perfect
    assert r.failures == [7], "the index is the whole point -- go look at it"


def test_declining_to_predict_is_skipped_not_counted_wrong():
    from athanor.ccarc3 import predict

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]), _tr(1, 0, "ACTION2", [[1]], [[9]])]
    r = predict(lambda b, a, p: np.array([[b[0][0] + 1]]) if a == "ACTION1" else None, ts)
    assert (r.correct, r.wrong, r.skipped) == (1, 0, 1)


def test_boards_replaced_and_wasted_actions_are_skipped_automatically():
    """No forward model can predict a level swap; counting it wrong hides real bugs."""
    from athanor.ccarc3 import predict

    ts = [
        _tr(0, 0, "ACTION1", [[0]], [[1]]),
        _tr(1, 1, "ACTION1", [[1]], [[9]], crosses_level=True),
        _tr(2, 1, "ACTION1", [[9]], [[9]], wasted=True),
        _tr(3, 0, "ACTION1", [[1]], [[9]], full_reset=True),
    ]
    r = predict(lambda b, a, p: np.array([[b[0][0] + 1]]), ts)
    assert r.correct == 1 and r.wrong == 0 and r.skipped == 3


def test_a_model_that_crashes_has_not_predicted():
    from athanor.ccarc3 import predict

    def boom(b, a, p):
        raise ZeroDivisionError

    r = predict(boom, [_tr(0, 0, "ACTION1", [[0]], [[1]])])
    assert r.skipped == 1 and r.wrong == 0


def test_the_first_transition_has_no_before_and_is_skipped():
    from athanor.ccarc3 import predict

    r = predict(lambda b, a, p: b, [_tr(0, 0, "RESET", None, [[1]])])
    assert r.skipped == 1


def test_monotone_rows_finds_a_display_that_ticks_every_action():
    """How a depleting resource is found without knowing how it is drawn."""
    from athanor.ccarc3 import monotone_rows

    pairs = []
    board = np.zeros((8, 8), dtype=int)
    board[7, :] = 11                       # a full "energy bar" on the last row
    for i in range(6):
        after = board.copy()
        after[7, i] = 3                    # drains one cell per action
        after[i % 4, 0] = 2                # the avatar moves about, irregularly
        pairs.append((board.copy(), after))
        board = after
    assert 7 in monotone_rows(pairs)


def test_monotone_rows_ignores_rows_that_only_sometimes_change():
    from athanor.ccarc3 import monotone_rows

    a = np.zeros((4, 4), dtype=int)
    pairs = []
    for i in range(10):
        b = a.copy()
        if i % 5 == 0:
            b[2, 2] = 7                    # changes rarely
        pairs.append((a.copy(), b))
    assert monotone_rows(pairs) == []


def test_monotone_rows_on_nothing_is_empty():
    from athanor.ccarc3 import monotone_rows

    assert monotone_rows([]) == []


def test_monotone_rows_counts_the_threshold_as_met():
    """``>=``, not ``>``. A row that ticks on exactly 90% of actions qualifies.

    The default threshold is a documented number, so its boundary is part of
    the contract; ``>`` passed every test there was.
    """
    from athanor.ccarc3 import monotone_rows

    a = np.zeros((4, 4), dtype=int)
    pairs = []
    for i in range(10):
        b = a.copy()
        if i != 0:                         # changes on 9 of 10 -- exactly 0.9
            b[2, 2] = 7
        pairs.append((a.copy(), b))
    assert monotone_rows(pairs) == [2]
    assert monotone_rows(pairs, threshold=0.95) == []


def test_monotone_rows_does_not_count_a_pair_it_skipped():
    """A shape mismatch is skipped, so it must not dilute the denominator.

    Counting it lowers every row's ratio and can hide the display this function
    exists to find -- a level boundary in the input would be enough.
    """
    from athanor.ccarc3 import monotone_rows

    a = np.zeros((4, 4), dtype=int)
    pairs = []
    for _ in range(5):
        b = a.copy()
        b[2, 2] = 7
        pairs.append((a.copy(), b))
    pairs.append((np.zeros((4, 4), dtype=int), np.zeros((5, 5), dtype=int)))
    # 5/5 clears 0.9; counting the skipped pair makes it 5/6 = 0.83 and hides
    # the row. Nine-and-one does NOT discriminate -- 9/10 is exactly 0.9 and
    # passes either way, which is how the first draft of this test passed
    # against the mutant it was written to kill.
    assert monotone_rows(pairs) == [2]


def test_monotone_rows_comes_back_sorted():
    from athanor.ccarc3 import monotone_rows

    # np.nonzero already returns ascending indices, so rows first seen in one
    # pair arrive sorted and Counter order matches sorted order by accident. A
    # LATER pair has to introduce a SMALLER row for the two to diverge.
    a = np.zeros((6, 6), dtype=int)
    pairs = []
    for i in range(4):
        b = a.copy()
        b[5, 0] = 7                        # row 5 changes on every pair
        if i >= 2:
            b[2, 0] = 7                    # row 2 shows up only later
        pairs.append((a.copy(), b))
    assert monotone_rows(pairs, threshold=0.5) == [2, 5]


# --------------------------------------------------------------------------- #
# effective_actions — available is not the same as effective
# --------------------------------------------------------------------------- #


def test_effective_actions_separates_the_no_ops_from_the_ones_that_work():
    """The failed run's signature: some available actions do nothing at all.

    ``available_actions`` said seven were accepted; one in five actions changed
    no cell. This is the read that would have shown it, and it costs no actions.
    """
    from athanor.ccarc3 import effective_actions

    ts = [
        _tr(0, 0, "ACTION1", [[0]], [[1]]),
        _tr(1, 0, "ACTION1", [[1]], [[2]]),
        _tr(2, 0, "ACTION6", [[2]], [[2]]),
        _tr(3, 0, "ACTION6", [[2]], [[2]]),
        _tr(4, 0, "ACTION6", [[2]], [[2]]),
    ]
    assert effective_actions(ts) == {"ACTION1": (2, 2), "ACTION6": (0, 3)}


def test_a_level_swap_would_make_every_action_look_effective():
    """A board replacement changes ~every cell, so counting it credits the

    action that happened to be in flight when the level ended. On `ls20` that
    single transition was enough to make a refuted rule read 13/1."""
    from athanor.ccarc3 import effective_actions

    ts = [
        _tr(0, 0, "ACTION6", [[0]], [[0]]),
        _tr(1, 1, "ACTION6", [[0]], [[9]], crosses_level=True),
        _tr(2, 0, "ACTION6", [[9]], [[0]], full_reset=True),
    ]
    assert effective_actions(ts) == {"ACTION6": (0, 1)}, "only the in-level one counts"


def test_actions_burned_after_a_death_are_not_evidence_about_the_action():
    """A wasted action never reached the game; it says nothing about the action."""
    from athanor.ccarc3 import effective_actions

    ts = [
        _tr(0, 0, "ACTION3", [[0]], [[1]]),
        _tr(1, 0, "ACTION3", [[1]], [[1]], wasted=True),
    ]
    assert effective_actions(ts) == {"ACTION3": (1, 1)}


def test_effective_actions_can_be_asked_about_one_level():
    """Which actions matter is level-scoped, like every other rule here."""
    from athanor.ccarc3 import effective_actions

    ts = [
        _tr(0, 0, "ACTION1", [[0]], [[1]]),
        _tr(1, 1, "ACTION1", [[1]], [[1]]),
        _tr(2, 1, "ACTION2", [[1]], [[5]]),
    ]
    assert effective_actions(ts, level=0) == {"ACTION1": (1, 1)}
    assert effective_actions(ts, level=1) == {"ACTION1": (0, 1), "ACTION2": (1, 1)}


def test_the_first_transition_has_no_before_so_nothing_can_be_said_about_it():
    from athanor.ccarc3 import effective_actions

    assert effective_actions([_tr(0, 0, "RESET", None, [[1]])]) == {}


# --------------------------------------------------------------------------- #
# level_pace — what "normal" costs, measured
# --------------------------------------------------------------------------- #


def test_level_pace_reports_spent_baseline_and_ratio_per_level():
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(11)]
    ts += [_tr(0, 1, "ACTION1", [[0]], [[1]]) for _ in range(5)]
    assert level_pace(ts, [22, 20]) == {0: (11, 22, 0.5), 1: (5, 20, 0.25)}


def test_level_pace_counts_only_the_final_playthrough():
    """The bug this exists to prevent, on real numbers.

    Measuring `ls20`'s 860-action trace whole reported level 0 at 1.55x, level
    2 at 1.84x and level 4 at 1.73x -- three levels apparently run well over
    baseline by a solver that in fact cleared every one of them under it. The
    trace holds two playthroughs, so every level was counted twice and every
    ratio doubled. Split at the full reset, they read 0.77x, 0.92x and 0.86x.
    """
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(8)]
    ts.append(_tr(0, 0, "RESET", [[1]], [[0]], full_reset=True))
    ts += [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(3)]
    spent, base, ratio = level_pace(ts, [10])[0]
    assert (spent, base) == (4, 10), "the restart plus 3, not all 12"
    assert ratio == 0.4


def test_a_level_without_a_published_baseline_is_omitted():
    """Inventing a denominator is worse than reporting nothing."""
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, 9, "ACTION1", [[0]], [[1]])]
    assert level_pace(ts, [22, 20]) == {}


def test_a_zero_baseline_is_omitted_rather_than_divided_by():
    """A published list can carry a 0; the ratio would be a ZeroDivisionError.

    Dropping the truthiness check on ``baselines[level]`` survived the suite --
    nothing had ever handed this function a zero.
    """
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]),
          _tr(1, 1, "ACTION1", [[0]], [[1]])]
    assert level_pace(ts, [0, 20]) == {1: (1, 20, 0.05)}


def test_a_negative_level_does_not_index_the_baselines_from_the_end():
    """``level < len(baselines)`` alone is true for -1, which is Python's *last*
    baseline -- a level would silently borrow another level's denominator."""
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, -1, "ACTION1", [[0]], [[1]])]
    assert level_pace(ts, [22, 20]) == {}


def test_the_cut_is_the_last_full_reset_not_the_first():
    """Two restarts in one trace: only the final playthrough counts.

    Cutting at the *first* reset still drops the opening attempt, so a
    single-restart fixture -- the one already here -- cannot tell the two
    apart. It takes two.
    """
    from athanor.ccarc3 import level_pace

    ts = [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(8)]
    ts.append(_tr(0, 0, "RESET", [[1]], [[0]], full_reset=True))
    ts += [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(5)]
    ts.append(_tr(0, 0, "RESET", [[1]], [[0]], full_reset=True))
    ts += [_tr(0, 0, "ACTION1", [[0]], [[1]]) for _ in range(2)]
    spent, base, _ = level_pace(ts, [10])[0]
    assert (spent, base) == (3, 10), "the second restart plus 2, not 9 and not 16"


def test_cell_boundaries_rejects_a_single_grid_with_a_useful_message():
    """It takes an iterable of grids; one 2-D array iterates its rows instead.

    A real solver hit this as "expected a 2-D grid, got shape (64,)", which says
    nothing about the actual mistake.
    """
    from athanor.ccarc3 import cell_boundaries

    with pytest.raises(ValueError, match="iterable of grids, not one grid"):
        cell_boundaries(np.zeros((64, 64), dtype=int))
    # wrapped, it works
    assert cell_boundaries([np.zeros((8, 8), dtype=int)]) == ([0], [0])


def test_a_trace_being_written_right_now_is_still_readable(tmp_path):
    """`report` is most useful *during* a batch, which is when the last line is
    half-written. This raised JSONDecodeError on a live run rather than
    reporting the complete actions before it."""
    path = tmp_path / "t.jsonl"
    good = json.dumps({"i": 0, "level": 0, "action": "RESET", "params": {},
                       "frames": [[[1]]], "score": 0, "state": "NOT_FINISHED",
                       "full_reset": False, "available_actions": []})
    path.write_text(good + "\n" + good.replace('"i": 0', '"i": 1')[:40])
    assert [t.index for t in load(path)] == [0]


def test_a_bad_line_in_the_middle_is_corruption_and_is_refused(tmp_path):
    """Silently dropping it would shorten a level's action count and change a score."""
    path = tmp_path / "t.jsonl"
    good = json.dumps({"i": 0, "level": 0, "action": "RESET", "params": {},
                       "frames": [[[1]]], "score": 0, "state": "NOT_FINISHED",
                       "full_reset": False, "available_actions": []})
    path.write_text(good + "\n{ broken\n" + good + "\n")
    with pytest.raises(ValueError, match="line 2 of 3"):
        load(path)


def test_verify_reads_the_level_in_play_not_the_deepest_ever_reached():
    """After a full reset the ledger still holds the abandoned play's levels.

    `ledger.load` concatenates every playthrough, so a trace that reached level 3
    and then reset to level 0 has `max(level) == 3` while the level actually
    being played is 0. `verify` defaulted to the max and returned a confident
    verdict about a level nobody was on — the module docstring's own named
    catastrophe with the levels swapped.

    The existing default test uses levels [0, 3] in ascending order, where max
    and last are the same value, so it could not see this.
    """
    transitions = [
        _t(0, 0, "ACTION1", [[0]], [[2]]),   # play 1, level 0: holds
        _t(1, 3, "ACTION1", [[2]], [[2]]),   # play 1, level 3: violated
        _t(2, 0, "ACTION1", [[0]], [[2]]),   # play 2 after a reset: level 0 again
    ]
    result = verify(RED_MOVES, transitions)
    assert result.level == 0, (
        f"scoped to level {result.level}, but the game is being played on level 0 "
        "— the deeper level belongs to a playthrough that was abandoned")
    assert not result.refuted, (
        "a rule that holds on the level in play was refuted using evidence from "
        "a discarded playthrough")
