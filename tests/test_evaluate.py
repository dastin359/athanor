"""Evaluation semantics — these must match the flagship harness."""

from __future__ import annotations

from conftest import IDENTITY_SOLVE, MIRROR_SOLVE, MIRROR_TASK

from athanor.cc_harness.evaluate import (
    detect_hardcoding,
    grid_diff,
    is_valid_grid,
    normalize_candidates,
    pixel_accuracy,
    run_solution_isolated,
)


class TestGridValidation:
    def test_accepts_rectangular_int_grid(self):
        assert is_valid_grid([[0, 1], [2, 3]])

    def test_rejects_ragged_grid(self):
        assert not is_valid_grid([[0, 1], [2]])

    def test_rejects_empty(self):
        assert not is_valid_grid([])
        assert not is_valid_grid([[]])

    def test_rejects_boolean_cells(self):
        # bool subclasses int; a boolean mask is a bug, not a colour grid.
        assert not is_valid_grid([[True, False]])


class TestCandidateNormalization:
    def test_bare_grid_becomes_single_candidate(self):
        candidates, error = normalize_candidates([[1, 2], [3, 4]])
        assert error is None
        assert candidates == [[[1, 2], [3, 4]]]

    def test_two_candidates_are_preserved(self):
        candidates, error = normalize_candidates([[[1]], [[2]]])
        assert error is None
        assert len(candidates) == 2

    def test_duplicate_candidates_collapse(self):
        candidates, error = normalize_candidates([[[1, 1]], [[1, 1]]])
        assert error is None
        assert candidates == [[[1, 1]]]

    def test_too_many_candidates_rejected(self):
        candidates, error = normalize_candidates([[[1]], [[2]], [[3]]], max_candidates=2)
        assert candidates is None
        assert "more than 2" in error

    def test_non_grid_rejected(self):
        candidates, error = normalize_candidates("nope")
        assert candidates is None
        assert error


class TestPixelAccuracy:
    def test_exact_match(self):
        assert pixel_accuracy([[1, 2]], [[1, 2]]) == 1.0

    def test_partial_match(self):
        assert pixel_accuracy([[1, 0]], [[1, 2]]) == 0.5

    def test_shape_mismatch_scores_against_expected_area(self):
        # 1 of the 4 expected cells is reproduced.
        assert pixel_accuracy([[1]], [[1, 2], [3, 4]]) == 0.25

    def test_none_is_zero(self):
        assert pixel_accuracy(None, [[1]]) == 0.0


class TestGridDiff:
    def test_reports_shape_mismatch(self):
        diff = grid_diff([[1]], [[1, 2]])
        assert diff["shape_match"] is False

    def test_lists_differing_cells_with_bbox(self):
        diff = grid_diff([[1, 9], [3, 4]], [[1, 2], [3, 4]])
        assert diff["num_diff"] == 1
        assert diff["cells"] == [[0, 1, 2, 9]]
        assert diff["bbox"] == [0, 1, 0, 1]

    def test_truncates_long_listings(self):
        predicted = [[0] * 10 for _ in range(10)]
        expected = [[1] * 10 for _ in range(10)]
        diff = grid_diff(predicted, expected, limit=5)
        assert diff["num_diff"] == 100
        assert len(diff["cells"]) == 5
        assert diff["truncated"] is True


class TestIsolatedExecution:
    def test_correct_solution_passes_all_training(self):
        result = run_solution_isolated(MIRROR_SOLVE, MIRROR_TASK["train"], MIRROR_TASK["test"])
        assert result["status"] == "ok"
        assert result["all_train_correct"] is True
        assert result["train_correct"] == 3
        assert result["test"][0]["candidates"] == [[[3, 0, 6], [0, 4, 0]]]

    def test_wrong_solution_reports_partial_credit(self):
        result = run_solution_isolated(IDENTITY_SOLVE, MIRROR_TASK["train"], MIRROR_TASK["test"])
        assert result["all_train_correct"] is False
        assert 0.0 < result["train_pixel_accuracy"] < 1.0

    def test_test_outputs_are_never_passed_to_solution_code(self):
        # The worker receives test inputs only; a solution that reaches for an
        # 'output' key on a test sample cannot find one.
        code = "def solve(grid):\n    return grid\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], MIRROR_TASK["test"])
        assert result["status"] == "ok"

    def test_syntax_error_is_reported_not_raised(self):
        result = run_solution_isolated("def solve(grid)\n    return grid", MIRROR_TASK["train"], [])
        assert result["status"] == "error"
        assert "SyntaxError" in result["error"]

    def test_missing_solve_is_reported(self):
        result = run_solution_isolated("x = 1", MIRROR_TASK["train"], [])
        assert result["status"] == "error"
        assert "solve(grid)" in result["error"]

    def test_exception_inside_solve_is_per_example(self):
        code = "def solve(grid):\n    raise ValueError('boom')\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], MIRROR_TASK["test"])
        assert result["status"] == "ok"
        assert all("boom" in row["error"] for row in result["train"])

    def test_infinite_loop_is_killed(self):
        code = "def solve(grid):\n    while True:\n        pass\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], [], timeout_seconds=3)
        assert result["status"] == "error"
        assert result["timed_out"] is True

    def test_sys_exit_in_solution_does_not_kill_the_gate(self):
        code = "import sys\nsys.exit(3)\ndef solve(grid):\n    return grid\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], [])
        assert result["status"] == "error"

    def test_solution_stdout_is_captured_and_bounded(self):
        code = "print('x' * 20000)\ndef solve(grid):\n    return grid\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], [])
        assert result["status"] == "ok"
        assert "omitted" in result["stdout"]

    def test_multi_candidate_on_training_is_flagged(self):
        code = "def solve(grid):\n    return [[row[::-1] for row in grid], grid]\n"
        result = run_solution_isolated(code, MIRROR_TASK["train"], MIRROR_TASK["test"])
        assert result["multi_candidate_on_train"] is True
        # Training still scores the first candidate only, so the rule passes.
        assert result["all_train_correct"] is True


class TestHardcodingDetection:
    def test_flags_verbatim_training_output(self):
        code = "def solve(grid):\n    return [[9, 8, 7], [2, 1, 1]]\n"
        findings = detect_hardcoding(code, MIRROR_TASK["train"])
        assert any("verbatim" in f for f in findings)

    def test_flags_reaching_for_puzzle_data(self):
        code = "def solve(grid):\n    return train_samples[0]['output']\n"
        findings = detect_hardcoding(code, MIRROR_TASK["train"])
        assert any("train_samples" in f for f in findings)

    def test_clean_solution_is_silent(self):
        assert detect_hardcoding(MIRROR_SOLVE, MIRROR_TASK["train"]) == []
