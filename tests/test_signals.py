"""Mechanical generalization signals.

These stand in for the cheap half of the dropped reviewer, so the bar is: never
fire unless every training output agrees and the prediction disagrees. A false
positive spends the solver's attention and teaches it to discount the channel.
"""

from __future__ import annotations

from athanor.cc_harness.signals import output_space_signals

FIXED_SHAPE_PAIRS = [
    {"input": [[1, 2], [3, 4]], "output": [[1, 1], [1, 1]]},
    {"input": [[5, 6], [7, 8]], "output": [[5, 5], [5, 5]]},
]

SAME_AS_INPUT_PAIRS = [
    {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
    {"input": [[4, 5, 6, 7]], "output": [[7, 6, 5, 4]]},
]


class TestShapeSignals:
    def test_flags_a_prediction_breaking_a_fixed_output_shape(self):
        findings = output_space_signals(FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[9, 9, 9]]]])
        assert any("every training output is 2x2" in f for f in findings)

    def test_silent_when_the_shape_matches(self):
        # The prediction must also stay inside its own input's palette, which
        # every training output does — so predict in colour 9, not colour 1.
        findings = output_space_signals(FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[9, 9], [9, 9]]]])
        assert findings == []

    def test_flags_a_prediction_that_drops_the_input_shape_relation(self):
        findings = output_space_signals(SAME_AS_INPUT_PAIRS, [[[1, 2, 3, 4, 5]]], [[[[1, 2]]]])
        assert any("every training output has its input's shape" in f for f in findings)

    def test_silent_when_the_input_shape_relation_holds(self):
        findings = output_space_signals(SAME_AS_INPUT_PAIRS, [[[1, 2, 3, 4, 5]]], [[[[5, 4, 3, 2, 1]]]])
        assert findings == []

    def test_flags_a_non_square_prediction_when_training_is_always_square(self):
        pairs = [
            {"input": [[1, 2, 3]], "output": [[1, 1], [1, 1]]},
            {"input": [[4, 5]], "output": [[4, 4, 4], [4, 4, 4], [4, 4, 4]]},
        ]
        findings = output_space_signals(pairs, [[[7]]], [[[[7, 7, 7]]]])
        assert any("square" in f for f in findings)


FIXED_PALETTE_PAIRS = [
    {"input": [[1, 2], [3, 4]], "output": [[0, 5], [5, 0]]},
    {"input": [[6, 7], [8, 9]], "output": [[5, 0], [0, 5]]},
]


class TestPaletteSignals:
    def test_flags_a_departure_from_a_fixed_output_palette(self):
        findings = output_space_signals(FIXED_PALETTE_PAIRS, [[[1, 1], [1, 1]]], [[[[0, 5], [5, 3]]]])
        assert any("every training output uses exactly [0, 5]" in f and "[3]" in f for f in findings)

    def test_does_not_flag_a_recolouring_rule_using_the_test_input_s_colours(self):
        """A union-of-training-outputs check would fire here, wrongly.

        These outputs recolour from their own input, so a test input carrying
        colours no training pair contained produces a legitimate novel-colour
        output. Only an identical palette across every training output says
        anything about the test output.
        """
        findings = output_space_signals(FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[9, 9], [9, 9]]]])
        assert findings == []

    def test_flags_a_colour_absent_from_the_prediction_s_own_input(self):
        pairs = [
            {"input": [[1, 2]], "output": [[2, 1]]},
            {"input": [[3, 4]], "output": [[4, 3]]},
        ]
        # 2 exists in a training output, so the union check passes; but every
        # training output draws only on its own input's palette, and 2 is not
        # in this test input.
        findings = output_space_signals(pairs, [[[3, 1]]], [[[[1, 2]]]])
        assert any("not present in its own" in f for f in findings)

    def test_silent_when_the_palette_is_consistent(self):
        findings = output_space_signals(SAME_AS_INPUT_PAIRS, [[[1, 2, 3]]], [[[[3, 2, 1]]]])
        assert findings == []


class TestAbstention:
    """Disagreement in the training set means the regularity is not real."""

    def test_no_shape_signal_when_training_shapes_differ(self):
        pairs = [
            {"input": [[1]], "output": [[1, 1]]},
            {"input": [[2]], "output": [[2, 2, 2]]},
        ]
        findings = output_space_signals(pairs, [[[3]]], [[[[3, 3, 3, 3, 3]]]])
        assert all("training output is" not in f for f in findings)

    def test_no_signal_without_training_pairs(self):
        assert output_space_signals([], [[[1]]], [[[[1]]]]) == []

    def test_no_signal_for_an_empty_candidate_list(self):
        assert output_space_signals(FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[]]) == []

    def test_handles_a_missing_test_input(self):
        # Fewer inputs than candidate lists must not raise.
        assert output_space_signals(FIXED_SHAPE_PAIRS, [], [[[[1, 1], [1, 1]]]]) == []


class TestDuplicateCandidates:
    def test_flags_two_identical_candidates(self):
        findings = output_space_signals(
            FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]]
        )
        assert any("second attempt is wasted" in f for f in findings)

    def test_silent_for_two_genuinely_different_candidates(self):
        findings = output_space_signals(
            FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[1, 1], [1, 1]], [[5, 5], [5, 5]]]]
        )
        assert all("wasted" not in f for f in findings)

    def test_labels_each_candidate_separately(self):
        findings = output_space_signals(
            FIXED_SHAPE_PAIRS, [[[9, 9], [9, 9]]], [[[[1, 1], [1, 1]], [[7, 7, 7]]]]
        )
        assert any("candidate 2" in f for f in findings)
