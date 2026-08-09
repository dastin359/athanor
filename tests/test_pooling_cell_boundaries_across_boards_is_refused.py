"""``cell_boundaries`` pools one board, and says so when handed two.

The function is documented to take a whole trace -- "pooling across a trace is
the whole mechanism" -- and ARC-AGI-3 changes board shape at a level boundary.
So the ordinary way to call it is also the way to mix two cell grids into one
answer. It did that silently, bounding the result by whichever frame came last,
which is the failure ``diff`` already refuses to commit: "a shape change is a
real event in these games and silently returning 'everything changed' would
bury it."

Measured before the fix: pooling a 12x12 frame with a boundary at row 6 and a
6x6 frame with a boundary at row 3 returned ``[0, 3]`` -- the 12x12's real
boundary silently dropped for being out of range of a board it never belonged
to, and the survivor reported as an index into the wrong frame.

Two mutants motivated this: ``r <= height`` (which only differs from ``r <
height`` on ragged input, so nothing caught it) and the missing guard itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from athanor.ccarc3 import cell_boundaries


def _banded(n: int, split: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.int16)
    g[split:, :] = 1
    return g


def test_two_boards_of_different_shape_are_refused():
    with pytest.raises(ValueError, match="pools one board"):
        cell_boundaries([_banded(12, 6), _banded(6, 3)])


def test_the_refusal_names_both_shapes_and_the_fix():
    with pytest.raises(ValueError) as excinfo:
        cell_boundaries([_banded(12, 6), _banded(6, 3)])
    message = str(excinfo.value)
    assert "(12, 12)" in message and "(6, 6)" in message
    assert "level" in message, "the message should point at the actual fix"


def test_a_shrinking_board_is_refused_in_either_order():
    """Whichever came last would have been the one silently believed."""
    with pytest.raises(ValueError, match="pools one board"):
        cell_boundaries([_banded(6, 3), _banded(12, 6)])


def test_one_board_across_many_frames_still_pools():
    """The refusal must not cost the feature the function exists for."""
    a = np.zeros((12, 12), dtype=np.int16)
    a[0:6, 0:6] = 1
    b = np.zeros((12, 12), dtype=np.int16)
    b[6:12, 6:12] = 1
    rows, cols = cell_boundaries([a, b])
    assert rows == [0, 6] and cols == [0, 6]


def test_an_empty_frame_does_not_set_the_shape():
    """A zero-size grid is skipped, so it cannot make a real pair look ragged."""
    empty = np.zeros((0, 0), dtype=np.int16)
    rows, cols = cell_boundaries([empty, _banded(12, 6), empty, _banded(12, 6)])
    assert rows == [0, 6]


def test_boundaries_index_the_first_row_of_each_cell():
    """The off-by-one that ``r < height`` alone cannot catch.

    A 12-row frame that switches colour between rows 5 and 6 has its cell
    boundary at 6 -- the first row of the new cell -- not at 5. Reported one
    early, every cell a solver reads out of the frame is shifted.
    """
    rows, _ = cell_boundaries([_banded(12, 6)])
    assert rows == [0, 6]
    frame = _banded(12, 6)
    assert frame[rows[1], 0] == 1 and frame[rows[1] - 1, 0] == 0


def test_a_board_that_changes_only_in_width_is_refused_too():
    """The guard compares the whole shape, not just the height.

    Checking ``shape[0]`` alone survived every square-fixture test above -- and
    every fixture above is square, because that is the shape a test reaches for
    without thinking about it.
    """
    tall_wide = np.zeros((8, 12), dtype=np.int16)
    tall_wide[4:, :] = 1
    tall_narrow = np.zeros((8, 6), dtype=np.int16)
    tall_narrow[4:, :] = 1
    with pytest.raises(ValueError, match="pools one board"):
        cell_boundaries([tall_wide, tall_narrow])


def test_a_board_that_changes_only_in_height_is_refused_too():
    wide_tall = np.zeros((12, 8), dtype=np.int16)
    wide_tall[6:, :] = 1
    wide_short = np.zeros((6, 8), dtype=np.int16)
    wide_short[3:, :] = 1
    with pytest.raises(ValueError, match="pools one board"):
        cell_boundaries([wide_tall, wide_short])
