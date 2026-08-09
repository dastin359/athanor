"""A grid transform returns a fresh array, on every branch.

``logical()`` copied on the branch that reduces and returned the caller's own
array on the branch that does not. Both branches are reachable from one call
site with the same code, so which contract applied depended on the *data*: a
solver that edited the result was editing the ledger's frame whenever the grid
happened to have no block factor.

That is not the rare case. ``grids.logical`` says so itself -- a board scaled
into the 64x64 viewport usually has no integer factor -- and the doctrine
repeats it: "real frames rarely have a uniform block structure, so ``logical()``
often declines to reduce at all". So the aliasing branch is the one real games
take, and the copying branch is the one a synthetic ``np.kron`` fixture takes,
which is why every existing test saw the safe half.

The mutation that exposed it (``return arr[::k, ::k]`` instead of ``.copy()``)
survived the whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from athanor.ccarc3 import collapse, logical


def _rendered(board: np.ndarray, k: int) -> np.ndarray:
    return np.kron(board, np.ones((k, k), dtype=np.int16)).astype(np.int16)


def test_logical_does_not_alias_when_it_declines_to_reduce():
    """The k == 1 branch -- the one a real 64x64 frame takes."""
    frame = np.array([[1, 2], [3, 4]], dtype=np.int16)
    out = logical(frame)
    assert not np.shares_memory(out, frame)
    out[0, 0] = 9
    assert frame[0, 0] == 1, "writing to logical()'s result reached the frame"


def test_logical_does_not_alias_when_it_does_reduce():
    """The other branch, so the guarantee is stated for the whole function."""
    board = np.array([[1, 2], [3, 4]], dtype=np.int16)
    frame = _rendered(board, 4)
    out = logical(frame)
    assert not np.shares_memory(out, frame)
    out[0, 0] = 9
    assert frame[0, 0] == 1


def test_collapse_does_not_alias_on_an_empty_grid():
    # np.shares_memory is not decisive between two empty buffers, so identity
    # is the assertion that means anything here.
    frame = np.zeros((0, 0), dtype=np.int16)
    assert collapse(frame) is not frame


def test_collapse_does_not_alias_a_grid_it_cannot_reduce():
    frame = np.array([[1, 2], [3, 4]], dtype=np.int16)
    out = collapse(frame)
    assert not np.shares_memory(out, frame)
    out[0, 0] = 9
    assert frame[0, 0] == 1


@pytest.mark.parametrize("fn", [logical, collapse])
def test_the_result_is_writable(fn):
    """A copy that came back read-only would trade one surprise for another."""
    out = fn(np.array([[1, 2], [3, 4]], dtype=np.int16))
    out[0, 0] = 7
    assert out[0, 0] == 7
