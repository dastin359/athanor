"""Grid representation primitives for ARC-AGI-3.

ARC-AGI-3 hands back 64x64 integer grids over a sixteen-colour palette. Rendered
the way the SDK does it -- ``[5, 5, 5, ...]`` rows, one per line -- a single grid
costs roughly 6k tokens, and a game produces several per action across hundreds
of actions. These helpers exist so no solver has to re-derive its own encoding,
and so every run's traces stay comparable with every other run's.

Nothing here imports ``arc_agi_3``. Grids are plain nested lists or arrays, so
this module is usable (and testable) without the SDK or an API key.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

PALETTE_SIZE = 16
"""ARC-AGI-3 uses sixteen colours, not ARC-AGI-2's ten."""

DEFAULT_BACKGROUND = 5
"""The SDK renders 5 as black and treats it as the default backdrop.

Passed as the default to :func:`objects` only when the caller asks for it
explicitly; the automatic mode infers the background per grid instead, which
survives games that use a different backdrop.
"""

_CHARS = "0123456789abcdef"

__all__ = [
    "PALETTE_SIZE",
    "DEFAULT_BACKGROUND",
    "Change",
    "Object",
    "as_grid",
    "render",
    "diff",
    "block_size",
    "logical",
    "objects",
]


def as_grid(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Coerce a nested sequence to a 2-D ``int16`` array, validating the palette."""
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {arr.shape}")
    if arr.size and (arr.min() < 0 or arr.max() >= PALETTE_SIZE):
        raise ValueError(
            f"grid values must lie in [0, {PALETTE_SIZE}); "
            f"got [{arr.min()}, {arr.max()}]"
        )
    return arr


def render(grid: Sequence[Sequence[int]] | np.ndarray) -> str:
    """Render a grid as one character per cell, ``0-9a-f``.

    A 64x64 grid becomes 64 lines of 64 characters: 4,159 characters against
    12,416 for the bracketed-integer rows, a 3x reduction (measured, not
    estimated). The margin against the SDK's ``pretty_print_3d`` is larger,
    since that adds a header and two-space indent per grid -- and one action can
    return several grids. The saving is why a solver can afford to look at more
    than a handful of frames.
    """
    arr = as_grid(grid)
    return "\n".join("".join(_CHARS[v] for v in row) for row in arr)


@dataclass(frozen=True)
class Change:
    """A single cell that differs between two grids."""

    y: int
    x: int
    before: int
    after: int


def diff(
    before: Sequence[Sequence[int]] | np.ndarray,
    after: Sequence[Sequence[int]] | np.ndarray,
) -> list[Change]:
    """Cells that differ, in row-major order.

    Raises if the two grids disagree on shape -- a shape change is a real event
    in these games and silently returning "everything changed" would bury it.
    """
    a, b = as_grid(before), as_grid(after)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ys, xs = np.nonzero(a != b)
    return [Change(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs)]


def block_size(grid: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Largest ``k`` for which the grid is exactly a ``k x k``-block image.

    Games are *drawn* at 64x64 but their logical board is usually far coarser,
    each logical cell painted as a solid block. Recovering ``k`` is the single
    biggest token win available, and it is not something a solver should be
    eyeballing.

    Returns 1 when no larger factor makes every block uniform, so the result is
    always safe to divide by.

    Note: the algorithm is exact -- it never claims a ``k`` that does not hold --
    but whether real games actually render on a uniform block grid is listed as
    an open question in ``docs/ccarc3_design.md`` (§7.5) and has only been
    checked against locally authored games so far.
    """
    arr = as_grid(grid)
    h, w = arr.shape
    if h == 0 or w == 0:
        return 1
    for k in range(min(h, w), 1, -1):
        if h % k or w % k:
            continue
        blocks = arr.reshape(h // k, k, w // k, k)
        # Every block is uniform iff its min and max agree everywhere.
        if np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            return k
    return 1


def logical(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Collapse a block-rendered grid to its logical board.

    Idempotent on grids that are already logical, since :func:`block_size`
    falls back to 1.
    """
    arr = as_grid(grid)
    k = block_size(arr)
    return arr if k == 1 else arr[::k, ::k].copy()


@dataclass(frozen=True)
class Object:
    """A connected run of same-coloured cells."""

    colour: int
    cells: frozenset[tuple[int, int]]
    """``(y, x)`` pairs."""

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """``(top, left, bottom, right)``, inclusive."""
        ys = [y for y, _ in self.cells]
        xs = [x for _, x in self.cells]
        return min(ys), min(xs), max(ys), max(xs)

    @property
    def size(self) -> int:
        return len(self.cells)


_NEIGHBOURS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBOURS_8 = _NEIGHBOURS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def objects(
    grid: Sequence[Sequence[int]] | np.ndarray,
    *,
    background: int | None = None,
    connectivity: int = 4,
) -> list[Object]:
    """Connected components of equal colour, largest first.

    ``background=None`` infers the backdrop as the most common colour, which is
    what survives games that do not use :data:`DEFAULT_BACKGROUND`. Pass an
    explicit colour to override, or ``-1`` to treat every colour as foreground.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    arr = as_grid(grid)
    if arr.size == 0:
        return []
    if background is None:
        background = Counter(arr.ravel().tolist()).most_common(1)[0][0]

    offsets = _NEIGHBOURS_4 if connectivity == 4 else _NEIGHBOURS_8
    h, w = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)
    found: list[Object] = []

    for y0 in range(h):
        for x0 in range(w):
            colour = int(arr[y0, x0])
            if seen[y0, x0] or colour == background:
                continue
            cells: list[tuple[int, int]] = []
            queue = deque([(y0, x0)])
            seen[y0, x0] = True
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                        if arr[ny, nx] == colour:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            found.append(Object(colour=colour, cells=frozenset(cells)))

    found.sort(key=lambda o: (-o.size, sorted(o.cells)[0]))
    return found


def counts(grid: Sequence[Sequence[int]] | np.ndarray) -> dict[int, int]:
    """Cell count per colour, for the colours actually present."""
    arr = as_grid(grid)
    return dict(Counter(arr.ravel().tolist()))


def flatten_frames(frames: Iterable[Sequence[Sequence[int]]]) -> list[np.ndarray]:
    """Coerce the grid sequence a single ARC-AGI-3 action returns.

    ``FrameData.frame`` is a *list* of grids -- the engine renders every frame
    until the action completes -- so callers that assume one grid per action
    quietly drop the intermediate states where the mechanics are visible.
    """
    return [as_grid(f) for f in frames]
