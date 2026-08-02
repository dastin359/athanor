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

PALETTE = (
    "#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
    "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
    "#FF851B", "#921231", "#4FCC30", "#A356D6",
)
"""The official sixteen colours, read out of the SDK's own renderer.

0-5 is a white-to-black greyscale ramp, which is why 5 (black) reads as the
default backdrop; 6-15 are the chromatic colours.
"""

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
    "collapse",
    "cell_boundaries",
    "png",
    "PALETTE",
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
    """Collapse a block-rendered grid to its logical board, losslessly.

    Exact and safe: it only divides by a factor :func:`block_size` has proved,
    and is idempotent on grids that are already logical.

    **It is usually not the function you want on a real frame.** ARC-AGI-3
    renders into a fixed 64x64 viewport, and a board whose side does not divide
    64 is scaled non-uniformly -- a 10-wide board gives cells 6 or 7 pixels
    wide, so no integer factor holds and this correctly refuses to claim one,
    returning a barely-reduced grid. Use :func:`collapse` there.
    """
    arr = as_grid(grid)
    k = block_size(arr)
    return arr if k == 1 else arr[::k, ::k].copy()


def collapse(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Collapse runs of identical adjacent rows and columns.

    **This preserves structure and destroys position.** It is not a downscale
    and the result is not the logical board. Measured on the local bench, one
    frame of a 10x10 game goes 64x64 -> 32x32 under :func:`logical` and
    64x64 -> **5x5** here: every band of identical background rows between the
    two sprites merges into a single row, however wide it was.

    So it answers "what objects are there and how are they arranged relative to
    one another" cheaply, and answers "where is the avatar" wrongly. For the
    logical board with its metric intact, recover the cell grid across a whole
    trace with :func:`cell_boundaries` and index into the raw frame.
    """
    arr = as_grid(grid)
    if arr.size == 0:
        return arr
    keep_rows = np.ones(arr.shape[0], dtype=bool)
    keep_rows[1:] = (arr[1:] != arr[:-1]).any(axis=1)
    arr = arr[keep_rows]
    keep_cols = np.ones(arr.shape[1], dtype=bool)
    keep_cols[1:] = (arr[:, 1:] != arr[:, :-1]).any(axis=0)
    return arr[:, keep_cols].copy()


def png(
    grid: Sequence[Sequence[int]] | np.ndarray,
    path: str | "Path",
    *,
    scale: int = 8,
    grid_lines: bool = False,
) -> str:
    """Write a grid as a PNG and return the path, so you can *look* at it.

    Reading a 64x64 character grid is a poor way to see spatial structure --
    corridors, enclosures, symmetry and "which thing moved" all read instantly
    as an image and slowly as text. You can open the file with the Read tool
    and view it directly.

    Use it alongside :func:`render`, not instead of it: the image is better for
    grasping layout, the text better for reading exact cell values and for
    diffing. When a hypothesis is about *shape*, look; when it is about
    *coordinates*, read.

    ``scale`` is pixels per cell. ``grid_lines`` overlays faint cell borders,
    which helps when counting cells but clutters a dense frame.
    """
    from pathlib import Path as _Path

    from PIL import Image, ImageDraw

    arr = as_grid(grid)
    h, w = arr.shape
    img = Image.new("RGB", (w * scale, h * scale), PALETTE[5])
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            draw.rectangle(
                [x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1],
                fill=PALETTE[int(arr[y, x])],
            )
    if grid_lines and scale >= 4:
        for x in range(w + 1):
            draw.line([(x * scale, 0), (x * scale, h * scale)], fill="#444444")
        for y in range(h + 1):
            draw.line([(0, y * scale), (w * scale, y * scale)], fill="#444444")

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def cell_boundaries(
    grids: Iterable[Sequence[Sequence[int]] | np.ndarray],
) -> tuple[list[int], list[int]]:
    """Recover the render's cell grid by pooling boundaries across many frames.

    Returns ``(row_starts, col_starts)`` into the raw frame.

    Single-frame inference cannot do this. ARC-AGI-3 scales a board into a fixed
    64x64 viewport, so a 10-wide board gives cells of 6 and 7 pixels -- no
    integer factor for :func:`block_size` to find -- and any one frame only
    reveals the boundaries where its own contents happen to change colour. A
    board that is mostly empty reveals almost none.

    Pooling fixes it. Across a trace, sprites move and occupy different cells,
    so the union of observed boundaries converges on the true grid. This is why
    it takes an iterable of grids and not a grid: the extra frames are the whole
    mechanism, and calling it on one frame will usually under-report.

    Boundaries only ever appear where the raw pixels change, so the result is
    always a subset of the true cell grid -- it under-reports rather than
    inventing splits. Cross-check ``len(row_starts)`` against a board size you
    have independent reason to believe before trusting it as complete.
    """
    rows: set[int] = {0}
    cols: set[int] = {0}
    height = width = 0
    for g in grids:
        arr = as_grid(g)
        if arr.size == 0:
            continue
        height, width = arr.shape
        changed_rows = (arr[1:] != arr[:-1]).any(axis=1)
        rows.update(int(i) + 1 for i in np.nonzero(changed_rows)[0])
        changed_cols = (arr[:, 1:] != arr[:, :-1]).any(axis=0)
        cols.update(int(i) + 1 for i in np.nonzero(changed_cols)[0])
    if not height:
        return [], []
    return sorted(r for r in rows if r < height), sorted(c for c in cols if c < width)


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
