"""arc — observation and verification helpers for this puzzle workspace.

Import it the same way from anywhere in the workspace::

    from arc import train_samples, test_samples, verify, check, show, diff

That works from a script under ``explore/`` (``python explore/foo.py``), from a
one-liner at the workspace root (``python -c "from arc import ..."``), and from
a module invocation (``python -m explore.foo``). No ``sys.path`` boilerplate is
needed in any of them.

Design rule, and it is deliberate: **this module contains no transformation
primitives.** No rotate, no flood-fill, no connected components. Building those
is the reasoning work, and handing them over would change what is being
measured. What you get here is the ability to *look* and to *check*.

The important function is :func:`verify`. Anything you believe about this puzzle
should pass through it, because a claim that has been executed is worth more
than a claim that has been asserted — and it costs a fraction of the tokens.
Every call is appended to ``.athanor/invariants.jsonl``, which survives context
compaction and is replayed by ``python gate.py status``.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "WORKSPACE",
    "train_samples",
    "test_samples",
    "verify",
    "check",
    "show",
    "diff",
    "shape",
    "colors",
    "histogram",
    "png",
    "invariants",
]

Grid = list[list[int]]


def _find_workspace() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".athanor" / "state.json").is_file():
            return candidate
    return here


#: Root of this run's workspace.
WORKSPACE: Path = _find_workspace()

_TASK = json.loads((WORKSPACE / "task" / "task.json").read_text(encoding="utf-8"))

#: Training pairs — ``{'input': grid, 'output': grid}``.
train_samples: list[dict[str, Grid]] = _TASK.get("train") or []

#: Test inputs — ``{'input': grid}``. Test outputs are not in this workspace.
test_samples: list[dict[str, Grid]] = _TASK.get("test") or []

#: ARC colour names, for readable reports.
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "lightblue", 9: "maroon",
}

_PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
    5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
}


# ── verification ─────────────────────────────────────────────────────────────

def _caller_source() -> str:
    """Best-effort workspace-relative path of the script that called verify()."""
    for frame in inspect.stack()[1:]:
        filename = frame.filename or ""
        if filename == __file__ or filename.startswith("<"):
            continue
        try:
            return str(Path(filename).resolve().relative_to(WORKSPACE))
        except ValueError:
            return os.path.basename(filename)
    return "<unknown>"


def verify(
    claim: str,
    condition: bool | Callable[[], Any] = False,
    note: str | None = None,
    *,
    retract: bool = False,
) -> bool:
    """Establish a claim about this puzzle by executing it.

    ``condition`` is either a boolean or a zero-argument callable (use a
    callable when the check itself might raise — the exception is recorded as a
    refutation rather than crashing your script)::

        verify("every output is 20x20",
               all(len(s['output']) == 20 and len(s['output'][0]) == 20
                   for s in train_samples))

        verify("background colour is always 8",
               lambda: {s['input'][0][0] for s in train_samples} == {8})

    Prints ``[VERIFIED]`` or ``[REFUTED]``, records the result, and returns the
    boolean so you can branch on it.

    The ledger is append-only and **the most recent entry for a claim wins**, so
    re-verifying a claim supersedes the earlier record. If a check was wrong —
    a condition that was accidentally a tautology, say — withdraw it::

        verify("outputs are always square", retract=True)

    A retracted claim disappears from ``invariants()`` and from
    ``gate.py status``. Withdrawing a bad invariant matters: the whole point of
    the ledger is that everything in it has been executed, and one claim that
    only looks verified poisons the rest.
    """
    error = ""
    if retract:
        holds = False
    elif callable(condition):
        try:
            holds = bool(condition())
        except Exception as exc:  # noqa: BLE001 - a check that blows up has not held
            holds, error = False, f"{type(exc).__name__}: {exc}"
    else:
        holds = bool(condition)

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": str(claim),
        "holds": holds,
        "source": _caller_source(),
    }
    if retract:
        entry["retracted"] = True
    if note:
        entry["note"] = str(note)
    if error:
        entry["error"] = error

    ledger = WORKSPACE / ".athanor" / "invariants.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # a read-only workspace must not break exploration

    label = "RETRACTED" if retract else ("VERIFIED " if holds else "REFUTED  ")
    suffix = f"  ({note})" if note else ""
    if error:
        suffix += f"  [{error}]"
    print(f"[{label}] {claim}{suffix}")
    return holds


def invariants() -> list[dict[str, Any]]:
    """Live invariants: most recent entry per claim, retractions removed."""
    ledger = WORKSPACE / ".athanor" / "invariants.jsonl"
    if not ledger.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("claim"))] = entry
    return [entry for entry in latest.values() if not entry.get("retracted")]


# ── free dry-run ─────────────────────────────────────────────────────────────

def check(solve_fn: Callable[[Grid], Any], *, verbose: bool = True) -> dict[str, Any]:
    """Score a candidate ``solve`` against the training pairs. Costs nothing.

    Use this as many times as you like while iterating. It does **not** consume
    an iteration and does **not** record anything — ``python gate.py submit`` is
    the formal checkpoint, and this exists so you never spend one on a bug you
    could have caught here::

        from arc import check
        def solve(grid): ...
        check(solve)
    """
    import copy

    rows: list[dict[str, Any]] = []
    correct = 0
    total_pixel = 0.0

    for idx, sample in enumerate(train_samples):
        expected = sample["output"]
        row: dict[str, Any] = {"index": idx, "correct": False, "pixel_accuracy": 0.0}
        try:
            predicted = _first_candidate(solve_fn(copy.deepcopy(sample["input"])))
            row["predicted"] = predicted
            row["correct"] = predicted == expected
            row["pixel_accuracy"] = _pixel_accuracy(predicted, expected)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        correct += 1 if row["correct"] else 0
        total_pixel += row["pixel_accuracy"]
        rows.append(row)

    test_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(test_samples):
        row = {"index": idx}
        try:
            raw = solve_fn(copy.deepcopy(sample["input"]))
            candidates = raw if _looks_like_candidate_list(raw) else [raw]
            row["candidates"] = candidates
            row["shapes"] = [shape(c) for c in candidates]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        test_rows.append(row)

    summary = {
        "train": rows,
        "test": test_rows,
        "train_correct": correct,
        "train_total": len(train_samples),
        "train_pixel_accuracy": total_pixel / len(train_samples) if train_samples else 0.0,
        "all_train_correct": bool(train_samples) and correct == len(train_samples),
    }

    if verbose:
        for row in rows:
            if row.get("error"):
                print(f"train {row['index']}  ERROR  {row['error']}")
            elif row["correct"]:
                print(f"train {row['index']}  PASS   {shape(row.get('predicted'))}")
            else:
                wrong = _count_wrong(row.get("predicted"), train_samples[row["index"]]["output"])
                print(
                    f"train {row['index']}  FAIL   got {shape(row.get('predicted'))} "
                    f"want {shape(train_samples[row['index']]['output'])}  "
                    f"wrong={wrong}  pixel={row['pixel_accuracy']:.3f}"
                )
        for row in test_rows:
            if row.get("error"):
                print(f"test  {row['index']}  ERROR  {row['error']}")
            else:
                print(f"test  {row['index']}  -> {len(row.get('candidates') or [])} candidate(s) "
                      f"{row.get('shapes')}")
        print(
            f"== {summary['train_correct']}/{summary['train_total']} training examples reproduced "
            f"(mean pixel {summary['train_pixel_accuracy']:.3f})"
        )
        if summary["all_train_correct"]:
            hypothesis = WORKSPACE / "solution" / "hypothesis.md"
            if hypothesis.is_file() and hypothesis.read_text(encoding="utf-8").strip():
                print("== ready for `python gate.py submit`")
            else:
                print("== write solution/hypothesis.md, then `python gate.py submit`")
    return summary


def _looks_like_candidate_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(isinstance(v, list) and v and isinstance(v[0], list) for v in value)
    )


def _first_candidate(value: Any) -> Any:
    return value[0] if _looks_like_candidate_list(value) else value


def _count_wrong(predicted: Any, expected: Grid) -> int | str:
    if not isinstance(predicted, list) or shape(predicted) != shape(expected):
        return "n/a (shape mismatch)"
    return sum(
        1
        for r in range(len(expected))
        for c in range(len(expected[0]))
        if predicted[r][c] != expected[r][c]
    )


def _pixel_accuracy(predicted: Any, expected: Grid) -> float:
    try:
        exp_rows, exp_cols = len(expected), len(expected[0])
        total = exp_rows * exp_cols
        if not total:
            return 0.0
        pred_rows = len(predicted)
        pred_cols = len(predicted[0]) if pred_rows else 0
        return sum(
            1
            for r in range(min(pred_rows, exp_rows))
            for c in range(min(pred_cols, exp_cols))
            if predicted[r][c] == expected[r][c]
        ) / total
    except Exception:  # noqa: BLE001
        return 0.0


# ── observation ──────────────────────────────────────────────────────────────

def shape(grid: Any) -> str:
    """``"HxW"`` for a grid, ``"?"`` for anything that is not one."""
    try:
        return f"{len(grid)}x{len(grid[0])}"
    except Exception:  # noqa: BLE001
        return "?"


def colors(grid: Grid) -> list[int]:
    """Sorted distinct colours present in a grid."""
    return sorted({int(cell) for row in grid for cell in row})


def histogram(grid: Grid) -> dict[int, int]:
    """``{colour: count}``, most useful for spotting the background."""
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[int(cell)] = counts.get(int(cell), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def show(grid: Grid, title: str | None = None, ruler: bool = True) -> None:
    """Print a grid with row and column indices."""
    if title:
        print(title)
    if not grid:
        print("<empty>")
        return
    height, width = len(grid), len(grid[0])
    pad = len(str(height - 1))
    if ruler:
        tens = "".join(str((c // 10) % 10) if c >= 10 else " " for c in range(width))
        ones = "".join(str(c % 10) for c in range(width))
        if width > 10:
            print(" " * (pad + 1) + tens)
        print(" " * (pad + 1) + ones)
    for r, row in enumerate(grid):
        print(f"{r:>{pad}} " + "".join(str(cell) for cell in row))
    print(f"({height}x{width}, colours {colors(grid)})")


def diff(a: Grid, b: Grid, *, limit: int = 40, labels: tuple[str, str] = ("a", "b")) -> None:
    """Print where two grids disagree."""
    if shape(a) != shape(b):
        print(f"shape mismatch: {labels[0]}={shape(a)} {labels[1]}={shape(b)}")
        return
    entries = [
        (r, c, a[r][c], b[r][c])
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    ]
    if not entries:
        print(f"identical ({shape(a)})")
        return
    rows = sorted({e[0] for e in entries})
    cols = sorted({e[1] for e in entries})
    print(
        f"{len(entries)} differing cells in {shape(a)}; "
        f"rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]}"
    )
    for r, c, va, vb in entries[:limit]:
        print(f"  ({r},{c}) {labels[0]}={va} ({COLOR_NAMES.get(va, '?')}) "
              f"{labels[1]}={vb} ({COLOR_NAMES.get(vb, '?')})")
    if len(entries) > limit:
        print(f"  … {len(entries) - limit} more")


def png(grid: Grid, path: str | os.PathLike[str], cell: int = 24) -> str | None:
    """Render a grid to a PNG you can then open with the Read tool.

    Returns the path, or None when Pillow is unavailable. Useful for looking at
    a predicted output the way you looked at the puzzle: as a picture.
    """
    if not grid:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # The interpreter running this script has no Pillow, but the one that
        # built the workspace did — it rendered task/images/. Borrow it rather
        # than making the doctrine's "look at your prediction" advice a dead end.
        return _png_via_harness_python(grid, path, cell)

    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell, height * cell), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for r in range(height):
        for c in range(width):
            draw.rectangle(
                [c * cell, r * cell, c * cell + cell, r * cell + cell],
                fill=_PALETTE.get(int(grid[r][c]), (128, 128, 128)),
                outline=(50, 50, 50),
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", compress_level=1)
    return str(target)


def _png_via_harness_python(grid: Grid, path: str | os.PathLike[str], cell: int) -> str | None:
    """Render through the interpreter that built this workspace."""
    import subprocess

    recorded = WORKSPACE / ".athanor" / "harness_python"
    if not recorded.is_file():
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None
    executable = recorded.read_text(encoding="utf-8").strip()
    if not executable:
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None

    payload = json.dumps({"grid": grid, "path": str(path), "cell": int(cell), "palette": _PALETTE})
    source = (
        "import json,sys\n"
        "from PIL import Image, ImageDraw\n"
        "a=json.loads(sys.stdin.read())\n"
        "g=a['grid']; c=a['cell']; pal={int(k):tuple(v) for k,v in a['palette'].items()}\n"
        "h,w=len(g),len(g[0])\n"
        "im=Image.new('RGB',(w*c,h*c),(255,255,255)); d=ImageDraw.Draw(im)\n"
        "for r in range(h):\n"
        "    for x in range(w):\n"
        "        d.rectangle([x*c,r*c,x*c+c,r*c+c],fill=pal.get(int(g[r][x]),(128,128,128)),outline=(50,50,50))\n"
        "import os\n"
        "os.makedirs(os.path.dirname(os.path.abspath(a['path'])) or '.',exist_ok=True)\n"
        "im.save(a['path'],format='PNG',compress_level=1)\n"
    )
    try:
        done = subprocess.run(
            [executable, "-c", source], input=payload, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not render PNG via {executable}: {exc}")
        return None
    if done.returncode != 0:
        print(f"Could not render PNG via {executable}: {(done.stderr or '').strip()[:200]}")
        return None
    return str(path)


def _pairs() -> Iterable[tuple[Grid, Grid]]:
    for sample in train_samples:
        yield sample["input"], sample["output"]


if __name__ == "__main__":  # `python arc.py` prints a quick orientation dump
    print(f"workspace: {WORKSPACE}")
    print(f"{len(train_samples)} training pairs, {len(test_samples)} test input(s)\n")
    for i, (inp, out) in enumerate(_pairs()):
        print(f"train {i}: {shape(inp)} -> {shape(out)}   "
              f"colours in {colors(inp)} -> out {colors(out)}")
    for i, sample in enumerate(test_samples):
        print(f"test  {i}: {shape(sample['input'])}          colours {colors(sample['input'])}")
