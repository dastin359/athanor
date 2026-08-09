"""The sweep driver must not re-play a game already on the shared card.

Found by mutation: disabling the `if banked.exists()` gate in `_run_one` left
all 1079 tests green. Nothing asserted the guard that a one-card sweep depends
on most.

A scorecard cannot un-play a game. A second play of an already-banked game lands
beside the first, the per-game row count stops matching the result, and
`scoring.card_disagreement` starts reporting a split that is real — after the
money is spent. At ~$26 a game across 25 games, the guard failing at game 14 is
the expensive way to find out.

Two companions are pinned here for the same reason: a dead endpoint must abort
the queue rather than burn every remaining game against it, and a proofread that
could not run must say so rather than bank silently.

`run_game` is monkeypatched to a recorder, so these assert what the driver DID,
not what it printed.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PY_BIN = REPO / ".venv" / "bin" / "python"


def _drive(scratch: pathlib.Path, *, banked: bool) -> dict:
    """Run `_run_one` for one game with `run_game` replaced by a recorder."""
    gid = "bp35-fixture"
    out = scratch / "clean_rollouts" / gid
    out.mkdir(parents=True)
    if banked:
        (out / "clean_result.json").write_text(
            json.dumps({"game_id": gid, "won": True,
                        "levels_reached": 4, "levels_total": 4}), encoding="utf-8")

    code = f'''
import sys, json, types, pathlib
sys.path[:0] = [{str(REPO / "tools")!r}, {str(REPO / "src")!r}]
import clean_rollouts as cr

played = []
cr.run_game = lambda cfg: played.append(getattr(cfg, "game_id", "?"))

class _Info:
    levels = 4
    def suggested_budget(self, m): return 100

cr._run_one({gid!r}, {{{gid!r}: _Info()}})
print("PLAYED=" + json.dumps(played))
'''
    r = subprocess.run(
        [str(PY_BIN), "-c", code], capture_output=True, text=True, timeout=180,
        env={**os.environ, "CCARC3_SCRATCH": str(scratch), "ARC_API_KEY": "dummy"},
    )
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("PLAYED="))
    return {"played": json.loads(line.removeprefix("PLAYED=")), "out": r.stdout}


def test_a_banked_game_is_not_played_again(tmp_path: pathlib.Path) -> None:
    got = _drive(tmp_path, banked=True)
    assert got["played"] == [], (
        "the driver played a game that already had a banked clean result. On a "
        "shared scorecard that second play cannot be undone: it lands beside the "
        f"first and the card stops matching the result.\n{got['out']}"
    )
    assert "skipping" in got["out"]


def test_an_unbanked_game_IS_played(tmp_path: pathlib.Path) -> None:
    """The other direction, or a driver that plays nothing would pass above."""
    got = _drive(tmp_path, banked=False)
    assert got["played"] == ["bp35-fixture"], (
        f"the driver skipped a game with no banked result: {got['out']}"
    )


def test_a_dead_endpoint_aborts_the_queue() -> None:
    """`_aborted` is what reaches the other threads; without it they all run.

    Every remaining game fails identically in milliseconds against a dead
    endpoint, so the pass burns its retries on games it never really launched.
    """
    src = (REPO / "tools" / "clean_rollouts.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    i = code.index("Connection refused")
    window = code[i:i + 900]
    assert "_aborted.set()" in window, (
        "a Connection-refused failure no longer sets the abort flag, so the "
        "other threads keep launching against a dead endpoint"
    )
    assert code.index("_aborted.set()", i) < code.index("raise SystemExit", i), (
        "the abort flag is set after the raise unwinds this thread, so it never "
        "reaches the others"
    )


def test_a_proofread_that_could_not_run_is_announced() -> None:
    """Silence here means a run banks unchecked and reads as checked."""
    src = (REPO / "tools" / "clean_rollouts.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "elif p.returncode not in (0, 1, 2):" in code, (
        "the proofread's did-not-run branch is gone; a crashed proofread would "
        "bank the run with no verdict and no message"
    )
    assert "PROOFREAD DID NOT RUN" in code
