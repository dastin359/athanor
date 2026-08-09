"""`verdict()` decides whether a finished run counts, and nothing exercised it.

The comment above `install_strip` says why the strip is called from `main()` and
not at import: at module scope, importing this driver started a proxy and
demanded an API key, "so `verdict()` -- the function that decides whether a run
counts -- could not be imported to test."

The affordance was built. The test was not written. Coverage over the whole suite
on 2026-08-09 shows lines 299-312 -- the entire body of `verdict` -- never
executed, and `salvage`, which banks a finished attempt found on restart, calls
straight through it.

This is the shape one level up from the usual one: not a check that passes by not
running, but a check nothing runs at all, sitting behind a refactor performed
specifically to make running it possible.

Every branch below is a discard/bank decision on real money -- a wrong "clean"
publishes a contaminated or uncorroborated score, and a wrong "interrupted"
re-runs a game that already finished, at $57-79 a game.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import clean_rollouts as cr  # noqa: E402

GID = "zz99-deadbeef"


@pytest.fixture
def quiet(monkeypatch):
    """Neutralise the two checks that shell out, so each test names its own."""
    monkeypatch.setattr(cr, "uncorroborated", lambda game_dir, data: "")
    monkeypatch.setattr(cr, "fails_proofread", lambda game_dir: "")


def _run(tmp_path: pathlib.Path, result: dict | None, *, raw: str | None = None):
    game_dir = tmp_path / GID
    game_dir.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (game_dir / "result.json").write_text(raw, encoding="utf-8")
    elif result is not None:
        (game_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return game_dir


def test_a_finished_run_with_no_result_is_not_banked(tmp_path, quiet):
    state, data = cr.verdict(_run(tmp_path, None))
    assert (state, data) == ("no result", None)


def test_an_unreadable_result_is_not_banked(tmp_path, quiet):
    """A truncated write is a discard, never a silent zero."""
    state, data = cr.verdict(_run(tmp_path, None, raw='{"game_id": "zz99'))
    assert (state, data) == ("unreadable", None)


def test_an_interrupted_run_is_discarded_and_carries_its_data(tmp_path, quiet):
    """`collect_outcome` marks signal death, crash and wall clock with `error`."""
    state, data = cr.verdict(_run(tmp_path, {
        "game_id": GID, "error": "wall clock fired", "levels_reached": 5}))
    assert state == "interrupted"
    assert data and data["levels_reached"] == 5, (
        "the discard threw away the data; salvage and the log both need it"
    )


def test_a_genuine_loss_is_clean(tmp_path, quiet):
    """Losing is a datum. Only an interruption is not."""
    state, data = cr.verdict(_run(tmp_path, {
        "game_id": GID, "won": False, "levels_reached": 2, "levels_total": 6}))
    assert state == "clean"
    assert data["levels_reached"] == 2


def test_a_run_the_card_does_not_corroborate_is_not_clean(tmp_path, monkeypatch):
    """`sb26` finished 8/8 against a card frozen at level 3 and looked perfect."""
    monkeypatch.setattr(cr, "uncorroborated",
                        lambda game_dir, data: "card says 3/8, we say 8/8")
    monkeypatch.setattr(cr, "fails_proofread", lambda game_dir: "")
    state, data = cr.verdict(_run(tmp_path, {"game_id": GID, "won": True}))
    assert state.startswith("uncorroborated"), state
    assert "card says 3/8" in state, "the reason was dropped from the verdict"
    assert data is not None, "an uncorroborated run still has to be reportable"


def test_a_run_that_fails_the_proofread_is_not_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "uncorroborated", lambda game_dir, data: "")
    monkeypatch.setattr(cr, "fails_proofread",
                        lambda game_dir: "own per-level array in a tool result")
    state, data = cr.verdict(_run(tmp_path, {"game_id": GID, "won": True}))
    assert state.startswith("proofread failed"), state
    assert "own per-level array" in state


def test_corroboration_is_checked_before_the_proofread_shells_out(tmp_path, monkeypatch):
    """Ordering is not cosmetic: the proofread spawns a 300s subprocess."""
    calls = []
    monkeypatch.setattr(cr, "uncorroborated",
                        lambda game_dir, data: calls.append("card") or "no card")
    monkeypatch.setattr(cr, "fails_proofread",
                        lambda game_dir: calls.append("proofread") or "")
    cr.verdict(_run(tmp_path, {"game_id": GID, "won": True}))
    assert calls == ["card"], (
        f"the proofread ran for a run already discarded: {calls}"
    )


def test_salvage_banks_a_clean_earlier_attempt(tmp_path, quiet):
    """A restart must pick up a finished attempt rather than pay for it again.

    This is the bug that motivated `salvage`: `out_dir=attempt_N` built the
    workspace at `attempt_N/<game_id>`, the driver looked in `attempt_N`, every
    attempt read as "no result", and finished runs were re-run.
    """
    game_dir = tmp_path / GID
    for n, payload in ((1, {"game_id": GID, "error": "killed"}),
                       (2, {"game_id": GID, "won": True, "levels_reached": 6})):
        ws = game_dir / f"attempt_{n}" / GID
        ws.mkdir(parents=True)
        (ws / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    banked = cr.salvage(game_dir, GID)

    assert banked and banked["levels_reached"] == 6, (
        "a finished attempt sitting on disk was not salvaged; the driver would "
        "re-run a game it had already paid for"
    )


def test_salvage_declines_when_every_attempt_was_interrupted(tmp_path, quiet):
    game_dir = tmp_path / GID
    for n in (1, 2):
        ws = game_dir / f"attempt_{n}" / GID
        ws.mkdir(parents=True)
        (ws / "result.json").write_text(
            json.dumps({"game_id": GID, "error": "killed"}), encoding="utf-8")
    assert cr.salvage(game_dir, GID) is None
