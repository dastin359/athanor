"""A card succession mid-sweep must be visible, and two guards could not see one.

**The mid-sweep check compared a value against itself.** It read the finished
game's `trace.state.json` and compared its `card_id` to the live shared card —
but `_run_one` passes `fresh=True`, `build_workspace` unlinks the state file when
fresh, so `ArcClient._resume` never runs, and `_resume` is the only place
`card_id` can become anything other than the injected sweep card. `open()` takes
its lent-card branch and `_save_state` writes back exactly the id it was handed.
`got == want` by construction, for every game, always.

**And `sweep_card`'s refusal read `attempt_1` and meant "any attempt".** A game
banked on `attempt_2` — because its first was interrupted, which is the ordinary
case this driver is built around — counted as zero games on the card, so the
refusal to remint over a card carrying finished work failed open on the one
decision the driver explicitly declines to make for an operator.

The split that can really happen is a succession across driver restarts: the card
is reaped, the restart mints another, half the sweep on each. It is visible only
in the sweep's own history on disk, so that is what both guards now read.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

import clean_rollouts as cr    # noqa: E402


def _attempt(out: pathlib.Path, gid: str, n: int, card_id: str) -> None:
    ws = out / gid / f"attempt_{n}" / gid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "trace.state.json").write_text(json.dumps({"game_id": gid, "card_id": card_id}))


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "OUT", tmp_path)
    return tmp_path


def test_one_card_across_many_attempts_reads_as_one(out):
    _attempt(out, "aa11-x", 1, "card-A")
    _attempt(out, "aa11-x", 2, "card-A")
    _attempt(out, "bb22-y", 1, "card-A")
    assert cr._cards_seen() == {"card-A": ["aa11-x", "bb22-y"]}


def test_a_succession_is_visible_as_two_cards(out):
    _attempt(out, "aa11-x", 1, "card-A")     # before the reap
    _attempt(out, "bb22-y", 1, "card-B")     # after the restart
    assert cr._cards_seen() == {"card-A": ["aa11-x"], "card-B": ["bb22-y"]}


def test_a_game_banked_on_a_later_attempt_still_counts(out):
    """The `attempt_1` hole. Its first attempt was interrupted and left nothing;
    the run that scored on the card is `attempt_2`."""
    _attempt(out, "aa11-x", 2, "card-A")
    assert cr._cards_seen() == {"card-A": ["aa11-x"]}, (
        "a guard that reads attempt_1 counts this game as zero and continues"
    )


def test_the_startup_refusal_sees_a_game_banked_on_attempt_2(out, monkeypatch, capsys):
    """`sweep_card` must refuse to remint over a card carrying finished games —
    end to end, with the card read failing as it does when a card is reaped."""
    _attempt(out, "aa11-x", 2, "card-A")
    monkeypatch.setattr(cr, "SHARED_CARD_FILE", out / "shared_card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", out / "history.jsonl")
    (out / "shared_card.json").write_text(json.dumps({"card_id": "card-A", "cookies": {}}))
    monkeypatch.setattr(cr.sc, "load", lambda p: cr.sc.SharedCard("card-A", {}))
    monkeypatch.setattr(cr.sc, "read_card",
                        lambda card, gid: (_ for _ in ()).throw(RuntimeError("404 not found")))
    monkeypatch.delenv("CCARC3_NO_SHARED_CARD", raising=False)

    with pytest.raises(SystemExit) as exc:
        cr.sweep_card()
    assert "1 game(s) were scored on it" in str(exc.value)
    assert "aa11" in str(exc.value)


def test_an_empty_card_is_reminted_without_a_fuss(out, monkeypatch):
    """The common case: the card is opened at driver start and reaped before the
    first game reaches it. Five such remints in 6.5 hours, all harmless."""
    monkeypatch.setattr(cr, "SHARED_CARD_FILE", out / "shared_card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", out / "history.jsonl")
    (out / "shared_card.json").write_text(json.dumps({"card_id": "card-A", "cookies": {}}))
    monkeypatch.setattr(cr.sc, "load", lambda p: cr.sc.SharedCard("card-A", {}))
    monkeypatch.setattr(cr.sc, "read_card",
                        lambda card, gid: (_ for _ in ()).throw(RuntimeError("404 not found")))
    monkeypatch.setattr(cr.sc, "open_card", lambda tags: cr.sc.SharedCard("card-B", {"x": "y"}))
    monkeypatch.setattr(cr.sc, "save", lambda card, path: None)
    monkeypatch.delenv("CCARC3_NO_SHARED_CARD", raising=False)

    assert cr.sweep_card().card_id == "card-B"


def test_the_refusal_survives_a_driver_restart(out, monkeypatch):
    """**The hard stop must not delete its own trigger.**

    `sweep_card` retired `shared_card.json` and raised afterwards, so the
    refusal lasted exactly one process: the supervisor relaunches within ten
    minutes, the next call finds no card file, skips the whole reaped-card
    branch and silently mints a fresh one. The operator decision this driver
    explicitly declines to make was therefore made on a timer.
    """
    _attempt(out, "aa11-x", 1, "card-A")
    monkeypatch.setattr(cr, "SHARED_CARD_FILE", out / "shared_card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", out / "history.jsonl")
    (out / "shared_card.json").write_text(json.dumps({"card_id": "card-A", "cookies": {}}))
    monkeypatch.setattr(cr.sc, "load", lambda p: cr.sc.SharedCard("card-A", {}))
    monkeypatch.setattr(cr.sc, "read_card",
                        lambda card, gid: (_ for _ in ()).throw(RuntimeError("404 not found")))
    monkeypatch.setattr(cr.sc, "open_card",
                        lambda tags: pytest.fail("minted a new card over a refusal"))
    monkeypatch.delenv("CCARC3_NO_SHARED_CARD", raising=False)

    for attempt in (1, 2, 3):
        with pytest.raises(SystemExit) as exc:
            cr.sweep_card()
        assert "cannot be moved to a new card" in str(exc.value), (
            f"restart {attempt} did not refuse"
        )
        assert (out / "shared_card.json").exists(), (
            f"restart {attempt}: the refusal retired its own trigger, so the next "
            "process would mint a fresh card silently"
        )
