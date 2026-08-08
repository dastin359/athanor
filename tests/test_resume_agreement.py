"""A resume must not proceed when the server is not where the ledger thinks.

**The failure this prevents cost 370 actions and was invisible while it
happened.** `session.py::_record_resume_state` describes it: "A resume once
preserved the ledger but not the game -- trace indices continued from 370 while
the server replayed levels 0-5 on a fresh scorecard." That function was the
response, and it only *snapshots* what the resume inherited: the loss becomes
reconstructable afterwards rather than impossible. Detect-instead-of-enforce,
which is the shape that has cost this project a run at nearly every layer.

The mismatch is observable before one action is spent, and free — reading a
scorecard is a GET the proxy allows and the server does not bill.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from athanor.ccarc3.client import ArcClient, GameInfo

INFO = GameInfo("zz99-deadbeef", "Test", ("click",), (17, 38, 31))


def _client(tmp_path, level: int, card: dict | Exception):
    # **The key is passed, not inherited.** Without it these six tests read
    # `ARC_API_KEY` from the ambient environment and `ArcClient.__post_init__`
    # raises before a single assertion runs -- so the whole file passed or
    # errored depending on whether the shell that launched pytest happened to
    # have sourced `.env`. A test whose verdict turns on an ambient secret is
    # not testing what it says it is.
    c = ArcClient("zz99-deadbeef", trace_path=tmp_path / "trace.jsonl", info=INFO,
                  api_key="test-key-not-used-offline")
    c._resumed = True
    c.card_id = "card-1"
    c.level = level
    c.actions_used = 370
    def scorecard():
        if isinstance(card, Exception):
            raise card
        return card
    c.scorecard = scorecard          # type: ignore[method-assign]
    return c


def test_a_replayed_game_is_refused(tmp_path):
    """Ledger at level 6, server's card at level 0 — the exact 370-action case."""
    c = _client(tmp_path, 6, {"cards": {"zz99-deadbeef": {"levels_completed": [0]}}})
    with pytest.raises(RuntimeError, match="the server's .*card says 0"):
        c.open()


def test_an_agreeing_card_resumes(tmp_path):
    c = _client(tmp_path, 6, {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}})
    assert c.open() is c


def test_a_server_ahead_of_the_ledger_is_allowed(tmp_path):
    """Only *behind* is the corruption. Ahead means the ledger lost a write,
    which loses bookkeeping rather than the game, and is recoverable."""
    c = _client(tmp_path, 5, {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}})
    assert c.open() is c


def test_an_unreadable_card_refuses_rather_than_assuming(tmp_path):
    """A resume that cannot be verified is one that should not proceed: this
    call is the only thing between a mismatched card and a re-spent run."""
    c = _client(tmp_path, 6, RuntimeError("404"))
    with pytest.raises(RuntimeError, match="could not read the scorecard"):
        c.open()


def test_a_card_without_the_field_does_not_block_the_resume(tmp_path):
    """Absent evidence is not evidence of corruption — do not invent a verdict."""
    c = _client(tmp_path, 6, {"cards": {"zz99-deadbeef": {}}})
    assert c.open() is c


def test_a_resume_with_nothing_spent_does_not_demand_a_card_read(tmp_path):
    """No ledger, nothing to corrupt.

    Requiring a card read at level 0 with 0 actions would make an offline or
    just-opened game unresumable and protect nothing — the 370-action loss
    required 370 actions to already exist.
    """
    c = _client(tmp_path, 0, RuntimeError("no server here"))
    c.actions_used = 0
    assert c.open() is c


# ==========================================================================
# Two holes the shared scorecard opened in this guard, found 2026-08-08 by an
# adversarial hunt and reproduced by running the real code.
# ==========================================================================

def test_a_card_with_no_entry_for_this_game_refuses(tmp_path):
    """**The `or card` fallback read the other games' aggregate.**

    A 200 whose `cards` map has no entry for this game is the exact "this trace
    was never played on this card" state the guard exists to catch. The old code
    fell through to the card-level `levels_completed` — on a 25-game shared card,
    a number produced by the other 24 — and because that aggregate is an int
    rather than the per-play list it skipped the `isinstance` branch and was
    compared directly, so a large number silently satisfied the check.
    """
    for body in (
        {"levels_completed": 9, "total_actions": 400},          # no `cards` key
        {"cards": {"other-game": {"levels_completed": [9]}}, "levels_completed": 9},
        {"cards": {}},
    ):
        c = _client(tmp_path, 6, body)
        with pytest.raises(RuntimeError, match="carries no entry"):
            c.open()


def test_a_game_idle_past_the_reap_deadline_refuses(tmp_path):
    """**A readable card is not evidence the game is alive.**

    The two are on different clocks: a game idle past ~18 minutes is reaped, a
    card is not, and under one shared card the other 24 games keep it warm. The
    reaped play's row still records the level the ledger claims, so the level
    comparison is `5 < 5` and cannot fire — every bad resume on record was caught
    by a 404 on a per-game card, and the shared card removes that detector.
    """
    agreeing = {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}}
    c = _client(tmp_path, 6, agreeing)
    c.last_touched = time.time() - (ArcClient.REAP_DEADLINE_S + 60)
    with pytest.raises(RuntimeError, match="reap deadline"):
        c.open()


def test_a_gap_the_record_says_survived_still_resumes(tmp_path):
    """The deadline is the UPPER edge of the measured bracket, deliberately.

    `sk48` resumed cleanly across a gap of at least 13.6 minutes, so refusing at
    the lower edge would reject a gap this project has on record as survivable.
    """
    agreeing = {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}}
    c = _client(tmp_path, 6, agreeing)
    c.last_touched = time.time() - (13.7 * 60)
    c.open()                                  # must not raise


def test_a_state_file_without_the_stamp_resumes_as_before(tmp_path):
    """Every run banked before this field existed has no stamp. A guard that
    breaks them on deploy is worse than the hole it closes."""
    agreeing = {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}}
    c = _client(tmp_path, 6, agreeing)
    c.last_touched = 0.0                      # the legacy shape
    c.open()                                  # must not raise


def test_the_stamp_advances_only_when_the_server_answered(tmp_path):
    """**Where the stamp is written decides whether the guard means anything.**

    The reap clock measures time since the SERVER last answered this game.
    `_save_state` runs on paths with no server contact at all — `open()`'s
    lent-card branch, and `gate.on_change`, which `gate.check()` can fire at the
    start of `_send` before the request goes out. Stamping there refreshes the
    clock without touching ARC, so an idle game would look freshly alive and the
    reap check above could never fire: the proxy-for-the-thing mistake, inside
    the fix for a proxy-for-the-thing mistake.

    So: persisting state must carry the stamp forward unchanged, never renew it.
    """
    c = _client(tmp_path, 6, {"cards": {"zz99-deadbeef": {"levels_completed": [6]}}})
    stale = time.time() - (60 * 60)
    c.last_touched = stale

    c._save_state()

    saved = json.loads((tmp_path / "trace.state.json").read_text())
    assert saved["last_touched"] == pytest.approx(stale, abs=1.0), (
        "saving state renewed the reap clock without the server having answered: "
        f"stored {saved['last_touched']}, expected {stale}"
    )
    assert time.time() - saved["last_touched"] > ArcClient.REAP_DEADLINE_S, (
        "the persisted stamp no longer reads as idle, so the guard is disarmed"
    )
