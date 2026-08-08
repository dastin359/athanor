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

import pathlib
import sys

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
