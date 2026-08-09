"""A container replacement must not silently start a second scorecard.

`sweep_card()` has a careful ladder of refusals: an unreadable pin is an operator
decision, a card that 404s is probed three times before being believed dead, and
a dead card carrying banked games is a hard stop because "they cannot be moved to
a new card, so continuing would build a submission that silently omits them".

**Every rung of it reads `SP/shared_card.json` or `SP/shared_card_history.jsonl`,
and `SP` is the store that reverts.** When the container is replaced both files
vanish together, `SHARED_CARD_FILE.exists()` is False, the whole ladder is
skipped, and a fresh card is minted without a word. Confirmed on 2026-08-09:
after a replacement, both files absent — including the one the driver's own error
text points at ("the card_id needed to recover is in the history file").

It is the same defect the `dead` rename already carries a comment about — "the
next `sweep_card()` finds no card file at all, skips this whole branch, and
silently mints a fresh card" — arriving through a different door. That one was
fixed by reordering two statements. This one cannot be: the trigger is deleted by
the platform, so the evidence has to live somewhere the platform does not touch.

The committed history is that somewhere, and it holds **card ids only**. A pinned
card carries `GAMESESSION` and four `AWSALBAPP-*` cookies, and `evidence/` is
pushed to GitHub, so writing them there would publish live credentials. The
cookies are also the only route back to a card — so a replaced container
genuinely cannot resume the old one. What it can do is refuse to pretend the card
never existed, which is the entire decision `sweep_card` says it will not make
for you.
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

CARD = "ebb808ef-0693-4d6e-a7e4-3ef12210e2f2"
OTHER = "a0c879d0-af88-4095-bc89-9eb931af00df"


@pytest.fixture
def replaced(tmp_path, monkeypatch):
    """A box whose scratchpad reverted: no pin, no scratchpad history."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    durable = tmp_path / "evidence" / "ccarc3" / "sweep_card"
    durable.mkdir(parents=True)
    monkeypatch.setattr(cr, "SHARED_CARD_FILE", scratch / "shared_card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", scratch / "shared_card_history.jsonl")
    monkeypatch.setattr(cr, "DURABLE_CARD_DIR", durable)
    monkeypatch.setattr(cr, "DURABLE_CARD_HISTORY", durable / "history.jsonl")
    monkeypatch.delenv("CCARC3_NO_SHARED_CARD", raising=False)

    def _never_mint(*a, **k):
        raise AssertionError("sweep_card minted a card instead of refusing")

    monkeypatch.setattr(cr.sc, "open_card", _never_mint)
    return scratch, durable


def _history(durable: pathlib.Path, *cards: str) -> None:
    (durable / "history.jsonl").write_text(
        "".join(json.dumps({"card_id": c, "event": "opened", "at": 1.0}) + "\n"
                for c in cards), encoding="utf-8")


def test_it_refuses_when_the_lost_card_carries_banked_games(replaced, monkeypatch):
    scratch, durable = replaced
    _history(durable, CARD)
    monkeypatch.setattr(cr, "_cards_seen",
                        lambda: {CARD: ["bp35-0a0ad940", "su15-1944f8ab"]})

    with pytest.raises(SystemExit) as exc:
        cr.sweep_card()

    message = str(exc.value)
    assert CARD in message, "the refusal did not name the stranded card"
    assert "bp35" in message and "su15" in message, "it did not name the games"
    assert "will not choose for you" in message


def test_it_proceeds_when_the_lost_card_never_carried_a_game(replaced, monkeypatch, capsys):
    """An empty card reaped before the first game costs nothing — five such on 2026-08-08."""
    scratch, durable = replaced
    _history(durable, CARD)
    monkeypatch.setattr(cr, "_cards_seen", lambda: {})
    minted = []

    class _Card:
        card_id = "fresh-card"

    monkeypatch.setattr(cr.sc, "open_card", lambda **k: minted.append(1) or _Card())
    monkeypatch.setattr(cr.sc, "save", lambda card, path: path)

    cr.sweep_card()

    assert minted, "it refused over a card with nothing on it"
    out = capsys.readouterr().out
    assert "nothing is stranded" in out, "it minted without saying why that was safe"


def test_a_genuinely_first_run_still_mints(replaced, monkeypatch):
    """No pin and no history is a new sweep, which is the ordinary case."""
    scratch, durable = replaced
    monkeypatch.setattr(cr, "_cards_seen", lambda: {})
    minted = []

    class _Card:
        card_id = "fresh-card"

    monkeypatch.setattr(cr.sc, "open_card", lambda **k: minted.append(1) or _Card())
    monkeypatch.setattr(cr.sc, "save", lambda card, path: path)

    cr.sweep_card()

    assert minted, "a first run was refused"


def test_every_card_in_the_history_is_checked_not_just_the_last(replaced, monkeypatch):
    """A sweep that legitimately succeeded one card can still strand an earlier one."""
    scratch, durable = replaced
    _history(durable, CARD, OTHER)
    monkeypatch.setattr(cr, "_cards_seen", lambda: {CARD: ["bp35-0a0ad940"]})

    with pytest.raises(SystemExit) as exc:
        cr.sweep_card()

    assert CARD in str(exc.value), "only the most recent card was considered"


def test_the_durable_history_never_receives_a_cookie(replaced):
    """`evidence/` is pushed to GitHub; a session cookie there is a live credential."""
    scratch, durable = replaced
    cr._remember_card(CARD, "opened")

    body = (durable / "history.jsonl").read_text(encoding="utf-8")
    row = json.loads(body.strip())
    assert set(row) == {"card_id", "event", "at"}, f"unexpected fields: {sorted(row)}"
    for forbidden in ("cookie", "GAMESESSION", "AWSALB", "value", "api_key"):
        assert forbidden.lower() not in body.lower(), (
            f"{forbidden!r} reached the committed history"
        )


def test_both_histories_are_written_so_neither_alone_is_load_bearing(replaced):
    scratch, durable = replaced
    cr._remember_card(CARD, "opened")
    assert (durable / "history.jsonl").exists(), "the durable history was not written"
    assert (scratch / "shared_card_history.jsonl").exists(), (
        "the scratchpad history was not written; the local recovery text points at it"
    )


def test_an_unwritable_durable_store_does_not_kill_the_sweep(replaced, monkeypatch, capsys):
    """Recording history is bookkeeping. Losing it must not lose the run."""
    scratch, durable = replaced
    monkeypatch.setattr(cr, "DURABLE_CARD_HISTORY",
                        pathlib.Path("/proc/definitely/not/writable/history.jsonl"))

    cr._remember_card(CARD, "opened")

    assert "could not record card history" in capsys.readouterr().out
    assert (scratch / "shared_card_history.jsonl").exists(), (
        "a failure writing one history stopped the other being written"
    )
