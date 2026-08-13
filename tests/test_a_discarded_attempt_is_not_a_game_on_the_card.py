"""`_cards_seen()` must count games that were *banked*, not attempts that merely started.

`sweep_card()` hard-stops the sweep when the pinned card is dead and games were
scored onto it: "they cannot be moved to a new card, so continuing would build a
submission that silently omits them". That refusal is correct and load-bearing.
It is also driven entirely by `_cards_seen()`, so anything that inflates that
answer converts a recoverable interruption into a sweep that cannot restart
itself.

`trace.state.json` is written by `open()` **before a single action**. The rule
that keeps the two apart is therefore "a `result.json` beside the state file is
the cheapest evidence the attempt got somewhere", and `_cards_seen` says so in a
comment.

**It applied that rule to one of its two loops.** The live loop under `OUT`
checked for `result.json`; the durable loop over
`evidence/ccarc3/<sweep>/**/trace.state.json.gz`, twelve lines above, did not.
The two scan different stores for the same fact, and the fix landed on the one
that was being read at the time.

Measured 2026-08-12. The WSL VM was force-terminated 13 minutes into a `bp35`
run by an MSIX platform update. The attempt produced no `result.json` and was
correctly discarded; `preserve_evidence.sh` had already gzipped its
`trace.state.json` into the evidence tree; the card 404'd once the VM's network
identity changed. On restart `sweep_card` refused — *"1 game(s) were scored on it
(bp35)"* — over a discarded attempt that had banked nothing and left nothing on
the card worth keeping. The durable store outlives the interruption by design,
which is exactly why the unguarded loop is the one that fires after a crash.

The defect class is the usual one and the sibling-loop shape has bitten this repo
before: a check that names one thing (*games scored on the card*) and reads a
proxy for it (*any attempt ever preserved*). Here it fails **closed**, which is
the safe direction and therefore the one nobody notices until it blocks a run.

So these tests exercise the real function against a real tree, in both stores and
in both directions — an unfinished attempt must not count, and a finished one
must still count, or the refusal this feeds would stop protecting anything.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import clean_rollouts as cr  # noqa: E402

CARD = "ebb808ef-0693-4d6e-a7e4-3ef12210e2f2"
GAME = "bp35-0a0ad940"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A sweep with both stores wired up, empty.

    `_cards_seen` derives the durable path as `DURABLE_CARD_DIR.parent / OUT.name`,
    so both have to move together or the scan reads the real evidence tree.
    """
    out = tmp_path / "scratchpad" / "clean_rollouts_fixture"
    out.mkdir(parents=True)
    durable_card_dir = tmp_path / "evidence" / "ccarc3" / "sweep_card"
    durable_card_dir.mkdir(parents=True)
    monkeypatch.setattr(cr, "OUT", out)
    monkeypatch.setattr(cr, "DURABLE_CARD_DIR", durable_card_dir)
    durable = durable_card_dir.parent / out.name
    return out, durable


def _durable_attempt(durable: pathlib.Path, *, banked: bool) -> None:
    d = durable / GAME / "attempt_1" / GAME
    d.mkdir(parents=True)
    (d / "trace.state.json.gz").write_bytes(
        gzip.compress(json.dumps({"card_id": CARD, "state": "GAME_OVER"}).encode()))
    if banked:
        (d / "result.json.gz").write_bytes(gzip.compress(b"{}"))


def _live_attempt(out: pathlib.Path, *, banked: bool) -> None:
    d = out / GAME / "attempt_1" / GAME
    d.mkdir(parents=True)
    (d / "trace.state.json").write_text(
        json.dumps({"card_id": CARD, "state": "GAME_OVER"}), encoding="utf-8")
    if banked:
        (d / "result.json").write_text("{}", encoding="utf-8")


# ── the durable store: the loop that was unguarded ───────────────────────────

def test_a_durable_trace_without_a_result_is_not_a_game_on_the_card(tree):
    """The regression. This is the state a VM kill leaves behind."""
    _out, durable = tree
    _durable_attempt(durable, banked=False)

    assert cr._cards_seen() == {}, (
        "an interrupted, discarded attempt counted as a game scored on the card; "
        "sweep_card would hard-stop the sweep over work that was never banked"
    )


def test_a_durable_trace_with_a_result_still_counts(tree):
    """The positive control, and the whole point of the guard.

    Without this, the test above could pass because the durable loop stopped
    working altogether — and the refusal it feeds is what stands between a dead
    card and a submission that silently omits the games on it.
    """
    _out, durable = tree
    _durable_attempt(durable, banked=True)

    assert cr._cards_seen() == {CARD: [GAME]}, (
        "a banked game in the durable store was not attributed to its card; the "
        "stranded-card refusal has stopped protecting anything"
    )


# ── the live store: already correct, pinned so it cannot regress ─────────────

def test_a_live_trace_without_a_result_is_not_a_game_on_the_card(tree):
    out, _durable = tree
    _live_attempt(out, banked=False)
    assert cr._cards_seen() == {}


def test_a_live_trace_with_a_result_counts(tree):
    out, _durable = tree
    _live_attempt(out, banked=True)
    assert cr._cards_seen() == {CARD: [GAME]}


# ── the two stores must agree, which is the property that failed ─────────────

@pytest.mark.parametrize("banked", [False, True])
def test_both_stores_answer_the_same_question(tree, banked):
    """The durable and live scans read different files for one fact.

    They disagreed for months in the `banked=False` case, and only the durable
    one is populated after the event that makes the answer matter. Asserting the
    agreement directly is what stops a future fix landing on one loop again.
    """
    out, durable = tree
    _durable_attempt(durable, banked=banked)
    from_durable = cr._cards_seen()

    for child in list(durable.iterdir()):
        __import__("shutil").rmtree(child)
    _live_attempt(out, banked=banked)
    from_live = cr._cards_seen()

    assert from_durable == from_live, (
        f"durable store said {from_durable} and live store said {from_live} for "
        f"the same attempt (banked={banked})"
    )
