"""The card check must read only the plays *this attempt* produced.

**The guard silently changed meaning when the shared scorecard landed.**
`levels_completed` holds one entry per play, and a shared card keeps appending to
the same `cards[game_id]` entry across every *attempt* of a game -- so
`max(levels_completed)`, which all three copies of this guard used, carries the
high-water mark of attempts that were thrown away. A game whose attempt_1 reached
level 8 and was discarded, and whose attempt_2 has a card that stopped updating
at level 3, reads `best=8 >= reached=8` and banks as corroborated. That is the
exact `sb26` failure the guard exists to catch, made invisible by the fix to an
unrelated problem.

Three tools carry this check and none of them had a test. They now share one
implementation, and this drives each of them from disk.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))

import clean_rollouts as cr            # noqa: E402
import restore_clean_rollouts as rcr   # noqa: E402
from athanor.ccarc3 import scoring     # noqa: E402

GID = "sb26-7fbdac44"

# attempt_1 got to level 8 and was discarded; attempt_2 is the one being banked
# and its card froze at 3. Both live in one card entry.
POISONED = {"cards": {GID: {"levels_completed": [8, 3], "total_actions": 400}}}
BANKED = {"game_id": GID, "levels_reached": 8, "playthroughs": 1}


def _card_dir(tmp_path, card, name="ws") -> pathlib.Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorecard.json").write_text(json.dumps(card))
    return d


def _evidence_dir(tmp_path, card, name="ev") -> pathlib.Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "scorecard.json.gz", "wt") as fh:
        json.dump(card, fh)
    return d


def test_a_frozen_card_is_not_excused_by_a_discarded_attempt():
    """`max()` over the whole entry passes this; the attempt's own plays do not."""
    assert max(POISONED["cards"][GID]["levels_completed"]) == 8, "the old reading"
    assert scoring.card_disagreement(POISONED, GID, BANKED) == "card 3 vs result 8 levels"


def test_the_sweep_driver_refuses_it(tmp_path):
    assert cr.uncorroborated(_card_dir(tmp_path, POISONED), BANKED)


def test_the_restore_refuses_it(tmp_path):
    assert rcr.card_corroborates(_evidence_dir(tmp_path, POISONED), BANKED)


def test_a_card_holding_fewer_plays_than_the_run_is_a_disagreement():
    """A card that stopped listening mid-run leaves fewer entries than the trace
    recorded resets. That is the named failure, not a shrug."""
    card = {"cards": {GID: {"levels_completed": [3]}}}
    result = {"game_id": GID, "levels_reached": 3, "playthroughs": 3}
    assert scoring.card_disagreement(card, GID, result) == (
        "card holds 1 play(s), result claims 3")


def test_a_real_multi_play_attempt_still_corroborates(tmp_path):
    """The control, from preserved evidence: `su15` attempt_3 ran three plays and
    its card holds exactly those three. Nothing here may reject it."""
    ev = pathlib.Path("evidence/ccarc3/clean_rollouts/su15-1944f8ab/attempt_3/su15-1944f8ab")
    if not (ev / "scorecard.json.gz").exists():
        return                            # evidence not checked out in this tree
    card = json.loads(gzip.open(ev / "scorecard.json.gz").read())
    result = json.loads(gzip.open(ev / "result.json.gz").read())
    assert result["playthroughs"] == 3 and result["levels_reached"] == 9
    assert scoring.card_disagreement(card, result["game_id"], result) == ""
    assert cr.uncorroborated(_card_dir(tmp_path, card), result) == ""


def test_the_three_copies_share_one_implementation():
    """They drifted apart once already. Each must reach the same function."""
    import ast, inspect                   # noqa: PLC0415
    for fn in (cr.uncorroborated, rcr.card_corroborates):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        body = ast.unparse(tree.body[0].body[1:])   # drop the docstring; it
        assert "card_disagreement" in body           # quotes the broken reading
        assert "max(" not in body, "a local re-implementation is how this broke"


# ==========================================================================
# The half the shared card left open, found 2026-08-08.
# ==========================================================================

def test_a_frozen_card_on_a_RETRIED_game_is_caught(tmp_path):
    """**`done[-plays:]` cannot fire on a retry, because the predecessor pads it.**

    The server appends every attempt of a game to one `cards[game_id]` entry, so
    `len(done) < plays` — the only safety net — is always satisfied by prior
    attempts' rows. When the BANKED attempt contributes zero rows, which is the
    frozen-card case this whole function exists for, the slice silently returns
    the DISCARDED attempt's rows and corroborates the run with them. Whether that
    passed depended only on how far the thrown-away run happened to get:
    `[5, 8]` banked clean, `[5, 3]` was caught.

    `ArcClient.open` snapshots the row count against the lent card, so the
    attempt knows where its own rows begin.
    """
    card = {"cards": {GID: {"levels_completed": [5, 8]}}}   # attempt_1 got to 8
    banked = {"game_id": GID, "levels_reached": 8, "playthroughs": 1,
              "card_plays_at_open": 2}                      # ours start after row 2
    why = scoring.card_disagreement(card, GID, banked)
    assert "0 play(s) for this attempt" in why, (
        f"a frozen card on a retried game banked as corroborated: {why!r}"
    )


def test_the_boundary_does_not_reject_a_healthy_retry(tmp_path):
    """A retry whose own play IS on the card must still bank."""
    card = {"cards": {GID: {"levels_completed": [5, 8, 9]}}}
    banked = {"game_id": GID, "levels_reached": 9, "playthroughs": 1,
              "card_plays_at_open": 2}
    assert scoring.card_disagreement(card, GID, banked) == ""


def test_an_unknown_boundary_falls_back_rather_than_refusing(tmp_path):
    """Every run banked before this field existed has no boundary, and a check
    that voids the corpus on deploy is a check that gets switched off. The
    fallback is the old behaviour, which is weaker on a retry and correct on the
    per-game cards those runs actually used."""
    card = {"cards": {GID: {"levels_completed": [9, 9, 9]}}}
    banked = {"game_id": GID, "levels_reached": 9, "playthroughs": 3}
    assert scoring.card_disagreement(card, GID, banked) == ""


def test_the_client_records_the_boundary_against_a_lent_card(tmp_path, monkeypatch):
    """End to end: the number has to come off the real card at open time, since
    that is the only moment it is knowable."""
    from athanor.ccarc3.client import ArcClient, GameInfo

    info = GameInfo(GID, "Test", ("click",), (10, 20, 30))
    c = ArcClient(GID, trace_path=tmp_path / "trace.jsonl", info=info,
                  api_key="test-key-not-used-offline", card_id="lent-card")
    monkeypatch.setattr(
        ArcClient, "scorecard",
        lambda self: {"cards": {GID: {"levels_completed": [4, 7]}}})

    c.open()
    assert c.card_plays_at_open == 2, (
        "the attempt did not record where its own rows begin"
    )
    saved = json.loads((tmp_path / "trace.state.json").read_text())
    assert saved["card_plays_at_open"] == 2, "the boundary was not persisted"


def test_an_unreadable_card_leaves_the_boundary_unknown_not_wrong(tmp_path, monkeypatch):
    """A failed read must not break the attempt: `sweep_card` probes the card at
    driver start and `_assert_server_agrees` re-checks it on resume, so an
    unreachable card is caught where it can be acted on."""
    from athanor.ccarc3.client import ArcClient, GameInfo

    info = GameInfo(GID, "Test", ("click",), (10, 20, 30))
    c = ArcClient(GID, trace_path=tmp_path / "trace.jsonl", info=info,
                  api_key="test-key-not-used-offline", card_id="lent-card")
    monkeypatch.setattr(
        ArcClient, "scorecard",
        lambda self: (_ for _ in ()).throw(RuntimeError("404 not found")))

    c.open()                                   # must not raise
    assert c.card_plays_at_open == -1
