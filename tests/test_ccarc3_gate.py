"""Tests for the level-boundary gate.

The gate exists because knowledge that survives a level boundary is the only
thing standing between a solver and re-deriving the game seven times over, and
the action budget cannot pay for that. So the tests that matter are the ones
that check it actually *refuses* -- an advisory gate is not a gate.
"""

from __future__ import annotations

import json

import pytest

from athanor.ccarc3 import ArcClient, GateRefusal, LevelGate, RuleBook
from athanor.ccarc3 import client as client_mod


def test_the_gate_is_open_before_any_level_advance(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    g.observe(0)
    g.check()  # must not raise -- nothing has been learned yet
    assert not g.held


def test_a_level_advance_holds_the_gate(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    g.observe(0)
    g.observe(1)
    assert g.held
    with pytest.raises(GateRefusal, match="rule book has not been updated"):
        g.check()


def test_acknowledging_opens_the_gate_and_writes_the_rule_book(tmp_path):
    path = tmp_path / "rules.json"
    g = LevelGate(path)
    g.observe(1)
    g.acknowledge(
        "avatar moves on 1-4; the goal colour is level-specific",
        mechanics=["contact with the goal colour advances the level"],
        refuted=["ACTION5 has any effect"],
    )
    g.check()  # open again
    assert not g.held

    book = RuleBook.load(path)
    assert book.verified[0]["scope"] == "game"
    assert book.refuted[0]["rule"] == "ACTION5 has any effect"
    assert json.loads(path.read_text())["verified"]


def test_an_empty_summary_is_refused(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    g.observe(1)
    with pytest.raises(GateRefusal, match="a claim, not a token"):
        g.acknowledge("   ")
    assert g.held, "a rejected acknowledgement must leave the gate held"


def test_acknowledging_nothing_pending_is_an_error(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    with pytest.raises(GateRefusal, match="nothing pending"):
        g.acknowledge("hello")


def test_a_level_with_nothing_portable_is_recorded_not_dropped(tmp_path):
    """Saying "nothing carries" is a real answer; a run full of them is a signal."""
    path = tmp_path / "rules.json"
    g = LevelGate(path)
    g.observe(1)
    g.acknowledge("nothing here generalised")
    assert RuleBook.load(path).open_questions == ["level 0: nothing here generalised"]


def test_a_full_reset_clears_a_pending_hold_but_keeps_the_rule_book(tmp_path):
    """Progress is rewound; knowledge is not. That split is the whole design."""
    path = tmp_path / "rules.json"
    g = LevelGate(path)
    g.observe(1)
    g.acknowledge("learned something", mechanics=["gravity applies"])
    g.observe(2)
    assert g.held
    g.observe(0)  # full reset
    assert not g.held
    assert RuleBook.load(path).verified


def test_refusals_are_counted(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    g.observe(1)
    for _ in range(3):
        with pytest.raises(GateRefusal):
            g.check()
    assert g.refusals == 3


# --------------------------------------------------------------------------- #
# wired into the client
# --------------------------------------------------------------------------- #


def _frame(**kw):
    base = {
        "game_id": "g1",
        "frame": [[[1, 1], [1, 1]]],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 3,
        "action_input": {"id": 1, "data": {}, "reasoning": None},
        "guid": "guid-1",
        "full_reset": False,
        "available_actions": [1, 2, 3, 4],
    }
    base.update(kw)
    return base


def test_the_client_refuses_the_next_action_after_a_level_advance(monkeypatch, tmp_path):
    replies = [_frame(levels_completed=1), _frame(levels_completed=1)]
    sent = []

    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "c"}
        if url.endswith("/scorecard/close"):
            return {}
        sent.append(url)
        return replies.pop(0)

    monkeypatch.setattr(client_mod, "_post", fake_post)
    monkeypatch.setenv("ARC_API_KEY", "k")

    gate = LevelGate(tmp_path / "rules.json")
    c = ArcClient("g1", trace_path=tmp_path / "t.jsonl", gate=gate).open()
    c.act(1)  # advances to level 1
    assert gate.held

    before = len(sent)
    with pytest.raises(GateRefusal):
        c.act(2)
    assert len(sent) == before, "a gated action must never reach the server"

    gate.acknowledge("level 0 done", mechanics=["moving right is possible"])
    c.act(2)
    assert len(sent) == before + 1


def test_a_client_without_a_gate_is_unaffected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        client_mod, "_post",
        lambda url, payload, key, **kw: (
            {"card_id": "c"} if url.endswith("open") else _frame(levels_completed=1)
        ),
    )
    monkeypatch.setenv("ARC_API_KEY", "k")
    c = ArcClient("g1", trace_path=tmp_path / "t.jsonl").open()
    c.act(1)
    c.act(2)  # no gate, no refusal
    assert c.level == 1


def test_untested_is_recorded_apart_from_refuted(tmp_path):
    """A real solver filed "never tested yet" under refuted, having nowhere else.

    Tested-and-false and never-exercised are different states; collapsing them
    makes a solver stop asking about an open question. Same reason
    arc.unreached() exists on ARC-AGI-2.
    """
    path = tmp_path / "rules.json"
    g = LevelGate(path)
    g.observe(1)
    g.acknowledge(
        "level 0 done",
        mechanics=["ACTION1 moves up"],
        refuted=["ACTION5 is available"],
        untested=["what happens when moving into a wall"],
    )
    book = RuleBook.load(path)
    assert [r["rule"] for r in book.refuted] == ["ACTION5 is available"]
    assert any("UNTESTED" in q and "into a wall" in q for q in book.open_questions)
    assert not any("into a wall" in r["rule"] for r in book.refuted)


def test_untested_alone_still_opens_the_gate(tmp_path):
    g = LevelGate(tmp_path / "rules.json")
    g.observe(1)
    g.acknowledge("learned only what I do not know", untested=["everything"])
    assert not g.held


def test_an_acknowledgement_survives_the_process_that_made_it(monkeypatch, tmp_path):
    """The solver had to write a helper around this. Its docstring read:
    "Gate state is per-process; re-clear it after re-importing session."

    The client persists after each *action*, so acknowledging and then exiting
    left gate_pending set on disk and the next process restored a gate that was
    already satisfied -- refusing an action for a boundary already recorded.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"

    gate = LevelGate(tmp_path / "rules.json")
    c = ArcClient("g", trace_path=path, gate=gate)
    c.card_id = "c1"
    gate.observe(1)
    c._save_state()
    assert json.loads(c.state_path.read_text())["gate_pending"] == 1

    gate.acknowledge("done", mechanics=["m"])
    assert json.loads(c.state_path.read_text())["gate_pending"] is None

    gate2 = LevelGate(tmp_path / "rules.json")
    ArcClient("g", trace_path=path, gate=gate2)
    assert not gate2.held, "the next process must not re-refuse a recorded boundary"
    assert gate2.acknowledged == {1: "done"}


def test_a_boundary_already_acknowledged_is_not_gated_again():
    """What makes a replay affordable.

    After a full reset the solver re-crosses every boundary it has already
    documented. Demanding a fresh rule-book entry at each one would cost a turn
    apiece to restate what the book already holds — and the gate exists to catch
    knowledge about to be lost, not to bill for knowledge already kept.
    """
    from athanor.ccarc3 import GateRefusal, LevelGate

    gate = LevelGate(rulebook_path="rules.json")
    gate.observe(1)
    with pytest.raises(GateRefusal):
        gate.check()
    gate.acknowledge("level 0 established the push mechanic")
    gate.check()

    gate.observe(0)          # full reset for the replay
    gate.observe(1)          # re-crossing a boundary already recorded
    gate.check()             # must not raise

    gate.observe(2)          # a boundary never seen before still gates
    with pytest.raises(GateRefusal):
        gate.check()
