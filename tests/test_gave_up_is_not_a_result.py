"""A solver that stopped with its allowance untouched did not lose — it quit.

Fourth face of one failure. Interrupted, crashed, clocked out and now quit all
look identical in `result.json` — `won: false`, no error — and all four are the
harness failing to record that the environment was never actually contested.
`if prior and not prior.get("error"): skip` then makes the false loss permanent.
"""
import json

import pytest

from athanor.ccarc3 import session as S
from athanor.ccarc3.client import GameInfo
from athanor.ccarc3.session import Ccarc3Config, Workspace


def _ws(tmp_path, *, levels=10, baselines=(32, 81, 60, 71, 205, 148, 244, 109, 164, 225),
        meta_budget=None):
    root = tmp_path / "ws"
    root.mkdir()
    meta = {"game_id": "lf52-x", "levels": levels}
    if meta_budget is not None:
        meta["action_budget"] = meta_budget
    (root / "meta.json").write_text(json.dumps(meta))
    info = GameInfo(game_id="lf52-x", title="LF52", tags=(), baseline_actions=tuple(baselines))
    return Workspace(root=root, config=Ccarc3Config("lf52-x", budget_multiple=5.0),
                     info=info, initial_prompt="")


def test_the_budget_survives_the_baseline_strip(tmp_path):
    """The strip removes `action_budget` from meta.json on purpose, which zeroed
    this for all 25 clean rollouts and silently disabled the guard that reads it."""
    ws = _ws(tmp_path)                       # stripped: no action_budget in meta

    assert S._action_budget(ws) == 1339 * 5, "must fall back to the parent's own arithmetic"


def test_meta_still_wins_when_it_has_one(tmp_path):
    """An unstripped run keeps reporting the multiple it actually got."""
    assert S._action_budget(_ws(tmp_path, meta_budget=4242)) == 4242


def test_quitting_with_the_allowance_untouched_is_not_a_result(tmp_path, monkeypatch):
    """`lf52`: exit 0, no timeout, 7 of 10 levels, 865 of 6695 actions."""
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 865, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    out = S.collect_outcome(ws, exit_code=0, timed_out=False)

    assert "gave up" in out["error"] and "7 of 10" in out["error"]


def test_a_solver_that_spent_its_allowance_is_a_real_loss(tmp_path, monkeypatch):
    """Grinding to the ceiling and losing is a measurement of the game, and must
    stay banked — otherwise every hard environment retries forever."""
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 5000, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error")


def test_the_retry_is_bounded(tmp_path, monkeypatch):
    """Unlike an interruption, quitting may be telling us the game is hard. At
    ~$60 a run, 12 passes is $720 to hear the same answer three times."""
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 865, "levels_reached": 7})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": S.GIVE_UP_ATTEMPTS - 1})
    assert S.collect_outcome(ws, exit_code=0, timed_out=False).get("error")

    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": S.GIVE_UP_ATTEMPTS})
    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error"), (
        "after the bound the loss stands as real"
    )


def test_a_win_is_never_second_guessed(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts",
                        lambda p: {"actions_used": 100, "levels_reached": 10, "won": True})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error")
