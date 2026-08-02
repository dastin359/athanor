"""Tests for CCARC3 workspace construction and outcome collection.

All offline. The one thing worth stating outright: ``collect_outcome`` reads the
ledger, never the solver's own account of how it did. On ARC-AGI-2 the gate was
what made a claimed solve and a real one the same thing; here the trace plays
that role, so it must be the only thing consulted.
"""

from __future__ import annotations

import json

import pytest

from athanor.ccarc3 import Ccarc3Config, GameInfo, build_workspace
from athanor.ccarc3.session import DEFAULT_EFFORT, DEFAULT_MODEL, build_cli_args, collect_outcome

INFO = GameInfo("ls20-test", "LS20", ("keyboard",), (22, 123, 73))


@pytest.fixture
def ws(tmp_path):
    return build_workspace(Ccarc3Config("ls20-test", out_dir=tmp_path), INFO)


def test_workspace_has_everything_the_solver_needs(ws):
    names = {p.name for p in ws.root.iterdir()}
    assert {"CLAUDE.md", "DOCTRINE.md", "session.py", "meta.json", "notes"} <= names


def test_the_action_budget_is_derived_from_the_game_not_guessed(ws):
    meta = json.loads((ws.root / "meta.json").read_text())
    assert meta["action_budget"] == int(218 * 4.0)
    assert str(meta["action_budget"]) in (ws.root / "CLAUDE.md").read_text()


def test_a_short_game_still_gets_a_workable_floor(tmp_path):
    tiny = GameInfo("t", baseline_actions=(3, 4))
    w = build_workspace(Ccarc3Config("t", out_dir=tmp_path), tiny)
    assert json.loads((w.root / "meta.json").read_text())["action_budget"] == 200


def test_the_generated_session_is_prewired_to_this_game(ws):
    src = (ws.root / "session.py").read_text()
    assert "'ls20-test'" in src or '"ls20-test"' in src
    assert "LevelGate" in src and "ArcClient" in src
    assert "(22, 123, 73)" in src, "baselines must reach the solver"


def test_the_workspace_puts_athanor_on_the_path(ws):
    assert "athanor" in ws.env["PYTHONPATH"] or ws.env["PYTHONPATH"].endswith("src")


def test_doctrine_carries_the_findings_that_contradict_instinct(ws):
    doctrine = (ws.root / "DOCTRINE.md").read_text().lower()
    assert "dying is cheap" in doctrine
    assert "never make reset your first action after completing a level" in doctrine
    assert "not refutation unless the rule was applicable" in doctrine


def test_the_run_defaults_are_the_ones_the_project_requires():
    """Opus 5 at high effort, so runs stay comparable across the project."""
    assert DEFAULT_MODEL == "claude-opus-5"
    assert DEFAULT_EFFORT == "high"
    cfg = Ccarc3Config("g")
    assert cfg.model == DEFAULT_MODEL and cfg.effort == DEFAULT_EFFORT


def test_cli_args_carry_model_and_prompt(ws):
    args = build_cli_args(ws)
    assert "--model" in args and DEFAULT_MODEL in args
    assert "-p" in args
    assert "--output-format" in args and "stream-json" in args


# --------------------------------------------------------------------------- #
# outcome collection
# --------------------------------------------------------------------------- #


def _trace(ws, rows):
    with ws.trace_path.open("w", encoding="utf-8") as fh:
        for i, (action, level, state, frames) in enumerate(rows):
            fh.write(json.dumps({
                "i": i, "level": level, "action": action, "params": {},
                "frames": frames, "score": level, "state": state,
                "full_reset": False, "available_actions": ["ACTION1"],
            }) + "\n")


def test_outcome_is_read_from_the_ledger_not_from_any_claim(ws):
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 0, "NOT_FINISHED", [[[2]]]),
        ("ACTION1", 1, "NOT_FINISHED", [[[3]]]),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["levels_reached"] == 1
    assert out["levels_total"] == 3
    assert out["actions_used"] == 3
    assert out["won"] is False
    assert json.loads((ws.root / "result.json").read_text())["levels_reached"] == 1


def test_a_win_is_detected_from_the_state(ws):
    _trace(ws, [("RESET", 0, "NOT_FINISHED", [[[1]]]), ("ACTION1", 3, "WIN", [[[2]]])])
    assert collect_outcome(ws, exit_code=0, timed_out=False)["won"] is True


def test_deaths_count_episodes_not_frames(ws):
    """Three frames while dead is one death plus two wasted actions."""
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 0, "GAME_OVER", [[[2]]]),
        ("ACTION1", 0, "GAME_OVER", []),
        ("ACTION1", 0, "GAME_OVER", []),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["deaths"] == 1
    assert out["wasted_actions"] == 2


def test_an_empty_run_collects_without_crashing(ws):
    out = collect_outcome(ws, exit_code=1, timed_out=True)
    assert out["actions_used"] == 0 and out["timed_out"] is True


def test_the_rule_book_is_summarised_when_present(ws):
    _trace(ws, [("RESET", 0, "NOT_FINISHED", [[[1]]])])
    ws.rules_path.write_text(json.dumps({
        "verified": [{"rule": "a"}, {"rule": "b"}],
        "refuted": [{"rule": "c"}],
        "open_questions": [],
    }))
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["mechanics_recorded"] == 2
    assert out["refutations_recorded"] == 1


def test_bypass_permissions_is_downgraded_under_root(ws, monkeypatch):
    """--dangerously-skip-permissions is refused as root; the run dies empty."""
    import athanor.ccarc3.session as sess

    monkeypatch.setattr(sess, "resolve_permission_mode", lambda m: "acceptEdits")
    args = sess.build_cli_args(ws)
    assert "acceptEdits" in args
    assert "bypassPermissions" not in args
