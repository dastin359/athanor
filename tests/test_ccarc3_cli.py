"""Tests for the CCARC3 command line.

The reporting assertion is the one that matters: games span 171 to 1843 baseline
actions, so a pooled "levels solved" figure mostly reports which games happened
to be in the batch. Per-game rows, always.
"""

from __future__ import annotations

import json

import pytest

from athanor.ccarc3 import cli


def _result(game_id, **kw):
    base = {
        "game_id": game_id, "levels_reached": 2, "levels_total": 7,
        "won": False, "actions_used": 300, "baseline_total": 776,
        "deaths": 3, "wasted_actions": 0, "full_resets": 0,
        "exit_code": 0, "timed_out": False,
    }
    base.update(kw)
    return base


def test_report_reads_results_off_disk(tmp_path, capsys):
    for g in ("ls20", "cd82"):
        d = tmp_path / g
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_result(g)))
    assert cli.main(["report", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ls20" in out and "cd82" in out
    assert "4/14 levels reached" in out


def test_report_shows_a_row_per_game_not_just_a_pooled_number(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "result.json").write_text(json.dumps(_result("a", levels_reached=7, won=True)))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "result.json").write_text(json.dumps(_result("b", levels_reached=0)))
    cli.main(["report", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "7/7" in out and "0/7" in out
    assert "WON" in out


def test_report_flags_the_silent_failures(tmp_path, capsys):
    d = tmp_path / "x"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(
        _result("x", wasted_actions=12, full_resets=1, timed_out=True)))
    cli.main(["report", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "wasted=12" in out and "FULLRESET=1" in out and "TIMEOUT" in out


def test_a_failed_run_is_reported_and_excluded_from_totals(tmp_path, capsys):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "result.json").write_text(json.dumps(_result("ok")))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "result.json").write_text(json.dumps({"game_id": "bad", "error": "boom"}))
    cli.main(["report", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "ERROR" in out and "1 run(s) failed" in out
    assert "2/7 levels reached" in out, "the failed run must not dilute the total"


def test_report_on_an_empty_directory_says_so(tmp_path, capsys):
    assert cli.main(["report", "--out-dir", str(tmp_path)]) == 1
    assert "no finished runs" in capsys.readouterr().out


def test_trace_summarises_a_run(tmp_path, capsys):
    run = tmp_path / "r"
    run.mkdir()
    rows = [
        {"i": 0, "level": 0, "action": "RESET", "params": {}, "frames": [[[1]]],
         "score": 0, "state": "NOT_FINISHED", "full_reset": False, "available_actions": []},
        {"i": 1, "level": 0, "action": "ACTION1", "params": {}, "frames": [[[2]]],
         "score": 0, "state": "NOT_FINISHED", "full_reset": False, "available_actions": []},
        {"i": 2, "level": 1, "action": "ACTION1", "params": {}, "frames": [],
         "score": 1, "state": "GAME_OVER", "full_reset": False, "available_actions": []},
    ]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert cli.main(["trace", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "3 actions" in out
    assert "wasted (issued while dead): 1" in out


def test_trace_on_a_missing_run_fails_cleanly(tmp_path, capsys):
    assert cli.main(["trace", "--run", str(tmp_path)]) == 1


def test_the_parser_defaults_to_the_required_model_and_effort():
    args = cli.build_parser().parse_args(["run", "--game", "g"])
    assert args.model == "claude-opus-5"
    assert args.effort == "high"


def test_ccarc3_is_reachable_from_the_athanor_cli():
    from athanor.cli import build_parser

    args = build_parser().parse_args(["ccarc3", "run", "--game", "g"])
    assert args.command == "ccarc3"
    assert args.ccarc3_command == "run"
    assert args.game == "g"
