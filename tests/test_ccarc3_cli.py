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


def test_report_re_derives_from_the_trace_and_corrects_a_stale_result(tmp_path, capsys):
    """`result.json` is what the run computed then; the trace is the record.

    A harness fix can make the stored file wrong after the fact. When full
    resets became detectable, every stored `ls20` figure still read 860 actions
    and zero full resets, so the batch summary rated a game won in 490 actions
    as if it had cost 860.
    """
    d = tmp_path / "g"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(
        _result("g", actions_used=999, levels_reached=0, full_resets=0, won=False)))
    with (d / "trace.jsonl").open("w") as fh:
        for i, (level, state) in enumerate([(0, "NOT_FINISHED"), (1, "NOT_FINISHED"),
                                            (0, "NOT_FINISHED"), (1, "WIN")]):
            fh.write(json.dumps({
                "i": i, "level": level, "action": "ACTION1", "params": {},
                "frames": [[[i]]], "score": level, "state": state,
                "full_reset": False, "available_actions": ["ACTION1"],
            }) + "\n")

    cli.main(["report", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "999" not in out, "the stale total must not survive"
    assert "WON" in out and "FULLRESET=1" in out
    assert "the last 2 of 4" in out


def test_report_still_works_for_a_run_whose_trace_is_gone(tmp_path, capsys):
    """Archived runs keep result.json; falling back beats reporting nothing."""
    d = tmp_path / "g"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_result("g", actions_used=42)))
    assert cli.main(["report", "--out-dir", str(tmp_path)]) == 0
    assert "42" in capsys.readouterr().out


def _run_dir(root, game, trace=None, **kw):
    """``trace`` is a list of levels, one per action, so a test can exercise the
    final-playthrough preference. Without it `_compare` falls back to the stored
    totals and the day's total-vs-final correction is never reached."""
    d = root / game
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(_result(game, **kw)))
    if trace is not None:
        with (d / "trace.jsonl").open("w") as fh:
            for i, level in enumerate(trace):
                fh.write(json.dumps({
                    "i": i, "level": level, "action": "ACTION1", "params": {},
                    "frames": [[[i]]], "score": level, "state": "NOT_FINISHED",
                    "full_reset": False, "available_actions": ["ACTION1"],
                }) + "\n")
    return d


def test_compare_uses_the_final_playthrough_on_both_sides(tmp_path, capsys):
    """Exercises the correction `_run_dir` alone cannot reach.

    The 'before' run climbed to level 3, was wiped, and ended on 1 having spent
    3 of its 6 actions after the reset. Reporting 3 levels against 3 actions
    would credit progress a full reset destroyed.
    """
    old, new = tmp_path / "old", tmp_path / "new"
    _run_dir(old, "g", trace=[0, 1, 2, 3, 0, 1], levels_total=6, baseline_total=100)
    _run_dir(new, "g", trace=[0, 1, 2], levels_total=6, baseline_total=100)
    cli.main(["report", "--out-dir", str(new), "--against", str(old)])
    out = capsys.readouterr().out
    assert "1->2/6" in out, "before ended on level 1, not the 3 it once reached"
    assert "3->2" not in out.split("paired against")[-1].split("\n")[2]


def test_report_pairs_a_game_against_an_earlier_batch(tmp_path, capsys):
    """The comparison written by hand three times in one session."""
    old, new = tmp_path / "old", tmp_path / "new"
    _run_dir(old, "cd82", levels_reached=0, levels_total=6, actions_used=337, baseline_total=171)
    _run_dir(new, "cd82", levels_reached=6, levels_total=6, actions_used=121,
             baseline_total=171, won=True)
    assert cli.main(["report", "--out-dir", str(new), "--against", str(old)]) == 0
    out = capsys.readouterr().out
    assert "0->6/6" in out and "337->121" in out
    assert "1.97x->0.71x" in out


def test_a_game_only_in_the_new_batch_is_named_not_dropped(tmp_path, capsys):
    old, new = tmp_path / "old", tmp_path / "new"
    _run_dir(old, "cd82")
    _run_dir(new, "cd82")
    _run_dir(new, "sb26")
    cli.main(["report", "--out-dir", str(new), "--against", str(old)])
    assert "unpaired: sb26" in capsys.readouterr().out


def test_comparing_against_a_directory_with_no_overlap_says_so(tmp_path, capsys):
    old, new = tmp_path / "old", tmp_path / "new"
    _run_dir(old, "aaaa")
    _run_dir(new, "bbbb")
    cli.main(["report", "--out-dir", str(new), "--against", str(old)])
    assert "nothing in" in capsys.readouterr().out


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


def test_levels_and_actions_describe_the_same_playthrough(tmp_path, capsys):
    """Pairing max-over-all-playthroughs levels with final-playthrough actions
    credits a run with progress a full reset destroyed, at the price of the
    progress that survived — the flattering half of each."""
    d = tmp_path / "g"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_result("g", levels_total=6)))
    with (d / "trace.jsonl").open("w") as fh:
        # reaches level 3, is wiped back to 0, then only gets to level 1
        for i, level in enumerate([0, 1, 2, 3, 0, 1]):
            fh.write(json.dumps({
                "i": i, "level": level, "action": "ACTION1", "params": {},
                "frames": [[[i]]], "score": level, "state": "NOT_FINISHED",
                "full_reset": False, "available_actions": ["ACTION1"],
            }) + "\n")

    cli.main(["report", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "1/6" in out, "the surviving playthrough reached level 1, not 3"
    assert "3/6" not in out, "the discarded playthrough's levels must not be credited"
    assert "FULLRESET=1" in out
