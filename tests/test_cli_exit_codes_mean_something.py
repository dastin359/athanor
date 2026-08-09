"""`cmd_run`'s exit code and `cmd_batch`'s failure record are what a caller sees.

Mutation found four untested decisions in `cli.py` at once -- more than the nine
other modules produced between them:

  * `cmd_run` always returning 0
  * its success threshold widened from `> 0` to `>= 0`
  * `cmd_batch` dropping a failed game instead of recording the error
  * its id parsing keeping blank tokens

The exit code is the whole interface for a shell caller: `0` means the solver
reached at least one level, `1` means it reached none. A run that returns 0
unconditionally reads as a win from every script that checks it, and the batch
summary is what tells an operator which games failed -- a dropped entry makes a
failed game indistinguishable from one never requested.

`run_game` is monkeypatched throughout: these assert what the CLI RETURNED and
RECORDED, not what the solver would have done.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from athanor.ccarc3 import cli as cli_mod


def _args(tmp_path: Path, **over) -> argparse.Namespace:
    base = dict(out_dir=str(tmp_path), model="claude-opus-5", effort="high",
                budget_multiple=4.0, timeout=60.0, fresh=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_a_run_that_reached_a_level_exits_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli_mod, "run_game",
                        lambda cfg: {"game_id": "g", "levels_reached": 1})
    assert cli_mod.cmd_run(_args(tmp_path, game="g")) == 0


def test_a_run_that_reached_nothing_exits_one(tmp_path, monkeypatch) -> None:
    """The load-bearing half: `return 0` unconditionally passes the case above."""
    monkeypatch.setattr(cli_mod, "run_game",
                        lambda cfg: {"game_id": "g", "levels_reached": 0})
    assert cli_mod.cmd_run(_args(tmp_path, game="g")) == 1, (
        "a run that reached no level exited 0 -- every shell caller that checks "
        "the status reads a total failure as a success"
    )


def test_a_missing_level_count_exits_one(tmp_path, monkeypatch) -> None:
    """A result without the field must not read as a win."""
    monkeypatch.setattr(cli_mod, "run_game", lambda cfg: {"game_id": "g"})
    assert cli_mod.cmd_run(_args(tmp_path, game="g")) == 1


def test_batch_records_a_failed_game_rather_than_dropping_it(
    tmp_path, monkeypatch, capsys
) -> None:
    """A dropped entry is indistinguishable from a game never requested."""
    seen = []

    def flaky(cfg):
        seen.append(cfg.game_id)
        if cfg.game_id == "bad":
            raise RuntimeError("boom")
        return {"game_id": cfg.game_id, "levels_reached": 2}

    monkeypatch.setattr(cli_mod, "run_game", flaky)
    captured = {}
    monkeypatch.setattr(cli_mod, "_summarise",
                        lambda results: captured.setdefault("rows", results))

    rc = cli_mod.cmd_batch(_args(tmp_path, games="good,bad"))
    assert rc == 0
    assert seen == ["good", "bad"], "a failure ended the batch early"
    rows = captured["rows"]
    assert len(rows) == 2, f"the failed game was dropped from the summary: {rows}"
    bad = next(r for r in rows if r["game_id"] == "bad")
    assert "boom" in bad.get("error", ""), (
        f"the failure was recorded without its cause: {bad}"
    )


def test_batch_skips_blank_ids(tmp_path, monkeypatch) -> None:
    """`--games "a,,b"` and a trailing comma must not queue an empty game."""
    seen = []
    monkeypatch.setattr(cli_mod, "run_game",
                        lambda cfg: seen.append(cfg.game_id) or
                        {"game_id": cfg.game_id, "levels_reached": 1})
    monkeypatch.setattr(cli_mod, "_summarise", lambda results: None)
    cli_mod.cmd_batch(_args(tmp_path, games=" a , , b ,"))
    assert seen == ["a", "b"], f"blank ids were queued as games: {seen}"
